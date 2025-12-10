# LFM2-8B GPU 추론 파이프라인 개발 로그

---

## 전체 추론 파이프라인

```
ALGORITHM LFM2_Inference(input_ids, model_weights)
  INPUT:
    - input_ids: int32 배열 [batch_size × seq_len]
    - model_weights: 사전 로드된 GPU 가중치
  
  OUTPUT:
    - logits: float32 배열 [batch_size × seq_len × vocab_size]
  
  BEGIN
    hidden_states ← EmbeddingLookup(input_ids, embed_table)
    
    FOR stage = 0 TO 3 DO
      SetDevice(GPU[stage])
      
      FOR layer = stage*6 TO stage*6+5 DO
        normed ← RMSNorm(hidden_states)
        
        IF LAYER_TYPES[layer] == 0 THEN
          Q ← TiledGEMM(normed, W_q)
          K ← TiledGEMM(normed, W_k)
          V ← TiledGEMM(normed, W_v)
          
          Q, K ← ApplyRoPE(Q, K, position)
          attn_out ← FusedGQA(Q, K, V, causal_mask)
          layer_out ← TiledGEMM(attn_out, W_o)
          
        ELSE
          layer_out ← ShortConv(normed, conv_weight, conv_cache)
        END IF
        
        hidden_states ← hidden_states + layer_out
        normed ← RMSNorm(hidden_states)
        
        router_logits ← GEMM(normed, router_weight)
        top_k_experts, weights ← TopK(Softmax(router_logits), k=4)
        moe_out ← PersistentMoE(normed, top_k_experts, weights, expert_weights)
        hidden_states ← hidden_states + moe_out
      END FOR
      
      IF stage < 3 THEN
        EventRecord(transfer_event, compute_stream)
        StreamWaitEvent(next_gpu_stream, transfer_event)
        P2PTransfer(hidden_states, GPU[stage], GPU[stage+1])
      END IF
    END FOR
    
    hidden_states ← RMSNorm(hidden_states)
    logits ← LMHeadGEMM(hidden_states, lm_head_weight)
    
    RETURN logits
  END
```

---

---

12/01 (월): 프로젝트 초기 설정

먼저 모델 파라미터를 정의했다.

```cpp
#pragma once
#include <cstddef>

constexpr size_t VOCAB_SIZE = 65536;
constexpr size_t HIDDEN_SIZE = 2048;
constexpr size_t INTERMEDIATE_SIZE = 7168;
constexpr size_t NUM_HIDDEN_LAYERS = 24;
constexpr size_t NUM_ATTENTION_HEADS = 32;
constexpr size_t NUM_KEY_VALUE_HEADS = 8;  // GQA
constexpr size_t HEAD_DIM = HIDDEN_SIZE / NUM_ATTENTION_HEADS;  // 64
```

GQA(Grouped Query Attention)를 사용하기 때문에 KV heads가 8개뿐이다. Query heads 32개가 KV heads 8개를 공유한다.

MoE 파라미터도 추가:

```cpp
constexpr size_t NUM_EXPERTS = 32;
constexpr size_t NUM_EXPERTS_PER_TOK = 4;  // Top-4
constexpr size_t MOE_INTERMEDIATE_SIZE = 1792;
```

레이어 타입 배열 정의. 0=Attention, 1=Conv:

```cpp
constexpr int LAYER_TYPES[] = {
    1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1,
    1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1
};
```

24개 레이어 중 Attention이 6개(layer 2,6,10,14,18,21), 나머지 18개는 Conv.

---

---

12/04 (목): Tensor, 모델 로더 구현 및 헤더 정의

Tensor 클래스: `owns_data_` 플래그로 메모리 소유권 관리. Move semantics 지원.

ModelLoader: 텐서 인덱스 방식으로 offset 기반 로드. Row-major layout.

레이어 클래스: RMSNorm, Attention, ShortConv, SparseMoeBlock, DecoderLayer, LFM2Model

---

12/07 (일): RMSNorm, 기본 GEMM 커널 구현

성능 최적화 과정

시도 1: Naive GEMM - Global memory 접근이 병목

시도 2: Tiled GEMM (64x64x32) + Float4 벡터 로드 + Double Buffering
- Shared Memory 활용으로 Global Memory 접근 최소화
- Float4로 메모리 대역폭 효율 향상 (16바이트 정렬 필요)
- 다음 타일 미리 로드하면서 현재 타일 연산

GEMM 커널 (attn_matmul_tiled_kernel)

실제 구현된 최적화 GEMM:

```cuda
__global__ void attn_matmul_tiled_kernel(float* out, const float* A,
                                         const float* B, int M, int N, int K) {
  const int BM = 128, BN = 128, BK = 16;
  const int TM = 8, TN = 8;
  
  __shared__ float As[2][BM][BK + 4];
  __shared__ float Bs[2][BK][BN + 4];
  
  float threadResults[TM * TN] = {0.0f};
  float regM[TM], regN[TN];
  
  const float4* A_vec = reinterpret_cast<const float4*>(A);
  const float4* B_vec = reinterpret_cast<const float4*>(B);
  
  for (int i = 0; i < 2; ++i) {
    int r = load_a_row + i * 64;
    float4 loaded_a = A_vec[(global_row_a * K + global_col_a) / 4];
    As[0][r][load_a_col + 0] = loaded_a.x;
    As[0][r][load_a_col + 1] = loaded_a.y;
    As[0][r][load_a_col + 2] = loaded_a.z;
    As[0][r][load_a_col + 3] = loaded_a.w;
  }
  
  for (int k = 0; k < K; k += BK) {
    int cur_buf = (k / BK) % 2;
    int nxt_buf = ((k / BK) + 1) % 2;
    
    for (int kk = 0; kk < BK; ++kk) {
      for (int m = 0; m < TM; ++m) regM[m] = As[cur_buf][...][kk];
      for (int n = 0; n < TN; ++n) regN[n] = Bs[cur_buf][kk][...];
      for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n)
          threadResults[m * TN + n] += regM[m] * regN[n];
    }
    __syncthreads();
  }
}
```

핵심 최적화:
- Double buffering: 메모리 로드와 연산 오버랩
- Float4 vectorized load: 한 번에 128bit 읽기
- Register-level tiling: 각 스레드가 TM x TN 결과 계산
- Shared memory padding: Bank conflict 방지

---

12/07 (일): Attention, Conv, MoE 커널 완성

Attention 레이어에서 segfault 발생. 원인: 입력 텐서 shape (0, 32, 64) 초기화 누락.

misaligned address 에러. float4 로드 시 16바이트 정렬 필요.

```cpp
float4* ptr = reinterpret_cast<float4*>(x + offset);

float4* ptr = reinterpret_cast<float4*>(__builtin_assume_aligned(x + offset, 16));
```

float로 하나씩 로드하는 방식으로 변경.

RMSNorm 구현

수식:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma$$

여기서:
- $x$: 입력 벡터
- $n$: hidden dimension
- $\epsilon$: 수치 안정성을 위한 작은 값 (1e-5)
- $\gamma$: 학습된 가중치 (weight)

```cuda
__global__ void rms_norm_kernel(float *out, const float *in,
                                const float *weight, int total_rows,
                                int hidden_dim) {
    int row_idx = blockIdx.x;
    const float *src = in + row_idx * hidden_dim;
    float *dst = out + row_idx * hidden_dim;
    
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        sum_sq += src[i] * src[i];
    }
    
    // Warp Reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    
    // Block Reduction + RMS 계산 후 적용
}
```

RMSNorm 최적화 과정:

첫 번째 버전에서 reduce 구현이 잘못됐다:
```cuda
__shared__ float shared[256];
shared[threadIdx.x] = sum_sq;
__syncthreads();
if (threadIdx.x == 0) {
    for (int i = 1; i < 256; i++) sum_sq += shared[i];
}
```

Warp-level reduction으로 변경:
```cuda
for (int offset = 16; offset > 0; offset /= 2) {
    sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
}
```

RoPE 구현

RoPE (Rotary Position Embedding) 수식:
$$\text{RoPE}(x, m) = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} \otimes \begin{pmatrix} \cos(m\theta) \\ \cos(m\theta) \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \end{pmatrix} \otimes \begin{pmatrix} \sin(m\theta) \\ \sin(m\theta) \end{pmatrix}$$

구체적으로:
$$x'_d = x_d \cdot \cos(m\theta_d) - x_{d+\frac{D}{2}} \cdot \sin(m\theta_d)$$
$$x'_{d+\frac{D}{2}} = x_d \cdot \sin(m\theta_d) + x_{d+\frac{D}{2}} \cdot \cos(m\theta_d)$$

여기서:
- $m$: 위치 인덱스 (position)
- $\theta_d = \text{base}^{-2d/D}$, base=1000000 (ROPE_THETA)
- $D$: head dimension

부호 오류 수정:

```cuda
out[d + half] = x1 * sin_val + x2 * cos_val;

out[d]        = x1 * cos_val - x2 * sin_val;
out[d + half] = x1 * sin_val + x2 * cos_val;
```

Bank Conflict 문제 발견

Attention QK^T 계산에서 성능이 예상보다 낮았다.

```cuda
__shared__ float As[128][16];
float val = As[threadIdx.x][k];
```

Bank conflict 시 직렬화되어 32배 느려짐.

해결 시도들:

시도 1: Padding +1, +2 - 불충분

시도 2: Padding +4 - Bank conflict 해결
```cpp
constexpr int ATTN_PADDING = 4;
```
stride가 20이면 20 % 32 = 20, conflict 없음. 메모리 25% 증가지만 성능 이득이 더 큼.

Fused GQA Kernel 구현

Query-Key 연산, Softmax, Value 가중합을 하나의 커널로 통합:

```cuda
__global__ void fused_attn_gqa_kernel(float *output, const float *Q, const float *K, const float *V,
                                       int seq_len, int head_dim, int num_heads, int num_kv_heads, float scale) {
}
```

Softmax에서 NaN 발생:
```
Output contains NaN!
```

Softmax 수식:
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}$$

수치 안정성을 위한 안정화 버전:
$$\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n}e^{x_j - \max(x)}}$$

원인: exp(1000) = inf. max subtraction 추가로 해결:
```cuda
float max_val = -INFINITY;
for (int i = 0; i < seq_len; i++) max_val = fmaxf(max_val, scores[i]);
for (int i = 0; i < seq_len; i++) scores[i] = expf(scores[i] - max_val);
```

RoPE + Transpose 통합

위치 인코딩과 텐서 변환을 동시 수행하여 커널 런치 오버헤드 감소.

---

12/07 (일) 계속: MoE 커널 개발

MoE 메모리 문제: 8개 expert 출력 전부 할당 시 OOM. 해결: Top-k expert만 계산.

MoE 런타임 에러: expert_indices가 -1인 경우 처리 안 함. `if (expert >= 0)` 체크 추가.

scatter 로직 버그: 같은 토큰을 두 번 처리. 수정 완료.

Router 구현

```cuda
__global__ void moe_router_topk_kernel(
    int *expert_counts, int *topk_indices_out, float *topk_weights_out,
    const float *logits, const float *bias, int num_experts, int k_experts,
    int num_tokens, float routed_scale, bool use_bias) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tokens) return;
    
    float max_scores[4];
    int max_ids[4];
    
    // Top-k 선택 (삽입 정렬)
    for (int e = 0; e < num_experts; ++e) {
        float sum = logits[tid * num_experts + e];
        float prob = sigmoid(sum);
        float score = prob + (use_bias ? bias[e] : 0.0f);
        // 삽입 정렬로 top-k 유지
    }
}
```

Expert Load Imbalance 문제

Expert별 토큰 수 불균형이 심함. Persistent kernel 방식으로 해결.

Persistent Kernel 구현

```cuda
__global__ void moe_persistent_fused_w1w3_kernel(
    float *inter_out, const float *x, float **w1_ptrs, float **w3_ptrs,
    const int *tile_offsets, const int *offsets, const int *counts,
    int *task_counter, int hidden_size, int inter_size, int num_experts) {
    while (true) {
        __shared__ int task_id_s;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            task_id_s = atomicAdd(task_counter, 1);
        }
        __syncthreads();
        int task_id = task_id_s;
        
        int total_tiles = tile_offsets[num_experts];
        if (task_id >= total_tiles) break;
        
        // Binary search로 expert 찾기 후 처리
    }
}
```

deadlock 발생. `__syncthreads()` 위치 수정.

일부 스레드만 `__syncthreads()`에 도달해서 영원히 대기. 조건문 밖으로 이동.

동적 작업 분배

atomicAdd로 task_counter를 관리하여 Expert별 로드 밸런싱.

W1/W3 Fusion

MoE에서 W1, W3가 같은 입력을 사용:

```cuda
y1 = matmul(x, W1);
y3 = matmul(x, W3);
```

Fusion 시도들:

시도 1: 단순 연결/스트림 연속 실행 - 오버헤드로 느려짐

시도 2: 커널 내부에서 두 output 동시 계산 - 채택

---

12/07 (일) 계속: 멀티GPU 파이프라인 구현

Multi-GPU 메모리 문제: 각 GPU에 동일한 버퍼 할당 시 OOM. 해결: 레이어 균등 분배 (각 GPU에 6개씩).

Multi-GPU 런타임 에러: GPU 1,2,3 출력이 all zeros. 원인: 스트림 동기화 문제.

```cpp
cudaMemcpyAsync(gpu1_input, gpu0_output, size, cudaMemcpyDeviceToDevice, stream);
kernel<<<...>>>(gpu1_input, ...);

cudaEventRecord(event, transfer_stream);
cudaStreamWaitEvent(compute_stream, event);
kernel<<<...>>>(gpu1_input, ...);
```

레이어 분배

```
GPU 0: Layer 0-5   (Embed + 초기 레이어)
GPU 1: Layer 6-11
GPU 2: Layer 12-17
GPU 3: Layer 18-23 (+ LM Head)
```

P2P 통신 설정

```cpp
void setup_p2p() {
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < NUM_GPUS; j++) {
            if (i != j) {
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }
}
```

P2P 활성화 필요. `cudaDeviceEnablePeerAccess` 추가.

Double Stream으로 compute와 transfer 오버랩. `NUM_STREAMS_PER_STAGE = 2`

Embedding Lookup 커널

```cuda
__global__ void embedding_lookup_kernel(float *hidden_states, const int *input_ids,
                                        const float *embed_table, int seq_len, int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = seq_len * hidden_size;
  if (idx < total_elements) {
    int token_idx = idx / hidden_size;
    int dim_idx = idx % hidden_size;
    int token_id = input_ids[token_idx];
    hidden_states[idx] = embed_table[token_id * hidden_size + dim_idx];
  }
}
```

LM Head GEMM 커널

```cuda
__global__ void lm_head_gemm_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
  const int BM = GEMM_TILE_M, BN = GEMM_TILE_N, BK = GEMM_TILE_K;
  
  float acc[GEMM_ROWS_PER_THREAD][GEMM_COLS_PER_THREAD] = {0.0f};
  __shared__ float Bs[BK][BN];
  __shared__ float As[BM][BK];
  
  for (int k = 0; k < K; k += BK) {
    __syncthreads();
    
    for (int kk = 0; kk < BK; ++kk) {
      for (int r = 0; r < GEMM_ROWS_PER_THREAD; ++r) {
        float a_val = As[ty * GEMM_ROWS_PER_THREAD + r][kk];
        acc[r][0] = fmaf(a_val, b_val0, acc[r][0]);
        acc[r][1] = fmaf(a_val, b_val1, acc[r][1]);
      }
    }
    __syncthreads();
  }
}
```

출력 logits 계산용. `fmaf` (fused multiply-add) 사용으로 정밀도 향상.

배치 크기 실험 결과 BUFFER_SEQ_LEN=32 선택 (메모리와 성능 균형).

`fmaf` vs 일반 연산 비교:
```cuda
acc = acc + a * b;

acc = fmaf(a, b, acc);
```

장점:
- 연산 1개로 줄어 throughput 향상
- Rounding error 감소로 수치 정밀도 향상
- 대부분의 GPU에서 동일 latency

메모리 최적화

OOM 해결 방법:
- 전체 모델 로드: 레이어별 순차 로드로 변경
- K/V 캐시 크기: 실제 사용할 seq_len으로 축소
- Double buffering: 버퍼 크기 조정 (48KB 제한)
- MoE expert 병렬: 2개씩 나눠서 실행
- Multi-GPU 버퍼: 필요한 stage만 할당

가중치 사전 로드: init_gpu()에서 모든 가중치를 GPU로 미리 전송.

버퍼 사전 할당: 중간 결과 버퍼를 미리 할당하여 런타임 cudaMalloc 오버헤드 제거.

---

12/10 (수): 빌드 및 테스트 완료

12/10 (수) 계속: 튜닝 및 문서 시도

파라미터 확정

```cpp
constexpr int GEMM_TILE_M = 64;
constexpr int GEMM_TILE_N = 64;
constexpr int GEMM_TILE_K = 32;

constexpr int ATTN_TILE_M = 128;
constexpr int ATTN_TILE_N = 128;
constexpr int ATTN_TILE_K = 16;
constexpr int ATTN_THREAD_TILE_M = 8;
constexpr int ATTN_THREAD_TILE_N = 8;
constexpr int ATTN_PADDING = 4;

constexpr int NUM_GPUS = 4;
constexpr int NUM_STREAMS_PER_STAGE = 2;

constexpr int BUFFER_SEQ_LEN = 32;
```

---

문제 해결 기록 (실패 로그)

주요 실패 원인:

1. 런타임 에러 - segfault, illegal memory access, misaligned address
2. 결과 오류 - 계산 버그, 인덱싱 버그, 초기화 누락
3. 동기화 문제 - race condition, deadlock, 스트림 동기화

세부 기록:

- RoPE 부호 오류 - 수식 검토로 발견
- Bank Conflict - 프로파일링으로 발견
- MoE Deadlock - `__syncthreads()` 위치 문제
- P2P 실패 - `cudaDeviceEnablePeerAccess` 누락
- Softmax Overflow - NaN 발생. exp(1000) = inf 때문
- Race Condition - 결과가 실행마다 달라서 발견. atomicAdd로 해결
- 스트림 동기화 - GPU 간 데이터 전송 문제
- Shared memory 초과 - 타일 크기 조정

---

파일 구조

```
├── include/
│   ├── config.h         # 모델 파라미터 정의
│   ├── tensor.h         # Tensor 클래스 선언
│   ├── layer.h          # 레이어 클래스 선언 (RMSNorm, Attention, Conv, MoE)
│   ├── model.h          # LFM2Model 클래스 선언
│   └── model_loader.h   # ModelLoader 클래스 선언
├── src/
│   ├── tensor.cu        # Tensor 구현
│   ├── layer.cu         # CUDA 커널 구현 (GEMM, Attention, MoE 등)
│   ├── model.cu         # 모델 forward, GPU 파이프라인
│   ├── model_loader.cpp # 모델 가중치 로딩
│   └── main.cpp         # 메인 진입점, 벤치마크
├── obj/                 # 컴파일된 오브젝트 파일
├── tests/
│   ├── attn/            # Attention 유닛 테스트
│   ├── conv/            # Conv 유닛 테스트
│   └── moe/             # MoE 유닛 테스트
├── data/
│   ├── inputs.bin       # 테스트 입력
│   ├── outputs.bin      # 모델 출력
│   └── answers.bin      # 정답
├── Makefile             # 빌드 스크립트
└── main                 # 실행 파일
```

---

성능 결과: 검증 완료 (1024/1024 정확)

개발 기간: 2025년 12월 1일 ~ 12월 10일
