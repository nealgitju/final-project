#include "layer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

__global__ void attn_matmul_tiled_kernel(float* out, const float* A,
                                         const float* B, int M, int N, int K) {
  const int BM = ATTN_TILE_M;
  const int BN = ATTN_TILE_N;
  const int BK = ATTN_TILE_K;
  const int TM = ATTN_THREAD_TILE_M;
  const int TN = ATTN_THREAD_TILE_N;
  int bx = blockIdx.x, by = blockIdx.y;
  int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0 ~ 255

  __shared__ float As[2][BM][BK + ATTN_PADDING];
  __shared__ float Bs[2][BK][BN + ATTN_PADDING];

  float threadResults[TM * TN] = {0.0f};
  float regM[TM];
  float regN[TN];

  int load_a_row = tid / (BK / 4); 
  int load_a_col = (tid % (BK / 4)) * 4; 
  
  int load_b_row = tid / (BK / 4); 
  int load_b_col = (tid % (BK / 4)) * 4; 

  const float4* A_vec = reinterpret_cast<const float4*>(A);
  const float4* B_vec = reinterpret_cast<const float4*>(B);

  {
      int k = 0;
      #pragma unroll
      for(int i = 0; i < 2; ++i) {
          int r = load_a_row + i * 64;
          int global_row_a = by * BM + r;
          int global_col_a = k + load_a_col;
          float4 loaded_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          if (global_row_a < M && global_col_a < K) {
             loaded_a = A_vec[(global_row_a * K + global_col_a) / 4];
          }
          As[0][r][load_a_col + 0] = loaded_a.x;
          As[0][r][load_a_col + 1] = loaded_a.y;
          As[0][r][load_a_col + 2] = loaded_a.z;
          As[0][r][load_a_col + 3] = loaded_a.w;
      }

      #pragma unroll
      for(int i = 0; i < 2; ++i) {
          int r = load_b_row + i * 64; 
          int global_row_b = bx * BN + r;
          int global_col_b = k + load_b_col;
          float4 loaded_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          if (global_row_b < N && global_col_b < K) {
             loaded_b = B_vec[(global_row_b * K + global_col_b) / 4];
          }          
          Bs[0][load_b_col + 0][r] = loaded_b.x;
          Bs[0][load_b_col + 1][r] = loaded_b.y;
          Bs[0][load_b_col + 2][r] = loaded_b.z;
          Bs[0][load_b_col + 3][r] = loaded_b.w;
      }
  }
    __syncthreads();

  // Main Loop
  for (int k = 0; k < K; k += BK) {
    int cur_buf = (k / BK) % 2;
    int nxt_buf = ((k / BK) + 1) % 2;
    int next_k = k + BK;

    // Prefetch Next Tile
    if (next_k < K) {
        #pragma unroll
        for(int i=0; i<2; ++i) {
          int r = load_a_row + i * 64;
          int global_row_a = by * BM + r;
          int global_col_a = next_k + load_a_col;
          float4 loaded_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          if (global_row_a < M && global_col_a < K) {
             loaded_a = A_vec[(global_row_a * K + global_col_a) / 4];
          }
          As[nxt_buf][r][load_a_col + 0] = loaded_a.x;
          As[nxt_buf][r][load_a_col + 1] = loaded_a.y;
          As[nxt_buf][r][load_a_col + 2] = loaded_a.z;
          As[nxt_buf][r][load_a_col + 3] = loaded_a.w;
        }

        #pragma unroll
        for(int i=0; i<2; ++i) {
          int r = load_b_row + i * 64;
          int global_row_b = bx * BN + r;
          int global_col_b = next_k + load_b_col;
          float4 loaded_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          if (global_row_b < N && global_col_b < K) {
             loaded_b = B_vec[(global_row_b * K + global_col_b) / 4];
          }
          Bs[nxt_buf][load_b_col + 0][r] = loaded_b.x;
          Bs[nxt_buf][load_b_col + 1][r] = loaded_b.y;
          Bs[nxt_buf][load_b_col + 2][r] = loaded_b.z;
          Bs[nxt_buf][load_b_col + 3][r] = loaded_b.w;
        }
    }

    #pragma unroll
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      #pragma unroll
      for (int i = 0; i < TM; ++i) {
          regM[i] = As[cur_buf][threadIdx.y * TM + i][dotIdx];
      }
      
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
          regN[j] = Bs[cur_buf][dotIdx][threadIdx.x * TN + j];
      }

      #pragma unroll
      for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
           threadResults[i*TN + j] += regM[i] * regN[j];
        }
      }
    }
    
        __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      int globalRow = by * BM + threadIdx.y * TM + i;
      int globalCol = bx * BN + threadIdx.x * TN + j;
      if (globalRow < M && globalCol < N) {
        out[globalRow * N + globalCol] = threadResults[i * TN + j];
      }
    }
  }
}
  

__global__ void rms_norm_kernel(float *out, const float *in,
                                const float *weight, int total_rows,
                                int hidden_dim) {
  int row_idx = blockIdx.x;
  int tid = threadIdx.x;

  if (row_idx >= total_rows)
    return;

  const float *src = in + row_idx * hidden_dim;
  float *dst = out + row_idx * hidden_dim;

  float sum_sq = 0.0f;
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    sum_sq += src[i] * src[i];
  }

  // Warp Reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

  static __shared__ float shared_sum[32];
  int lane = tid % warpSize;
  int wid = tid / warpSize;
  if (lane == 0)
    shared_sum[wid] = sum_sq;
        __syncthreads();

  if (tid < (blockDim.x / warpSize)) {
    sum_sq = shared_sum[tid];
  } else {
    sum_sq = 0.0f;
  }

  if (wid == 0) {
    for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2)
      sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
  }
  __syncthreads();

  if (tid == 0)
    shared_sum[0] = sum_sq;
  __syncthreads();

  float rms = 1.0f / sqrtf(shared_sum[0] / hidden_dim + 1e-5f);

  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    dst[i] = src[i] * rms * weight[i];
  }
}

__global__ void rope_transpose_kernel(float *out, const float *in,
                                      const float *cos, const float *sin,
                                      int batch, int seq_len, int num_heads,
                                      int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half_dim = head_dim / 2;
  int total = batch * num_heads * seq_len * half_dim;

  if (idx < total) {
    int d = idx % half_dim;
    int rem = idx / half_dim;
    int s = rem % seq_len;
    rem = rem / seq_len;
    int h = rem % num_heads;
    int b = rem / num_heads;

    int in_idx = b * seq_len * num_heads * head_dim + s * num_heads * head_dim +
                 h * head_dim;
    float q1 = in[in_idx + d];
    float q2 = in[in_idx + d + half_dim];

    // CPU Random compatible load
    float c1 = cos[s * head_dim + d];
    float c2 = cos[s * head_dim + d + half_dim];
    float s1 = sin[s * head_dim + d];
    float s2 = sin[s * head_dim + d + half_dim];

    int out_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim +
                  s * head_dim;

    out[out_idx + d] = q1 * c1 - q2 * s1;
    out[out_idx + d + half_dim] = q2 * c2 + q1 * s2;
  }
}

__global__ void transpose_v_kernel(float *out, const float *in, int batch,
                                   int seq_len, int num_heads, int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * seq_len * num_heads * head_dim;

  if (idx < total) {
    int d = idx % head_dim;
    int rem = idx / head_dim;
    int h = rem % num_heads;
    rem = rem / num_heads;
    int s = rem % seq_len;
    int b = rem / seq_len;

    int in_idx = b * seq_len * num_heads * head_dim + s * num_heads * head_dim +
                 h * head_dim + d;
    int out_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim +
                  s * head_dim + d;

    out[out_idx] = in[in_idx];
  }
}

#define AT_WARP_SIZE 32
__device__ inline float warpReduceSum(float val) {
  for (int offset = AT_WARP_SIZE / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void fused_attn_gqa_kernel(float *__restrict__ output,
                                      const float *__restrict__ Q,
                                      const float *__restrict__ K,
                                      const float *__restrict__ V, int seq_len,
                                      int head_dim, int num_heads,
                                      int num_kv_heads, float scale) {
  int b = blockIdx.x;
  int h_q = blockIdx.y;
  int tid_x = threadIdx.x; // 0 ~ 31
  int tid_y = threadIdx.y; // 0 ~ 15

  int group_size = num_heads / num_kv_heads;
  int h_kv = h_q / group_size;

  extern __shared__ float smem[];
  float *smem_K = smem;
  float *smem_V = smem + seq_len * head_dim;

  int lin_idx = tid_y * 32 + tid_x;
  int total_elements = seq_len * head_dim;

  // Load K (Shared for the whole group) using float4
  const float4* K_vec = reinterpret_cast<const float4*>(K);
  float4* smem_K_vec = reinterpret_cast<float4*>(smem_K);
  
  int num_float4 = total_elements / 4;
  for (int i = lin_idx; i < num_float4; i += 512) {   
      int s = i / (head_dim / 4);
      int d_vec = i % (head_dim / 4);
      
      int k_offset_vec = (b * num_kv_heads * seq_len + h_kv * seq_len + s) * (head_dim / 4) + d_vec;
      smem_K_vec[i] = K_vec[k_offset_vec];
  }

  // Load V using float4
  const float4* V_vec = reinterpret_cast<const float4*>(V);
  float4* smem_V_vec = reinterpret_cast<float4*>(smem_V);

  for (int i = lin_idx; i < num_float4; i += 512) {
      int s = i / (head_dim / 4);
      int d_vec = i % (head_dim / 4);
      
      int v_offset_vec = (b * num_kv_heads * seq_len + h_kv * seq_len + s) * (head_dim / 4) + d_vec;
      smem_V_vec[i] = V_vec[v_offset_vec];
  }
  __syncthreads();

  int q_offset = b * num_heads * seq_len * head_dim + h_q * seq_len * head_dim +
                 tid_y * head_dim + tid_x;
  float q_val[2];
  q_val[0] = Q[q_offset];
  q_val[1] = Q[q_offset + 32];

  float logits[16];
  float max_val = -1e30f;

  for (int j = 0; j < seq_len; ++j) {
    float part = q_val[0] * smem_K[j * head_dim + tid_x] +
                 q_val[1] * smem_K[j * head_dim + tid_x + 32];
    float score = warpReduceSum(part);

    if (tid_x == 0) {
      score *= scale;
      if (j > tid_y)
        score = -1e30f;
      logits[j] = score;
      if (score > max_val)
        max_val = score;
    }
  }

  float inv_sum = 0.0f;
  if (tid_x == 0) {
    double sum_exp = 0.0;
    for (int j = 0; j < seq_len; ++j) {
      // Use double precision exp
      float val = (float)exp((double)(logits[j] - max_val));
      logits[j] = val;
      sum_exp += (double)val;
    }
    inv_sum = (float)(1.0 / sum_exp);
  }

  float out_val[2] = {0.0f, 0.0f};
  for (int j = 0; j < seq_len; ++j) {
    float prob =
        __shfl_sync(0xffffffff, (tid_x == 0 ? logits[j] * inv_sum : 0.0f), 0);
    out_val[0] += prob * smem_V[j * head_dim + tid_x];
    out_val[1] += prob * smem_V[j * head_dim + tid_x + 32];
  }

  int out_idx = b * seq_len * num_heads * head_dim +
                tid_y * num_heads * head_dim + h_q * head_dim + tid_x;
  output[out_idx] = out_val[0];
  output[out_idx + 32] = out_val[1];
}

// ============================================================================
// KERNELS - CONV
// ============================================================================

__global__ void conv_transpose_split_kernel(float *Bx, float *C,
                                            const float *in_proj, int batch,
                                            int seq_len, int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * seq_len * hidden_size;

  if (idx < total) {
    int h = idx % hidden_size;
    int rem = idx / hidden_size;
    int s = rem % seq_len;
    int b = rem / seq_len;

    int out_idx = b * hidden_size * seq_len + h * seq_len + s;
    int base_in = b * seq_len * 3 * hidden_size + s * 3 * hidden_size;

    float val_B = in_proj[base_in + h];
    float val_C = in_proj[base_in + hidden_size + h];
    float val_gate = in_proj[base_in + 2 * hidden_size + h];

    Bx[out_idx] = val_B * val_gate;
    C[out_idx] = val_C;
  }
}

__global__ void causal_conv1d_kernel(float *out, const float *input,
                                     const float *weight, int batch,
                                     int hidden_size, int seq_len,
                                     int kernel_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * hidden_size * seq_len;

  if (idx < total) {
    int s = idx % seq_len;
    int rem = idx / seq_len;
    int h = rem % hidden_size;
    int b = rem / hidden_size;

    float sum = 0.0f;
    int input_base = b * hidden_size * seq_len + h * seq_len;
    int weight_base = h * kernel_size;

    for (int k = 0; k < kernel_size; ++k) {
      int pos = s - (kernel_size - 1) + k;
      if (pos >= 0) {
        sum += input[input_base + pos] * weight[weight_base + k];
      }
    }
    out[idx] = sum;
  }
}

__global__ void conv_gating_transpose_kernel(float *out, const float *C,
                                             const float *conv_out, int batch,
                                             int hidden_size, int seq_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * hidden_size * seq_len;

  if (idx < total) {
    int s = idx % seq_len;
    int rem = idx / seq_len;
    int h = rem % hidden_size;
    int b = rem / hidden_size;

    int in_idx = b * hidden_size * seq_len + h * seq_len + s;
    float val = C[in_idx] * conv_out[in_idx];

    int out_idx = b * seq_len * hidden_size + s * hidden_size + h;
    out[out_idx] = val;
  }
}

// ============================================================================
// KERNELS - MOE
// ============================================================================

__device__ inline float d_sigmoid(float x) { return 1.0f / (1.0f + (float)exp(-(double)x)); }
__device__ inline float d_silu(float x) { return x / (1.0f + (float)exp(-(double)x)); }

// 어떤 토큰을 어떤 Expert로 보낼지 (Top-4)
__global__ void moe_router_topk_kernel(
    int *__restrict__ expert_counts, int *__restrict__ topk_indices_out,
    float *__restrict__ topk_weights_out, const float *__restrict__ logits,
    const float *__restrict__ bias, int num_experts, int k_experts,
    int num_tokens, float routed_scale, bool use_bias) {
  // 1 Thread -> 1 Token 
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_tokens)
    return;

  float max_scores[4]; // Score
  float max_probs[4];  // Prob
  int max_ids[4];

  for (int i = 0; i < 4; ++i) {
    max_scores[i] = -1e30f;
    max_probs[i] = 0.0f;
    max_ids[i] = -1;
  }

  for (int e = 0; e < num_experts; ++e) {
    float sum = logits[tid * num_experts + e]; // GEMM으로 점수계산
    float prob = d_sigmoid(sum); // Sigmoid 함수로 확률 계산
    float score = prob + (use_bias ? bias[e] : 0.0f); // bias 추가

    // 삽입 정렬
    if (score > max_scores[3]) { // 4등보다 점수가 높으면
      int pos = 3;
      while (pos > 0 && score > max_scores[pos - 1]) { // 위치 이동
        max_scores[pos] = max_scores[pos - 1];
        max_probs[pos] = max_probs[pos - 1];
        max_ids[pos] = max_ids[pos - 1];
        pos--;
      } // 삽입
      max_scores[pos] = score;
      max_probs[pos] = prob;
      max_ids[pos] = e;
    }
  }

  // Normalize
  float sum_prob = 0.0f;
  for (int i = 0; i < k_experts; ++i)
    sum_prob += max_probs[i];

  float scale = (sum_prob > 1e-6f) ? (1.0f / sum_prob) : 1.0f;

  // Store
  for (int i = 0; i < k_experts; ++i) {
    int expert_id = max_ids[i];
    float weight = max_probs[i] * scale * routed_scale;

    topk_indices_out[tid * k_experts + i] = expert_id;
    topk_weights_out[tid * k_experts + i] = weight;

    if (expert_id >= 0 && expert_id < num_experts) {
      // Race Condition 방지(특정 Expert count++)
      atomicAdd(&expert_counts[expert_id], 1);
    }
  }
}
// 메모리 상 데이터 재배치 (Expert별로 데이터를 모아서 GEMM)
__global__ void moe_scatter_kernel(
    float *__restrict__ sorted_x, int *__restrict__ sorted_idx_map,
    float *__restrict__ sorted_weights, int *__restrict__ expert_offsets,
    int *__restrict__ token_expert_pos, 
    const float *__restrict__ x, const int *__restrict__ topk_indices,
    const float *__restrict__ topk_weights, int hidden_size, int k_experts,
    int num_tokens) {
  
  // 1 Block = 1 Token
  int tid = blockIdx.x; 
  if (tid >= num_tokens) return;

  int lane = threadIdx.x; // Copy Worker

  // Shared memory to broadcast write_pos
  __shared__ int write_pos_s;

  for (int k = 0; k < k_experts; ++k) {
    if (lane == 0) {
        int expert_id = topk_indices[tid * k_experts + k];
        float weight = topk_weights[tid * k_experts + k];
        
        int wp = atomicAdd(&expert_offsets[expert_id], 1);
        write_pos_s = wp;

        sorted_idx_map[wp] = tid;
        sorted_weights[wp] = weight;
        token_expert_pos[tid * k_experts + k] = wp;
    }
    __syncthreads();

    int write_pos = write_pos_s;
    int src_base = tid * hidden_size;
    int dst_base = write_pos * hidden_size;

    // Parallel Copy
    for (int i = lane; i < hidden_size; i += blockDim.x) {
        sorted_x[dst_base + i] = x[src_base + i];
    }
    __syncthreads();
  }
}

__global__ void moe_gemm_kernel(
    float* __restrict__ C, 
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    int M, int N, int K) {
    
    // Tiling Parameters
    // A: MxK, B: NxK (Weights), C: MxN = A @ B^T
    const int BM = 64;
    const int BN = 64;
    const int BK = 16;
    const int TM = 4;
    const int TN = 4;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x; // 0..255

    // Thread Mapping: 256 threads compute 64x64 output.
    int ty = tid / 16; // 0..15
    int tx = tid % 16; // 0..15

    int row_start = ty * TM; 
    int col_start = tx * TN; 

    // Double Buffering: [2]
    __shared__ float As[2][BM][BK]; 
    __shared__ float Bs[2][BN][BK]; 

    float accum[TM][TN] = {0.0f};
    float regM[TM]; 
    float regN[TN]; 

    // Prologue (Load Tile 0)
    int load_idx = tid * 4;
    
    {
       int k=0;
       // Load A
       #pragma unroll
       for(int i=0; i<4; ++i) {
           int cur = load_idx + i;
           int r = cur / BK; // 0..63
           int c = cur % BK; // 0..15
           int global_r = by * BM + r;
           int global_c = k + c;
           if(global_r < M && global_c < K) As[0][r][c] = A[global_r * K + global_c];
           else As[0][r][c] = 0.0f;
       }
       // Load B
       #pragma unroll
       for(int i=0; i<4; ++i) {
           int cur = load_idx + i; 
           int r = cur / BK; 
           int c = cur % BK; 
           int global_r = bx * BN + r;
           int global_c = k + c;
           if(global_r < N && global_c < K) Bs[0][r][c] = B[global_r * K + global_c];
           else Bs[0][r][c] = 0.0f;
       }
    }
    __syncthreads();

    // Main Loop
    for (int k = 0; k < K; k += BK) {
        int cur = (k/BK)%2;
        int nxt = ((k/BK)+1)%2;
        int next_k = k + BK;

        // Prefetch Next (if exists)
        if (next_k < K) {
           // Load A
           #pragma unroll
           for(int i=0; i<4; ++i) {
               int cur_idx = load_idx + i;
               int r = cur_idx / BK; 
               int c = cur_idx % BK; 
               int global_r = by * BM + r;
               int global_c = next_k + c;
               if(global_r < M && global_c < K) As[nxt][r][c] = A[global_r * K + global_c];
               else As[nxt][r][c] = 0.0f;
           }
           // Load B
           #pragma unroll
           for(int i=0; i<4; ++i) {
               int cur_idx = load_idx + i;
               int r = cur_idx / BK;
               int c = cur_idx % BK;
               int global_r = bx * BN + r;
               int global_c = next_k + c;
               if(global_r < N && global_c < K) Bs[nxt][r][c] = B[global_r * K + global_c];
               else Bs[nxt][r][c] = 0.0f;
           }
        }

        // Compute
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for(int i=0; i<TM; ++i) regM[i] = As[cur][row_start+i][kk];
            #pragma unroll
            for(int j=0; j<TN; ++j) regN[j] = Bs[cur][col_start+j][kk];
            
            #pragma unroll
            for(int i=0; i<TM; ++i)
              #pragma unroll
              for(int j=0; j<TN; ++j)
                 accum[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    // Store
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_row = by * BM + row_start + i;
            int global_col = bx * BN + col_start + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = accum[i][j];
            }
        }
    }
}

__global__ void moe_swiglu_kernel(float *interm, const float *w1_out,
                                  const float *w3_out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int main_dim = size / 4;
  
  if (idx < main_dim) {
      float4 v1 = reinterpret_cast<const float4*>(w1_out)[idx];
      float4 v3 = reinterpret_cast<const float4*>(w3_out)[idx];
      float4 res;
      res.x = d_silu(v1.x) * v3.x;
      res.y = d_silu(v1.y) * v3.y;
      res.z = d_silu(v1.z) * v3.z;
      res.w = d_silu(v1.w) * v3.w;
      reinterpret_cast<float4*>(interm)[idx] = res;
  }
  
  // Handle tail
  if (idx == 0) {
      for(int i = main_dim * 4; i < size; ++i) {
         interm[i] = d_silu(w1_out[i]) * w3_out[i];
      }
  }
}

// Gather (원래 순서대로 Reduce)
__global__ void moe_gather_deterministic_kernel(
    float *__restrict__ output, const float *__restrict__ sorted_out,
    const int *__restrict__ token_expert_pos,
    const float *__restrict__ topk_weights, int num_tokens, int hidden_size,
    int k_experts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Total threads = num_tokens * hidden_size 쓰레드 당 하나
  if (idx >= num_tokens * hidden_size)
    return;

  int token_idx = idx / hidden_size;
  int col = idx % hidden_size;

  float sum = 0.0f;
  for (int k = 0; k < k_experts; ++k) {
    // Scatter 때 기록해둔 위치
    int read_pos_row = token_expert_pos[token_idx * k_experts + k];
    
    // Expert가 처리한 결과값 Read
    float val = sorted_out[read_pos_row * hidden_size + col];
    float weight = topk_weights[token_idx * k_experts + k];
    // Weighted Sum
    sum += val * weight;
  }
  
  output[idx] = sum;
}

// ============================================================================
// MOE OPTIMIZATIONS
// ============================================================================

__device__ inline float d_sigmoid_opt(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float d_silu_opt(float x) { return x / (1.0f + expf(-x)); }
// workload 분배 스케줄링
__global__ void moe_scan_kernel(int *offsets, int *write_pos, int *tile_offsets,
                                int *task_counters, const int *counts,
                                int num_experts, int tile_size) {
  // warp 1개만 (32 threads)
  int tid = threadIdx.x;
  if (tid >= 32) return;

  int count = (tid < num_experts) ? counts[tid] : 0;
  int tiles = (count + tile_size - 1) / tile_size;

  int val_off = count; // 토큰 누적합용
  int val_tile = tiles; // 타일 누적합용
  // Warp Shuffle: 메모리 X. Register끼리 연산
  #pragma unroll
  for (int i = 1; i < 32; i *= 2) {
    int n_off = __shfl_up_sync(0xffffffff, val_off, i);
    int n_tile = __shfl_up_sync(0xffffffff, val_tile, i);
    if (tid >= i) {
        val_off += n_off; // 모든 Expert의 토큰 합
        val_tile += n_tile; // 모든 Expert의 타일 합
    }
  }
  // 결과 저장 및 초기화
  if (tid < num_experts) {
    int start = val_off - count;
    offsets[tid] = start;
    write_pos[tid] = start; 
    tile_offsets[tid] = val_tile - tiles;
  }
  
  // Sentinel for binary search
  if (tid == num_experts - 1) {
    // 다음 커널에서 몇 번째 타일을 처리할지 알려주는 값
    tile_offsets[num_experts] = val_tile;
    // 초기화
    task_counters[0] = 0;
    task_counters[1] = 0;
  }
}

// SiLU(XW1) * W3X 계산
// SiLU(XW1) * W3X 계산
// Optimized: Vectorized Loads (float4) + Double Buffering
__global__ void moe_persistent_fused_w1w3_kernel(
    float *__restrict__ inter_out, 
    const float *__restrict__ x,   
    float **__restrict__ w1_ptrs, 
    float **__restrict__ w3_ptrs,
    const int *__restrict__ tile_offsets, 
    const int *__restrict__ offsets,
    const int *__restrict__ counts,
    int *__restrict__ task_counter,
    int hidden_size, int inter_size, int num_experts) 
{
    while (true) {
        __shared__ int task_id_s;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            task_id_s = atomicAdd(task_counter, 1);
        }
        __syncthreads();
        int task_id = task_id_s;

        int total_tiles = tile_offsets[num_experts];
        if (task_id >= total_tiles) break;

        // Binary search
        int e = 0;
        if (task_id >= tile_offsets[16]) e = 16;
        if (task_id >= tile_offsets[e+8]) e += 8;
        if (task_id >= tile_offsets[e+4]) e += 4;
        if (task_id >= tile_offsets[e+2]) e += 2;
        if (task_id >= tile_offsets[e+1]) e += 1;
        
        int tile_idx = task_id - tile_offsets[e];
        int count = counts[e];
        // 64x64 Tiling
        int start_row = tile_idx * 64; 
        int num_rows = min(64, count - start_row);
        
        int global_row_offset = offsets[e] + start_row;
        const float *in_ptr = x + global_row_offset * hidden_size;
        float *out_ptr = inter_out + global_row_offset * inter_size;
        const float *w1 = w1_ptrs[e];
        const float *w3 = w3_ptrs[e];
        
        int tid = threadIdx.x + threadIdx.y * blockDim.x; // 0..255
        
        // Map tid to (ty, tx) for 32x8 block size usage
        int ty = tid / 32; // 0..7
        int tx = tid % 32; // 0..31

        // Vectorized Loading (64x16 tile)
        int load_r = tid / 4; // 0..63
        int load_c_vec = tid % 4; // 0..3
        
        for (int n_base = 0; n_base < inter_size; n_base += 64) {
            float sum1[16] = {0.0f}; 
            float sum3[16] = {0.0f};
            
            // Padding [17]
            __shared__ float smem_x[2][64][17]; 
            __shared__ float smem_w1[2][64][17]; 
            __shared__ float smem_w3[2][64][17];

            // Prologue: Load Tile 0
            {
                if (load_r < num_rows) {
                    float4 val = reinterpret_cast<const float4*>(in_ptr + load_r * hidden_size)[load_c_vec];
                    smem_x[0][load_r][load_c_vec*4] = val.x;
                    smem_x[0][load_r][load_c_vec*4+1] = val.y;
                    smem_x[0][load_r][load_c_vec*4+2] = val.z;
                    smem_x[0][load_r][load_c_vec*4+3] = val.w;
                } else {
                    smem_x[0][load_r][load_c_vec*4] = 0.0f;
                    smem_x[0][load_r][load_c_vec*4+1] = 0.0f;
                    smem_x[0][load_r][load_c_vec*4+2] = 0.0f;
                    smem_x[0][load_r][load_c_vec*4+3] = 0.0f;
                }
                
                if ((n_base + load_r) < inter_size) {
                    float4 mc1 = reinterpret_cast<const float4*>(w1 + (n_base + load_r) * hidden_size)[load_c_vec];
                    smem_w1[0][load_r][load_c_vec*4] = mc1.x;
                    smem_w1[0][load_r][load_c_vec*4+1] = mc1.y;
                    smem_w1[0][load_r][load_c_vec*4+2] = mc1.z;
                    smem_w1[0][load_r][load_c_vec*4+3] = mc1.w;

                    float4 mc3 = reinterpret_cast<const float4*>(w3 + (n_base + load_r) * hidden_size)[load_c_vec];
                    smem_w3[0][load_r][load_c_vec*4] = mc3.x;
                    smem_w3[0][load_r][load_c_vec*4+1] = mc3.y;
                    smem_w3[0][load_r][load_c_vec*4+2] = mc3.z;
                    smem_w3[0][load_r][load_c_vec*4+3] = mc3.w;
                } else {
                    #pragma unroll
                    for(int z=0; z<4; ++z) {
                        smem_w1[0][load_r][load_c_vec*4+z] = 0.0f;
                        smem_w3[0][load_r][load_c_vec*4+z] = 0.0f;
                    }
                }
            }
            __syncthreads();

            int num_k_steps = (hidden_size + 15) / 16;
            
            for (int k_step = 0; k_step < num_k_steps; ++k_step) {
                int cur = k_step % 2;
                int nxt = (k_step + 1) % 2;
                int k_next_base = (k_step + 1) * 16;
                
                if (k_step < num_k_steps - 1) {
                    if (load_r < num_rows) {
                        float4 val = reinterpret_cast<const float4*>(in_ptr + load_r * hidden_size + k_next_base)[load_c_vec];
                        smem_x[nxt][load_r][load_c_vec*4] = val.x;
                        smem_x[nxt][load_r][load_c_vec*4+1] = val.y;
                        smem_x[nxt][load_r][load_c_vec*4+2] = val.z;
                        smem_x[nxt][load_r][load_c_vec*4+3] = val.w;
                    } else {
                        smem_x[nxt][load_r][load_c_vec*4] = 0.0f;
                        smem_x[nxt][load_r][load_c_vec*4+1] = 0.0f;
                        smem_x[nxt][load_r][load_c_vec*4+2] = 0.0f;
                        smem_x[nxt][load_r][load_c_vec*4+3] = 0.0f;
                    }

                    if ((n_base + load_r) < inter_size) {
                        float4 mc1 = reinterpret_cast<const float4*>(w1 + (n_base + load_r) * hidden_size + k_next_base)[load_c_vec];
                        smem_w1[nxt][load_r][load_c_vec*4] = mc1.x;
                        smem_w1[nxt][load_r][load_c_vec*4+1] = mc1.y;
                        smem_w1[nxt][load_r][load_c_vec*4+2] = mc1.z;
                        smem_w1[nxt][load_r][load_c_vec*4+3] = mc1.w;

                        float4 mc3 = reinterpret_cast<const float4*>(w3 + (n_base + load_r) * hidden_size + k_next_base)[load_c_vec];
                        smem_w3[nxt][load_r][load_c_vec*4] = mc3.x;
                        smem_w3[nxt][load_r][load_c_vec*4+1] = mc3.y;
                        smem_w3[nxt][load_r][load_c_vec*4+2] = mc3.z;
                        smem_w3[nxt][load_r][load_c_vec*4+3] = mc3.w;
                    } else {
                        #pragma unroll
                        for(int z=0; z<4; ++z) {
                            smem_w1[nxt][load_r][load_c_vec*4+z] = 0.0f;
                            smem_w3[nxt][load_r][load_c_vec*4+z] = 0.0f;
                        }
                    }
                }

                #pragma unroll
                for (int k = 0; k < 16; ++k) {
                    float val_x_0 = smem_x[cur][ty*8 + 0][k];
                    float val_x_1 = smem_x[cur][ty*8 + 1][k];
                    float val_x_2 = smem_x[cur][ty*8 + 2][k];
                    float val_x_3 = smem_x[cur][ty*8 + 3][k];
                    float val_x_4 = smem_x[cur][ty*8 + 4][k];
                    float val_x_5 = smem_x[cur][ty*8 + 5][k];
                    float val_x_6 = smem_x[cur][ty*8 + 6][k];
                    float val_x_7 = smem_x[cur][ty*8 + 7][k];

                    float val_w1_0 = smem_w1[cur][tx*2 + 0][k];
                    float val_w1_1 = smem_w1[cur][tx*2 + 1][k];
                    
                    float val_w3_0 = smem_w3[cur][tx*2 + 0][k];
                    float val_w3_1 = smem_w3[cur][tx*2 + 1][k];
                    
                    sum1[0] += val_x_0 * val_w1_0; sum1[1] += val_x_1 * val_w1_0;
                    sum1[2] += val_x_2 * val_w1_0; sum1[3] += val_x_3 * val_w1_0;
                    sum1[4] += val_x_4 * val_w1_0; sum1[5] += val_x_5 * val_w1_0;
                    sum1[6] += val_x_6 * val_w1_0; sum1[7] += val_x_7 * val_w1_0;
                    
                    sum1[8] += val_x_0 * val_w1_1; sum1[9] += val_x_1 * val_w1_1;
                    sum1[10] += val_x_2 * val_w1_1; sum1[11] += val_x_3 * val_w1_1;
                    sum1[12] += val_x_4 * val_w1_1; sum1[13] += val_x_5 * val_w1_1;
                    sum1[14] += val_x_6 * val_w1_1; sum1[15] += val_x_7 * val_w1_1;

                    sum3[0] += val_x_0 * val_w3_0; sum3[1] += val_x_1 * val_w3_0;
                    sum3[2] += val_x_2 * val_w3_0; sum3[3] += val_x_3 * val_w3_0;
                    sum3[4] += val_x_4 * val_w3_0; sum3[5] += val_x_5 * val_w3_0;
                    sum3[6] += val_x_6 * val_w3_0; sum3[7] += val_x_7 * val_w3_0;
                    
                    sum3[8] += val_x_0 * val_w3_1; sum3[9] += val_x_1 * val_w3_1;
                    sum3[10] += val_x_2 * val_w3_1; sum3[11] += val_x_3 * val_w3_1;
                    sum3[12] += val_x_4 * val_w3_1; sum3[13] += val_x_5 * val_w3_1;
                    sum3[14] += val_x_6 * val_w3_1; sum3[15] += val_x_7 * val_w3_1;
                }
                
                __syncthreads();
            }
            
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int c = tx * 2 + j;
                if ((n_base + c) < inter_size) {
                    #pragma unroll
                    for(int i=0; i<8; ++i) {
                         int r = ty * 8 + i;
                         if (r < num_rows) {
                             float s1 = sum1[j*8 + i];
                             float s3 = sum3[j*8 + i];
                             float val = d_silu_opt(s1) * s3;
                             out_ptr[r * inter_size + n_base + c] = val;
                         }
                    }
                }
            }
        }
    }
}

__global__ void moe_persistent_w2_kernel(
    float *__restrict__ final_out, 
    const float *__restrict__ inter_in, 
    float **__restrict__ w2_ptrs,  
    const int *__restrict__ tile_offsets, 
    const int *__restrict__ offsets,
    const int *__restrict__ counts,
    int *__restrict__ task_counter,
    int hidden_size, int inter_size, int num_experts) 
{
    while (true) {
        __shared__ int task_id_s;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            task_id_s = atomicAdd(task_counter, 1);
        }
        __syncthreads();
        int task_id = task_id_s;

        int total_tiles = tile_offsets[num_experts];
        if (task_id >= total_tiles) break;

        int e = 0;
        if (task_id >= tile_offsets[16]) e = 16;
        if (task_id >= tile_offsets[e+8]) e += 8;
        if (task_id >= tile_offsets[e+4]) e += 4;
        if (task_id >= tile_offsets[e+2]) e += 2;
        if (task_id >= tile_offsets[e+1]) e += 1;
        
        int tile_idx = task_id - tile_offsets[e];
        int count = counts[e];
        int start_row = tile_idx * 64; 
        int num_rows = min(64, count - start_row);
        
        int global_row_offset = offsets[e] + start_row;
        const float *in_ptr = inter_in + global_row_offset * inter_size;
        float *out_ptr = final_out + global_row_offset * hidden_size;
        const float *w2 = w2_ptrs[e];
        
        int tid = threadIdx.x + threadIdx.y * blockDim.x; 
        int ty = tid / 32; 
        int tx = tid % 32; 
        
        int load_r = tid / 4; 
        int load_c_vec = tid % 4; 
        
        for (int h_base = 0; h_base < hidden_size; h_base += 64) {
            float sum[16] = {0.0f};
            
            __shared__ float smem_x[2][64][17]; 
            __shared__ float smem_w[2][64][17]; 
            
            {
                if (load_r < num_rows) {
                     float4 val = reinterpret_cast<const float4*>(in_ptr + load_r * inter_size)[load_c_vec];
                     smem_x[0][load_r][load_c_vec*4] = val.x;
                     smem_x[0][load_r][load_c_vec*4+1] = val.y;
                     smem_x[0][load_r][load_c_vec*4+2] = val.z;
                     smem_x[0][load_r][load_c_vec*4+3] = val.w;
                } else {
                     smem_x[0][load_r][load_c_vec*4] = 0.0f;
                     smem_x[0][load_r][load_c_vec*4+1] = 0.0f;
                     smem_x[0][load_r][load_c_vec*4+2] = 0.0f;
                     smem_x[0][load_r][load_c_vec*4+3] = 0.0f;
                }
                
                if ((h_base + load_r) < hidden_size) {
                     float4 w = reinterpret_cast<const float4*>(w2 + (h_base + load_r) * inter_size)[load_c_vec];
                     smem_w[0][load_r][load_c_vec*4] = w.x;
                     smem_w[0][load_r][load_c_vec*4+1] = w.y;
                     smem_w[0][load_r][load_c_vec*4+2] = w.z;
                     smem_w[0][load_r][load_c_vec*4+3] = w.w;
                } else {
                     #pragma unroll
                     for(int z=0; z<4; ++z) smem_w[0][load_r][load_c_vec*4+z] = 0.0f;
                }
            }
            __syncthreads();
            
            int num_k_steps = (inter_size + 15) / 16;
            for (int k_step = 0; k_step < num_k_steps; ++k_step) {
                int cur = k_step % 2;
                int nxt = (k_step + 1) % 2;
                int k_next_base = (k_step + 1) * 16;
                
                if (k_step < num_k_steps - 1) {
                     if (load_r < num_rows) {
                         float4 val = reinterpret_cast<const float4*>(in_ptr + load_r * inter_size + k_next_base)[load_c_vec];
                         smem_x[nxt][load_r][load_c_vec*4] = val.x;
                         smem_x[nxt][load_r][load_c_vec*4+1] = val.y;
                         smem_x[nxt][load_r][load_c_vec*4+2] = val.z;
                         smem_x[nxt][load_r][load_c_vec*4+3] = val.w;
                     } else {
                         smem_x[nxt][load_r][load_c_vec*4] = 0.0f;
                         smem_x[nxt][load_r][load_c_vec*4+1] = 0.0f;
                         smem_x[nxt][load_r][load_c_vec*4+2] = 0.0f;
                         smem_x[nxt][load_r][load_c_vec*4+3] = 0.0f;
                     }
                     
                     if ((h_base + load_r) < hidden_size) {
                         float4 w = reinterpret_cast<const float4*>(w2 + (h_base + load_r) * inter_size + k_next_base)[load_c_vec];
                         smem_w[nxt][load_r][load_c_vec*4] = w.x;
                         smem_w[nxt][load_r][load_c_vec*4+1] = w.y;
                         smem_w[nxt][load_r][load_c_vec*4+2] = w.z;
                         smem_w[nxt][load_r][load_c_vec*4+3] = w.w;
                     } else {
                         #pragma unroll
                         for(int z=0; z<4; ++z) smem_w[nxt][load_r][load_c_vec*4+z] = 0.0f;
                     }
                }
                
                #pragma unroll
                for (int k = 0; k < 16; ++k) {
                    float val_x_0 = smem_x[cur][ty*8 + 0][k];
                    float val_x_1 = smem_x[cur][ty*8 + 1][k];
                    float val_x_2 = smem_x[cur][ty*8 + 2][k];
                    float val_x_3 = smem_x[cur][ty*8 + 3][k];
                    float val_x_4 = smem_x[cur][ty*8 + 4][k];
                    float val_x_5 = smem_x[cur][ty*8 + 5][k];
                    float val_x_6 = smem_x[cur][ty*8 + 6][k];
                    float val_x_7 = smem_x[cur][ty*8 + 7][k];

                    float val_w_0 = smem_w[cur][tx*2 + 0][k];
                    float val_w_1 = smem_w[cur][tx*2 + 1][k];
                    
                    sum[0] += val_x_0 * val_w_0; sum[1] += val_x_1 * val_w_0;
                    sum[2] += val_x_2 * val_w_0; sum[3] += val_x_3 * val_w_0;
                    sum[4] += val_x_4 * val_w_0; sum[5] += val_x_5 * val_w_0;
                    sum[6] += val_x_6 * val_w_0; sum[7] += val_x_7 * val_w_0;
                    
                    sum[8] += val_x_0 * val_w_1; sum[9] += val_x_1 * val_w_1;
                    sum[10] += val_x_2 * val_w_1; sum[11] += val_x_3 * val_w_1;
                    sum[12] += val_x_4 * val_w_1; sum[13] += val_x_5 * val_w_1;
                    sum[14] += val_x_6 * val_w_1; sum[15] += val_x_7 * val_w_1;
                }
                __syncthreads();
            }
            
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int c = tx * 2 + j;
                if ((h_base + c) < hidden_size) {
                    #pragma unroll
                    for(int i=0; i<8; ++i) {
                         int r = ty * 8 + i;
                         if (r < num_rows) {
                             out_ptr[r * hidden_size + h_base + c] = sum[j*8 + i];
                         }
                    }
                }
            }
        }
    }
}

RMSNorm::RMSNorm(const std::string &weight_file) {
  weight_ = Tensor::load_from_file(weight_file);
}

RMSNorm::~RMSNorm() {
  if (weight_gpu_)
    cudaFree(weight_gpu_);
}

void RMSNorm::init_gpu() {
  size_t size = weight_.size() * sizeof(float);
  CHECK_CUDA(cudaMalloc(&weight_gpu_, size));
  CHECK_CUDA(
      cudaMemcpy(weight_gpu_, weight_.data(), size, cudaMemcpyHostToDevice));
}

void RMSNorm::forward_gpu(float *x, float *y, int rows, int hidden_dim,
                          cudaStream_t stream) {
  rms_norm_kernel<<<rows, 256, 0, stream>>>(y, x, weight_gpu_, rows,
                                            hidden_dim);
}

void RMSNorm::forward(const Tensor &x, Tensor &y) {}

// ----------------------------------------------------------------------------
// RotaryEmbedding Implementation
// ----------------------------------------------------------------------------
RotaryEmbedding::RotaryEmbedding() : max_seq_len_(MAX_POSITION_EMBEDDINGS) {
  cos_cached_ = Tensor({max_seq_len_, HEAD_DIM});
  sin_cached_ = Tensor({max_seq_len_, HEAD_DIM});
}

RotaryEmbedding::~RotaryEmbedding() {
  if (cos_gpu_)
    cudaFree(cos_gpu_);
  if (sin_gpu_)
    cudaFree(sin_gpu_);
}

void RotaryEmbedding::init_gpu() {
  size_t head_dim = HEAD_DIM;
  size_t max_req = max_seq_len_;

  for (size_t i = 0; i < max_req; ++i) {
    for (size_t j = 0; j < head_dim / 2; ++j) {
      float theta = powf(ROPE_THETA, -2.0f * j / head_dim);
      float angle = i * theta;
      cos_cached_.at(i, j) = cos(angle);
      cos_cached_.at(i, j + head_dim / 2) = cos(angle);
      sin_cached_.at(i, j) = sin(angle);
      sin_cached_.at(i, j + head_dim / 2) = sin(angle);
    }
  }

  CHECK_CUDA(cudaMalloc(&cos_gpu_, max_req * head_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&sin_gpu_, max_req * head_dim * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(cos_gpu_, cos_cached_.data(),
                        max_req * head_dim * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(sin_gpu_, sin_cached_.data(),
                        max_req * head_dim * sizeof(float),
                        cudaMemcpyHostToDevice));
}

void RotaryEmbedding::forward_gpu(size_t seq_len, float *cos, float *sin,
                                  cudaStream_t stream) {}

void RotaryEmbedding::forward(size_t seq_len, Tensor &cos, Tensor &sin) {}

// ----------------------------------------------------------------------------
// MLP Implementation
// ----------------------------------------------------------------------------
MLP::MLP(const std::string &w1_file, const std::string &w2_file,
         const std::string &w3_file) {
  w1_ = Tensor::load_from_file(w1_file);
  w2_ = Tensor::load_from_file(w2_file);
  w3_ = Tensor::load_from_file(w3_file);
}

MLP::~MLP() {
  if (w1_gpu_)
    cudaFree(w1_gpu_);
  if (w2_gpu_)
    cudaFree(w2_gpu_);
  if (w3_gpu_)
    cudaFree(w3_gpu_);
}

void MLP::init_gpu() {
  CHECK_CUDA(cudaMalloc(&w1_gpu_, w1_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(w1_gpu_, w1_.data(), w1_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&w2_gpu_, w2_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(w2_gpu_, w2_.data(), w2_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&w3_gpu_, w3_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(w3_gpu_, w3_.data(), w3_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}


void MLP::forward(const Tensor &x, Tensor &y) {}

void MLP::forward_gpu(float *x, float *y, int batch, int seq_len,
                      int hidden_size, cudaStream_t stream) {

  if (!temp_buf_gpu_) {}

  int num_tokens = batch * seq_len;
  int inter_size = INTERMEDIATE_SIZE;

  // Gate: x @ w1 -> temp (part 1)
  // Up: x @ w3 -> temp (part 2)
  // Need 2 * inter_size * num_tokens space.

  float *input_buf = x;
  float *gate_buf = temp_buf_gpu_;
  float *up_buf = temp_buf_gpu_ + num_tokens * inter_size;

  // 1. Gate Projection (W1) 
  dim3 block(16, 16);
  dim3 grid_gate((inter_size + 127) / 128, (num_tokens + 127) / 128);
  attn_matmul_tiled_kernel<<<grid_gate, block, 0, stream>>>(
      gate_buf, input_buf, w1_gpu_, num_tokens, inter_size, hidden_size);

  // 2. Up Projection (W3)
  attn_matmul_tiled_kernel<<<grid_gate, block, 0, stream>>>(
      up_buf, input_buf, w3_gpu_, num_tokens, inter_size, hidden_size);

  // 3. SwiGLU: gate = silu(gate) * up. Reuse gate_buf.
  int total_elems = num_tokens * inter_size;
  moe_swiglu_kernel<<<(total_elems + 255) / 256, 256, 0, stream>>>(
      gate_buf, gate_buf, up_buf, total_elems);

  // 4. Down Projection (W2). w2 is [Hidden, Inter].
  // Input is gate_buf [Tokens, Inter]. Output [Tokens, Hidden].
  dim3 grid_down((hidden_size + 127) / 128, (num_tokens + 127) / 128);
  attn_matmul_tiled_kernel<<<grid_down, block, 0, stream>>>(
      y, gate_buf, w2_gpu_, num_tokens, hidden_size, inter_size);
}

// ----------------------------------------------------------------------------
// SparseMoeBlock Implementation
// ----------------------------------------------------------------------------
SparseMoeBlock::SparseMoeBlock(int layer_idx) : layer_idx_(layer_idx) {
  // 게이트 가중치(expert 결정) CPU Load
  std::stringstream ss;
  ss << "layers." << layer_idx << ".feed_forward.gate.weight";
  gate_ = Tensor::load_from_file(ss.str());

  // Expert 가중치 Load
  experts_.reserve(NUM_EXPERTS);
  for (size_t i = 0; i < NUM_EXPERTS; i++) {
    std::stringstream ss_w1, ss_w2, ss_w3;
    ss_w1 << "layers." << layer_idx << ".feed_forward.experts." << i
          << ".w1.weight";
    ss_w2 << "layers." << layer_idx << ".feed_forward.experts." << i
          << ".w2.weight";
    ss_w3 << "layers." << layer_idx << ".feed_forward.experts." << i
          << ".w3.weight";

    experts_.emplace_back(ss_w1.str(), ss_w2.str(), ss_w3.str());
  }

  // Expert Bias 값 Load
  if (USE_EXPERT_BIAS) {
    std::stringstream ss_bias;
    ss_bias << "layers." << layer_idx << ".feed_forward.expert_bias";
    expert_bias_ = Tensor::load_from_file(ss_bias.str());
  }
}

SparseMoeBlock::~SparseMoeBlock() {
  if (gate_gpu_)
    cudaFree(gate_gpu_);
  if (expert_bias_gpu_)
    cudaFree(expert_bias_gpu_);

  if (expert_w1_gpu_ptrs_)
    cudaFree(expert_w1_gpu_ptrs_);
  if (expert_w2_gpu_ptrs_)
    cudaFree(expert_w2_gpu_ptrs_);
  if (expert_w3_gpu_ptrs_)
    cudaFree(expert_w3_gpu_ptrs_);

  if (expert_counts_gpu_)
    cudaFree(expert_counts_gpu_);
  if (expert_offsets_gpu_)
    cudaFree(expert_offsets_gpu_);
  if (sorted_indices_gpu_)
    cudaFree(sorted_indices_gpu_);
  if (sorted_weights_gpu_)
    cudaFree(sorted_weights_gpu_);
  if (sorted_x_gpu_)
    cudaFree(sorted_x_gpu_);
  if (sorted_inter_gpu_)
    cudaFree(sorted_inter_gpu_);
  if (sorted_out_gpu_)
    cudaFree(sorted_out_gpu_);
  if (topk_indices_gpu_)
    cudaFree(topk_indices_gpu_);
  if (topk_weights_gpu_)
    cudaFree(topk_weights_gpu_);
  if (token_expert_pos_gpu_)
    cudaFree(token_expert_pos_gpu_);

  // Optimizations
  if (expert_tile_offsets_gpu_)
    cudaFree(expert_tile_offsets_gpu_);
  if (task_counters_gpu_)
    cudaFree(task_counters_gpu_);
  if (expert_write_pos_gpu_)
    cudaFree(expert_write_pos_gpu_);

  if (h_expert_counts_)
    free(h_expert_counts_);
  if (h_expert_offsets_)
    free(h_expert_offsets_);

  if (expert_w1_ptrs_host_)
    free(expert_w1_ptrs_host_);
  if (expert_w2_ptrs_host_)
    free(expert_w2_ptrs_host_);
  if (expert_w3_ptrs_host_)
    free(expert_w3_ptrs_host_);
}

void SparseMoeBlock::init_gpu() {
  // 1. Gate와 Bias를 GPU로
  CHECK_CUDA(cudaMalloc(&gate_gpu_, gate_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(gate_gpu_, gate_.data(), gate_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  if (expert_bias_.size() > 0) {
    CHECK_CUDA(
        cudaMalloc(&expert_bias_gpu_, expert_bias_.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(expert_bias_gpu_, expert_bias_.data(),
                          expert_bias_.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  // Expert 초기화
  for (auto &expert : experts_) {
    expert.init_gpu(); // 각 Expert의 w1,w2,w3을 GPU로
  }

  // 3. Expert의 가중치 포인터
  expert_w1_ptrs_host_ = (float **)malloc(NUM_EXPERTS * sizeof(float *));
  expert_w2_ptrs_host_ = (float **)malloc(NUM_EXPERTS * sizeof(float *));
  expert_w3_ptrs_host_ = (float **)malloc(NUM_EXPERTS * sizeof(float *));

  for (size_t i = 0; i < NUM_EXPERTS; ++i) {
    expert_w1_ptrs_host_[i] = experts_[i].w1_gpu_;
    expert_w2_ptrs_host_[i] = experts_[i].w2_gpu_;
    expert_w3_ptrs_host_[i] = experts_[i].w3_gpu_;
  }

  CHECK_CUDA(cudaMalloc(&expert_w1_gpu_ptrs_, NUM_EXPERTS * sizeof(float *)));
  CHECK_CUDA(cudaMalloc(&expert_w2_gpu_ptrs_, NUM_EXPERTS * sizeof(float *)));
  CHECK_CUDA(cudaMalloc(&expert_w3_gpu_ptrs_, NUM_EXPERTS * sizeof(float *)));

  CHECK_CUDA(cudaMemcpy(expert_w1_gpu_ptrs_, expert_w1_ptrs_host_,
                        NUM_EXPERTS * sizeof(float *), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(expert_w2_gpu_ptrs_, expert_w2_ptrs_host_,
                        NUM_EXPERTS * sizeof(float *), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(expert_w3_gpu_ptrs_, expert_w3_ptrs_host_,
                        NUM_EXPERTS * sizeof(float *), cudaMemcpyHostToDevice));


  size_t max_tokens = 4096; // Batch 256 * Seq 16
  size_t expanded = max_tokens * NUM_EXPERTS_PER_TOK;
  size_t hidden = HIDDEN_SIZE;
  size_t inter = MOE_INTERMEDIATE_SIZE;

  CHECK_CUDA(cudaMalloc(&expert_counts_gpu_, NUM_EXPERTS * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&expert_offsets_gpu_, NUM_EXPERTS * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&sorted_indices_gpu_, expanded * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&sorted_weights_gpu_, expanded * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&sorted_x_gpu_, expanded * hidden * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&sorted_inter_gpu_,
                        expanded * inter * 2 * sizeof(float))); 
  CHECK_CUDA(cudaMalloc(&sorted_out_gpu_, expanded * hidden * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&topk_indices_gpu_,
                        max_tokens * NUM_EXPERTS_PER_TOK * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&topk_weights_gpu_,
                        max_tokens * NUM_EXPERTS_PER_TOK * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&token_expert_pos_gpu_,
                        max_tokens * NUM_EXPERTS_PER_TOK * sizeof(int)));
  // 추가 버퍼
  CHECK_CUDA(cudaMalloc(&expert_tile_offsets_gpu_, (NUM_EXPERTS + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&task_counters_gpu_, 2 * sizeof(int)));
  CHECK_CUDA(cudaMemset(task_counters_gpu_, 0, 2 * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&expert_write_pos_gpu_, NUM_EXPERTS * sizeof(int)));

  h_expert_counts_ = (int *)malloc(NUM_EXPERTS * sizeof(int));
  h_expert_offsets_ = (int *)malloc(NUM_EXPERTS * sizeof(int));
}

void SparseMoeBlock::forward_gpu(float *x, float *y, int batch, int seq_len,
                                 int hidden_size, cudaStream_t stream) {
  int num_tokens = batch * seq_len;
  // 출력 버퍼
  CHECK_CUDA(cudaMemsetAsync(y, 0, num_tokens * hidden_size * sizeof(float), stream));
  // Expert Count 버퍼
  CHECK_CUDA(cudaMemsetAsync(expert_counts_gpu_, 0, NUM_EXPERTS * sizeof(int),
                             stream));

  // x * gate_W = Expert logits
  float *router_logits = sorted_x_gpu_; 
  
  {
      int m = num_tokens;
      int n = NUM_EXPERTS;
      int k = hidden_size;
      dim3 grid((n + 127) / 128, (m + 127) / 128);
      dim3 block(16, 16); 
      // BM x BN = 128 x 128 크기의 결과 블록을 계산
      // TM x TN = 8 x 8. Thread당 64개 계산 => 128x128 / 8x8 = 16x16
      attn_matmul_tiled_kernel<<<grid, block, 0, stream>>>(
          router_logits, x, gate_gpu_, m, n, k);
  }

  // 4개 선정 및 카운트(Expert 별 처리할 토큰)
  int threads = 256;
  int blocks = (num_tokens + threads - 1) / threads;
  moe_router_topk_kernel<<<blocks, threads, 0, stream>>>(
      expert_counts_gpu_, topk_indices_gpu_, topk_weights_gpu_,
      router_logits, expert_bias_gpu_, NUM_EXPERTS, NUM_EXPERTS_PER_TOK,
      num_tokens, 1.0f, true);
  
  // Offset 스캔. 각 Expert가 어느 메모리 쓸지 계산
  // CPU Sync X
  moe_scan_kernel<<<1, 32, 0, stream>>>(expert_offsets_gpu_, expert_write_pos_gpu_, expert_tile_offsets_gpu_,
                                        task_counters_gpu_, expert_counts_gpu_,
                                        NUM_EXPERTS, 64); 

  // Scatter. Expert 별로 토큰 분배(재배치)
  // Optimized: Block-per-token launch
  moe_scatter_kernel<<<num_tokens, 256, 0, stream>>>(
      sorted_x_gpu_, sorted_indices_gpu_, sorted_weights_gpu_,
      expert_write_pos_gpu_, token_expert_pos_gpu_, x, topk_indices_gpu_,
      topk_weights_gpu_, hidden_size, NUM_EXPERTS_PER_TOK, num_tokens);

  // 연산 (W1, W3, SwiGLU, W2)
  int num_sms = 72; // TITAN SM 72개
  // Persistent Kernel: 일을 미리 나누지않고, idle한 block이 생길 때마다 일을 할당 (쉴틈없이)
  int persistent_blocks = num_sms * 4; 
  dim3 dimBlock(32, 8); // 256개. 가로 32 -> warp 1개가 coalesced된 memory read

  // Fusion: W1 + W3
  // task_counters_gpu_[0] 사용
  moe_persistent_fused_w1w3_kernel<<<persistent_blocks, dimBlock, 0, stream>>>(
      sorted_inter_gpu_, sorted_x_gpu_, expert_w1_gpu_ptrs_, expert_w3_gpu_ptrs_,
      expert_tile_offsets_gpu_, expert_offsets_gpu_, expert_counts_gpu_,
      &task_counters_gpu_[0], hidden_size, MOE_INTERMEDIATE_SIZE, NUM_EXPERTS);
  
  // W2 (Down Projection)
  moe_persistent_w2_kernel<<<persistent_blocks, dimBlock, 0, stream>>>(
      sorted_out_gpu_, sorted_inter_gpu_, expert_w2_gpu_ptrs_,
      expert_tile_offsets_gpu_, expert_offsets_gpu_, expert_counts_gpu_,
      &task_counters_gpu_[1], hidden_size, MOE_INTERMEDIATE_SIZE, NUM_EXPERTS);

  // Gather (결과를 원래 토큰 위치로 복원 및 Weighted Sum)
  int output_size = num_tokens * hidden_size;
  int gather_blocks = (output_size + 255) / 256;
  moe_gather_deterministic_kernel<<<gather_blocks, 256, 0, stream>>>(
      y, sorted_out_gpu_, token_expert_pos_gpu_, topk_weights_gpu_, num_tokens,
      hidden_size, NUM_EXPERTS_PER_TOK);
}
void SparseMoeBlock::forward(const Tensor &x, Tensor &y,
                             Tensor &router_logits) {}
void SparseMoeBlock::route_tokens(const Tensor &router_logits,
                                  std::vector<int> &top_k_indices,
                                  std::vector<float> &top_k_weights) {}

// ----------------------------------------------------------------------------
// Attention Implementation
// ----------------------------------------------------------------------------
Attention::Attention(int layer_idx) : layer_idx_(layer_idx) {
  std::stringstream ss_q, ss_k, ss_v, ss_o, ss_q_ln, ss_k_ln;
  ss_q << "layers." << layer_idx << ".self_attn.q_proj.weight";
  ss_k << "layers." << layer_idx << ".self_attn.k_proj.weight";
  ss_v << "layers." << layer_idx << ".self_attn.v_proj.weight";
  ss_o << "layers." << layer_idx << ".self_attn.out_proj.weight";
  ss_q_ln << "layers." << layer_idx << ".self_attn.q_layernorm.weight";
  ss_k_ln << "layers." << layer_idx << ".self_attn.k_layernorm.weight";

  q_proj_ = Tensor::load_from_file(ss_q.str());
  k_proj_ = Tensor::load_from_file(ss_k.str());
  v_proj_ = Tensor::load_from_file(ss_v.str());
  o_proj_ = Tensor::load_from_file(ss_o.str());

  q_layernorm_ = std::make_unique<RMSNorm>(ss_q_ln.str());
  k_layernorm_ = std::make_unique<RMSNorm>(ss_k_ln.str());
}

Attention::~Attention() {
  if (q_proj_gpu_)
    cudaFree(q_proj_gpu_);
  if (k_proj_gpu_)
    cudaFree(k_proj_gpu_);
  if (v_proj_gpu_)
    cudaFree(v_proj_gpu_);
  if (o_proj_gpu_)
    cudaFree(o_proj_gpu_);

  if (q_proj_out_gpu_)
    cudaFree(q_proj_out_gpu_);
  if (k_proj_out_gpu_)
    cudaFree(k_proj_out_gpu_);
  if (v_proj_out_gpu_)
    cudaFree(v_proj_out_gpu_);
  if (q_normed_gpu_)
    cudaFree(q_normed_gpu_);
  if (k_normed_gpu_)
    cudaFree(k_normed_gpu_);
  if (q_transposed_gpu_)
    cudaFree(q_transposed_gpu_);
  if (k_transposed_gpu_)
    cudaFree(k_transposed_gpu_);
  if (v_transposed_gpu_)
    cudaFree(v_transposed_gpu_);
  if (attn_out_transposed_gpu_)
    cudaFree(attn_out_transposed_gpu_);
}

void Attention::init_gpu() {
  q_layernorm_->init_gpu();
  k_layernorm_->init_gpu();

  constexpr int num_heads = NUM_ATTENTION_HEADS;
  constexpr int num_kv_heads = NUM_KEY_VALUE_HEADS;
  constexpr int head_dim = HEAD_DIM;

  CHECK_CUDA(cudaMalloc(&q_proj_gpu_, q_proj_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(q_proj_gpu_, q_proj_.data(),
                        q_proj_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&k_proj_gpu_, k_proj_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(k_proj_gpu_, k_proj_.data(),
                        k_proj_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&v_proj_gpu_, v_proj_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(v_proj_gpu_, v_proj_.data(),
                        v_proj_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&o_proj_gpu_, o_proj_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(o_proj_gpu_, o_proj_.data(),
                        o_proj_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Buffers max size
  int max_tokens = 4096; // Batch 256 * Seq 16
  CHECK_CUDA(cudaMalloc(&q_proj_out_gpu_,
                        max_tokens * num_heads * head_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&k_proj_out_gpu_,
                        max_tokens * num_kv_heads * head_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_proj_out_gpu_,
                        max_tokens * num_kv_heads * head_dim * sizeof(float)));

  CHECK_CUDA(cudaMalloc(&q_normed_gpu_,
                        max_tokens * num_heads * head_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&k_normed_gpu_,
                        max_tokens * num_kv_heads * head_dim * sizeof(float)));

  CHECK_CUDA(cudaMalloc(&q_transposed_gpu_,
                        max_tokens * num_heads * head_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&k_transposed_gpu_,
                        max_tokens * num_kv_heads * head_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_transposed_gpu_,
                        max_tokens * num_kv_heads * head_dim * sizeof(float)));

  CHECK_CUDA(cudaMalloc(&attn_out_transposed_gpu_,
                        max_tokens * num_heads * head_dim * sizeof(float)));
}

void Attention::forward_gpu(float *x, float *cos, float *sin, float *output,
                            int batch, int seq_len, int hidden_size,
                            cudaStream_t stream) {
  int num_heads = NUM_ATTENTION_HEADS;
  int num_kv_heads = NUM_KEY_VALUE_HEADS;
  int head_dim = HEAD_DIM;

  // 1. Projections
  dim3 block(16, 16);
  dim3 grid_q((num_heads * head_dim + 127) / 128,
              (batch * seq_len + 127) / 128);
  attn_matmul_tiled_kernel<<<grid_q, block, 0, stream>>>(
      q_proj_out_gpu_, x, q_proj_gpu_, batch * seq_len, num_heads * head_dim,
      hidden_size);

  dim3 grid_k((num_kv_heads * head_dim + 127) / 128,
              (batch * seq_len + 127) / 128);
  attn_matmul_tiled_kernel<<<grid_k, block, 0, stream>>>(
      k_proj_out_gpu_, x, k_proj_gpu_, batch * seq_len, num_kv_heads * head_dim,
      hidden_size);
  attn_matmul_tiled_kernel<<<grid_k, block, 0, stream>>>(
      v_proj_out_gpu_, x, v_proj_gpu_, batch * seq_len, num_kv_heads * head_dim,
      hidden_size);

  // 2. Norm
  q_layernorm_->forward_gpu(q_proj_out_gpu_, q_normed_gpu_,
                            batch * seq_len * num_heads, head_dim, stream);
  k_layernorm_->forward_gpu(k_proj_out_gpu_, k_normed_gpu_,
                            batch * seq_len * num_kv_heads, head_dim, stream);

  // 3. RoPE
  int rope_q_total = batch * num_heads * seq_len * (head_dim / 2);
  rope_transpose_kernel<<<(rope_q_total + 255) / 256, 256, 0, stream>>>(
      q_transposed_gpu_, q_normed_gpu_, cos, sin, batch, seq_len, num_heads,
      head_dim);

  int rope_k_total = batch * num_kv_heads * seq_len * (head_dim / 2);
  rope_transpose_kernel<<<(rope_k_total + 255) / 256, 256, 0, stream>>>(
      k_transposed_gpu_, k_normed_gpu_, cos, sin, batch, seq_len, num_kv_heads,
      head_dim);

  // 4. V Transpose
  int v_total = batch * num_kv_heads * seq_len * head_dim;
  transpose_v_kernel<<<(v_total + 255) / 256, 256, 0, stream>>>(
      v_transposed_gpu_, v_proj_out_gpu_, batch, seq_len, num_kv_heads,
      head_dim);

  // 5. Attn
  dim3 grid_attn(batch, num_heads);
  dim3 block_attn(32, seq_len > 1024 ? 1024: seq_len); 
  block_attn = dim3(32, 16); 
  // Shared mem needs to hold K, V: 2 * seq_len * head_dim.
  size_t smem_size = 2 * seq_len * head_dim * sizeof(float);

  fused_attn_gqa_kernel<<<grid_attn, block_attn, smem_size, stream>>>(
      attn_out_transposed_gpu_, q_transposed_gpu_, k_transposed_gpu_,
      v_transposed_gpu_, seq_len, head_dim, num_heads, num_kv_heads,
      1.0f / sqrtf(head_dim));

  // 6. Out Proj
  dim3 grid((hidden_size + 127) / 128, (batch * seq_len + 127) / 128);
  attn_matmul_tiled_kernel<<<grid, block, 0, stream>>>(
      output, attn_out_transposed_gpu_, o_proj_gpu_, batch * seq_len,
      hidden_size, hidden_size);
}
void Attention::forward(const Tensor &x, const Tensor &cos, const Tensor &sin,
                        const Tensor *attention_mask, Tensor &output) {}

// ----------------------------------------------------------------------------
// ShortConv Implementation
// ----------------------------------------------------------------------------
ShortConv::ShortConv(int layer_idx) : layer_idx_(layer_idx) {
  std::stringstream ss_conv, ss_in, ss_out;
  ss_conv << "layers." << layer_idx << ".conv.conv.weight";
  ss_in << "layers." << layer_idx << ".conv.in_proj.weight";
  ss_out << "layers." << layer_idx << ".conv.out_proj.weight";

  conv_weight_ = Tensor::load_from_file(ss_conv.str());
  in_proj_weight_ = Tensor::load_from_file(ss_in.str());
  out_proj_weight_ = Tensor::load_from_file(ss_out.str());

  // Bias loading skipped as per instruction "Kernel didn't have bias"
}

ShortConv::~ShortConv() {
  if (conv_weight_gpu_)
    cudaFree(conv_weight_gpu_);
  if (in_proj_weight_gpu_)
    cudaFree(in_proj_weight_gpu_);
  if (out_proj_weight_gpu_)
    cudaFree(out_proj_weight_gpu_);

  if (in_proj_out_gpu_)
    cudaFree(in_proj_out_gpu_);
  if (Bx_gpu_)
    cudaFree(Bx_gpu_);
  if (C_gpu_)
    cudaFree(C_gpu_);
  if (conv_out_gpu_)
    cudaFree(conv_out_gpu_);
  if (y_transposed_gpu_)
    cudaFree(y_transposed_gpu_);
}

void ShortConv::init_gpu() {
  CHECK_CUDA(
      cudaMalloc(&conv_weight_gpu_, conv_weight_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(conv_weight_gpu_, conv_weight_.data(),
                        conv_weight_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(
      cudaMalloc(&in_proj_weight_gpu_, in_proj_weight_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(in_proj_weight_gpu_, in_proj_weight_.data(),
                        in_proj_weight_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&out_proj_weight_gpu_,
                        out_proj_weight_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(out_proj_weight_gpu_, out_proj_weight_.data(),
                        out_proj_weight_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  int max_tokens = 4096; // Batch 256 * Seq 16
  int hidden = HIDDEN_SIZE;
  CHECK_CUDA(
      cudaMalloc(&in_proj_out_gpu_, max_tokens * 3 * hidden * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&Bx_gpu_, max_tokens * hidden * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu_, max_tokens * hidden * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv_out_gpu_, max_tokens * hidden * sizeof(float)));
  CHECK_CUDA(
      cudaMalloc(&y_transposed_gpu_, max_tokens * hidden * sizeof(float)));
}

void ShortConv::forward_gpu(float *x, float *y, int batch, int seq_len,
                            int hidden_size, cudaStream_t stream) {
  // 1. In Proj
  dim3 block(16, 16);
  dim3 grid_in((3 * hidden_size + 127) / 128, (batch * seq_len + 127) / 128);

  // Using moe_gemm_kernel or attn_matmul? attn_matmul is "A @ B^T".
  // in_proj_weight is loaded from file. Shape [3*H, H].
  // x is [B*S, H].
  // We want output [B*S, 3*H] = x @ in_proj_weight^T.
  // Yes, attn_matmul matches.
  attn_matmul_tiled_kernel<<<grid_in, block, 0, stream>>>(
      in_proj_out_gpu_, x, in_proj_weight_gpu_, batch * seq_len,
      3 * hidden_size, hidden_size);

  // 2. Transpose Split
  int total_elements = batch * seq_len * hidden_size;
  conv_transpose_split_kernel<<<(total_elements + 255) / 256, 256, 0, stream>>>(
      Bx_gpu_, C_gpu_, in_proj_out_gpu_, batch, seq_len, hidden_size);

  // 3. Conv
  causal_conv1d_kernel<<<(total_elements + 255) / 256, 256, 0, stream>>>(
      conv_out_gpu_, Bx_gpu_, conv_weight_gpu_, batch, hidden_size, seq_len,
      CONV_KERNEL);

  // 4. Gating Transpose
  conv_gating_transpose_kernel<<<(total_elements + 255) / 256, 256, 0,
                                 stream>>>(
      y_transposed_gpu_, C_gpu_, conv_out_gpu_, batch, hidden_size, seq_len);

  // 5. Out Proj
  dim3 grid_out((hidden_size + 127) / 128, (batch * seq_len + 127) / 128);
  attn_matmul_tiled_kernel<<<grid_out, block, 0, stream>>>(
      y, y_transposed_gpu_, out_proj_weight_gpu_, batch * seq_len, hidden_size,
      hidden_size);
}
void ShortConv::forward(const Tensor &x, Tensor &y) {}

// ----------------------------------------------------------------------------
// DecoderLayer Implementation
// ----------------------------------------------------------------------------
DecoderLayer::DecoderLayer(int layer_idx, bool is_attention_layer)
    : layer_idx_(layer_idx), is_attention_layer_(is_attention_layer) {

  // Load normalization layers
  std::stringstream ss_norm1, ss_norm2;
  ss_norm1 << "layers." << layer_idx << ".operator_norm.weight";
  ss_norm2 << "layers." << layer_idx << ".ffn_norm.weight";

  input_layernorm_ = std::make_unique<RMSNorm>(ss_norm1.str());
  post_attention_layernorm_ = std::make_unique<RMSNorm>(ss_norm2.str());

  // Load attention or conv
  if (is_attention_layer) {
    self_attn_ = std::make_unique<Attention>(layer_idx);
  } else {
    short_conv_ = std::make_unique<ShortConv>(layer_idx);
  }

  // Load MoE block or MLP
  if (static_cast<size_t>(layer_idx) >= NUM_DENSE_LAYERS) {
    moe_block_ = std::make_unique<SparseMoeBlock>(layer_idx);
  } else {
    std::stringstream ss_w1, ss_w2, ss_w3;
    ss_w1 << "layers." << layer_idx << ".feed_forward.w1.weight";
    ss_w2 << "layers." << layer_idx << ".feed_forward.w2.weight";
    ss_w3 << "layers." << layer_idx << ".feed_forward.w3.weight";
    dense_mlp_ = std::make_unique<MLP>(ss_w1.str(), ss_w2.str(), ss_w3.str());
  }
}

DecoderLayer::~DecoderLayer() {
  if (normed_input_gpu_)
    cudaFree(normed_input_gpu_);
  if (attn_output_gpu_)
    cudaFree(attn_output_gpu_);
  if (normed_hidden_gpu_)
    cudaFree(normed_hidden_gpu_);
  if (ffn_output_gpu_)
    cudaFree(ffn_output_gpu_);
}

void DecoderLayer::init_gpu(int device_id) {
  CHECK_CUDA(cudaSetDevice(device_id));

  input_layernorm_->init_gpu();
  post_attention_layernorm_->init_gpu();

  if (self_attn_)
    self_attn_->init_gpu();
  if (short_conv_)
    short_conv_->init_gpu();

  if (moe_block_)
    moe_block_->init_gpu();
  if (dense_mlp_)
    dense_mlp_->init_gpu();

  // Alloc buffers
  int max_tokens = 4096; // Batch 256 * Seq 16
  CHECK_CUDA(
      cudaMalloc(&normed_input_gpu_, max_tokens * HIDDEN_SIZE * sizeof(float)));
  CHECK_CUDA(
      cudaMalloc(&attn_output_gpu_, max_tokens * HIDDEN_SIZE * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&normed_hidden_gpu_,
                        max_tokens * HIDDEN_SIZE * sizeof(float)));
  CHECK_CUDA(
      cudaMalloc(&ffn_output_gpu_, max_tokens * HIDDEN_SIZE * sizeof(float)));

  // Alloc temp buffer for MLP if needed
  if (dense_mlp_) {
    int inter_size = INTERMEDIATE_SIZE;
    int needed = max_tokens * inter_size * 2; // Gate + Up
    CHECK_CUDA(cudaMalloc(&dense_mlp_->temp_buf_gpu_, needed * sizeof(float)));
  }
}

__global__ void add_kernel(float *out, const float *a, const float *b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int vec_n = n / 4;
  
  if (idx < vec_n) {
      float4 va = reinterpret_cast<const float4*>(a)[idx];
      float4 vb = reinterpret_cast<const float4*>(b)[idx];
      float4 res;
      res.x = va.x + vb.x;
      res.y = va.y + vb.y;
      res.z = va.z + vb.z;
      res.w = va.w + vb.w;
      reinterpret_cast<float4*>(out)[idx] = res;
  }
  
  if (idx == 0) {
      for(int i = vec_n * 4; i < n; ++i) {
          out[i] = a[i] + b[i];
      }
  }
}

void DecoderLayer::forward_gpu(float *x, float *cos, float *sin, float *output,
                               int batch, int seq_len, int hidden_size,
                               cudaStream_t stream) {
  // 1. Input Norm
  int num_elements = batch * seq_len * hidden_size;
  input_layernorm_->forward_gpu(x, normed_input_gpu_, batch * seq_len,
                                hidden_size, stream);

  // 2. Operator (Attn or Conv)
  if (is_attention_layer_) {
    self_attn_->forward_gpu(normed_input_gpu_, cos, sin, attn_output_gpu_,
                            batch, seq_len, hidden_size, stream);
  } else {
    short_conv_->forward_gpu(normed_input_gpu_, attn_output_gpu_, batch,
                             seq_len, hidden_size, stream);
  }

  // 3. Residual 1: x + attn_out -> x 
  add_kernel<<<(num_elements + 255) / 256, 256, 0, stream>>>(
      output, x, attn_output_gpu_, num_elements);

  // 4. Post Attn Norm (Norm of h)
  post_attention_layernorm_->forward_gpu(output, normed_hidden_gpu_,
                                         batch * seq_len, hidden_size, stream);

  // 5. FFN (MoE or MLP)
  if (moe_block_) {
    moe_block_->forward_gpu(normed_hidden_gpu_, ffn_output_gpu_, batch, seq_len,
                            hidden_size, stream);
  } else {
    dense_mlp_->forward_gpu(normed_hidden_gpu_, ffn_output_gpu_, batch, seq_len,
                            hidden_size, stream);
  }

  // 6. Residual 2: h + ffn_out -> output
  add_kernel<<<(num_elements + 255) / 256, 256, 0, stream>>>(
      output, output, ffn_output_gpu_, num_elements);
}
void DecoderLayer::forward(const Tensor &x, const Tensor &cos,
                           const Tensor &sin, const Tensor *attention_mask,
                           Tensor &output) {}

