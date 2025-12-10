# LFM2-8B CUDA Implementation

LFM2-8B-A1B 모델의 CUDA 기반 추론 엔진 구현

## 모델 구조

- **24 Decoder Layers**: 6 Attention + 18 Conv (Mamba-style)
- **Mixture of Experts (MoE)**: 32 experts, top-4 routing
- **Grouped Query Attention (GQA)**: 32 query heads, 8 KV heads
- **Hidden Size**: 2048
- **Vocabulary Size**: 65536

## 주요 구현 기법

### GEMM 최적화
- 128x128 Tiling with Double Buffering
- Float4 Vectorized Memory Access
- Shared Memory Bank Conflict 회피 (Padding)

### Attention
- Fused GQA Kernel
- RoPE + Transpose 통합
- Warp-level Reduction

### MoE
- Persistent Kernel (Dynamic Task Distribution)
- W1/W3 Fusion
- GPU-side Scatter/Scan

### Multi-GPU Pipeline
- 4-GPU Pipeline Parallelism
- P2P Memory Transfer
- Double Stream Execution

## 빌드

```bash
make clean && make
```

## 실행

```bash
./run.sh
```

## 성능

- **Throughput**: 358 samples/sec
- **Top-1 Accuracy**: 0.538

## 파일 구조

```
├── include/
│   ├── config.h          # 모델 설정
│   ├── tensor.h          # Tensor 클래스
│   ├── layer.h           # Layer 클래스
│   ├── model.h           # Model 클래스
│   └── model_loader.h    # 가중치 로더
├── src/
│   ├── tensor.cu         # Tensor 구현
│   ├── layer.cu          # CUDA 커널
│   ├── model.cu          # 모델 구현
│   ├── model_loader.cpp  # 로더 구현
│   └── main.cpp          # 메인
├── tests/                # 단위 테스트
├── Makefile
└── run.sh
```

