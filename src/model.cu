#include "model.h"
#include "model_loader.h"
#include "config.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>

std::unique_ptr<ModelLoader> g_model_loader;
bool g_use_fp16 = false;
__global__ void embedding_lookup_kernel(float *hidden_states,
                                        const int *input_ids,
                                        const float *embed_table, int seq_len,
                                        int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = seq_len * hidden_size;

  if (idx < total_elements) {
    int token_idx = idx / hidden_size;
    int dim_idx = idx % hidden_size;

    int token_id = input_ids[token_idx];
    hidden_states[idx] = embed_table[token_id * hidden_size + dim_idx];
  }
}


__device__ inline float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}


__global__ void lm_head_gemm_kernel(float* __restrict__ C, const float* __restrict__ A, const float* __restrict__ B,
                                    int M, int N, int K) {
    const int BM = GEMM_TILE_M;
    const int BN = GEMM_TILE_N;
    const int BK = GEMM_TILE_K;

    int output_n_start = blockIdx.x * BM;
    int output_m_start = blockIdx.y * BM;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * GEMM_THREADS_X + tx;
    
    float acc[GEMM_ROWS_PER_THREAD][GEMM_COLS_PER_THREAD] = {0.0f};
    
    __shared__ float Bs[BK][BN];
    __shared__ float As[BM][BK];
    
    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for(int i = 0; i < GEMM_LOAD_ITERS; ++i) {
            int lidx = tid + i * GEMM_THREADS_TOTAL;
            if (lidx < GEMM_SHARED_B_SIZE) {
                int r = lidx / BN;
                int c = lidx % BN;
                float val = 0.0f;
                if (output_n_start + c < N && k + r < K) {
                    val = B[(output_n_start + c) * K + (k + r)];
                }
                Bs[r][c] = val;
            }
        }
        
        #pragma unroll
        for(int i = 0; i < GEMM_LOAD_ITERS; ++i) {
            int lidx = tid + i * GEMM_THREADS_TOTAL;
            if (lidx < GEMM_SHARED_B_SIZE) {
                int r = lidx / BK;
                int c = lidx % BK;
                float val = 0.0f;
                if (output_m_start + r < M && k + c < K) {
                    val = A[(output_m_start + r) * K + (k + c)];
                }
                As[r][c] = val;
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float b_val0 = Bs[kk][tx];
            float b_val1 = Bs[kk][tx + GEMM_THREADS_X];
            
            #pragma unroll
            for (int r = 0; r < GEMM_ROWS_PER_THREAD; ++r) {
                float a_val = As[ty * GEMM_ROWS_PER_THREAD + r][kk];
                acc[r][0] = fmaf(a_val, b_val0, acc[r][0]);
                acc[r][1] = fmaf(a_val, b_val1, acc[r][1]);
            }
        }
        __syncthreads();
    }
    
    #pragma unroll
    for (int r = 0; r < GEMM_ROWS_PER_THREAD; ++r) {
        int global_r = output_m_start + ty * GEMM_ROWS_PER_THREAD + r;
        if (global_r < M) {
            int c0 = output_n_start + tx;
            int c1 = output_n_start + tx + GEMM_THREADS_X;
            if (c0 < N) C[global_r * N + c0] = acc[r][0];
            if (c1 < N) C[global_r * N + c1] = acc[r][1];
        }
    }
}

// FP16 conversion utilities
__global__ void convert_fp32_to_fp16_kernel(__half* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void convert_fp16_to_fp32_kernel(float* dst, const __half* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

// Helper function to convert FP32 to FP16
void convert_fp32_to_fp16(__half* dst, const float* src, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    convert_fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(dst, src, n);
}

// Helper function to convert FP16 to FP32
void convert_fp16_to_fp32(float* dst, const __half* src, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    convert_fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(dst, src, n);
}

LFM2Model::LFM2Model(const std::string& model_file) {
  std::cout << "Loading LFM2-8B-A1B model from " << model_file << std::endl;
  g_model_loader = std::make_unique<ModelLoader>(model_file);

  load_embeddings();
  load_layers();
  load_output_layers();

  std::cout << "Model loaded successfully!" << std::endl;

  init_gpu_pipeline();
}

LFM2Model::~LFM2Model() {
  stop_threads_ = true;
  for (auto &stage : stages_) {
    stage->cv.notify_all();
  }

  for (auto &thread : threads_) {
    if (thread.joinable())
      thread.join();
  }

  if (embed_tokens_gpu_)
    cudaFree(embed_tokens_gpu_);
  if (embed_tokens_gpu_fp16_)
    cudaFree(embed_tokens_gpu_fp16_);
  if (lm_head_gpu_)
    cudaFree(lm_head_gpu_);
  if (lm_head_gpu_fp16_)
    cudaFree(lm_head_gpu_fp16_);
  if (cublas_handle_)
    cublasDestroy(cublas_handle_);
  
  for(auto& stage : stages_) {
      if(stage->compute_event) cudaEventDestroy(stage->compute_event);
  }
    
  for (auto& stage_events : ipc_events_) {
      for (auto evt : stage_events) {
          cudaEventDestroy(evt);
      }
  }

  for (auto& stage_bufs : pipeline_buffers_.buffers) {
    for (float *buf : stage_bufs) {
      if (buf) cudaFree(buf);
    }
  }

  for (auto &stage_streams : streams_) {
    for (auto stream : stage_streams) {
      cudaStreamDestroy(stream);
    }
  }
}

void LFM2Model::load_embeddings() {
  std::cout << "Loading embeddings..." << std::endl;
  embed_tokens_ = Tensor::load_from_file("embed_tokens.weight");
  std::cout << "  Embeddings shape: " << embed_tokens_.size(0) << " x " << embed_tokens_.size(1) << std::endl;
}

void LFM2Model::load_layers() {
  std::cout << "Loading " << NUM_HIDDEN_LAYERS << " decoder layers..." << std::endl;

  layers_.reserve(NUM_HIDDEN_LAYERS);
  for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
    bool is_attention = (LAYER_TYPES[i] == 0);
    std::cout << "  Layer " << i << ": " << (is_attention ? "Attention" : "Conv") << std::endl;
    layers_.push_back(std::make_unique<DecoderLayer>(i, is_attention));
  }
}

void LFM2Model::load_output_layers() {
  std::cout << "Loading output layers..." << std::endl;

  norm_ = std::make_unique<RMSNorm>("embedding_norm.weight");

  if (g_model_loader->has_tensor("lm_head.weight")) {
    lm_head_ = Tensor::load_from_file("lm_head.weight");
  } else {
    lm_head_ = embed_tokens_;
    std::cout << "  Using tied weights for LM head" << std::endl;
  }
}

void LFM2Model::init_gpu_pipeline() {

  cudaGetDeviceCount(&num_gpus_);
  if (num_gpus_ < NUM_GPUS) {
    fprintf(stderr, "Error: Need at least %d GPUs for pipeline, found %d.\n", NUM_GPUS, num_gpus_);
    exit(1);
  }
  streams_.resize(NUM_STAGES);
  rotary_embs_.resize(NUM_STAGES);
  stages_.resize(NUM_STAGES);
  for(int i = 0; i < NUM_STAGES; ++i) {
      stages_[i] = std::make_unique<StageContext>();
  }
  
  for (int i = 0; i < num_gpus_; ++i) {
    CHECK_CUDA(cudaSetDevice(i));
    int next = (i + 1) % 4;
    
    cudaDeviceEnablePeerAccess(next, 0);
  }

  for (int i = 0; i < NUM_STAGES; ++i) {
    int device_id = i % num_gpus_;
    CHECK_CUDA(cudaSetDevice(device_id));
    for (int j = 0; j < NUM_STREAMS_PER_STAGE; ++j) {
        CHECK_CUDA(cudaStreamCreate(&streams_[i][j]));
    }
    CHECK_CUDA(cudaEventCreateWithFlags(&stages_[i]->compute_event, cudaEventDisableTiming));
  }

  size_t layers_per_stage = NUM_HIDDEN_LAYERS / NUM_STAGES; 
  
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(
      cudaMalloc(&embed_tokens_gpu_, embed_tokens_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(embed_tokens_gpu_, embed_tokens_.data(),
                        embed_tokens_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  if (g_use_fp16) {
    CHECK_CUDA(cudaMalloc(&embed_tokens_gpu_fp16_, embed_tokens_.size() * sizeof(__half)));
    convert_fp32_to_fp16(embed_tokens_gpu_fp16_, embed_tokens_gpu_, embed_tokens_.size());
  }

  for (int s = 0; s < NUM_STAGES; ++s) {
      int device_id = s % num_gpus_;
      CHECK_CUDA(cudaSetDevice(device_id));
      rotary_embs_[s] = std::make_unique<RotaryEmbedding>();
      rotary_embs_[s]->init_gpu();
      
      size_t start_layer = s * layers_per_stage;
      size_t end_layer = start_layer + layers_per_stage;
      
      for (size_t i = start_layer; i < end_layer; ++i) {
          layers_[i]->init_gpu(device_id);
      }
  }

  pipeline_buffers_.buffers.resize(NUM_STAGES);
  ipc_events_.resize(NUM_STAGES);
  
  size_t buffer_size_bytes = (size_t)BATCH_SIZE * BUFFER_SEQ_LEN * HIDDEN_SIZE * sizeof(float);

  for (int s = 0; s < NUM_STAGES; ++s) {
      int device_id = s % num_gpus_;
      CHECK_CUDA(cudaSetDevice(device_id));
      
      ipc_events_[s].reserve(PIPELINE_DEPTH);
      for(int i = 0; i < PIPELINE_DEPTH; ++i) {
          cudaEvent_t evt;
          CHECK_CUDA(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
          ipc_events_[s].push_back(evt);
      }
      
      if (s > 0) {
          pipeline_buffers_.buffers[s].reserve(PIPELINE_DEPTH);
          for (int i = 0; i < PIPELINE_DEPTH; ++i) {
              float *buf;
              CHECK_CUDA(cudaMalloc(&buf, buffer_size_bytes));
              pipeline_buffers_.buffers[s].push_back(buf);
          }
      }
  }

  CHECK_CUDA(cudaSetDevice(num_gpus_ - 1));
  norm_->init_gpu();
  
  CHECK_CUDA(cudaMalloc(&lm_head_gpu_, lm_head_.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(lm_head_gpu_, lm_head_.data(),
                        lm_head_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  if (g_use_fp16) {
    CHECK_CUDA(cudaMalloc(&lm_head_gpu_fp16_, lm_head_.size() * sizeof(__half)));
    convert_fp32_to_fp16(lm_head_gpu_fp16_, lm_head_gpu_, lm_head_.size());
    
    // Initialize cuBLAS handle for FP16 operations
    cublasCreate(&cublas_handle_);
    cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH);
  }

  stop_threads_ = false;
  threads_.reserve(NUM_STAGES);
  for(int i=0; i<NUM_STAGES; ++i) {
      threads_.emplace_back(&LFM2Model::worker_stage, this, i);
  }
}

void LFM2Model::forward(int sample_idx, int batch_size, const int* input_ids_ptr,
                               size_t seq_len, float* output_buffer) {
  active_tasks_++;
  Task task;
  task.sample_idx = sample_idx;
  task.batch_size = batch_size;
  task.input_ids = input_ids_ptr;
  task.seq_len = seq_len;
  task.output_buffer = output_buffer;
  task.event = nullptr; 

  {
    std::unique_lock<std::mutex> lock(stages_[0]->mutex);
    stages_[0]->queue.push(task);
  }
  stages_[0]->cv.notify_one();
}

void LFM2Model::sync() {
  while (true) {
    bool empty = true;
    for (int i = 0; i < NUM_STAGES; ++i) {
        std::unique_lock<std::mutex> lock(stages_[i]->mutex);
        if (!stages_[i]->queue.empty()) {
            empty = false;
            break;
        }
    }
    
    if (empty && active_tasks_ == 0) {
        for (int i = 0; i < NUM_STAGES; ++i) {
            int device_id = i % num_gpus_;
            cudaSetDevice(device_id);
            for (int j = 0; j < NUM_STREAMS_PER_STAGE; ++j) {
                cudaStreamSynchronize(streams_[i][j]);
            }
        }
        break;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void LFM2Model::worker_stage(int stage_id) {
  int device_id = stage_id % num_gpus_;
  CHECK_CUDA(cudaSetDevice(device_id));
  
  while (!stop_threads_) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(stages_[stage_id]->mutex);
      stages_[stage_id]->cv.wait(
          lock, [this, stage_id] { return !stages_[stage_id]->queue.empty() || stop_threads_; });
      if (stop_threads_ && stages_[stage_id]->queue.empty())
        return;
      task = stages_[stage_id]->queue.front();
      stages_[stage_id]->queue.pop();
    }

    int batch_id = task.sample_idx / task.batch_size;
    cudaStream_t stream = streams_[stage_id][batch_id % 2];

    int seq_len = task.seq_len;
    int batch_size = task.batch_size;
    int total_tokens = batch_size * seq_len;
    
    float *hidden_states = nullptr;

    if (stage_id == 0) {
        int *d_input_ids;
        CHECK_CUDA(cudaMallocAsync(&d_input_ids, total_tokens * sizeof(int), stream));
        CHECK_CUDA(cudaMemcpyAsync(d_input_ids, task.input_ids,
                                   total_tokens * sizeof(int), cudaMemcpyHostToDevice,
                                   stream));

        CHECK_CUDA(cudaMallocAsync(
            &hidden_states, total_tokens * HIDDEN_SIZE * sizeof(float), stream));

        const int EMBED_THREADS = 256;
        int num_blocks = (total_tokens * HIDDEN_SIZE + EMBED_THREADS - 1) / EMBED_THREADS;
        embedding_lookup_kernel<<<num_blocks, EMBED_THREADS, 0, stream>>>(
            hidden_states, d_input_ids, embed_tokens_gpu_, total_tokens, HIDDEN_SIZE); 

        CHECK_CUDA(cudaFreeAsync(d_input_ids, stream));
    } else {
        CHECK_CUDA(cudaStreamWaitEvent(stream, task.event, 0));
        CHECK_CUDA(cudaMallocAsync(
            &hidden_states, total_tokens * HIDDEN_SIZE * sizeof(float), stream));
        CHECK_CUDA(cudaMemcpyAsync(hidden_states, task.prev_stage_buffer,
                                   total_tokens * HIDDEN_SIZE * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    
    CHECK_CUDA(cudaStreamWaitEvent(stream, stages_[stage_id]->compute_event, 0));

    size_t layers_per_stage = NUM_HIDDEN_LAYERS / NUM_STAGES;
    size_t start_layer = stage_id * layers_per_stage;
    size_t end_layer = start_layer + layers_per_stage;

    for (size_t i = start_layer; i < end_layer; ++i) {
        layers_[i]->forward_gpu(hidden_states, rotary_embs_[stage_id]->cos_gpu_,
                                rotary_embs_[stage_id]->sin_gpu_, hidden_states, batch_size, seq_len,
                                HIDDEN_SIZE, stream);
    }
    
    CHECK_CUDA(cudaEventRecord(stages_[stage_id]->compute_event, stream));
    
    if (stage_id < NUM_STAGES - 1) {
        int next_stage = stage_id + 1;
        int slot = (task.sample_idx / task.batch_size) % PIPELINE_DEPTH;
        float* dst_buffer = pipeline_buffers_.buffers[next_stage][slot];
        
        CHECK_CUDA(cudaMemcpyAsync(
            dst_buffer, hidden_states,
            total_tokens * HIDDEN_SIZE * sizeof(float), cudaMemcpyDefault, stream));
            
        cudaEvent_t signal_evt = ipc_events_[stage_id][slot];
        CHECK_CUDA(cudaEventRecord(signal_evt, stream));
        task.event = signal_evt;
        CHECK_CUDA(cudaFreeAsync(hidden_states, stream));
        task.prev_stage_buffer = dst_buffer;
        
        {
            std::unique_lock<std::mutex> lock(stages_[next_stage]->mutex);
            stages_[next_stage]->queue.push(task);
        }
        stages_[next_stage]->cv.notify_one();
    } else {
        float *normed_out;
        CHECK_CUDA(cudaMallocAsync(
            &normed_out, total_tokens * HIDDEN_SIZE * sizeof(float), stream));
                
        norm_->forward_gpu(hidden_states, normed_out, total_tokens, HIDDEN_SIZE, stream);
      
        float* last_hiddens;
        CHECK_CUDA(
            cudaMallocAsync(&last_hiddens, batch_size * HIDDEN_SIZE * sizeof(float), stream));            
        
        CHECK_CUDA(cudaMemcpy2DAsync(
            last_hiddens,
            HIDDEN_SIZE * sizeof(float),
            normed_out + (seq_len - 1) * HIDDEN_SIZE,
            seq_len * HIDDEN_SIZE * sizeof(float),
            HIDDEN_SIZE * sizeof(float),
            batch_size,
            cudaMemcpyDeviceToDevice,
            stream));

        float *logits_gpu;
        CHECK_CUDA(
            cudaMallocAsync(&logits_gpu, batch_size * VOCAB_SIZE * sizeof(float), stream));
        
        int n_vocab = VOCAB_SIZE;
        int n_hidden = HIDDEN_SIZE;
        
        if (g_use_fp16) {
          // Use FP16 Tensor Core GEMM via cuBLAS
          __half *last_hiddens_fp16;
          __half *logits_fp16;
          CHECK_CUDA(cudaMallocAsync(&last_hiddens_fp16, batch_size * HIDDEN_SIZE * sizeof(__half), stream));
          CHECK_CUDA(cudaMallocAsync(&logits_fp16, batch_size * VOCAB_SIZE * sizeof(__half), stream));
          
          // Convert input to FP16
          convert_fp32_to_fp16(last_hiddens_fp16, last_hiddens, batch_size * HIDDEN_SIZE, stream);
          
          // cuBLAS FP16 GEMM: C = A * B^T
          // Original kernel: C[MxN] = A[MxK] * B^T[KxN] where:
          //   A: [batch_size x HIDDEN_SIZE] (row-major) = last_hiddens_fp16
          //   B: [VOCAB_SIZE x HIDDEN_SIZE] (row-major) = lm_head_gpu_fp16_
          //   C: [batch_size x VOCAB_SIZE] (row-major) = logits_fp16
          // cuBLAS computes: C = alpha * op(A) * op(B) + beta * C (column-major)
          // To get C_row = A_row * B_row^T, we use:
          //   op(A) = A_row (CUBLAS_OP_N), op(B) = B_row^T (CUBLAS_OP_T)
          //   Dimensions: m=batch_size, n=n_vocab, k=n_hidden
          const __half alpha = __float2half(1.0f);
          const __half beta = __float2half(0.0f);
          
          cublasSetStream(cublas_handle_, stream);
          // cuBLAS column-major: C = A * B^T
          // A: [batch_size x n_hidden] row-major, interpreted as [n_hidden x batch_size] col-major
          // B^T: [n_hidden x n_vocab] row-major view, interpreted as [n_vocab x n_hidden] col-major
          // C: [batch_size x n_vocab] row-major, interpreted as [n_vocab x batch_size] col-major
          cublasGemmEx(cublas_handle_,
                       CUBLAS_OP_N, CUBLAS_OP_T,
                       batch_size, n_vocab, n_hidden,
                       &alpha,
                       last_hiddens_fp16, CUDA_R_16F, n_hidden,   // A: [batch_size x n_hidden] row-major, lda=n_hidden
                       lm_head_gpu_fp16_, CUDA_R_16F, n_hidden,   // B: [n_vocab x n_hidden] row-major, lda=n_hidden
                       &beta,
                       logits_fp16, CUDA_R_16F, n_vocab,           // C: [batch_size x n_vocab] row-major, ldc=n_vocab
                       CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
          
          // Convert output back to FP32
          convert_fp16_to_fp32(logits_gpu, logits_fp16, batch_size * VOCAB_SIZE, stream);
          
          CHECK_CUDA(cudaFreeAsync(last_hiddens_fp16, stream));
          CHECK_CUDA(cudaFreeAsync(logits_fp16, stream));
        } else {
          dim3 grid((n_vocab + GEMM_TILE_N - 1) / GEMM_TILE_N, (batch_size + GEMM_TILE_M - 1) / GEMM_TILE_M);
          dim3 block(GEMM_THREADS_X, GEMM_THREADS_Y);

          lm_head_gemm_kernel<<<grid, block, 0, stream>>>(
               logits_gpu,
               last_hiddens,
               lm_head_gpu_,
               batch_size, n_vocab, n_hidden);
        }

        CHECK_CUDA(cudaMemcpyAsync(task.output_buffer, logits_gpu,
                                   batch_size * VOCAB_SIZE * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
                                   
        CHECK_CUDA(cudaFreeAsync(hidden_states, stream));
        CHECK_CUDA(cudaFreeAsync(normed_out, stream));
        CHECK_CUDA(cudaFreeAsync(last_hiddens, stream));
        CHECK_CUDA(cudaFreeAsync(logits_gpu, stream));
        
        active_tasks_--;
    }
  }
}

void LFM2Model::forward(const std::vector<int>& input_ids, Tensor& logits) {
  static bool warned = false;
  if (!warned) {
    std::cerr << "Warning: LFM2Model::forward (legacy/warmup) is not implemented for GPU pipeline." << std::endl;
    warned = true;
  }
}
