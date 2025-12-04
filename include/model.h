#pragma once

#include "config.h"
#include "layer.h"
#include "model_loader.h"
#include "tensor.h"
#include <condition_variable>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <array>

// Global model loader (defined in model.cu)
extern std::unique_ptr<ModelLoader> g_model_loader;

struct PipelineResult {
  int sample_idx;
  bool ready;
};

class LFM2Model {
public:
  LFM2Model(const std::string &model_file);
  ~LFM2Model();

  void forward(const std::vector<int>& input_ids, Tensor& logits);
  void forward(int sample_idx, int batch_size, const int* input_ids_ptr, size_t seq_len,
                     float* output_buffer);
  void sync();

private:
  std::unique_ptr<ModelLoader> loader_;

  Tensor embed_tokens_;
  float* embed_tokens_gpu_ = nullptr;
  __half* embed_tokens_gpu_fp16_ = nullptr;

  std::vector<std::unique_ptr<DecoderLayer>> layers_;
  std::unique_ptr<RMSNorm> norm_;
  Tensor lm_head_;
  float* lm_head_gpu_ = nullptr;
  __half* lm_head_gpu_fp16_ = nullptr;
  cublasHandle_t cublas_handle_ = nullptr;
  std::vector<std::unique_ptr<RotaryEmbedding>> rotary_embs_;

  void load_embeddings();
  void load_layers();
  void load_output_layers();
  void init_gpu_pipeline();
  static constexpr int NUM_STAGES = 24;
  static constexpr int PIPELINE_DEPTH = 16;
  static constexpr int BATCH_SIZE = 64;
  
  std::vector<std::array<cudaStream_t, 2>> streams_;

  struct Task {
    int sample_idx;
    int batch_size;
    const int* input_ids;
    size_t seq_len;
    float* output_buffer;
    float* prev_stage_buffer;
    float* current_stage_buffer;
    cudaEvent_t event;
  };

  struct StageContext {
    std::queue<Task> queue;
    std::mutex mutex;
    std::condition_variable cv;
    cudaEvent_t compute_event;
  };
  
  std::vector<std::unique_ptr<StageContext>> stages_;
  std::vector<std::thread> threads_;
  std::atomic<int> active_tasks_{0};
  bool stop_threads_ = false;

  struct PipelineBuffers {
    std::vector<std::vector<float*>> buffers;
  };
  PipelineBuffers pipeline_buffers_;
  
  std::vector<std::vector<cudaEvent_t>> ipc_events_;

  int num_gpus_;

  void worker_stage(int stage_id);
};
