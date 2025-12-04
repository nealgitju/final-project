#pragma once

#include "config.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

// RMSNorm Layer
class RMSNorm {
public:
  RMSNorm(const std::string &weight_file);
  ~RMSNorm();
  void forward(const Tensor &x, Tensor &y);

  // GPU support
  void init_gpu();
  void forward_gpu(float *x, float *y, int rows, int hidden_dim,
                   cudaStream_t stream = 0);

private:
  Tensor weight_;
  float *weight_gpu_ = nullptr;
};

// Rotary Position Embedding
class RotaryEmbedding {
public:
  RotaryEmbedding();
  ~RotaryEmbedding();
  void forward(size_t seq_len, Tensor &cos, Tensor &sin);

  // GPU support
  void init_gpu();
  void forward_gpu(size_t seq_len, float *cos, float *sin,
                   cudaStream_t stream = 0);

  float *cos_gpu_ = nullptr;
  float *sin_gpu_ = nullptr;

private:
  Tensor cos_cached_;
  Tensor sin_cached_;
  size_t max_seq_len_;
};

// MLP Layer (Feed-Forward Network)
class MLP {
public:
  MLP(const std::string &w1_file, const std::string &w2_file,
      const std::string &w3_file);
  ~MLP();
  void forward(const Tensor &x, Tensor &y);

  // GPU support
  void init_gpu();
  void forward_gpu(float *x, float *y, int batch, int seq_len, int hidden_size,
                   cudaStream_t stream = 0);

  float *w1_gpu_ = nullptr;
  float *w3_gpu_ = nullptr;
  float *w2_gpu_ = nullptr;
  float *temp_buf_gpu_ = nullptr;

private:
  Tensor w1_; // up projection
  Tensor w3_; // gate projection
  Tensor w2_; // down projection
};

// Sparse MoE Block
class SparseMoeBlock {
public:
  SparseMoeBlock(int layer_idx);
  ~SparseMoeBlock();
  void forward(const Tensor &x, Tensor &y, Tensor &router_logits);

  // GPU support
  void init_gpu();
  void forward_gpu(float *x, float *y, int batch, int seq_len, int hidden_size,
                   cudaStream_t stream = 0);

private:
  Tensor gate_; // router
  std::vector<MLP> experts_;
  Tensor expert_bias_; // optional

  void route_tokens(const Tensor &router_logits,
                    std::vector<int> &top_k_indices,
                    std::vector<float> &top_k_weights);

  // GPU resources
  float *gate_gpu_ = nullptr;
  float *expert_bias_gpu_ = nullptr;

  float **expert_w1_gpu_ptrs_ = nullptr;
  float **expert_w2_gpu_ptrs_ = nullptr;
  float **expert_w3_gpu_ptrs_ = nullptr;

  int *expert_counts_gpu_ = nullptr;
  int *expert_offsets_gpu_ = nullptr;
  int *sorted_indices_gpu_ = nullptr;
  float *sorted_weights_gpu_ = nullptr;
  float *sorted_x_gpu_ = nullptr;
  float *sorted_inter_gpu_ = nullptr;
  float *sorted_out_gpu_ = nullptr;
  int *topk_indices_gpu_ = nullptr;
  float *topk_weights_gpu_ = nullptr;

  int *h_expert_counts_ = nullptr;
  int *h_expert_offsets_ = nullptr;
  int *token_expert_pos_gpu_ = nullptr;

  float **expert_w1_ptrs_host_ = nullptr;
  float **expert_w2_ptrs_host_ = nullptr;
  float **expert_w3_ptrs_host_ = nullptr;

  // Optimizations
  int *expert_tile_offsets_gpu_ = nullptr;
  int *task_counters_gpu_ = nullptr;
  int *expert_write_pos_gpu_ = nullptr;
  int layer_idx_;
};

// Multi-Head Attention
class Attention {
public:
  Attention(int layer_idx);
  ~Attention();
  void forward(const Tensor &x, const Tensor &cos, const Tensor &sin,
               const Tensor *attention_mask, Tensor &output);

  // GPU support
  void init_gpu();
  void forward_gpu(float *x, float *cos, float *sin, float *output, int batch,
                   int seq_len, int hidden_size, cudaStream_t stream = 0);

private:
  Tensor q_proj_;
  Tensor k_proj_;
  Tensor v_proj_;
  Tensor o_proj_;
  std::unique_ptr<RMSNorm> q_layernorm_;
  std::unique_ptr<RMSNorm> k_layernorm_;
  int layer_idx_;

  float *q_proj_gpu_ = nullptr;
  float *k_proj_gpu_ = nullptr;
  float *v_proj_gpu_ = nullptr;
  float *o_proj_gpu_ = nullptr;

  float *q_proj_out_gpu_ = nullptr;
  float *k_proj_out_gpu_ = nullptr;
  float *v_proj_out_gpu_ = nullptr;
  float *q_normed_gpu_ = nullptr;
  float *k_normed_gpu_ = nullptr;
  float *q_transposed_gpu_ = nullptr;
  float *k_transposed_gpu_ = nullptr;
  float *v_transposed_gpu_ = nullptr;
  float *attn_out_transposed_gpu_ = nullptr;

  // GPU Norm pointers are managed by q_layernorm_ / k_layernorm_
};

// Short Convolution (Mamba-style)
class ShortConv {
public:
  ShortConv(int layer_idx);
  ~ShortConv();
  void forward(const Tensor &x, Tensor &y);

  // GPU support
  void init_gpu();
  void forward_gpu(float *x, float *y, int batch, int seq_len, int hidden_size,
                   cudaStream_t stream = 0);

private:
  Tensor conv_weight_;
  Tensor conv_bias_;
  Tensor in_proj_weight_;
  Tensor in_proj_bias_;
  Tensor out_proj_weight_;
  Tensor out_proj_bias_;
  int layer_idx_;

  float *conv_weight_gpu_ = nullptr;
  float *in_proj_weight_gpu_ = nullptr;
  float *out_proj_weight_gpu_ = nullptr;

  float *in_proj_out_gpu_ = nullptr;
  float *Bx_gpu_ = nullptr;
  float *C_gpu_ = nullptr;
  float *conv_out_gpu_ = nullptr;
  float *y_transposed_gpu_ = nullptr;
};

// Decoder Layer
class DecoderLayer {
public:
  DecoderLayer(int layer_idx, bool is_attention_layer);
  ~DecoderLayer();
  void forward(const Tensor &x, const Tensor &cos, const Tensor &sin,
               const Tensor *attention_mask, Tensor &output);

  // GPU support
  void init_gpu(int device_id);
  void forward_gpu(float *x, float *cos, float *sin, float *output, int batch,
                   int seq_len, int hidden_size, cudaStream_t stream = 0);

  bool is_attention_layer() const { return is_attention_layer_; }

private:
  int layer_idx_;
  bool is_attention_layer_;

  // Components
  std::unique_ptr<RMSNorm> input_layernorm_;
  std::unique_ptr<RMSNorm> post_attention_layernorm_;

  // Either attention or conv
  std::unique_ptr<Attention> self_attn_;
  std::unique_ptr<ShortConv> short_conv_;

  // Either MoE block (layers >= 2) or dense MLP (layers 0-1)
  std::unique_ptr<SparseMoeBlock> moe_block_;
  std::unique_ptr<MLP> dense_mlp_;

  // Intermediate buffers
  float *normed_input_gpu_ = nullptr;
  float *attn_output_gpu_ = nullptr;
  float *normed_hidden_gpu_ = nullptr;
  float *ffn_output_gpu_ = nullptr;
};
