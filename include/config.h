#pragma once

#include <cstddef>
constexpr size_t VOCAB_SIZE = 65536;
constexpr size_t HIDDEN_SIZE = 2048;
constexpr size_t INTERMEDIATE_SIZE = 7168;
constexpr size_t NUM_HIDDEN_LAYERS = 24;
constexpr size_t NUM_ATTENTION_HEADS = 32;
constexpr size_t NUM_KEY_VALUE_HEADS = 8;
constexpr size_t MAX_POSITION_EMBEDDINGS = 128000;
constexpr float RMS_NORM_EPS = 1e-5f;
constexpr float ROPE_THETA = 1000000.0f;

constexpr size_t NUM_EXPERTS = 32;
constexpr size_t NUM_EXPERTS_PER_TOK = 4;
constexpr size_t NUM_DENSE_LAYERS = 2;
constexpr size_t MOE_INTERMEDIATE_SIZE = 1792;
constexpr float ROUTED_SCALING_FACTOR = 1.0f;
constexpr bool NORM_TOPK_PROB = true;
constexpr bool USE_EXPERT_BIAS = true;

constexpr size_t CONV_L_CACHE = 3;
constexpr bool USE_CONV_BIAS = false;

constexpr bool ATTENTION_BIAS = false;
constexpr float ATTENTION_DROPOUT = 0.0f;
constexpr size_t HEAD_DIM = HIDDEN_SIZE / NUM_ATTENTION_HEADS;
constexpr size_t NUM_KEY_VALUE_GROUPS = NUM_ATTENTION_HEADS / NUM_KEY_VALUE_HEADS;
constexpr size_t CONV_KERNEL = CONV_L_CACHE;

constexpr int LAYER_TYPES[] = {1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1};

constexpr int GEMM_TILE_M = 64;
constexpr int GEMM_TILE_N = 64;
constexpr int GEMM_TILE_K = 32;
constexpr int GEMM_THREADS_X = 32;
constexpr int GEMM_THREADS_Y = 8;
constexpr int GEMM_THREADS_TOTAL = 256;
constexpr int GEMM_ROWS_PER_THREAD = 8;
constexpr int GEMM_COLS_PER_THREAD = 2;
constexpr int GEMM_LOAD_ITERS = 8;
constexpr int GEMM_SHARED_B_SIZE = 2048;

constexpr int ATTN_TILE_M = 128;
constexpr int ATTN_TILE_N = 128;
constexpr int ATTN_TILE_K = 16;
constexpr int ATTN_THREAD_TILE_M = 8;
constexpr int ATTN_THREAD_TILE_N = 8;
constexpr int ATTN_PADDING = 4;

constexpr int NUM_GPUS = 4;
constexpr int NUM_STREAMS_PER_STAGE = 2;
constexpr int BUFFER_SEQ_LEN = 32;

// FP16 Configuration - runtime flag
extern bool g_use_fp16;
