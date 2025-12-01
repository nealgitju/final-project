#pragma once

#include <cstddef>

// Model parameters - LFM2-8B
constexpr size_t VOCAB_SIZE = 65536;
constexpr size_t HIDDEN_SIZE = 2048;
constexpr size_t INTERMEDIATE_SIZE = 7168;
constexpr size_t NUM_HIDDEN_LAYERS = 24;
constexpr size_t NUM_ATTENTION_HEADS = 32;
constexpr size_t NUM_KEY_VALUE_HEADS = 8;
constexpr size_t MAX_POSITION_EMBEDDINGS = 128000;
constexpr float RMS_NORM_EPS = 1e-5f;
constexpr float ROPE_THETA = 1000000.0f;

// Layer types: 0=Attention, 1=Conv
constexpr int LAYER_TYPES[] = {
    1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1,
    1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1
};

// GEMM parameters (초기 버전 - 작은 타일)
constexpr int GEMM_TILE_M = 32;
constexpr int GEMM_TILE_N = 32;
constexpr int GEMM_TILE_K = 8;

