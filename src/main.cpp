#include "model.h"
#include "config.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <cstring>

bool g_use_fp16 = false;

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --model PATH     Model file path (default: model.bin)" << std::endl;
    std::cout << "  --input PATH     Input file path (default: data/inputs.bin)" << std::endl;
    std::cout << "  --output PATH    Output file path (default: data/outputs.bin)" << std::endl;
    std::cout << "  --answer PATH    Answer file path for validation (default: data/answers.bin)" << std::endl;
    std::cout << "  --benchmark      Run benchmark mode" << std::endl;
    std::cout << "  --fp16           Use FP16 inference" << std::endl;
    std::cout << "  --help           Show this help" << std::endl;
}

int main(int argc, char** argv) {
    std::string model_path = "model.bin";
    std::string input_path = "data/inputs.bin";
    std::string output_path = "data/outputs.bin";
    std::string answer_path = "data/answers.bin";
    bool benchmark = false;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--answer") == 0 && i + 1 < argc) {
            answer_path = argv[++i];
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark = true;
        } else if (strcmp(argv[i], "--fp16") == 0) {
            g_use_fp16 = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "LFM2-8B GPU Inference Pipeline" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Print configuration
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Hidden size: " << HIDDEN_SIZE << std::endl;
    std::cout << "  Num layers: " << NUM_HIDDEN_LAYERS << std::endl;
    std::cout << "  Attention heads: " << NUM_ATTENTION_HEADS << std::endl;
    std::cout << "  KV heads (GQA): " << NUM_KEY_VALUE_HEADS << std::endl;
    std::cout << "  Num experts: " << NUM_EXPERTS << std::endl;
    std::cout << "  Top-k experts: " << NUM_EXPERTS_PER_TOK << std::endl;
    std::cout << "  Num GPUs: " << NUM_GPUS << std::endl;
    std::cout << "  FP16: " << (g_use_fp16 ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;
    
    // Load model
    LFM2Model model;
    try {
        model.load_weights(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        std::cerr << "Running in test mode without model weights..." << std::endl;
    }
    
    // Load input data
    std::cout << std::endl << "Loading input data..." << std::endl;
    Tensor input_ids;
    try {
        input_ids = Tensor::load_from_file(input_path);
        std::cout << "  Input shape: [" << input_ids.shape()[0] << ", " << input_ids.shape()[1] << "]" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: " << e.what() << std::endl;
        // Create dummy input for testing
        input_ids = Tensor({1, 32});
        for (size_t i = 0; i < 32; i++) {
            input_ids.at(0, i) = static_cast<float>(i + 1);
        }
        std::cout << "  Using dummy input for testing" << std::endl;
    }
    
    // Run inference
    std::cout << std::endl << "Running inference..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Tensor output;
    int num_samples = input_ids.shape()[0];
    int warmup_runs = benchmark ? 10 : 0;
    int benchmark_runs = benchmark ? 100 : 1;
    
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        output = model.forward(input_ids);
    }
    
    // Benchmark
    auto bench_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; i++) {
        output = model.forward(input_ids);
    }
    auto bench_end = std::chrono::high_resolution_clock::now();
    
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_time = std::chrono::duration<double>(end - start).count();
    double bench_time = std::chrono::duration<double>(bench_end - bench_start).count();
    double throughput = (num_samples * benchmark_runs) / bench_time;
    
    std::cout << "  Completed!" << std::endl;
    std::cout << std::endl;
    
    // Save output
    try {
        output.save_to_file(output_path);
        std::cout << "Output saved to " << output_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not save output: " << e.what() << std::endl;
    }
    
    // Validate
    std::cout << std::endl << "Validation:" << std::endl;
    try {
        Tensor answers = Tensor::load_from_file(answer_path);
        
        int correct = 0;
        float max_diff = 0.0f;
        float sum_diff = 0.0f;
        
        for (size_t i = 0; i < output.size(); i++) {
            float diff = std::abs(output.data()[i] - answers.data()[i]);
            max_diff = std::max(max_diff, diff);
            sum_diff += diff;
            if (diff < 1e-3) correct++;
        }
        
        float mean_diff = sum_diff / output.size();
        float accuracy = 100.0f * correct / output.size();
        
        std::cout << "  Samples: " << num_samples << std::endl;
        std::cout << "  Accuracy: " << accuracy << "%" << std::endl;
        std::cout << "  Max diff: " << max_diff << std::endl;
        std::cout << "  Mean diff: " << mean_diff << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Skipped (no answer file)" << std::endl;
    }
    
    // Performance report
    std::cout << std::endl << "Performance:" << std::endl;
    std::cout << "  Total time: " << total_time << " sec" << std::endl;
    std::cout << "  Throughput: " << throughput << " samples/sec" << std::endl;
    
    if (benchmark) {
        std::cout << std::endl << "Benchmark Details:" << std::endl;
        std::cout << "  Warmup runs: " << warmup_runs << std::endl;
        std::cout << "  Benchmark runs: " << benchmark_runs << std::endl;
        std::cout << "  Avg latency: " << (bench_time / benchmark_runs * 1000) << " ms" << std::endl;
    }
    
    std::cout << std::endl << "========================================" << std::endl;
    std::cout << "Done!" << std::endl;
    
    return 0;
}



