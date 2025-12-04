#pragma once

#include "tensor.h"
#include <string>
#include <unordered_map>
#include <memory>
#include <fstream>

// Model loader for single binary file format
class ModelLoader {
public:
    ModelLoader(const std::string& model_file);
    ~ModelLoader();
    
    // Load a tensor by name
    Tensor load_tensor(const std::string& name);
    
    // Check if tensor exists
    bool has_tensor(const std::string& name) const;
    
    // Get tensor info without loading
    std::vector<size_t> get_shape(const std::string& name) const;
    
private:
    struct TensorInfo {
        std::string name;
        std::vector<size_t> shape;
        uint64_t offset;
        uint64_t size;
    };
    
    std::string model_file_;
    std::unordered_map<std::string, TensorInfo> index_;
    
    void load_index();
};
