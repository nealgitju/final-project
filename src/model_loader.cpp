#include "model_loader.h"
#include "tensor.h"
#include <iostream>
#include <cstring>

std::unique_ptr<ModelLoader> g_model_loader;

ModelLoader::ModelLoader(const std::string& model_path) : model_path_(model_path) {
    file_.open(model_path, std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot open model file: " + model_path);
    }
    build_index();
}

ModelLoader::~ModelLoader() {
    if (file_.is_open()) {
        file_.close();
    }
}

void ModelLoader::build_index() {
    // Read header
    char magic[4];
    file_.read(magic, 4);
    if (std::strncmp(magic, "LFM2", 4) != 0) {
        throw std::runtime_error("Invalid model file format");
    }
    
    // Read version
    uint32_t version;
    file_.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    
    // Read number of tensors
    uint32_t num_tensors;
    file_.read(reinterpret_cast<char*>(&num_tensors), sizeof(uint32_t));
    
    // Read tensor index
    for (uint32_t i = 0; i < num_tensors; i++) {
        // Read name length and name
        uint32_t name_len;
        file_.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        
        std::string name(name_len, '\0');
        file_.read(&name[0], name_len);
        
        // Read tensor info
        TensorInfo info;
        
        uint32_t ndim;
        file_.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));
        
        info.shape.resize(ndim);
        info.size = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint32_t dim;
            file_.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
            info.shape[d] = dim;
            info.size *= dim;
        }
        
        file_.read(reinterpret_cast<char*>(&info.offset), sizeof(size_t));
        
        tensor_index_[name] = info;
    }
    
    std::cout << "Loaded index with " << tensor_index_.size() << " tensors" << std::endl;
}

bool ModelLoader::has_tensor(const std::string& name) const {
    return tensor_index_.find(name) != tensor_index_.end();
}

Tensor ModelLoader::load_tensor(const std::string& name) {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    const TensorInfo& info = it->second;
    
    // Seek to tensor data
    file_.seekg(info.offset);
    
    // Create tensor and read data
    Tensor tensor(info.shape);
    file_.read(reinterpret_cast<char*>(tensor.data()), info.size * sizeof(float));
    
    return tensor;
}



