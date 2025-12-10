NVCC = nvcc
CXX = g++
CUDA_ARCH = -arch=sm_80

CXXFLAGS = -std=c++17 -O3 -Wall
NVCCFLAGS = -std=c++17 -O3 $(CUDA_ARCH) -Xcompiler -Wall

INCLUDES = -I./include
LIBS = -lcudart -lcublas

SRC_DIR = src
OBJ_DIR = obj
INC_DIR = include

# Source files
CU_SRCS = $(SRC_DIR)/tensor.cu $(SRC_DIR)/layer.cu $(SRC_DIR)/model.cu
CPP_SRCS = $(SRC_DIR)/model_loader.cpp $(SRC_DIR)/main.cpp

# Object files
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))
CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
OBJS = $(CU_OBJS) $(CPP_OBJS)

# Headers
HEADERS = $(wildcard $(INC_DIR)/*.h)

TARGET = main

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

test: $(TARGET)
	./$(TARGET) --benchmark

# Test targets
test-attn:
	cd tests/attn && make && ./main

test-conv:
	cd tests/conv && make && ./main

test-moe:
	cd tests/moe && make && ./main

test-all: test-attn test-conv test-moe



