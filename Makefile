

# Compiler
NVCC = nvcc
# CXX = g++
# Compiler Flags
# -DNDEBUG to disable asserts
# -rdc=true
NVCCFLAGS = -arch=all -Iinclude -O3 -std=c++20 -lineinfo
# CXXFLAGS = -Wall -Wextra -Iinclude -O3 -std=c++20

# Source and Include directories
SRC_DIR = src
INCLUDE_DIR = include

# Output
OUT_DIR = ./
LIBRARY = $(OUT_DIR)/libToucanGPURT.a
TEST_RUN_BIN = $(OUT_DIR)/test_run

# Source and Object files
CXX_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
OBJECTS = $(CXX_SOURCES:$(SRC_DIR)/%.cpp=$(SRC_DIR)/%.o) $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(SRC_DIR)/%.o)

# Default target
all: $(LIBRARY) $(TEST_RUN_BIN)

# Rule to make the shared library
$(LIBRARY): $(OBJECTS)
	ar rcs $@ $(OBJECTS)

$(TEST_RUN_BIN): $(LIBRARY)
	$(NVCC) $(OUT_DIR)/main.cpp -L$(OUT_DIR) -lToucanGPURT -I./include -o $(TEST_RUN_BIN)

# Rule to make object files
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(SRC_DIR)/*.o $(LIBRARY)

# Phony targets
.PHONY: all clean
