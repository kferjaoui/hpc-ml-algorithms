NVCC      ?= nvcc
CXX       ?= g++
ARCH      ?= sm_89  # for my RTX 4070 (adjust to your GPU architecture)

CXXFLAGS  ?= -O3 -std=c++20

INC       := -Iinclude

NVCCFLAGS := $(CXXFLAGS) -arch=$(ARCH) $(INC)
LDFLAGS   :=

#sources
GPU_SRCS := src/dot_kernels.cu src/dot_product.test.cu
CPU_SRCS := src/dot_threads.cpp

GPU_OBJS := $(GPU_SRCS:.cu=.o)
CPU_OBJS := $(CPU_SRCS:.cpp=.o)

#binaries
GPU_BIN := dot_gpu
CPU_BIN := dot_cpu

all: $(GPU_BIN) $(CPU_BIN)

#linking
$(GPU_BIN): $(GPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CPU_BIN): $(CPU_OBJS)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(LDFLAGS)

#compile rules
src/%.o: src/%.cu include/helpers.h include/dot_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

src/%.o: src/%.cpp include/CycleTimer.h
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

clean:
	rm -f $(GPU_OBJS) $(CPU_OBJS) $(GPU_BIN) $(CPU_BIN)

.PHONY: all clean
