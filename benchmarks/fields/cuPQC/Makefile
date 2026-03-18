NVCC       := nvcc
NVCC_FLAGS := -dlto -arch=native -std=c++17 -O3

SDK_VERSION := 0.4.1
SDK_URL     := https://developer.download.nvidia.com/compute/cupqc/redist/cupqc/cupqc-sdk-$(SDK_VERSION)-x86_64.tar.gz
SDK_DIR     := cupqc-sdk-$(SDK_VERSION)-x86_64

CUPQC_INCLUDE    := $(SDK_DIR)/include/cupqc
COMMONDX_INCLUDE := $(SDK_DIR)/include
LIB_DIR          := $(SDK_DIR)/lib

# Our CUDA headers
FIELDS_DIR   := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))/..
OUR_INCLUDE  := $(FIELDS_DIR)/cuda/include
CUDA_COMMON  := $(FIELDS_DIR)/../../crates/cuda-common/include

TARGET := bench

.PHONY: all run clean distclean

all: $(TARGET)

# Download and extract SDK if not present
$(SDK_DIR):
	@echo "Downloading cuPQC SDK $(SDK_VERSION)..."
	@wget -q "$(SDK_URL)" -O cupqc-sdk.tar.gz
	@tar -xzf cupqc-sdk.tar.gz
	@rm -f cupqc-sdk.tar.gz
	@echo "cuPQC SDK extracted to $(SDK_DIR)/"

$(TARGET): bench.cu $(SDK_DIR)
	$(NVCC) $(NVCC_FLAGS) \
		-I$(COMMONDX_INCLUDE) \
		-I$(CUPQC_INCLUDE) \
		-I$(CUDA_COMMON) \
		-I$(OUR_INCLUDE) \
		-L$(LIB_DIR) -lcupqc-hash \
		-o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

distclean: clean
	rm -rf $(SDK_DIR)
