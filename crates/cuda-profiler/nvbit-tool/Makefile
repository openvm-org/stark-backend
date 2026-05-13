# NVBit tool + offline aggregator. See README.md for full usage.
#
# Build:
#   make NVBIT_PATH=/path/to/nvbit_release_x86_64/core ARCH=sm_90
#
# Produces:
#   nvbit-tool.so  — LD_PRELOAD into the CUDA app to capture per-CTA records
#   aggregate      — fold the binary log into a small per-(SM, time-bucket) grid
#
# Run (LD_PRELOAD style):
#   SHADOW_PROFILER=1 \
#   SHADOW_PROFILER_OUT=./shadow_profile.bin \
#   LD_PRELOAD=./nvbit-tool.so \
#     ./your-cuda-binary
#
# `ARCH` matches the GPU you're running on. NVBit 1.8 supports through CC 12.1.

NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas

ifeq ($(DEBUG),1)
NVCC_OPT=-O0 -g
else
NVCC_OPT=-O3
endif

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)
ifeq ($(NVCC_VER_CHECK),0)
$(error nvcc >= $(NVCC_VER_REQ) required)
endif

PTXAS_VER_ADD_FLAG=12.3
PTXAS_VER=$(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK=$(shell echo "${PTXAS_VER} >= $(PTXAS_VER_ADD_FLAG)" | bc)
ifeq ($(PTXAS_VER_CHECK), 0)
MAXRREGCOUNT_FLAG=-maxrregcount=24
else
MAXRREGCOUNT_FLAG=
endif

NVBIT_PATH ?= /home/ubuntu/nvbit_release_x86_64/core
INCLUDES=-I$(NVBIT_PATH)
LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

# `all` includes every supported real arch the local nvcc knows about. For an
# H100-only build set ARCH=sm_90 on the make command line.
ARCH ?= all

NVBIT_TOOL=nvbit-tool.so
AGGREGATE=aggregate

all: $(NVBIT_TOOL) $(AGGREGATE)

$(NVBIT_TOOL): tool.o inject_funcs.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) $(NVCC_OPT) tool.o inject_funcs.o $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -lpthread -lzstd -shared -o $@

tool.o: tool.cu
	$(NVCC) -dc -c -std=c++17 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) $(NVCC_OPT) -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

# Offline aggregator: pure C with libzstd, no CUDA dependency. Decompresses
# the SHDWPROF stream (or reads it raw) and folds it into a small per-(SM,
# time-bucket) grid that `render_overview.py` consumes.
$(AGGREGATE): aggregate.c
	$(CC) -O3 -Wall -o $@ $< -lzstd

clean:
	rm -f *.so *.o $(AGGREGATE)
