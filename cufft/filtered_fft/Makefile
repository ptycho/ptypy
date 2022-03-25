NVCC = nvcc
NVCC_FLAGS += -dc -arch=sm_70 
CUDADIR = $(dir $(shell which nvcc) )/..

INCLUDES = $(shell python -m pybind11 --includes) -I$(CUDADIR)/include
PYMODEXT = $(shell python-config --extension-suffix)
CPPFLAGS += $(INCLUDES) -DMY_FFT_ROWS=128 -DMY_FFT_COLS=128
OPTFLAGS = -O3  -std=c++14
CXXFLAGS += -fPIC
LD_FLAGS += -L$(CUDADIR)/lib64 -lcufft_static -lculibos -ldl -lrt -lpthread -cudart shared
OBJ = filtered_fft.o 
OBJ_MOD = module.o 
MODULE = module$(PYMODEXT)

all: $(MODULE) $(EXE)

python: $(MODULE)

clean:
	rm -rf $(OBJ) $(EXE) $(MODULE) $(OBJ_EXE) $(OBJ_MOD)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(OPTFLAGS) -Xcompiler "$(CXXFLAGS)" $(CPPFLAGS) -c $< -o $@ 

%.o: %.cpp
	$(NVCC) $(NVCC_FLAGS) -x cu $(OPTFLAGS) -Xcompiler "$(CXXFLAGS)" $(CPPFLAGS) -c $< -o $@ 

$(MODULE): $(OBJ) $(OBJ_MOD)
	$(NVCC) $(OPTFLAGS) -shared $(LD_FLAGS) $(OBJ) $(OBJ_MOD) -o $@
