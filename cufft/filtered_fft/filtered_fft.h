#pragma once 

#include <thrust/complex.h>
#include <cuda_runtime.h>

using thrust::complex;

class FilteredFFT {
public:
    virtual void fft(complex<float>* input, complex<float>* output) = 0;
    virtual void ifft(complex<float>* input, complex<float>* output) = 0;
    virtual int getBatches() const = 0;
    virtual int getRows() const = 0;
    virtual int getColumns() const = 0;
    virtual bool isForward() const = 0;
    virtual void setStream(cudaStream_t stream) = 0;
    virtual cudaStream_t getStream() const = 0;
    virtual ~FilteredFFT() {}
};

// we fix the rows/columns at compile-time, so not passing them
// to the factory here
// Note that cudaStream_t (runtime API) and CUStream (driver API) are
// the same type
FilteredFFT* make_filtered(int batches, 
  int rows, int columns,
  bool symmetricScaling,
  bool isForward,
  complex<float>* prefilt, complex<float>* postfilt, 
  cudaStream_t stream);

void destroy_filtered(FilteredFFT* fft);


