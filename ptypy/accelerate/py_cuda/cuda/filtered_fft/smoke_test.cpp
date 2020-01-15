#include "errors.hpp"
#include "filtered_fft.hpp"
#include <cuda.h>
#include <iostream>
#include <chrono>

#ifndef MY_FFT_ROWS
# define MY_FFT_ROWS 128
# pragma GCC warning "MY_FFT_ROWS not set in preprocessor - defaulting to 128"
#endif

#ifndef MY_FFT_COLS
# define MY_FFT_COLS 128
# pragma GCC warning "MY_FFT_COLS not set in preprocessor - defaulting to 128"
#endif


///////////////// a quick smoke test
int main() {
    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));

    int batches = 2000;
    int rows = MY_FFT_ROWS;
    int cols = MY_FFT_COLS;
    complex<float> *pre, *post, *f;
    cudaCheck(cudaMalloc((void**)&pre, rows*cols*sizeof(complex<float>)));
    cudaCheck(cudaMalloc((void**)&post, rows*cols*sizeof(complex<float>)));
    cudaCheck(cudaMalloc((void**)&f, batches*rows*cols*sizeof(complex<float>)));

    auto fft = make_filtered(batches, true, pre, post, stream);

    if (rows != fft->getRows() || cols != fft->getColumns())
        throw std::runtime_error("Mismatch in rows/cols between smoke test and module");

    cudaCheck(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i)
        fft->fft(f, f);
    cudaCheck(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "FFT " << batches << "x" << rows << "x" << cols << 
        ": " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms\n";

    cudaCheck(cudaDeviceSynchronize());
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i)
        fft->ifft(f, f);
    cudaCheck(cudaDeviceSynchronize());
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "IFFT " << batches << "x" << rows << "x" << cols << 
        ": " << std::chrono::duration_cast<std::chrono::milliseconds>(end2-start2).count() << "ms\n";
    
    std::cout << "Done\n";

    destroy_filtered(fft);

    cudaStreamDestroy(stream);
    cudaFree(pre);
    cudaFree(post);
    cudaFree(f);
}
