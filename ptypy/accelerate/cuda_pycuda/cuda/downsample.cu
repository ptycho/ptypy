#include <thrust/complex.h>
using thrust::complex;

extern "C" __global__ void downsample(TYPE* A,
                                    const TYPE* B,
                                    int Brows,
                                    int Bcols,
                                    int factor)
{
    // shared mem is BDIMx x BDIMy, declared on the call-side
    extern __shared__ char sh_raw[];
    auto block = reinterpret_cast<TYPE*>(sh_raw);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = threadIdx.x + blockIdx.x * blockDim.x;
    const int by = threadIdx.y + blockIdx.y * blockDim.y;

    // offset A and B to the right point in first dimension
    A += blockIdx.z * Brows / factor * Bcols / factor;
    B += blockIdx.z * Brows * Bcols; 

    // read input data into shared memory
    if (bx < Bcols && by < Brows) {
        block[ty * blockDim.x + tx] = B[by * Bcols + bx];
    } else {
        block[ty * blockDim.x + tx] = TYPE(0);
    }
    __syncthreads();

    // reduce across the downsampling area
    // first in x direction
    int c = 1;
    while (c != factor) 
    {
        c *= 2;
        if (tx % c == 0) {
            block[ty * blockDim.x + tx] += block[ty * blockDim.x + tx + c/2];
        }
        __syncthreads();
    }

    // now in y direction
    c = 1;
    while (c != factor) 
    {
        c *= 2;
        if (ty % c == 0) {
            block[ty * blockDim.x + tx] += block[(ty +c/2) * blockDim.x + tx];
        }
        __syncthreads();
    }

    // now write out
    if (tx % factor == 0 && ty % factor == 0 && bx < Bcols && by < Brows) {
        const int ax = bx / factor;
        const int ay = by / factor;
        A[ay * Bcols / factor + ax] = block[ty * blockDim.x + tx];
    }
}