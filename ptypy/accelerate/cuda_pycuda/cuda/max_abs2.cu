/** max_abs2 kernel, calculating the maximum abs(x)**2 value in the last
 * two dimensions 
 * 
 * Data types:
 * - IN_TYPE: can be float/double or complex<float>/complex<double>
 */

#include <cmath>
#include <thrust/complex.h>
using thrust::complex;
using thrust::norm;

inline __device__ OUT_TYPE norm(const float& in) {
    return in*in;
}

inline __device__ OUT_TYPE norm(const double& in) {
    return in*in;
}

extern "C" __global__ void max_abs2(const IN_TYPE* a,
                                    int rows,
                                    int cols,
                                    OUT_TYPE* out)
{
    int bid = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // offset a to get to the current row
    a += bid * rows * cols;
    
    __shared__ OUT_TYPE sh[BDIM_X*BDIM_Y];

    // initialise to zero
    OUT_TYPE maxv = OUT_TYPE(0);

    for (int iy = ty; iy < rows; iy += blockDim.y)
    {
        for (int ix = tx; ix < cols; ix += blockDim.x)
        {
            auto v = norm(a[iy * cols + ix]);
            if (maxv < v)
                maxv = v;
        }
    }

    int txy = ty * BDIM_X + tx;
    sh[txy] = maxv;
    __syncthreads();

    // reduce:
    const int nt = BDIM_X*BDIM_Y;
    int c = nt;
    
    while (c > 1)
    {
        int half = c / 2;
        if (txy < half)
        {
            auto v = sh[c - txy - 1];
            if (maxv < v) {
                sh[txy] = v;
                maxv = v;
            }
        }
        __syncthreads();
        c = c - half;
    }

    if (txy == 0)
    {
        out[bid] = sh[0];
    }
}