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

extern "C" __global__ void max_abs2_step1(const IN_TYPE* a,
                                          int rows,
                                          int cols,
                                          OUT_TYPE* out)
{
    
    int bid = blockIdx.z;
    int tx = threadIdx.x;
    const int iy = blockIdx.y;
    
    // offset a to get to the current row
    a += bid * rows * cols;
    
    __shared__ OUT_TYPE sh[BDIM_X];

    
    OUT_TYPE maxv = OUT_TYPE(0);

    for (int ix = tx; ix < cols; ix += BDIM_X) {
        auto v = norm(a[iy * cols + ix]);
        if (v > maxv)
            maxv = v;
    }

    
    sh[tx] = maxv; 
    
    __syncthreads();

    // reduce:
    const int nt = BDIM_X;
    int c = nt;
    
    while (c > 1)
    {
        int half = c / 2;
        if (tx < half)
        {
            auto v = sh[c - tx - 1];
            if (maxv < v) {
                sh[tx] = v;
                maxv = v;
            }
        }
        __syncthreads();
        c = c - half;
    }

    if (tx == 0)
    {
        out[bid * gridDim.y + blockIdx.y] = sh[0];
    }
}

extern "C" __global__ void max_abs2_step2(const OUT_TYPE* in,
                                          int n,
                                          OUT_TYPE* out)
{
    int tx = threadIdx.x;
    int bid = blockIdx.x;

    in += blockIdx.x * n;

    __shared__ OUT_TYPE sh[BDIM_X];

    OUT_TYPE maxv = OUT_TYPE(0);
    for (int i = tx; i < n; ++i) {
        auto v = in[i];
        if (v > maxv)
            maxv = v;
    }
    sh[tx] = maxv;
    __syncthreads();

    // reduce:
    const int nt = BDIM_X;
    int c = nt;
    
    while (c > 1)
    {
        int half = c / 2;
        if (tx < half)
        {
            auto v = sh[c - tx - 1];
            if (maxv < v) {
                sh[tx] = v;
                maxv = v;
            }
        }
        __syncthreads();
        c = c - half;
    }

    if (tx == 0)
    {
        out[bid] = sh[0];
    }
}
