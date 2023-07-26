/** max_abs2 kernel, calculating the sum of abs(x)**2 value in the first dimension
 * and then the maximum across the last 2 dimensions
 * 
 * Data types:
 * - IN_TYPE: can be float/double or complex<float>/complex<double>
 */

#include "common.cuh"

inline __device__ OUT_TYPE norm(const float& in) {
    return in*in;
}

inline __device__ OUT_TYPE norm(const double& in) {
    return in*in;
}

extern "C" __global__ void max_abs2_step1(const IN_TYPE* a,
                                          int n,
                                          int rows,
                                          int cols,
                                          OUT_TYPE* out)
{
    int tx = threadIdx.x;
    const int iy = blockIdx.y;
    
    __shared__ OUT_TYPE sh[BDIM_X];
    
    OUT_TYPE maxv = OUT_TYPE(0);

    for (int ix = tx; ix < cols; ix += BDIM_X) {
        OUT_TYPE v = OUT_TYPE(0); 
        for (int in = 0; in < n; ++in) {
            v += norm(a[in * rows * cols + iy * cols + ix]);
        }
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
        out[iy] = sh[0];
    }
}

extern "C" __global__ void max_abs2_step2(const OUT_TYPE* in,
                                          int n,
                                          OUT_TYPE* out)
{
    int tx = threadIdx.x;

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
        out[0] = sh[0];
    }
}
