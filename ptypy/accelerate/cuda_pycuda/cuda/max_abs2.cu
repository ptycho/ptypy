/** max_abs2 kernel, calculating the maximum abs(x)**2 value in the last
 * two dimensions for each index in the address array
 * 
 * Data types:
 * - IN_TYPE: can be float/double or complex<float>/complex<double>
 */

#include <cmath>
#include <thrust/complex.h>
using thrust::complex;
using thrust::norm;

inline __device__ ACC_TYPE norm(const float& in) {
    return in*in;
}

inline __device__ ACC_TYPE norm(const double& in) {
    return in*in;
}

extern "C" __global__ void max_abs2(const IN_TYPE* a,
                                    int Y,
                                    int X,
                                    const int* __restrict addr,
                                    int addroffs,
                                    int rows,
                                    int cols,
                                    ACC_TYPE* out)
{
    int bid = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    const int addr_stride = 15;

    __shared__ ACC_TYPE sh[BDIM_X*BDIM_Y];

    const int* oa = addr + addroffs + bid * addr_stride;
    a += oa[0]*X*Y + oa[1]*X + oa[2];
    
    // initialise to zero
    ACC_TYPE maxv = ACC_TYPE(0);

    for (int iy = ty; iy < rows; iy += blockDim.y)
    {
        for (int ix = tx; ix < cols; ix += blockDim.x)
        {
            auto v = norm(a[iy * X + ix]);
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