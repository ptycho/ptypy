/** pr_norm_local.
*
* Data types:
* - IN_TYPE: the data type for the inputs (float or double)
* - OUT_TYPE: the data type for the outputs (float or double)
* - MATH_TYPE: the data type used for computation
*/

#include "common.cuh"

// specify max number of threads/block and min number of blocks per SM,
// to assist the compiler in register optimisations.
// We achieve a higher occupancy in this case, as less registers are used
// (guided by profiler)
extern "C" __global__ void __launch_bounds__(1024, 2) 
    pr_norm_local(OUT_TYPE *pr_norm,
                  int A,
                  int B,
                  int C,
                  const complex<IN_TYPE>* __restrict__ probe,
                  int D,
                  int E,
                  int F,
                  const int* __restrict__ addr)
{
    const int bid = blockIdx.z;
    const int tx = threadIdx.x;
    const int b = threadIdx.y + blockIdx.y * blockDim.y;
    const int addr_stride = 15;

    const int* pa = addr + 1 + (bid * D) * addr_stride;
    const int* da = addr + 9 + (bid * D) * addr_stride;

    probe += pa[0] * E * F + pa[1] * F + pa[2];
    pr_norm += da[0] * B * C;

    if (b >= B)
        return; 

    for (int c = tx; c < C; c += blockDim.x)
    {
        MATH_TYPE acc = MATH_TYPE(0);
        for (int idx = 0; idx < D; ++idx)
        {
            complex<MATH_TYPE> probe_val = probe[b * F + c + idx * E * F];
            MATH_TYPE abs_probe_val = abs(probe_val);
            acc += abs_probe_val *
                   abs_probe_val;  // if we do this manually (real*real +imag*imag)
                                   // we get differences to numpy due to rounding
        }
        pr_norm[b * C + c] = acc;
    }

}