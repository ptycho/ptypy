/** ob_norm_local.
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
    ob_norm_local(OUT_TYPE *ob_norm,
                  int A,
                  int B,
                  int C,
                  const complex<IN_TYPE>* __restrict__ obj,
                  int D,
                  int E,
                  int F,
                  const int* __restrict__ addr)
{
    const int bid = blockIdx.z;
    const int tx = threadIdx.x;
    const int b = threadIdx.y + blockIdx.y * blockDim.y;
    const int addr_stride = 15;

    const int* oa = addr + 3 + (bid * D) * addr_stride;
    const int* da = addr + 9 + (bid * D) * addr_stride;

    obj += oa[0] * E * F + oa[1] * F + oa[2];
    ob_norm += da[0] * B * C;

    if (b >= B)
        return; 

    for (int c = tx; c < C; c += blockDim.x)
    {
        MATH_TYPE acc = MATH_TYPE(0);
        for (int idx = 0; idx < D; ++idx)
        {
            complex<MATH_TYPE> obj_val = obj[b * F + c + idx * E * F];
            MATH_TYPE abs_obj_val = abs(obj_val);
            acc += abs_obj_val *
                   abs_obj_val;  // if we do this manually (real*real +imag*imag)
                                   // we get differences to numpy due to rounding
        }
        ob_norm[b * C + c] = acc;
    }

}