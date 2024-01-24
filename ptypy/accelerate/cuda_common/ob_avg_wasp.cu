/** ob_avg_wasp
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
    ob_avg_wasp(complex<OUT_TYPE> *obj,
                const complex<IN_TYPE>* __restrict__ obj_sum_nmr,
                const IN_TYPE* __restrict__ obj_sum_dnm,
                int A,
                int B,
                int C
                )
{
    const int bid = blockIdx.z;
    const int tx = threadIdx.x;
    const int b = threadIdx.y + blockIdx.y * blockDim.y;

    /*go to this object mode*/
    obj += bid * B * C;
    obj_sum_nmr += bid * B * C;
    obj_sum_dnm += bid * B * C;

    if (b >= B)
        return;

    for (int c = tx; c < C; c += blockDim.x) {
      if (obj_sum_dnm[b * C + c] != 0) {
        auto avg_val_tmp = obj_sum_nmr[b * C + c] / obj_sum_dnm[b * C + c];
        complex<OUT_TYPE> avg_val = avg_val_tmp;
        obj[b * C + c] = avg_val;
      }
      else {
        complex<OUT_TYPE> avg_val = obj_sum_nmr[b * C + c];
        obj[b * C + c] = avg_val;
      }
    }
}
