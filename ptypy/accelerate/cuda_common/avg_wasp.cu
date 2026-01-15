/** avg_wasp
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
    avg_wasp(complex<OUT_TYPE> *arr,
             const complex<IN_TYPE>* __restrict__ nmr,
             const IN_TYPE* __restrict__ dnm,
             int A,
             int B,
             int C
             )
{
    const int bid = blockIdx.z;
    const int tx = threadIdx.x;
    const int b = threadIdx.y + blockIdx.y * blockDim.y;

    /*go to this mode*/
    arr += bid * B * C;
    nmr += bid * B * C;
    dnm += bid * B * C;

    if (b >= B)
        return;

    for (int c = tx; c < C; c += blockDim.x) {
      if (dnm[b * C + c] != 0) {
        auto avg_val_tmp = nmr[b * C + c] / dnm[b * C + c];
        complex<OUT_TYPE> avg_val = avg_val_tmp;
        arr[b * C + c] = avg_val;
      }
      else {
        complex<OUT_TYPE> avg_val = nmr[b * C + c];
        arr[b * C + c] = avg_val;
      }
    }
}
