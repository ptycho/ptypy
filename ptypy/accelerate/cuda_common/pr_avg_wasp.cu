/** pr_avg_wasp
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
    pr_avg_wasp(complex<OUT_TYPE> *probe,
                const complex<IN_TYPE>* __restrict__ probe_sum_nmr,
                const IN_TYPE* __restrict__ probe_sum_dnm,
                int A,
                int B,
                int C
                )
{
    const int bid = blockIdx.z;
    const int tx = threadIdx.x;
    const int b = threadIdx.y + blockIdx.y * blockDim.y;

    /*go to this probe mode*/
    probe += bid * B * C;
    probe_sum_nmr += bid * B * C;
    probe_sum_dnm += bid * B * C;

    if (b >= B)
        return;

    for (int c = tx; c < C; c += blockDim.x) {
      if (probe_sum_dnm[b * C + c] != 0) {
        auto avg_val_tmp = probe_sum_nmr[b * C + c] / probe_sum_dnm[b * C + c];
        complex<OUT_TYPE> avg_val = avg_val_tmp;
        probe[b * C + c] = avg_val;
      }
      else {
        complex<OUT_TYPE> avg_val = probe_sum_nmr[b * C + c];
        probe[b * C + c] = avg_val;
      }
    }
}
