/** abs2sum kernel, calculating the sum of abs(x)**2 value in the first dimension
 *
 * Data types:
 * - IN_TYPE: can be float/double or complex<float>/complex<double>
 * - OUT_TYPE: can be float/double
 */

#include "common.cuh"

extern "C" __global__ void abs2sum(const IN_TYPE* a,
                                   const int n,
                                   const int rows,
                                   const int cols,
                                   OUT_TYPE* out)
{
    int tx = threadIdx.x;
    const int iy = blockIdx.y;

    for (int ix = tx; ix < cols; ix += BDIM_X) {
        OUT_TYPE acc = OUT_TYPE(0);
        for (int in = 0; in < n; ++in) {
            OUT_TYPE tmp = abs(a[in * rows * cols + iy * cols + ix]);
            acc += tmp * tmp;
        }
        out[iy * cols + ix] = acc;
    }
}

