/** max_real kernel: maximum of a real array, reduced by a single block so
 * that the result lands in a preallocated one-element buffer without any
 * temporary allocation (safe inside CUDA graph capture).
 *
 * Data types:
 * - IN_TYPE: float or double
 * - OUT_TYPE: float or double
 * - BDIM_X: number of threads of the (single) block
 */

extern "C" __global__ void max_real(const IN_TYPE* a,
                                    int n,
                                    OUT_TYPE* out)
{
    const int tx = threadIdx.x;
    __shared__ OUT_TYPE sh[BDIM_X];

    OUT_TYPE maxv = OUT_TYPE(0);
    for (int i = tx; i < n; i += BDIM_X) {
        OUT_TYPE v = OUT_TYPE(a[i]);
        if (v > maxv)
            maxv = v;
    }
    sh[tx] = maxv;
    __syncthreads();

    for (int c = BDIM_X / 2; c > 0; c /= 2) {
        if (tx < c) {
            OUT_TYPE v = sh[tx + c];
            if (v > sh[tx])
                sh[tx] = v;
        }
        __syncthreads();
    }
    if (tx == 0)
        out[0] = sh[0];
}
