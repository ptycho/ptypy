
extern "C" __global__ void fill_b(
    const FTYPE* A0,
    const FTYPE* A1,
    const FTYPE* A2,
    const FTYPE* w,
    FTYPE Brenorm,
    int size,
    FTYPE* out
)
{
    int tx = threadIdx.x;
    int ix = tx + blockIdx.x * blockDim.x;
    __shared__ FTYPE smem[3][BDIM_X];

    if (ix < size) {
        smem[0][tx] = w[ix] * A0[ix] * A0[ix];
        smem[1][tx] = w[ix] * FTYPE(2) * A0[ix] * A1[ix];
        smem[2][tx] = w[ix] * (A1[ix] * A1[ix] + FTYPE(2) * A0[ix] * A2[ix]);
    } else {
        smem[0][tx] = FTYPE(0);
        smem[1][tx] = FTYPE(0);
        smem[2][tx] = FTYPE(0);
    }
    __syncthreads();

    int nt = blockDim.x;
    int c = nt;
    while (c > 1)
    {
        int half = c / 2;
        if (tx < half)
        {
          smem[0][tx] += smem[0][c - tx - 1];
          smem[1][tx] += smem[1][c - tx - 1];
          smem[2][tx] += smem[2][c - tx - 1];
        }
        __syncthreads();
        c = c - half;
    }

    if (tx == 0) {
        out[blockIdx.x*3 + 0] = smem[0][0] * Brenorm;
        out[blockIdx.x*3 + 1] = smem[1][0] * Brenorm;
        out[blockIdx.x*3 + 2] = smem[2][0] * Brenorm;
    }
}