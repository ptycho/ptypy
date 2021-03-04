/** fill_b kernel.
 * Data types:
 * - IN_TYPE: the data type for the inputs
 * - OUT_TYPE: the data type for the outputs
 * - MATH_TYPE: the data type used for computation
 * - ACC_TYPE: the accumulator type for summing
 */

extern "C" __global__ void fill_b(const IN_TYPE* A0,
                                  const IN_TYPE* A1,
                                  const IN_TYPE* A2,
                                  const IN_TYPE* w,
                                  IN_TYPE Brenorm,
                                  int size,
                                  OUT_TYPE* out)
{
  int tx = threadIdx.x;
  int ix = tx + blockIdx.x * blockDim.x;
  __shared__ ACC_TYPE smem[3][BDIM_X];

  if (ix < size)
  {
    // MATHTYPE(2) to make sure it's float in single precision and doesn't
    // accidentally promote the equation to double
    MATH_TYPE t_a0 = A0[ix];
    MATH_TYPE t_a1 = A1[ix];
    MATH_TYPE t_a2 = A2[ix];
    MATH_TYPE t_w = w[ix];
    smem[0][tx] = t_w * t_a0 * t_a0;
    smem[1][tx] = t_w * MATH_TYPE(2) * t_a0 * t_a1;
    smem[2][tx] = t_w * (t_a1 * t_a1 + MATH_TYPE(2) * t_a0 * t_a2);
  }
  else
  {
    smem[0][tx] = ACC_TYPE(0);
    smem[1][tx] = ACC_TYPE(0);
    smem[2][tx] = ACC_TYPE(0);
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

  if (tx == 0)
  {
    out[blockIdx.x * 3 + 0] = MATH_TYPE(smem[0][0]) * MATH_TYPE(Brenorm);
    out[blockIdx.x * 3 + 1] = MATH_TYPE(smem[1][0]) * MATH_TYPE(Brenorm);
    out[blockIdx.x * 3 + 2] = MATH_TYPE(smem[2][0]) * MATH_TYPE(Brenorm);
  }
}