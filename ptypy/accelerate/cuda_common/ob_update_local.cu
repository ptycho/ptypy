/** ob_update_local - in DR algorithm.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - MATH_TYPE: the data type used for computation
 */

#include "common.cuh"

template <class T>
__device__ inline void atomicAdd(complex<T>* x, const complex<T>& y)
{
  auto xf = reinterpret_cast<T*>(x);
  atomicAdd(xf, y.real());
  atomicAdd(xf + 1, y.imag());
}

extern "C" __global__ void ob_update_local(
    const complex<IN_TYPE>* __restrict__ exit_wave,
    const complex<IN_TYPE>* __restrict__ aux,
    int A,
    int B,
    int C,
    const complex<IN_TYPE>* __restrict__ probe,
    int D,
    int E,
    int F,
    const IN_TYPE* __restrict__ pr_norm,
    complex<OUT_TYPE>* obj,
    int G,
    int H,
    int I,
    const int* __restrict__ addr,
    const IN_TYPE* pr_norm_max,
    const IN_TYPE A_,
    const IN_TYPE B_)
{
  const int bid = blockIdx.z;
  const int tx = threadIdx.x;
  const int b = threadIdx.y + blockIdx.y * blockDim.y;
  if (b >= B)
    return;
  const int addr_stride = 15;

  const int* oa = addr + 3 + bid * addr_stride;
  const int* pa = addr + bid * addr_stride;
  const int* ea = addr + 6 + bid * addr_stride;

  probe += pa[0] * E * F + pa[1] * F + pa[2];
  obj += oa[0] * H * I + oa[1] * I + oa[2];
  aux += bid * B * C;
  const MATH_TYPE pr_norm_max_val = pr_norm_max[0];
  const MATH_TYPE A_val = A_;
  const MATH_TYPE B_val = B_;
  
  assert(oa[0] * H * I + oa[1] * I + oa[2] + (B - 1) * I + C - 1 < G * H * I);

  exit_wave += ea[0] * B * C;

  for (int c = tx; c < C; c += blockDim.x)
  {
      complex<MATH_TYPE> probe_val = probe[b * F + c];
      complex<MATH_TYPE> exit_val = exit_wave[b * C + c];
      complex<MATH_TYPE> aux_val = aux[b * C + c];
      MATH_TYPE norm_val = (MATH_TYPE(1) - A_val) * pr_norm_max_val + A_val * pr_norm[b * F + c];

      auto add_val_m = (A_val + B_val) * conj(probe_val) * (exit_val - aux_val) / norm_val;
      complex<OUT_TYPE> add_val = add_val_m;
      atomicAdd(&obj[b * I + c], add_val);
  }
}
