/** pr_update_local - for DR algorithm.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - MATH_TYPE: the data type used for computation
 * - ACC_TYPE: data type used in norm calculation (input here)
 */

#include "common.cuh"

template <class T, class U>
__device__ inline void atomicAdd(complex<T>* x, const complex<U>& y)
{
  auto xf = reinterpret_cast<T*>(x);
  atomicAdd(xf, T(y.real()));
  atomicAdd(xf + 1, T(y.imag()));
}

extern "C" __global__ void pr_update_local(
    const complex<IN_TYPE>* __restrict__ exit_wave,
    const complex<IN_TYPE>* __restrict__ aux,
    int A,
    int B,
    int C,
    complex<OUT_TYPE>* probe,
    int D,
    int E,
    int F,
    const IN_TYPE* __restrict__ ob_norm,
    const complex<IN_TYPE>* __restrict__ obj,
    int G,
    int H,
    int I,
    const int* __restrict__ addr,
    const IN_TYPE* ob_norm_max,
    const IN_TYPE A_,
    const IN_TYPE B_)
{
  assert(B == E);  // prsh[1]
  assert(C == F);  // prsh[2]
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
  const MATH_TYPE ob_norm_max_val = ob_norm_max[0];
  const MATH_TYPE A_val = A_;
  const MATH_TYPE B_val = B_;

  assert(oa[0] * H * I + oa[1] * I + oa[2] + (B - 1) * I + C - 1 < G * H * I);

  exit_wave += ea[0] * B * C;

  for (int c = tx; c < C; c += blockDim.x)
  {
      complex<MATH_TYPE> obj_val = obj[b * I + c];
      complex<MATH_TYPE> exit_val = exit_wave[b * C + c];
      complex<MATH_TYPE> aux_val = aux[b * C + c];
      MATH_TYPE norm_val = (MATH_TYPE(1) - A_val) * ob_norm_max_val + A_val * ob_norm[b * C + c];

      complex<MATH_TYPE> add_val_m = (A_val + B_val) * conj(obj_val) * (exit_val - aux_val) / norm_val;
      complex<OUT_TYPE> add_val = add_val_m;
      atomicAdd(&probe[b * F + c], add_val);
  }

}
