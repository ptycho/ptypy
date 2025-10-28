/** pr_update_wasp - in WASP algorithm.
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

extern "C" __global__ void pr_update_wasp(
    const complex<IN_TYPE>* __restrict__ exit_wave,
    const complex<IN_TYPE>* __restrict__ aux,
    int A,
    int B,
    int C,
    complex<OUT_TYPE>* probe,
    complex<OUT_TYPE>* probe_sum_nmr,
    OUT_TYPE* probe_sum_dnm,
    int D,
    int E,
    int F,
    const complex<IN_TYPE>* __restrict__ obj,
    int G,
    int H,
    int I,
    const int* __restrict__ addr,
    const IN_TYPE beta_)
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
  probe_sum_nmr += pa[0] * E * F + pa[1] * F + pa[2];
  probe_sum_dnm += pa[0] * E * F + pa[1] * F + pa[2];
  obj += oa[0] * H * I + oa[1] * I + oa[2];
  aux += bid * B * C;
  const MATH_TYPE beta = beta_;

  assert(oa[0] * H * I + oa[1] * I + oa[2] + (B - 1) * I + C - 1 < G * H * I);

  exit_wave += ea[0] * B * C;

  for (int c = tx; c < C; c += blockDim.x)
  {
      complex<MATH_TYPE> obj_val = obj[b * I + c];
      MATH_TYPE obj_abs2_val = obj_val.real() * obj_val.real() + obj_val.imag() * obj_val.imag();
      complex<MATH_TYPE> exit_val = exit_wave[b * C + c];
      complex<MATH_TYPE> aux_val = aux[b * C + c];

      /*ob_conj * deltaEW / (beta + ob_abs2)*/
      auto add_val_0 = conj(obj_val) * (exit_val - aux_val) / (beta + obj_abs2_val);
      complex<OUT_TYPE> add_val = add_val_0;
      atomicAdd(&probe[b * F + c], add_val);

      /*ob_conj * ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]*/
      auto add_val_1 = conj(obj_val) * exit_val;
      complex<OUT_TYPE> add_val_nmr = add_val_1;
      atomicAdd(&probe_sum_nmr[b * F + c], add_val_nmr);

      /*ob_abs2*/
      OUT_TYPE add_val_dnm = obj_abs2_val;
      atomicAdd(&probe_sum_dnm[b * F + c], add_val_dnm);
  }
}
