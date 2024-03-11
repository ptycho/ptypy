/** ob_update_wasp - in WASP algorithm.
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

extern "C" __global__ void ob_update_wasp(
    const complex<IN_TYPE>* __restrict__ exit_wave,
    const complex<IN_TYPE>* __restrict__ aux,
    int A,
    int B,
    int C,
    const complex<IN_TYPE>* __restrict__ probe,
    const IN_TYPE* __restrict__ probe_abs2,
    int D,
    int E,
    int F,
    complex<OUT_TYPE>* obj,
    complex<OUT_TYPE>* obj_sum_nmr,
    OUT_TYPE* obj_sum_dnm,
    int G,
    int H,
    int I,
    const int* __restrict__ addr,
    const IN_TYPE* __restrict__ probe_abs2_mean,
    const IN_TYPE alpha_)
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
  probe_abs2 += pa[0] * E * F + pa[1] * F + pa[2];
  obj += oa[0] * H * I + oa[1] * I + oa[2];
  obj_sum_nmr += oa[0] * H * I + oa[1] * I + oa[2];
  obj_sum_dnm += oa[0] * H * I + oa[1] * I + oa[2];
  aux += bid * B * C;
  /*the abs2 mean of this probe mode*/
  const MATH_TYPE probe_abs2_mean_val = probe_abs2_mean[pa[0]];
  const MATH_TYPE alpha = alpha_;

  assert(oa[0] * H * I + oa[1] * I + oa[2] + (B - 1) * I + C - 1 < G * H * I);

  exit_wave += ea[0] * B * C;

  for (int c = tx; c < C; c += blockDim.x)
  {
      complex<MATH_TYPE> probe_val = probe[b * F + c];
      MATH_TYPE probe_abs2_val = probe_abs2[b * F + c];
      complex<MATH_TYPE> exit_val = exit_wave[b * C + c];
      complex<MATH_TYPE> aux_val = aux[b * C + c];

      /*(pr_abs2.mean() * alpha + pr_abs2)*/
      MATH_TYPE norm_val = probe_abs2_mean_val * alpha + probe_abs2_val;

      /*0.5 * pr_conj * deltaEW / (pr_abs2.mean() * alpha + pr_abs2)*/
      auto add_val_0 = MATH_TYPE(0.5) * conj(probe_val) * (exit_val - aux_val) / norm_val;
      complex<OUT_TYPE> add_val = add_val_0;
      atomicAdd(&obj[b * I + c], add_val);

      /*pr_conj * ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]*/
      auto add_val_1 = conj(probe_val) * exit_val;
      complex<OUT_TYPE> add_val_nmr = add_val_1;
      atomicAdd(&obj_sum_nmr[b * I + c], add_val_nmr);

      /*pr_abs2*/
      OUT_TYPE add_val_dnm = probe_abs2_val;
      atomicAdd(&obj_sum_dnm[b * I + c], add_val_dnm);
  }
}
