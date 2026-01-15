/** build_exit_alpha_tau kernel.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double - for aux wave)
 * - MATH_TYPE: the data type used for computation 
 */


#include <thrust/complex.h>
using thrust::complex;


extern "C" __global__ void build_exit_alpha_tau(
                                      complex<OUT_TYPE>* auxiliary_wave,
                                      complex<OUT_TYPE>* exit_wave,
                                      int B,
                                      int C,
                                      const complex<IN_TYPE>* __restrict__ probe,
                                      int E,
                                      int F,
                                      const complex<IN_TYPE>* __restrict__ obj,
                                      int H,
                                      int I,
                                      const int* __restrict__ addr,
                                      IN_TYPE alpha_,
                                      IN_TYPE tau_)
{
  int bid = blockIdx.z;
  int tx = threadIdx.x;
  const int b = threadIdx.y + blockIdx.y * blockDim.y;
  if (b >= B)
    return;
  const int addr_stride = 15;
  MATH_TYPE alpha = alpha_;
  MATH_TYPE tau = tau_;

  const int* oa = addr + 3 + bid * addr_stride;
  const int* pa = addr + bid * addr_stride;
  const int* ea = addr + 6 + bid * addr_stride;

  probe += pa[0] * E * F + pa[1] * F + pa[2];
  obj += oa[0] * H * I + oa[1] * I + oa[2];
  exit_wave += ea[0] * B * C;
  auxiliary_wave += ea[0] * B * C;

  for (int c = tx; c < C; c += blockDim.x)
  {
      complex<MATH_TYPE> t_aux = auxiliary_wave[b * C + c];
      complex<MATH_TYPE> t_probe = probe[b * F + c];
      complex<MATH_TYPE> t_obj = obj[b * I + c];
      complex<MATH_TYPE> t_ex = exit_wave[b * C + c];

      auto dex = tau * t_aux + (tau * alpha - MATH_TYPE(1)) * t_ex +
        (MATH_TYPE(1) - tau * (MATH_TYPE(1) + alpha)) * t_obj * t_probe;

      exit_wave[b * C + c] += dex;
      auxiliary_wave[b * C + c] = dex;
  }
}
