/** pr_update.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - MATH_TYPE: the data type used for computation
 * - ACC_TYPE: accumulator type for local pr array
 * 
 * NOTE: This version of ob_update goes over all tiles that need to be accumulated
 * in a single thread block to avoid global atomic additions (as in pr_update.cu).
 * This requires a local array of NUM_MODES size to store the local updates.
 * GPU registers per thread are limited (255 32bit registers on V100), 
 * and at some point the registers will spill into shared or global memory
 * and the kernel will get considerably slower.
 */

#include "common.cuh"

#define pr_dlayer(k) addr[(k)]
#define pr_roi_row(k) addr[1 * num_pods + (k)]
#define pr_roi_column(k) addr[2 * num_pods + (k)]
#define ex_dlayer(k) addr[6 * num_pods + (k)]
#define obj_dlayer(k) addr[3 * num_pods + (k)]
#define obj_roi_row(k) addr[4 * num_pods + (k)]
#define obj_roi_column(k) addr[5 * num_pods + (k)]


extern "C" __global__ void pr_update2(int pr_sh,
                                      int ob_sh_row,
                                      int ob_sh_col,
                                      int pr_modes,
                                      int ob_modes,
                                      int num_pods,
                                      complex<OUT_TYPE>* pr_g,
                                      OUT_TYPE* prn_g,
                                      const complex<IN_TYPE>* __restrict__ ob_g,
                                      const complex<IN_TYPE>* __restrict__ ex_g,
                                      const int* addr)
{
  int y = blockIdx.y * BDIM_Y + threadIdx.y;
  int dy = pr_sh;
  int z = blockIdx.x * BDIM_X + threadIdx.x;
  int dz = pr_sh;
  complex<ACC_TYPE> pr[NUM_MODES];
  ACC_TYPE prn[NUM_MODES];

  int txy = threadIdx.y * BDIM_X + threadIdx.x;
  assert(pr_modes <= NUM_MODES);

  if (y < pr_sh && z < pr_sh)
  {
#pragma unroll
    for (int i = 0; i < NUM_MODES; ++i)
    {
      auto idx = i * dy * dz + y * dz + z;
      assert(idx < pr_modes * pr_sh * pr_sh);
      pr[i] = pr_g[idx];
      prn[i] = prn_g[idx];
    }
  }

  __shared__ int addresses[BDIM_X * BDIM_Y * 5];

  for (int p = 0; p < num_pods; p += BDIM_X * BDIM_Y)
  {
    int mi = BDIM_X * BDIM_Y;
    if (mi > num_pods - p)
      mi = num_pods - p;

    if (p > 0)
      __syncthreads();

    if (txy < mi)
    {
      assert(p + txy < num_pods);
      assert(txy < BDIM_X * BDIM_Y);
      addresses[txy * 5 + 0] = pr_dlayer(p + txy);
      addresses[txy * 5 + 1] = ex_dlayer(p + txy);
      addresses[txy * 5 + 2] = obj_dlayer(p + txy);
      assert(obj_dlayer(p + txy) < NUM_MODES);
      assert(addresses[txy * 5 + 2] < NUM_MODES);
      addresses[txy * 5 + 3] = obj_roi_row(p + txy);
      addresses[txy * 5 + 4] = obj_roi_column(p + txy);
    }

    __syncthreads();

    if (y >= pr_sh || z >= pr_sh)
      continue;

#pragma unroll 4
    for (int i = 0; i < mi; ++i)
    {
      int* ad = addresses + i * 5;
      int v1 = y + ad[3];
      int v2 = z + ad[4];
      if (v1 >= 0 && v1 < ob_sh_row && v2 >= 0 && v2 < ob_sh_col)
      {
        auto obidx = ad[2] * ob_sh_row * ob_sh_col + v1 * ob_sh_col + v2;
        assert(obidx < ob_modes * ob_sh_row * ob_sh_col);
        complex<MATH_TYPE> ob = ob_g[obidx];

        int idx = ad[0];
        assert(idx < NUM_MODES);
        auto cob = conj(ob);
        complex<MATH_TYPE> ex_val = ex_g[ad[1] * pr_sh * pr_sh + y * pr_sh + z];
        complex<ACC_TYPE> add_val = cob * ex_val;
        pr[idx] += add_val;
        prn[idx] += ob.real() * ob.real() + ob.imag() * ob.imag();
      }
    }
  }

  if (y < pr_sh && z < pr_sh)
  {
    for (int i = 0; i < NUM_MODES; ++i)
    {
      pr_g[i * dy * dz + y * dz + z] = pr[i];
      prn_g[i * dy * dz + y * dz + z] = prn[i];
    }
  }
}
