/** ob_update.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - MATH_TYPE: the data type used for computation
 * - ACC_TYPE: accumulator type for the local ob accumulation
 * 
 * NOTE: This version of ob_update goes over all tiles that need to be accumulated
 * in a single thread block to avoid global atomic additions (as in ob_update.cu).
 * This requires a local array of NUM_MODES size to store the local updates.
 * GPU registers per thread are limited (255 32bit registers on V100), 
 * and at some point the registers will spill into shared or global memory
 * and the kernel will get considerably slower.
 */


#include "common.cuh"

#define pr_dlayer(k) addr[(k)]
#define ex_dlayer(k) addr[6 * num_pods + (k)]
#define obj_dlayer(k) addr[3 * num_pods + (k)]
#define obj_roi_row(k) addr[4 * num_pods + (k)]
#define obj_roi_column(k) addr[5 * num_pods + (k)]


extern "C" __global__ void ob_update2(
    int pr_sh,
    int ob_modes,
    int num_pods,
    int ob_sh_rows,
    int ob_sh_cols,
    int pr_modes,
    int ex_0,
    int ex_1,
    int ex_2,
    complex<OUT_TYPE>* ob_g,
    OUT_TYPE* obn_g,
    const complex<IN_TYPE>* __restrict__ pr_g,  // 2, 5, 5
    const complex<IN_TYPE>* __restrict__ ex_g,  // 16, 5, 5
    const int* addr)
{
  int y = blockIdx.y * BDIM_Y + threadIdx.y;
  int dy = ob_sh_rows;
  int z = blockIdx.x * BDIM_X + threadIdx.x;
  int dz = ob_sh_cols;
  complex<ACC_TYPE> ob[NUM_MODES];
  ACC_TYPE obn[NUM_MODES];

  int txy = threadIdx.y * BDIM_X + threadIdx.x;
  assert(ob_modes <= NUM_MODES);

  if (y < dy && z < dz)
  {
#pragma unroll
    for (int i = 0; i < NUM_MODES; ++i)
    {
      auto idx = i * dy * dz + y * dz + z;
      assert(idx < ob_modes * ob_sh_rows * ob_sh_cols);
      ob[i] = ob_g[idx];
      obn[i] = obn_g[idx];
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

    if (y >= dy || z >= dz)
      continue;

#pragma unroll 4
    for (int i = 0; i < mi; ++i)
    {
      int* ad = addresses + i * 5;
      int v1 = y - ad[3];
      int v2 = z - ad[4];
      if (v1 >= 0 && v1 < pr_sh && v2 >= 0 && v2 < pr_sh)
      {
        auto pridx = ad[0] * pr_sh * pr_sh + v1 * pr_sh + v2;
        assert(pridx < pr_modes * pr_sh * pr_sh);
        complex<MATH_TYPE> pr = pr_g[pridx];
        int idx = ad[2];
        assert(idx < NUM_MODES);
        auto cpr = conj(pr);
        auto exidx = ad[1] * pr_sh * pr_sh + v1 * pr_sh + v2;
        assert(exidx < ex_0 * ex_1 * ex_2);
        complex<MATH_TYPE> t_ex_g = ex_g[exidx];
        complex<ACC_TYPE> add_val = cpr * t_ex_g;
        ob[idx] += add_val;
        obn[idx] += pr.real() * pr.real() + pr.imag() * pr.imag();
      }
    }
  }

  if (y < dy && z < dz)
  {
    for (int i = 0; i < NUM_MODES; ++i)
    {
      ob_g[i * dy * dz + y * dz + z] = ob[i];
      obn_g[i * dy * dz + y * dz + z] = obn[i];
    }
  }
}
