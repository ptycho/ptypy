/** update_addr_error_state kernel.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 */

#include "common.cuh"

extern "C" __global__ void update_addr_error_state(int* __restrict addr,
                                                   const int* __restrict mangled_addr,
                                                   OUT_TYPE* error_state,
                                                   const IN_TYPE* __restrict error_sum,
                                                   int nmodes)
{
  int tx = threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // we're using one warp only in x direction, to get implicit
  // intra-warp sync between reading err_st and writing it
  assert(blockDim.x <= 32);

  addr += row * nmodes * 15;
  mangled_addr += row * nmodes * 15;

  auto err_sum = error_sum[row];
  auto err_st = error_state[row];

  if (err_sum < err_st)
  {
    for (int i = tx, e = nmodes * 15; i < e; i += blockDim.x)
    {
      addr[i] = mangled_addr[i];
    }
  }

  if (tx == 0 && err_sum < err_st)
  {
    error_state[row] = error_sum[row];
  }
}