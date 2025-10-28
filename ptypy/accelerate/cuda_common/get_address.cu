#include "common.cuh"

inline __device__ int minimum(int a, int b) { return a < b ? a : b; }

inline __device__ int maximum(int a, int b) { return a < b ? b : a; }

extern "C" __global__ void get_address(const int* addr_current,
                                       int* mangled_addr,
                                       int num_pods,
                                       const int* __restrict delta,
                                       int max_oby,
                                       int max_obx)
{
  // we use only one thread block
  const int tx = threadIdx.x;
  const int idx = tx % 2;  // even threads access y dim, odd threads x dim
  const int maxval = (idx == 0) ? max_oby : max_obx;

  const int addr_stride = 15;
  const int d = delta[idx];
  addr_current += 3 + idx + 1;
  mangled_addr += 3 + idx + 1;

  for (int ix = tx; ix < num_pods * 2; ix += blockDim.x)
  {
    const int bid = ix / 2;
    int cur = addr_current[bid * addr_stride] + d;
    int bound = maximum(0, minimum(maxval, cur));
    assert(bound >= 0);
    assert(bound <= maxval);
    mangled_addr[bid * addr_stride] = bound;
  }
}