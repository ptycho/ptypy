#include <thrust/complex.h>
#include <cassert>
using thrust::complex;
        

#define pr_dlayer(k) addr[(k)]
#define pr_roi_row(k) addr[1*num_pods + (k)]
#define pr_roi_column(k) addr[2*num_pods + (k)]
#define ex_dlayer(k) addr[6*num_pods + (k)]
#define obj_dlayer(k) addr[3*num_pods + (k)]
#define obj_roi_row(k) addr[4*num_pods + (k)]
#define obj_roi_column(k) addr[5*num_pods + (k)]

// #define NUM_MODES 8
// #define BDIM_X 16
// #define BDIM_Y 16

/*
pods: 16
address: (5, 3, 4, 4)
ex: (16, 5, 5)
prsh: [2, 5, 5]
ob: (2, 7, 7)
*/

extern "C" __global__ void pr_update2(int pr_sh,
                                      int ob_sh_row,
                                      int ob_sh_col,
                                      int pr_modes,
                                      int ob_modes,
                                      int num_pods,
                                      complex<float>* pr_g,
                                      complex<float>* prn_g,
                                      const complex<float>* __restrict__ ob_g,
                                      const complex<float>* __restrict__ ex_g,
                                      const int* addr)
{
  int y = blockIdx.y * BDIM_Y + threadIdx.y;
  int dy = pr_sh;
  int z = blockIdx.x * BDIM_X + threadIdx.x;
  int dz = pr_sh;
  complex<float> pr[NUM_MODES], prn[NUM_MODES];

  int txy = threadIdx.y * BDIM_X + threadIdx.x;
  assert(pr_modes <= NUM_MODES);

  if (y < pr_sh && z < pr_sh) {
    #pragma unroll
    for (int i = 0; i < NUM_MODES; ++i) {
      auto idx = i*dy*dz + y*dz + z;
      assert(idx < pr_modes * pr_sh * pr_sh);
      pr[i] = pr_g[idx];
      prn[i] = prn_g[idx];
    }
  }

  __shared__ int addresses[BDIM_X*BDIM_Y*5];

  for (int p = 0; p < num_pods; p += BDIM_X*BDIM_Y)
  {

    int mi = BDIM_X*BDIM_Y;
    if (mi > num_pods - p) mi = num_pods - p;

    if (p > 0) __syncthreads();

    
    if (txy < mi) {
      assert(p+txy < num_pods);
      assert(txy < BDIM_X * BDIM_Y);
      addresses[txy*5+0] = pr_dlayer(p+txy);
      addresses[txy*5+1] = ex_dlayer(p+txy);
      addresses[txy*5+2] = obj_dlayer(p+txy);
      assert(obj_dlayer(p+txy) < NUM_MODES);
      assert(addresses[txy*5+2] < NUM_MODES);
      addresses[txy*5+3] = pr_roi_row(p+txy);
      addresses[txy*5+4] = pr_roi_column(p+txy);
    }
    

    __syncthreads();

    if (y >= pr_sh || z >= pr_sh)
      continue;

    #pragma unroll 4
    for (int i = 0; i < mi; ++i){
      int* ad = addresses + i*5;
      int v1 = y - ad[3];
      int v2 = z - ad[4];
      if (v1 >= 0 && v1 < ob_sh_row && v2 >= 0 && v2 < ob_sh_col) {
        auto obidx = ad[2] * ob_sh_row * ob_sh_col + v1 * ob_sh_col + v2;
        assert(obidx < ob_modes * ob_sh_row * ob_sh_col);
        auto ob = ob_g[obidx];
        int idx = ad[0];
        assert(idx < NUM_MODES);
        auto cob = conj(ob);
        pr[idx] += cob * 
          ex_g[ad[1]*pr_sh*pr_sh +v1*pr_sh + v2];
        auto rr = prn[idx].real();
        rr += ob.real() * ob.real() + ob.imag() * ob.imag();
        prn[idx].real(rr);
      }
    }

  }

  if (y < pr_sh && z < pr_sh) {
    for (int i = 0; i < NUM_MODES; ++i){
      pr_g[i*dy*dz + y*dz + z] = pr[i];
      prn_g[i*dy*dz + y*dz + z] = prn[i];
    }
  }

}

