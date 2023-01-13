/*
This was a test to join the fourier update kernels,
and use shared memory across the modes.
But the performance
is 2x slower than individual as we have many idle threads here.
It is not used at the moment.
*/

#include "common.cuh"

extern "C" __global__ void fourier_update(int nmodes,
                                          complex<float> *f_d,
                                          const float *fmask_d,
                                          const float *fmag_d,
                                          float *fdev_d,
                                          float *ferr_d,
                                          const float *mask_sum,
                                          const int *addr,
                                          float *err_fmag,
                                          float pbound,
                                          int A,
                                          int B)
{
  // block in x/y are across the full tile (ix and iy go from 0 to A/B)
  // might go beyond if not divisible
  // blockDim.z is nmodes - we use z index to go over the modes accumulation
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int ix = tx + blockIdx.x * blockDim.x;
  int iy = ty + blockIdx.y * blockDim.y;
  int addr_stride = 15;
  assert(tz < nmodes);

  const int *ea = addr + 6 + (blockIdx.z * nmodes) * addr_stride;
  const int *da = addr + 9 + (blockIdx.z * nmodes) * addr_stride;
  const int *ma = addr + 12 + (blockIdx.z * nmodes) * addr_stride;

  // full offset for this thread
  auto f = f_d + ea[0] * A * B + iy * B + ix;
  auto fdev = fdev_d + da[0] * A * B + iy * B + ix;
  auto fmag = fmag_d + da[0] * A * B + iy * B + ix;
  auto fmask = fmask_d + ma[0] * A * B + iy * B + ix;
  auto ferr = ferr_d + da[0] * A * B + iy * B + ix;

  extern __shared__ float shm[];  // BX * BY * nmodes

  // offset so we have shmt[0..nmodes] to reduce in
  auto shmt =
      shm + threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z;

  // modes values
  if (ix < B && iy < A)
  {
    float abs_exit_wave = abs(f[tz * A * B]);
    shmt[tz] = abs_exit_wave *
               abs_exit_wave;  // if we do this manually (real*real +imag*imag)
                               // we get bad rounding errors
  }
  else
  {
    shmt[tz] = 0.0f;
  }
  __syncthreads();

  // accumulate across modes
  assert(nmodes == blockDim.z);
  int c = nmodes;
  while (c > 1)
  {
    int half = c / 2;
    if (tz < half)
    {
      shmt[tz] += shmt[c - tz - 1];
    }
    __syncthreads();
    c = c - half;
  }

  // now write outputs if we're the first thread in the block
  int tyrem = (A - iy) < int(blockDim.y) ? (A - iy) : blockDim.y;
  int txrem = (B - ix) < int(blockDim.x) ? (B - ix) : blockDim.x;
  int nt = tyrem * txrem;
  int shidx = ty * txrem + tx;
  if (tz == 0 && iy < A && ix < B)
  {
    auto acc = shmt[0];
    auto fdevv = sqrt(acc) - *fmag;
    *ferr = (*fmask * fdevv * fdevv) / mask_sum[ma[0]];
    *fdev = fdevv;

    shm[shidx] = *ferr;
  }

  ////////////// error reduce
  __syncthreads();
  c = nt;
  while (c > 1)
  {
    int half = c / 2;
    if (shidx < half && tz == 0)
    {
      shm[shidx] += shm[c - shidx - 1];
    }
    __syncthreads();
    c = c - half;
  }
  if (shidx == 0 && tz == 0)
  {
    err_fmag[blockIdx.z] = shm[0];
  }

  ///////////// fmag_all_update
  ea += tz * addr_stride;
  da += tz * addr_stride;
  ma += tz * addr_stride;

  fmask = fmask_d + ma[0] * A * B + iy * B + ix;
  float err = err_fmag[da[0]];  // RACE CONDITION!
  fdev = fdev_d + da[0] * A * B + iy * B + ix;
  fmag = fmag_d + da[0] * A * B + iy * B + ix;
  f = f_d + ea[0] * A * B + iy * B + ix;
  float renorm = sqrt(pbound / err);

  if (ix < B && iy < A && renorm < 1.0f)
  {
    auto m = *fmask;
    auto fmagv = *fmag;
    auto fdevv = *fdev;
    float fm =
        (1.0f - m) + m * ((fmagv + fdevv * renorm) / (fmagv + fdevv + 1e-10f));
    *f *= fm;
  }
}
