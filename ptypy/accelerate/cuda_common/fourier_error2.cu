/** This kernel was an experiment to use shared memory reduction across
 * the modes. It turned out to run about 2x slower than the one without
 * shared memory, so it's not used at this stage.
 */
#include "common.cuh"

extern "C" __global__ void fourier_error2(int nmodes,
                                          complex<float> *f,
                                          const float *fmask,
                                          const float *fmag,
                                          float *fdev,
                                          float *ferr,
                                          const float *mask_sum,
                                          const int *addr,
                                          int A,
                                          int B)
{
  // block in x/y are across the full tile (ix and iy go from 0 to A/B)
  // might go beyond if not divisible
  // blockDim.z is nmodes - we use z index to go over the modes accumulation

  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int tz = threadIdx.z;  // for all modes within block
  int addr_stride = 15;
  assert(tz < nmodes);

  const int *ea = addr + 6 + (blockIdx.z * nmodes) * addr_stride;
  const int *da = addr + 9 + (blockIdx.z * nmodes) * addr_stride;
  const int *ma = addr + 12 + (blockIdx.z * nmodes) * addr_stride;

  // full offset for this thread
  f += ea[0] * A * B + iy * B + ix;
  fdev += da[0] * A * B + iy * B + ix;
  fmag += da[0] * A * B + iy * B + ix;
  fmask += ma[0] * A * B + iy * B + ix;
  ferr += da[0] * A * B + iy * B + ix;

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
                               // we get differences to numpy due to rounding
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
  if (tz == 0 && iy < A && ix < B)
  {
    auto acc = shmt[0];
    auto fdevv = sqrt(acc) - *fmag;
    *ferr = (*fmask * fdevv * fdevv) / mask_sum[ma[0]];
    *fdev = fdevv;
  }
}
