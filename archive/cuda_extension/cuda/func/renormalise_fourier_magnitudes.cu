#include "renormalise_fourier_magnitudes.h"

#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include <cassert>
#include <cmath>
#include <cstdio>

/********** kernels *****************/

// template, to switch between pbound and not at compile-time
template <bool usePbound>
__global__ void renormalise_fourier_magnitudes_kernel(const complex<float> *f,
                                                      const float *af,
                                                      const float *fmag,
                                                      const unsigned char *mask,
                                                      const float *err_fmag,
                                                      const int *addr_info,
                                                      complex<float> *out,
                                                      float pbound,
                                                      int A,
                                                      int B)
{
  using std::sqrt;

  int batch = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  auto ea = addr_info + batch * 3 * 5 + 2 * 3;
  auto da = ea + 3;
  auto ma = da + 3;

  auto ea_0 = ea[0];
  auto ea_1 = 0;
  auto ea_2 = 0;

  auto da_0 = da[0];
  auto da_1 = 0;
  auto da_2 = 0;

  auto ma_0 = ma[0];
  auto ma_1 = 0;
  auto ma_2 = 0;

  for (int i = tx; i < A; i += blockDim.x)
  {
    for (int j = ty; j < B; j += blockDim.y)
    {
      auto maidx = ma_0 * A * B + (ma_1 + i) * B + (ma_2 + j);
      auto eaidx = ea_0 * A * B + (ea_1 + i) * B + (ea_2 + j);
      auto daidx = da_0 * A * B + (da_1 + i) * B + (da_2 + j);

      auto m = mask[maidx];
      auto magnitudes = fmag[daidx];
      auto absolute_magnitudes = af[daidx];
      auto fourier_space_solution = f[eaidx];
      auto fourier_error = err_fmag[da_0];

      if (!usePbound)
      {
        auto fm = m ? magnitudes / (absolute_magnitudes + 1e-10f) : 1.0f;
        auto v = fm * fourier_space_solution;
        out[eaidx] = v;
      }
      else if (fourier_error > pbound)
      {
        // power bound is applied
        auto fdev = absolute_magnitudes - magnitudes;
        auto renorm = sqrt(pbound / fourier_error);
        auto fm =
            m ? (magnitudes + fdev * renorm) / (absolute_magnitudes + 1e-10f)
              : 1.0f;
        out[eaidx] = fm * fourier_space_solution;
      }
      else
      {
        out[eaidx] = 0.0f;
      }
    }
  }
}

/********** class implementation ********/

RenormaliseFourierMagnitudes::RenormaliseFourierMagnitudes()
    : CudaFunction("renormalise_fourier_magnitudes")
{
}

void RenormaliseFourierMagnitudes::setParameters(int M, int N, int A, int B)
{
  M_ = M;
  N_ = N;
  A_ = A;
  B_ = B;
}

void RenormaliseFourierMagnitudes::setDeviceBuffers(complex<float> *d_f,
                                                    float *d_af,
                                                    float *d_fmag,
                                                    unsigned char *d_mask,
                                                    float *d_err_fmag,
                                                    int *d_addr_info,
                                                    complex<float> *d_out)
{
  d_f_ = d_f;
  d_af_ = d_af;
  d_fmag_ = d_fmag;
  d_mask_ = d_mask;
  d_err_fmag_ = d_err_fmag;
  d_addr_info_ = d_addr_info;
  d_out_ = d_out;
}

void RenormaliseFourierMagnitudes::allocate()
{
  ScopedTimer t(this, "allocate");
  d_f_.allocate(M_ * A_ * B_);
  d_af_.allocate(N_ * A_ * B_);
  d_fmag_.allocate(N_ * A_ * B_);
  d_mask_.allocate(N_ * A_ * B_);
  d_err_fmag_.allocate(N_);
  d_addr_info_.allocate(M_ * 5 * 3);
  d_out_.allocate(M_ * A_ * B_);
}

void RenormaliseFourierMagnitudes::updateErrorInput(float *d_err_fmag)
{
  d_err_fmag_ = d_err_fmag;
}

complex<float> *RenormaliseFourierMagnitudes::getOutput() const
{
  return d_out_.get();
}

void RenormaliseFourierMagnitudes::transfer_in(const complex<float> *f,
                                               const float *af,
                                               const float *fmag,
                                               const unsigned char *mask,
                                               const float *err_fmag,
                                               const int *addr_info)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_f_.get(), f, M_ * A_ * B_);
  gpu_memcpy_h2d(d_af_.get(), af, N_ * A_ * B_);
  gpu_memcpy_h2d(d_fmag_.get(), fmag, N_ * A_ * B_);
  gpu_memcpy_h2d(d_mask_.get(), mask, N_ * A_ * B_);
  gpu_memcpy_h2d(d_err_fmag_.get(), err_fmag, N_);
  gpu_memcpy_h2d(d_addr_info_.get(), addr_info, M_ * 5 * 3);
}

void RenormaliseFourierMagnitudes::run(float pbound, bool usePbound)
{
  ScopedTimer t(this, "run");

  // always use a 32x32 block of threads
  dim3 threadsPerBlock = {32, 32, 1u};
  dim3 blocks = {unsigned(M_), 1u, 1u};

  if (usePbound)
  {
    renormalise_fourier_magnitudes_kernel<true>
        <<<blocks, threadsPerBlock>>>(d_f_.get(),
                                      d_af_.get(),
                                      d_fmag_.get(),
                                      d_mask_.get(),
                                      d_err_fmag_.get(),
                                      d_addr_info_.get(),
                                      d_out_.get(),
                                      pbound,
                                      A_,
                                      B_);
    checkLaunchErrors();
  }
  else
  {
    renormalise_fourier_magnitudes_kernel<false>
        <<<blocks, threadsPerBlock>>>(d_f_.get(),
                                      d_af_.get(),
                                      d_fmag_.get(),
                                      d_mask_.get(),
                                      d_err_fmag_.get(),
                                      d_addr_info_.get(),
                                      d_out_.get(),
                                      pbound,
                                      A_,
                                      B_);
    checkLaunchErrors();
  }

  timing_sync();
}

void RenormaliseFourierMagnitudes::transfer_out(complex<float> *out)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(out, d_out_.get(), M_ * A_ * B_);
}

/********* interface function *********/

extern "C" void renormalise_fourier_magnitudes_c(
    const float *f_f,           // M x A x B
    const float *af,            // N x A x B
    const float *fmag,          // N x A x B
    const unsigned char *mask,  // N x A x B
    const float *err_fmag,      // N
    const int *addr_info,       // M x 5 x 3
    float pbound,
    float *f_out,  // M x A x B
    int M,
    int N,
    int A,
    int B,
    int usePbound)
{
  auto f = reinterpret_cast<const complex<float> *>(f_f);
  auto out = reinterpret_cast<complex<float> *>(f_out);

  auto rfm = gpuManager.get_cuda_function<RenormaliseFourierMagnitudes>(
      "renormalise_fourier_magnitudes", M, N, A, B);
  rfm->allocate();
  rfm->transfer_in(f, af, fmag, mask, err_fmag, addr_info);
  rfm->run(pbound, usePbound != 0);
  rfm->transfer_out(out);
}