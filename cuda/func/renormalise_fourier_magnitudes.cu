#include "renormalise_fourier_magnitudes.h"

#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include <cmath>

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
                                                      int M,
                                                      int N)
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

  for (int i = tx; i < M; i += blockDim.x)
  {
    for (int j = ty; j < N; j += blockDim.y)
    {
      auto maidx = ma_0 * M * N + (ma_1 + i) * N + (ma_2 + j);
      auto eaidx = ea_0 * M * N + (ea_1 + i) * N + (ea_2 + j);
      auto daidx = da_0 * M * N + (da_1 + i) * N + (da_2 + j);

      auto m = mask[maidx];
      auto magnitudes = fmag[daidx];
      auto absolute_magnitudes = af[daidx];
      auto fourier_space_solution = f[eaidx];
      auto fourier_error = err_fmag[da_0];

      if (!usePbound)
      {
        auto fm = m ? magnitudes / (absolute_magnitudes + 1e-10f) : 1.0f;
        out[eaidx] = fm * fourier_space_solution;
      }
      else if (err_fmag[ea_0] > pbound)
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

void RenormaliseFourierMagnitudes::setParameters(int i, int m, int n)
{
  i_ = i;
  m_ = m;
  n_ = n;
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
  d_f_.allocate(i_ * m_ * n_);
  d_af_.allocate(i_ * m_ * n_);
  d_fmag_.allocate(i_ * m_ * n_);
  d_mask_.allocate(i_ * m_ * n_);
  d_err_fmag_.allocate(i_);
  d_addr_info_.allocate(i_ * 5 * 3);
  d_out_.allocate(i_ * m_ * n_);
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
  gpu_memcpy_h2d(d_f_.get(), f, i_ * m_ * n_);
  gpu_memcpy_h2d(d_af_.get(), af, i_ * m_ * n_);
  gpu_memcpy_h2d(d_fmag_.get(), fmag, i_ * m_ * n_);
  gpu_memcpy_h2d(d_mask_.get(), mask, i_ * m_ * n_);
  gpu_memcpy_h2d(d_err_fmag_.get(), err_fmag, i_);
  gpu_memcpy_h2d(d_addr_info_.get(), addr_info, i_ * 5 * 3);
}

void RenormaliseFourierMagnitudes::run(float pbound, bool usePbound)
{
  ScopedTimer t(this, "run");

  // always use a 32x32 block of threads
  dim3 threadsPerBlock = {32, 32, 1u};
  dim3 blocks = {unsigned(i_), 1u, 1u};
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
                                      m_,
                                      n_);
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
                                      m_,
                                      n_);
    checkLaunchErrors();
  }

  timing_sync();
}

void RenormaliseFourierMagnitudes::transfer_out(complex<float> *out)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(out, d_out_.get(), i_ * m_ * n_);
}

/********* interface function *********/

extern "C" void renormalise_fourier_magnitudes_c(const float *f_f,
                                                 const float *af,
                                                 const float *fmag,
                                                 const unsigned char *mask,
                                                 const float *err_fmag,
                                                 const int *addr_info,
                                                 float pbound,
                                                 float *f_out,
                                                 int i,
                                                 int m,
                                                 int n,
                                                 int usePbound)
{
  auto f = reinterpret_cast<const complex<float> *>(f_f);
  auto out = reinterpret_cast<complex<float> *>(f_out);

  auto rfm = gpuManager.get_cuda_function<RenormaliseFourierMagnitudes>(
      "renormalise_fourier_magnitudes", i, m, n);
  rfm->allocate();
  rfm->transfer_in(f, af, fmag, mask, err_fmag, addr_info);
  rfm->run(pbound, usePbound != 0);
  rfm->transfer_out(out);
}