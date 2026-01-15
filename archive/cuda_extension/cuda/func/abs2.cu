#include "abs2.h"

#include "utils/Complex.h"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

/************ Kernels *********************/

// must be real, if this template fits
template <class T>
__global__ void abs2_kernel(const T *in, T *out, size_t n)
{
  size_t ti = threadIdx.x + blockIdx.x * blockDim.x;
  if (ti >= n)
    return;
  auto v = in[ti];
  out[ti] = v * v;
}

// complex to real
template <class T, class Tc>
__global__ void abs2_kernel(const Tc *in, T *out, size_t n)
{
  size_t ti = threadIdx.x + blockIdx.x * blockDim.x;
  if (ti >= n)
    return;
  auto v = in[ti];
  out[ti] = v.real() * v.real() + v.imag() * v.imag();
}

/************ Class methods *****************/

template <class Tin, class Tout>
Abs2<Tin, Tout>::Abs2() : CudaFunction("abs2")
{
}

template <class Tin, class Tout>
void Abs2<Tin, Tout>::setParameters(size_t n)
{
  n_ = n;
}

template <class Tin, class Tout>
void Abs2<Tin, Tout>::setDeviceBuffers(Tin *d_datain, Tout *d_dataout)
{
  d_datain_ = d_datain;
  d_dataout_ = d_dataout;
}

template <class Tin, class Tout>
void Abs2<Tin, Tout>::allocate()
{
  ScopedTimer t(this, "allocate");
  d_datain_.allocate(n_);
  d_dataout_.allocate(n_);
}

template <class Tin, class Tout>
Tout *Abs2<Tin, Tout>::getOutput() const
{
  return d_dataout_.get();
}

template <class Tin, class Tout>
void Abs2<Tin, Tout>::transfer_in(const Tin *datain)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_datain_.get(), datain, n_);
}

template <class Tin, class Tout>
void Abs2<Tin, Tout>::transfer_out(Tout *dataout)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(dataout, d_dataout_.get(), n_);
}

template <class Tin, class Tout>
void Abs2<Tin, Tout>::run()
{
  ScopedTimer t(this, "run");
  size_t block = 256;
  size_t blocks = (n_ + block - 1) / block;
  abs2_kernel<<<blocks, block>>>(d_datain_.get(), d_dataout_.get(), n_);
  checkLaunchErrors();

  // sync device if timing is enabled
  timing_sync();
}

/************** interface function *************/

// instantiate here to force creating all symbols
template class Abs2<float, float>;
template class Abs2<complex<float>, float>;
template class Abs2<double, double>;
template class Abs2<complex<double>, double>;

template <class Tin, class Tout>
static void entryFunc(const Tin *in, Tout *out, int n)
{
  auto abs2 = gpuManager.get_cuda_function<Abs2<Tin, Tout>>(
      "abs2<" + getTypeName<Tin>() + "," + getTypeName<Tout>() + ">", n);
  abs2->allocate();
  abs2->transfer_in(in);
  abs2->run();
  abs2->transfer_out(out);
}

extern "C" void abs2_c(const float *in, float *out, int n, int iisComplex)
{
  if (iisComplex != 0)
  {
    entryFunc(reinterpret_cast<const complex<float> *>(in), out, n);
  }
  else
  {
    entryFunc(in, out, n);
  }
}

extern "C" void abs2d_c(const double *in, double *out, int n, int iisComplex)
{
  if (iisComplex != 0)
  {
    entryFunc(reinterpret_cast<const complex<double> *>(in), out, n);
  }
  else
  {
    entryFunc(in, out, n);
  }
}
