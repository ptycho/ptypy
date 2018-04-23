#include "get_difference.h"

#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

/************* kernels ******************************/

template <bool usePbound, int BlockX, int BlockY>
__global__ void get_difference_kernel(
    const int *addr_info,
    float alpha,
    const complex<float> *backpropagated_solution,
    const float *err_fmag,
    const complex<float> *exit_wave,
    float pbound,
    const complex<float> *probe_obj,
    complex<float> *out,
    int m,
    int n)
{
  int batch = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // each of these are 3-d arrays with indices
  auto ea = addr_info + batch * 3 * 5 + 6;
  auto da = ea + 3;

  // these are the start indices for the batch item
  auto offset = ea[0] * m * n;

  auto da_0 = da[0];

#pragma unroll(2)
  for (int i = tx; i < m; i += BlockX)
  {
#pragma unroll(1)  // to make sure the compiler doesn't unroll
    for (int j = ty; j < n; j += BlockY)
    {
      auto outidx = offset + i * n + j;
      if (!usePbound || err_fmag[da_0] > pbound)
      {
        out[outidx] = backpropagated_solution[outidx] - probe_obj[outidx];
      }
      else
      {
        out[outidx] = alpha * (probe_obj[outidx] - exit_wave[outidx]);
      }
    }
  }
}

/************* class implementation *****************/

GetDifference::GetDifference() : CudaFunction("get_difference") {}

void GetDifference::setParameters(int i, int m, int n)
{
  i_ = i;
  m_ = m;
  n_ = n;
}

void GetDifference::setDeviceBuffers(int *d_addr_info,
                                     complex<float> *d_backpropagated_solution,
                                     float *d_err_fmag,
                                     complex<float> *d_exit_wave,
                                     complex<float> *d_probe_obj,
                                     complex<float> *d_out)
{
  d_addr_info_ = d_addr_info;
  d_backpropagated_solution_ = d_backpropagated_solution;
  d_err_fmag_ = d_err_fmag;
  d_exit_wave_ = d_exit_wave;
  d_probe_obj_ = d_probe_obj;
  d_out_ = d_out;
}
void GetDifference::allocate()
{
  ScopedTimer t(this, "allocate");
  d_addr_info_.allocate(i_ * 5 * 3);
  d_backpropagated_solution_.allocate(i_ * m_ * n_);
  d_err_fmag_.allocate(i_);
  d_exit_wave_.allocate(i_ * m_ * n_);
  d_probe_obj_.allocate(i_ * m_ * n_);
  d_out_.allocate(i_ * m_ * n_);
}

void GetDifference::updateErrorInput(float *d_err_fmag)
{
  d_err_fmag_ = d_err_fmag;
}

complex<float> *GetDifference::getOutput() const { return d_out_.get(); }

void GetDifference::transfer_in(const int *addr_info,
                                const complex<float> *backpropagated_solution,
                                const float *err_fmag,
                                const complex<float> *exit_wave,
                                const complex<float> *probe_obj)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_addr_info_.get(), addr_info, i_ * 5 * 3);
  gpu_memcpy_h2d(
      d_backpropagated_solution_.get(), backpropagated_solution, i_ * m_ * n_);
  gpu_memcpy_h2d(d_err_fmag_.get(), err_fmag, i_);
  gpu_memcpy_h2d(d_exit_wave_.get(), exit_wave, i_ * m_ * n_);
  gpu_memcpy_h2d(d_probe_obj_.get(), probe_obj, i_ * m_ * n_);
}

void GetDifference::run(float alpha, float pbound, bool usePbound)
{
  ScopedTimer t(this, "run");

  // TODO: is this really needed?
  checkCudaErrors(
      cudaMemset(d_out_.get(), 0, i_ * m_ * n_ * sizeof(*d_out_.get())));

  // always use a 32x32 block of threads
  dim3 threadsPerBlock = {32u, 32u, 1u};
  dim3 blocks = {unsigned(i_), 1u, 1u};
  if (usePbound)
  {
    get_difference_kernel<true, 32, 32>
        <<<blocks, threadsPerBlock>>>(d_addr_info_.get(),
                                      alpha,
                                      d_backpropagated_solution_.get(),
                                      d_err_fmag_.get(),
                                      d_exit_wave_.get(),
                                      pbound,
                                      d_probe_obj_.get(),
                                      d_out_.get(),
                                      m_,
                                      n_);
    checkLaunchErrors();
  }
  else
  {
    get_difference_kernel<false, 32, 32>
        <<<blocks, threadsPerBlock>>>(d_addr_info_.get(),
                                      alpha,
                                      d_backpropagated_solution_.get(),
                                      d_err_fmag_.get(),
                                      d_exit_wave_.get(),
                                      pbound,
                                      d_probe_obj_.get(),
                                      d_out_.get(),
                                      m_,
                                      n_);
    checkLaunchErrors();
  }

  timing_sync();
}

void GetDifference::transfer_out(complex<float> *out)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(out, d_out_.get(), i_ * m_ * n_);
}

/************* interface function *******************/

extern "C" void get_difference_c(const int *addr_info,
                                 float alpha,
                                 const float *fbackpropagated_solution,
                                 const float *err_fmag,
                                 const float *fexit_wave,
                                 float pbound,
                                 const float *fprobe_obj,
                                 float *fout,
                                 int i,
                                 int m,
                                 int n,
                                 int usePbound)
{
  auto backpropagated_solution =
      reinterpret_cast<const complex<float> *>(fbackpropagated_solution);
  auto exit_wave = reinterpret_cast<const complex<float> *>(fexit_wave);
  auto probe_obj = reinterpret_cast<const complex<float> *>(fprobe_obj);
  auto out = reinterpret_cast<complex<float> *>(fout);

  auto gd =
      gpuManager.get_cuda_function<GetDifference>("get_difference", i, m, n);
  gd->allocate();
  gd->transfer_in(
      addr_info, backpropagated_solution, err_fmag, exit_wave, probe_obj);
  gd->run(alpha, pbound, usePbound != 0);
  gd->transfer_out(out);
}