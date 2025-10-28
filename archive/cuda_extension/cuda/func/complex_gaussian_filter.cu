#include "complex_gaussian_filter.h"
#include "utils/GaussianWeights.h"

#include "utils/GpuManager.h"
#include "utils/Indexing.h"
#include "utils/ScopedTimer.h"

#include <cassert>
#include <numeric>

/******* kernels *****************/

__constant__ float c_Kernel[ComplexGaussianFilter::MAX_KERNEL_RADIUS + 1];

template <int BlockX, int BlockY>
__global__ void convolutionRowKernel(const complex<float>* in,
                                     complex<float>* out,
                                     int height,
                                     int width,
                                     int kernel_radius)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // offset for batch
  in += width * height * blockIdx.z;
  out += width * height * blockIdx.z;

  extern __shared__ char shm_raw[];
  auto shm = reinterpret_cast<complex<float>*>(shm_raw);

  // Offset to block start of core area
  int gbx = bx * BlockX;
  int gby = by * BlockY;
  int start = gbx * width + gby;
  in += start;
  out += start;
  // width of shared memory
  int shwidth = BlockY + 2 * kernel_radius;

  if (gbx + tx < height)
  {
    // main part - reflecting as needed
    IndexReflect ind(-gby, width - gby);
    shm[tx * shwidth + (kernel_radius + ty)] = in[tx * width + ind(ty)];

    // left halo (kernel radius before)
    for (int i = ty - kernel_radius; i < 0; i += BlockY)
    {
      shm[tx * shwidth + (i + kernel_radius)] = in[tx * width + ind(i)];
    }

    // right halo (kernel radius after)
    for (int i = ty + BlockY; i < BlockY + kernel_radius; i += BlockY)
    {
      shm[tx * shwidth + (i + kernel_radius)] = in[tx * width + ind(i)];
    }
  }

  __syncthreads();

  // safe to return now, after syncing
  if (gby + ty >= width || gbx + tx >= height)
    return;

  // compute
  auto sum = shm[tx * shwidth + (ty + kernel_radius)] * c_Kernel[0];
  for (int i = 1; i <= kernel_radius; ++i)
  {
    sum += (shm[tx * shwidth + (ty + i + kernel_radius)] +
            shm[tx * shwidth + (ty - i + kernel_radius)]) *
           c_Kernel[i];
  }

  out[tx * width + ty] = sum;
}

template <int BlockX, int BlockY>
__global__ void convolutionColumnsKernel(const complex<float>* in,
                                         complex<float>* out,
                                         int height,
                                         int width,
                                         int kernel_radius)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // offset for batch
  in += width * height * blockIdx.z;
  out += width * height * blockIdx.z;

  extern __shared__ char shm_raw[];
  auto shm = reinterpret_cast<complex<float>*>(shm_raw);
  // dims: BlockX + 2*kernel_radius, BlockY

  // Offset to block start of core area
  int gbx = bx * BlockX;
  int gby = by * BlockY;
  int start = gbx * width + gby;
  in += start;
  out += start;

  // only do this if column index is in range
  // (need to keep threads with us, so that synchthreads below doesn't deadlock)
  if (gby + ty < width)
  {
    // main data (center point for each thread) - reflecting if needed
    IndexReflect ind(-gbx, height - gbx);
    shm[(kernel_radius + tx) * BlockY + ty] = in[ind(tx) * width + ty];

    // upper halo (kernel radius before)
    for (int i = tx - kernel_radius; i < 0; i += BlockX)
    {
      shm[(i + kernel_radius) * BlockY + ty] = in[ind(i) * width + ty];
    }

    // lower halo (kernel radius after)
    for (int i = tx + BlockX; i < BlockX + kernel_radius; i += BlockX)
    {
      shm[(i + kernel_radius) * BlockY + ty] = in[ind(i) * width + ty];
    }
  }
  __syncthreads();

  // safe to return now, after syncing
  if (gby + ty >= width || gbx + tx >= height)
    return;

  // compute
  auto sum = shm[(tx + kernel_radius) * BlockY + ty] * c_Kernel[0];
  for (int i = 1; i <= kernel_radius; ++i)
  {
    sum += (shm[(tx + i + kernel_radius) * BlockY + ty] +
            shm[(tx - i + kernel_radius) * BlockY + ty]) *
           c_Kernel[i];
  }

  out[tx * width + ty] = sum;
}

/******* class implementation ********/

ComplexGaussianFilter::ComplexGaussianFilter()
    : CudaFunction("complex_gaussian_filter")
{
}

int ComplexGaussianFilter::totalSize() const
{
  return std::accumulate(shape_, shape_ + ndims_, 1, std::multiplies<int>());
}

std::vector<float> ComplexGaussianFilter::calcConvolutionKernel(float stddev,
                                                                int ndevs)
{
  auto lw = int(stddev * ndevs + 0.5f);
  return gaussian_kernel1d(stddev, lw);
}

void ComplexGaussianFilter::setParameters(int ndims,
                                          const int* shape,
                                          const float* mfs)
{
  ndims_ = ndims;
  std::copy(shape, shape + ndims, shape_);
  std::copy(mfs, mfs + ndims, mfs_);
}

void ComplexGaussianFilter::setDeviceBuffers(complex<float>* d_input,
                                             complex<float>* d_output)
{
  d_input_ = d_input;
  d_output_ = d_output;
}

void ComplexGaussianFilter::allocate()
{
  ScopedTimer t(this, "allocate");

  auto total = totalSize();
  d_input_.allocate(total);
  d_output_.allocate(total);
}

void ComplexGaussianFilter::transfer_in(const complex<float>* input)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_input_.get(), input, totalSize());
}

void ComplexGaussianFilter::run()
{
  ScopedTimer t(this, "run");

  // use separable convolution in each dim
  int batches = 1;
  int x = 1;
  int y = 1;
  float stdx = 0.0f, stdy = 0.0f;
  if (ndims_ == 3)
  {
    batches = shape_[0];
    x = shape_[1];
    y = shape_[2];
    stdx = mfs_[0];
    stdy = mfs_[1];
  }
  if (ndims_ == 2)
  {
    x = shape_[0];
    y = shape_[1];
    stdx = mfs_[0];
    stdy = mfs_[1];
  }
  if (ndims_ == 1)
  {
    stdy = mfs_[0];
    stdx = 0.0f;
    y = shape_[0];
  }

  if (stdx > 0.0f)
  {
    // construct convolution kernel in current dim
    auto weights = calcConvolutionKernel(stdx, NUM_STDDEVS);
    if (weights.size() - 1 > MAX_KERNEL_RADIUS)
      throw GPUException("Gaussian filter length too long: " +
                         std::to_string(weights.size()));

    checkCudaErrors(cudaMemcpyToSymbol(
        c_Kernel, weights.data(), weights.size() * sizeof(float)));

    // run colums kernel with right stride
    const int bx = 16;
    const int by = 4;
    dim3 threads(bx, by, 1);
    dim3 blocks((x + bx - 1) / bx, (y + by - 1) / by, batches);
    auto kernel_radius = weights.size() - 1;
    auto halos = kernel_radius * 2;
    auto shared = (bx + halos) * by * sizeof(complex<float>);
    if (shared > MAX_SHARED_PER_BLOCK)
    {
      throw GPUException("cannot run in kernel shared memory");
    }
    auto inp = d_input_.get();
    convolutionColumnsKernel<bx, by><<<blocks, threads, shared>>>(
        inp, d_output_.get(), x, y, kernel_radius);
    checkLaunchErrors();
  }

  // last dim is continuous, so we use a different kernel
  if (stdy > 0.0)
  {
    // construct convolution kernel in last dim
    auto weights = calcConvolutionKernel(stdy, NUM_STDDEVS);
    if (weights.size() - 1 > MAX_KERNEL_RADIUS)
      throw GPUException("Gaussian filter length too long: " +
                         std::to_string(weights.size()));
    checkCudaErrors(cudaMemcpyToSymbol(
        c_Kernel, weights.data(), weights.size() * sizeof(float)));

    // run y kernel
    const int bx = 4;
    const int by = 16;
    dim3 threads(bx, by, 1);
    dim3 blocks((x + bx - 1) / bx, (y + by - 1) / by, batches);
    auto kernel_radius = weights.size() - 1;
    auto halos = kernel_radius * 2;
    auto shared = (by + halos) * bx * sizeof(complex<float>);
    auto indata = d_output_.get();
    if (stdx <= 0.0f)
    {
      indata = d_input_.get();
    }
    convolutionRowKernel<bx, by><<<blocks, threads, shared>>>(
        indata, d_output_.get(), x, y, kernel_radius);

    checkLaunchErrors();
  }
  if (stdx == 0.0f && stdy == 0.0f)
  {
    gpu_memcpy_d2d(d_output_.get(), d_input_.get(), batches * x * y);
  }
  timing_sync();
}

void ComplexGaussianFilter::transfer_out(complex<float>* output)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(output, d_output_.get(), totalSize());
}

/******* interface functions *********/

extern "C" void complex_gaussian_filter_c(const float* f_input,
                                          float* f_output,
                                          const float* mfs,
                                          int ndims,
                                          const int* shape)
{
  auto input = reinterpret_cast<const complex<float>*>(f_input);
  auto output = reinterpret_cast<complex<float>*>(f_output);

  auto cgf = gpuManager.get_cuda_function<ComplexGaussianFilter>(
      "complex_gaussian_filter", ndims, shape, mfs);
  cgf->allocate();
  cgf->transfer_in(input);
  cgf->run();
  cgf->transfer_out(output);
}