#include "interpolated_shift.h"

#include "splines/bspline_kernel.cuh"
#include "splines/cubicPrefilter2D.cuh"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

/********* kernels ******************/

template <int BlockX, int BlockY>
__global__ void integer_shift_kernel(const complex<float>* in,
                                     complex<float>* out,
                                     int rows,
                                     int columns,
                                     int rowOffset,
                                     int colOffset)
{
  int tx = threadIdx.x + blockIdx.x * BlockX;
  int ty = threadIdx.y + blockIdx.y * BlockY;
  if (tx >= rows || ty >= columns)
    return;

  int item = blockIdx.z;
  in += item * rows * columns;
  out += item * rows * columns;

  int gid_old = tx * columns + ty;
  assert(gid_old < columns * rows);
  assert(gid_old >= 0);

  auto val = in[gid_old];

  int gid_new_x = tx + rowOffset;
  int gid_new_y = ty + colOffset;

  // write zero on the other end
  while (gid_new_x >= rows)
  {
    val = complex<float>();
    gid_new_x -= rows;
  }
  while (gid_new_x < 0)
  {
    val = complex<float>();
    gid_new_x += rows;
  }
  while (gid_new_y >= columns)
  {
    val = complex<float>();
    gid_new_y -= columns;
  }
  while (gid_new_y < 0)
  {
    val = complex<float>();
    gid_new_y += columns;
  }
  // do we need to do something with the corners?

  int gid_new = gid_new_x * columns + gid_new_y;
  assert(gid_new < rows * columns);
  assert(gid_new >= 0);

  out[gid_new] = val;
}

__device__ inline complex<float>& ascomplex(float2& f2)
{
  return reinterpret_cast<complex<float>&>(f2);
}

__device__ inline void calcWeights(float* weights, float fraction)
{
  if (fraction < 0.0)
  {
    weights[2] = -fraction;
    weights[1] = 1.0f + fraction;
    weights[0] = 0.0f;
  }
  else
  {
    weights[2] = 0.0f;
    weights[1] = 1.0f - fraction;
    weights[0] = fraction;
  }
}

template <int BlockX, int BlockY>
__global__ void linear_interpolate_kernel(const complex<float>* in,
                                          complex<float>* out,
                                          int rows,
                                          int columns,
                                          float offsetRow,
                                          float offsetColumn)
{
  int offsetRowInt = int(offsetRow);
  int offsetColInt = int(offsetColumn);
  float offsetRowFrac = offsetRow - offsetRowInt;  // positive or negative
  float offsetColFrac = offsetColumn - offsetColInt;

  // calculate convolutional weights
  float wx[3];
  calcWeights(wx, offsetRowFrac);
  float wy[3];
  calcWeights(wy, offsetColFrac);

  // indices
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int gx = tx + bx * BlockX;
  int gy = ty + by * BlockY;
  int gx_old = gx - offsetRowInt;
  int gy_old = gy - offsetColInt;

  // items index is blockIdx.z
  // we just advance the data
  int item = blockIdx.z;
  in += item * rows * columns;
  out += item * rows * columns;

  __shared__ float2 shr[BlockX + 2][BlockY + 2];

  // read top Halo
  if (tx == 0)
  {
    if (gx_old - 1 >= 0 && gx_old - 1 < rows && gy_old >= 0 && gy_old < columns)
    {
      ascomplex(shr[0][ty + 1]) = in[(gx_old - 1) * columns + gy_old];
    }
    else
    {
      ascomplex(shr[0][ty + 1]) = complex<float>();
    }
  }
  // read bottom Halo
  if (tx == BlockX - 1)
  {
    if (gx_old + 1 >= 0 && gx_old + 1 < rows && gy_old >= 0 && gy_old < columns)
    {
      ascomplex(shr[BlockX + 1][ty + 1]) = in[(gx_old + 1) * columns + gy_old];
    }
    else
    {
      ascomplex(shr[BlockX + 1][ty + 1]) = complex<float>();
    }
  }
  // read left Halo
  if (ty == 0)
  {
    if (gx_old >= 0 && gx_old < rows && gy_old - 1 >= 0 && gy_old - 1 < columns)
    {
      ascomplex(shr[tx + 1][0]) = in[gx_old * columns + gy_old - 1];
    }
    else
    {
      ascomplex(shr[tx + 1][0]) = complex<float>();
    }
  }
  // read right Halo
  if (ty == BlockY - 1)
  {
    if (gx_old >= 0 && gx_old < rows && gy_old + 1 >= 0 && gy_old + 1 < columns)
    {
      ascomplex(shr[tx + 1][BlockY + 1]) = in[gx_old * columns + gy_old + 1];
    }
    else
    {
      ascomplex(shr[tx + 1][BlockY + 1]) = complex<float>();
    }
  }
  // read the rest
  if (gx_old >= 0 && gx_old < rows && gy_old >= 0 && gy_old < columns)
  {
    ascomplex(shr[tx + 1][ty + 1]) = in[gx_old * columns + gy_old];
  }
  else
  {
    ascomplex(shr[tx + 1][ty + 1]) = complex<float>();
  }

  // now we have a block + halos in shared memory - do the interpolation
  __syncthreads();

  // interpolate rows in x
  __shared__ float2 shry[BlockX][BlockY + 2];

  ascomplex(shry[tx][ty + 1]) = wx[0] * ascomplex(shr[tx][ty + 1]) +
                                wx[1] * ascomplex(shr[tx + 1][ty + 1]) +
                                wx[2] * ascomplex(shr[tx + 2][ty + 1]);
  if (ty == 0)
  {
    ascomplex(shry[tx][0]) = wx[0] * ascomplex(shr[tx][0]) +
                             wx[1] * ascomplex(shr[tx + 1][0]) +
                             wx[2] * ascomplex(shr[tx + 2][0]);
  }
  if (ty == BlockY - 1)
  {
    ascomplex(shry[tx][BlockY + 1]) =
        wx[0] * ascomplex(shr[tx][BlockY + 1]) +
        wx[1] * ascomplex(shr[tx + 1][BlockY + 1]) +
        wx[2] * ascomplex(shr[tx + 2][BlockY + 1]);
  }

  __syncthreads();

  if (gx >= columns || gy >= rows)
  {
    return;
  }

  auto intv = wy[0] * ascomplex(shry[tx][ty]) +
              wy[1] * ascomplex(shry[tx][ty + 1]) +
              wy[2] * ascomplex(shry[tx][ty + 2]);

  // write back

  // if the point lies outside of the original frame and we're shifting in
  // that direction, it gets a zero value in any case
  // otherwise we take the interpolated value
  bool rightzero = offsetColFrac < 0.0f;
  bool leftzero = offsetColFrac > 0.0f;
  bool topzero = offsetRowFrac > 0.0f;
  bool bottomzero = offsetRowFrac < 0.0f;
  if ((gx_old == 0 && topzero) || (gx_old == rows - 1 && bottomzero) ||
      (gy_old == 0 && leftzero) || (gy_old == columns - 1 && rightzero))
  {
    out[gx * columns + gy] = complex<float>();
  }
  else
  {
    out[gx * columns + gy] = intv;
  }
}

/** this kernel is not working and shouldn't be used. It's also not optimised at
 * all. */
template <int BlockX, int BlockY>
__global__ void spline_interpolate_x(const complex<float>* in,
                                     complex<float>* out,
                                     int rows,
                                     int columns,
                                     float offsetRowFrac,
                                     float offsetColFrac)
{
  int tx = threadIdx.x + blockIdx.x * BlockX;
  int ty = threadIdx.y + blockIdx.y * BlockY;
  if (tx >= rows || ty >= columns)
    return;

  float w0, w1, w2, w3;
  bspline_weights(offsetRowFrac, w0, w1, w2, w3);
  int gid = tx * columns + ty;

  auto x_0 = tx > 1 ? in[gid - 2 * columns] : complex<float>();
  auto x_1 = tx > 0 ? in[gid - columns] : complex<float>();
  auto x_2 = in[gid];
  auto x_3 = tx < rows - 1 ? in[gid + 1 * columns] : complex<float>();

  out[gid] = w3 * x_0 + w2 * x_1 + w1 * x_2 + w0 * x_3;
}

/** this kernel is not working and shouldn't be used. It's also not optimised at
 * all. */
template <int BlockX, int BlockY>
__global__ void spline_interpolate_y(const complex<float>* in,
                                     complex<float>* out,
                                     int rows,
                                     int columns,
                                     float offsetRowFrac,
                                     float offsetColFrac)
{
  int tx = threadIdx.x + blockIdx.x * BlockX;
  int ty = threadIdx.y + blockIdx.y * BlockY;
  if (tx >= rows || ty >= columns)
    return;

  float w0, w1, w2, w3;
  bspline_weights(offsetColFrac, w0, w1, w2, w3);

  int gid = tx * columns + ty;

  auto y_0 = ty > 1 ? in[gid - 2] : complex<float>();
  auto y_1 = ty > 0 ? in[gid - 1] : complex<float>();
  auto y_2 = in[gid];
  auto y_3 = ty < columns - 1 ? in[gid + 1] : complex<float>();

  out[gid] = w3 * y_0 + w2 * y_1 + w1 * y_2 + w0 * y_3;
}

/********* class implementation *******/

InterpolatedShift::InterpolatedShift() : CudaFunction("interpolated_shift") {}

void InterpolatedShift::setParameters(int items, int rows, int columns)
{
  items_ = items;
  rows_ = rows;
  columns_ = columns;
}

void InterpolatedShift::setDeviceBuffers(complex<float>* d_in,
                                         complex<float>* d_out)
{
  d_in_ = d_in;
  d_out_ = d_out;
}

void InterpolatedShift::allocate()
{
  ScopedTimer t(this, "allocate");
  d_in_.allocate(items_ * rows_ * columns_);
  d_out_.allocate(items_ * rows_ * columns_);
}

void InterpolatedShift::transfer_in(const complex<float>* in)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_in_.get(), in, items_ * rows_ * columns_);
}

void InterpolatedShift::run(float offsetRow, float offsetColumn, bool do_linear)
{
  ScopedTimer t(this, "run");

  dim3 threadsPerBlock = {32u, 32u, 1u};
  dim3 blocks = {unsigned((rows_ + 31) / 32),
                 unsigned((columns_ + 31) / 32),
                 unsigned(items_)};

  // get fractional and integer parts
  auto offsetRowInt = int(offsetRow);
  auto offsetColInt = int(offsetColumn);
  auto offsetRowFrac = offsetRow - int(offsetRow);
  auto offsetColFrac = offsetColumn - int(offsetColumn);

  if (std::abs(offsetRowFrac) < 1e-6f && std::abs(offsetColFrac) < 1e-6f)
  {
    if (offsetRowInt == 0 && offsetColInt == 0)
    {
      // no transformation at all
      gpu_memcpy_d2d(d_out_.get(), d_in_.get(), items_ * rows_ * columns_);
    }
    else
    {
      // no fractional part, so we can just use a shifted copy
      integer_shift_kernel<32, 32>
          <<<blocks, threadsPerBlock>>>(d_in_.get(),
                                        d_out_.get(),
                                        rows_,
                                        columns_,
                                        int(offsetRow),
                                        int(offsetColumn));
      checkLaunchErrors();
    }
  }
  else
  {
    if (do_linear)
    {
      linear_interpolate_kernel<32, 32><<<blocks, threadsPerBlock>>>(
          d_in_.get(), d_out_.get(), rows_, columns_, offsetRow, offsetColumn);
      checkLaunchErrors();
    }
    else
    {
      // bicubic
      // this version has not been adapted for 3D yet

      // first, prefilter the data for cubic splines
      // Note: this prefilter does not match the scipy version completely
      // !!! IMPORTANT: this modifies the data in-place
      CubicBSplinePrefilter2D(
          d_in_.get(), columns_ * sizeof(complex<float>), columns_, rows_);
      checkLaunchErrors();

      // then interpolate in x and y directions
      // Note: these kernels are not working yet
      // and the buffers should be modified to avoid the final copy and to
      // avoid in-place operations as inputs might be re-used (?)
      spline_interpolate_y<32, 32><<<blocks, threadsPerBlock>>>(
          d_in_.get(), d_out_.get(), rows_, columns_, offsetRow, offsetColumn);
      checkLaunchErrors();
      spline_interpolate_x<32, 32><<<blocks, threadsPerBlock>>>(
          d_out_.get(), d_in_.get(), rows_, columns_, offsetRow, offsetColumn);
      checkLaunchErrors();

      // make sure result is in d_out_
      gpu_memcpy_d2d(d_in_.get(), d_out_.get(), rows_ * columns_);
    }
  }

  timing_sync();
}

void InterpolatedShift::transfer_out(complex<float>* out)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(out, d_out_.get(), items_ * rows_ * columns_);
}

/********* interface ************/

extern "C" void interpolated_shift_c(const float* f_in,
                                     float* f_out,
                                     int items,
                                     int rows,
                                     int columns,
                                     float offsetRow,
                                     float offsetCol,
                                     int ido_linear)
{
  auto in = reinterpret_cast<const complex<float>*>(f_in);
  auto out = reinterpret_cast<complex<float>*>(f_out);

  auto is = gpuManager.get_cuda_function<InterpolatedShift>(
      "interpolated_shift", items, rows, columns);
  is->allocate();
  is->transfer_in(in);
  is->run(offsetRow, offsetCol, ido_linear != 0);
  is->transfer_out(out);
}