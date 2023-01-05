#include "common.cuh"

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

extern "C" __global__ void integer_shift_kernel(const IN_TYPE* in,
                                                OUT_TYPE* out,
                                                int rows,
                                                int columns,
                                                int rowOffset,
                                                int colOffset)
{
  int tx = threadIdx.x + blockIdx.x * BDIM_X;
  int ty = threadIdx.y + blockIdx.y * BDIM_Y;
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

extern "C" __global__ void linear_interpolate_kernel(const IN_TYPE* in,
                                          OUT_TYPE* out,
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
  int gx = tx + bx * BDIM_X;
  int gy = ty + by * BDIM_Y;
  int gx_old = gx - offsetRowInt;
  int gy_old = gy - offsetColInt;

  // items index is blockIdx.z
  // we just advance the data
  int item = blockIdx.z;
  in += item * rows * columns;
  out += item * rows * columns;

  __shared__ float2 shr[BDIM_X + 2][BDIM_Y + 2];

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
  if (tx == BDIM_X - 1)
  {
    if (gx_old + 1 >= 0 && gx_old + 1 < rows && gy_old >= 0 && gy_old < columns)
    {
      ascomplex(shr[BDIM_X + 1][ty + 1]) = in[(gx_old + 1) * columns + gy_old];
    }
    else
    {
      ascomplex(shr[BDIM_X + 1][ty + 1]) = complex<float>();
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
  if (ty == BDIM_Y - 1)
  {
    if (gx_old >= 0 && gx_old < rows && gy_old + 1 >= 0 && gy_old + 1 < columns)
    {
      ascomplex(shr[tx + 1][BDIM_Y + 1]) = in[gx_old * columns + gy_old + 1];
    }
    else
    {
      ascomplex(shr[tx + 1][BDIM_Y + 1]) = complex<float>();
    }
  }
  // read top-left Halo
  if ((tx == 0) && (ty == 0))
  {
    if (gx_old - 1 >= 0 && gx_old - 1 < rows && gy_old - 1 >= 0 && gy_old - 1 < columns)
    {
      ascomplex(shr[0][0]) = in[(gx_old - 1) * columns + gy_old - 1];
    }
    else
    {
      ascomplex(shr[0][0]) = complex<float>();
    }
  }
  // read bottom-right Halo
  if ((tx == BDIM_X - 1) && (ty == BDIM_Y - 1))
  {
    if (gx_old + 1 >= 0 && gx_old + 1 < rows && gy_old + 1 >= 0 && gy_old + 1 < columns)
    {
      ascomplex(shr[BDIM_X + 1][BDIM_Y + 1]) = in[(gx_old + 1) * columns + gy_old + 1];
    }
    else
    {
      ascomplex(shr[BDIM_X + 1][BDIM_Y + 1]) = complex<float>();
    }
  }
  // read bottom-left Halo
  if ((ty == 0) && (tx == BDIM_X - 1))
  {
    if (gx_old + 1 >= 0 && gx_old + 1 < rows && gy_old - 1 >= 0 && gy_old - 1 < columns)
    {
      ascomplex(shr[BDIM_X + 1][0]) = in[(gx_old + 1) * columns + gy_old - 1];
    }
    else
    {
      ascomplex(shr[BDIM_X + 1][0]) = complex<float>();
    }
  }
  // read top-right Halo
  if ((ty == BDIM_Y - 1) && (tx == 0))
  {
    if (gx_old - 1 >= 0 && gx_old - 1 < rows && gy_old + 1 >= 0 && gy_old + 1 < columns)
    {
      ascomplex(shr[0][BDIM_Y + 1]) = in[(gx_old - 1) * columns + gy_old + 1];
    }
    else
    {
      ascomplex(shr[0][BDIM_Y + 1]) = complex<float>();
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
  __shared__ float2 shry[BDIM_X][BDIM_Y + 2];

  ascomplex(shry[tx][ty + 1]) = wx[0] * ascomplex(shr[tx][ty + 1]) +
                                wx[1] * ascomplex(shr[tx + 1][ty + 1]) +
                                wx[2] * ascomplex(shr[tx + 2][ty + 1]);
  if (ty == 0)
  {
    ascomplex(shry[tx][0]) = wx[0] * ascomplex(shr[tx][0]) +
                             wx[1] * ascomplex(shr[tx + 1][0]) +
                             wx[2] * ascomplex(shr[tx + 2][0]);
  }
  if (ty == BDIM_Y - 1)
  {
    ascomplex(shry[tx][BDIM_Y + 1]) =
        wx[0] * ascomplex(shr[tx][BDIM_Y + 1]) +
        wx[1] * ascomplex(shr[tx + 1][BDIM_Y + 1]) +
        wx[2] * ascomplex(shr[tx + 2][BDIM_Y + 1]);
  }

  __syncthreads();

  if (gx >= rows || gy >= columns)
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
