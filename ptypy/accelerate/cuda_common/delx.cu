/** difference along axes (last and mid axis kernels)
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs 
 * - OUT_TYPE: the data type for the outputs 
 */

#include "common.cuh"


/** Finite difference for forward/backward for any axis that is not the
 * last one, assuring that the reads and writes are coalesced.
 *
 * The idea is that arrays of any number of dimensions can be reshaped
 * (or just treated) as 3D, with the higher dimensions and lower dimensions
 * multiplied together and the axis along which we differentiate in the middle.
 * The higher dim might also be 1, capturing the case when we compute along
 * the zero axis.
 *
 * We use the following variables:
 * - higher_dim: sizes of the axes left of the diff axis multiplied together
 * - axis_dim: the size along the axis we diff over
 * - lower_dim: the sizes of the axes right of the diff axis multiplied together
 *
 * Examples:
 * - 5x3x10, axis=1: higher_dim=5, axis_dim=3, lower_dim=10
 * - 10x5x4x3, axis=1: higher_dim=10, axis_dim=5, lower_dim=12
 * - 10x5x4x3, axis=0: higher_dim=1, axis_dim=10, lower_dim=60
 * - 30x40, axis=0: higher_dim=1, axis_dim=30, lower_dim=40
 *
 * The thread/block dimensions are mapped as:
 *  z = high_dim,
 *  y = axis_dim,
 *  x = lower_dim
 *
 * We read tiles of the input into BDIM_Y x BDIM_X elements of shared memory,
 * always using a single thread block along the y dimension (axis_dim),
 * which iterates over the full axis in a loop. The other 2 dimensions are
 * fully parallelised in different thread blocks.
 * Data reads/writes into shared mem are coalesced since the ix index
 * corresponding to the threadIdx.x is used to read the lower_dim, with no
 * multiplier on the index.
 *
 * Once the tile is in shared memory, the difference is calculated -
 * depending on forward/backward diffs, and with special cases at the
 * end of the tile - either overlapping with next block or ensuring a
 * zero if it's the end of the input.
 *
 */
extern "C" __global__ void delx_mid(const IN_TYPE *__restrict__ input,
                                    OUT_TYPE *output,
                                    int lower_dim,   // x for 3D
                                    int higher_dim,  // z for 3D
                                    int axis_dim)
{
  // reinterpret to avoid compiler warning that
  // constructor of complex<float>() cannot be called if it's
  // shared memory - polluting the outputs
  __shared__ char shr[BDIM_X * BDIM_Y * sizeof(IN_TYPE)];
  auto shared_data = reinterpret_cast<IN_TYPE *>(shr);

  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int tz = threadIdx.z;  // only 0 here

  unsigned int ix = tx + blockIdx.x * BDIM_X;
  unsigned int iy = ty;
  unsigned int iz = tz + blockIdx.z * blockDim.z;

  // offset pointers for z dimension (higher-dim)
  input += iz * axis_dim * lower_dim;
  output += iz * axis_dim * lower_dim;

  // now read x/y tiles coalesced and perform difference along y,
  // letting this thread block iterate along the full y axis
  // to give a thread a bit more substantial work to do.
  auto maxblocks = (axis_dim + BDIM_Y - 1) / BDIM_Y;
  for (int bidx = 0; bidx < maxblocks; ++bidx)
  {
    iy = ty + bidx * BDIM_Y;

    if (iy < axis_dim && ix < lower_dim)
    {
      shared_data[ty * BDIM_X + tx] = input[iy * lower_dim + ix];
    }
    __syncthreads();

    if (iy < axis_dim && ix < lower_dim)
    {
      if (IS_FORWARD)
      {
        IN_TYPE plus1;
        if (ty < BDIM_Y - 1 &&
            iy < axis_dim - 1)  // we have a next element in shared data
        {
          plus1 = shared_data[(ty + 1) * BDIM_X + tx];
        }
        else if (iy == axis_dim - 1)  // end of axis 
        {
          plus1 = shared_data[ty * BDIM_X + tx];  // make sure it's zero
        }
        else  // end of block, but nore input is there
        {
          plus1 = input[(iy + 1) * lower_dim + ix];
        }
        output[iy * lower_dim + ix] = plus1 - shared_data[ty * BDIM_X + tx];
      }
      else
      {
        IN_TYPE minus1;
        if (ty > 0)  // we have a previous element in shared
        {
          minus1 = shared_data[(ty - 1) * BDIM_X + tx];
        }
        else if (iy == 0)  // use same as next to get zero
        {
          minus1 = shared_data[ty * BDIM_X + tx];
        }
        else  // read previous input (ty == 0 but iy > 0)
        {
          minus1 = input[(iy - 1) * lower_dim + ix];
        }
        output[iy * lower_dim + ix] = shared_data[ty * BDIM_X + tx] - minus1;
      }
    }
  }
}



/** This is the special case for when we diff along the last axis.
 * 
 * Here, flat_dim is all other dims multiplied together, and axis_dim
 * is the dimension along which we diff. 
 * To ensure that we stay coalesced (compared to delx_mid), 
 * we use the x index to iterate within each thread block (the loop).
 * Otherwise it follows the same ideas as delx_mid - please read the
 * description there.
  */
extern "C" __global__ void delx_last(const IN_TYPE *__restrict__ input,
                                     OUT_TYPE *output,
                                     int flat_dim,
                                     int axis_dim)
{
  // reinterpret to avoid constructor of complex<float>() + compiler warning
  __shared__ char shr[BDIM_X * BDIM_Y * sizeof(IN_TYPE)];
  auto shared_data = reinterpret_cast<IN_TYPE *>(shr);

  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;

  unsigned int ix = tx;
  unsigned int iy = ty + blockIdx.x * BDIM_Y;  // we always use x in grid

  int stride_y = axis_dim;

  auto maxblocks = (axis_dim + BDIM_X - 1) / BDIM_X;
  for (int bidx = 0; bidx < maxblocks; ++bidx)
  {
    ix = tx + bidx * BDIM_X;

    if (iy < flat_dim && ix < axis_dim)
    {
      shared_data[ty * BDIM_X + tx] = input[iy * stride_y + ix];
    }

    __syncthreads();

    if (iy < flat_dim && ix < axis_dim)
    {
      if (IS_FORWARD)
      {
        IN_TYPE plus1;
        if (tx < BDIM_X - 1 &&
            ix < axis_dim - 1)  // we have a next element in shared data
        {
          plus1 = shared_data[ty * BDIM_X + tx + 1];
        }
        else if (ix == axis_dim - 1)  // end of axis - same as current to get 0
        {
          plus1 = shared_data[ty * BDIM_X + tx];
        }
        else  // end of block, but nore input is there
        {
          plus1 = input[iy * stride_y + ix + 1];
        }

        output[iy * stride_y + ix] = plus1 - shared_data[ty * BDIM_X + tx];
      }
      else
      {
        IN_TYPE minus1;
        if (tx > 0)  // we have a previous element in shared
        {
          minus1 = shared_data[ty * BDIM_X + tx - 1];
        }
        else if (ix == 0)  // use same as next to get zero
        {
          minus1 = shared_data[ty * BDIM_X + tx];
        }
        else  // read previous input (ty == 0 but iy > 0)
        {
          minus1 = input[iy * stride_y + ix - 1];
        }
        output[iy * stride_y + ix] = shared_data[ty * BDIM_X + tx] - minus1;
      }
    }
  }
}