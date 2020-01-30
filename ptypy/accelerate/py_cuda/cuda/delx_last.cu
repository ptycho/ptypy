#include <thrust/complex.h>
using thrust::complex;

extern "C" __global__ void delx_last(
    const DTYPE *__restrict__ input,
    DTYPE *output,
    int flat_dim,
    int axis_dim)
{
    __shared__ DTYPE shared_data[BDIM_Y][BDIM_X];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int ix = tx;
    unsigned int iy = ty + blockIdx.x * BDIM_Y; // we always use x in grid

    int stride_y = axis_dim;

    auto maxblocks = (axis_dim + BDIM_X - 1) / BDIM_X;
    for (int bidx = 0; bidx < maxblocks; ++bidx)
    {
        ix = tx + bidx * BDIM_X;

        if (iy < flat_dim && ix < axis_dim)
        {
            shared_data[ty][tx] = input[iy * stride_y + ix];
        }

        __syncthreads();

        if (iy < flat_dim && ix < axis_dim)
        {
            if (IS_FORWARD)
            {
                DTYPE plus1;
                if (tx < BDIM_X - 1 && ix < axis_dim - 1) // we have a next element in shared data
                {
                    plus1 = shared_data[ty][tx + 1];
                }
                else if (ix == axis_dim - 1) // end of axis - next same as current to get 0
                {
                    plus1 = shared_data[ty][tx];
                }
                else // end of block, but nore input is there
                {
                    plus1 = input[iy * stride_y + ix + 1];
                }

                output[iy * stride_y + ix] = plus1 - shared_data[ty][tx];
            }
            else
            {
                DTYPE minus1;
                if (tx > 0) // we have a previous element in shared
                {
                    minus1 = shared_data[ty][tx - 1];
                }
                else if (ix == 0) // use same as next to get zero
                {
                    minus1 = shared_data[ty][tx];
                }
                else // read previous input (ty == 0 but iy > 0)
                {
                    minus1 = input[iy * stride_y + ix - 1];
                }
                output[iy * stride_y + ix] = shared_data[ty][tx] - minus1;
            }
        }
    }
}