#include <thrust/complex.h>
#include <stdio.h>
using thrust::complex;

extern "C" __global__ void delx_mid(
    const DTYPE *__restrict__ input,
    DTYPE *output,
    int lower_dim,  //x for 3D   // 1
    int higher_dim, //z for 3D   // 2
    int axis_dim)                // 3
{

    __shared__ DTYPE shared_data[BDIM_Y][BDIM_X];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int ix = tx + blockIdx.x * BDIM_X;
    unsigned int iy = ty;

    // offset pointers
    int xoffset = (ix / lower_dim) * lower_dim;
    input += ix + xoffset;
    output += ix + xoffset;

    auto maxblocks = (axis_dim + BDIM_Y - 1) / BDIM_Y;

    for (int bidx = 0; bidx < maxblocks; ++bidx)
    {
        iy = ty + bidx * BDIM_Y;

        if (iy < axis_dim && ix < lower_dim * higher_dim)
        {
            shared_data[ty][tx] = input[iy * lower_dim];
            //printf("%d, %d: %f\n", ty, tx, shared_data[ty][tx]);
        }
        __syncthreads();

        if (iy < axis_dim && ix < lower_dim * higher_dim)
        {
            if (IS_FORWARD)
            {
                DTYPE plus1;
                if (ty < BDIM_Y - 1 && iy < axis_dim - 1) // we have a next element in shared data
                {
                    plus1 = shared_data[ty + 1][tx];
                }
                else if (iy == axis_dim - 1) // end of axis - next same as current to get 0
                {
                    plus1 = shared_data[ty][tx];
                }
                else                        // end of block, but nore input is there
                {
                    plus1 = input[(iy + 1) * lower_dim];
                }
                //printf("%d, %d: %f - %f; ix=%d, xoffset=%d, iy=%d\n", ty, tx, plus1, shared_data[ty][tx], ix, xoffset, iy);
                output[iy * lower_dim] = plus1 - shared_data[ty][tx];
            }
            else
            {
                DTYPE minus1;
                if (ty > 0) // we have a previous element in shared
                {
                    minus1 = shared_data[ty - 1][tx];
                }
                else if (iy == 0) // use same as next to get zero
                {
                    minus1 = shared_data[ty][tx];
                }
                else // read previous input (ty == 0 but iy > 0)
                {
                    minus1 = input[(iy-1) * lower_dim];
                }
                output[iy * lower_dim] = shared_data[ty][tx] - minus1;
            }
        }
    }
}
