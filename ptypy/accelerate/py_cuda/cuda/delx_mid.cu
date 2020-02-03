#include <thrust/complex.h>
#include <stdio.h>
using thrust::complex;

extern "C" __global__ void delx_mid(
    const DTYPE *__restrict__ input,
    DTYPE *output,
    int lower_dim,  //x for 3D   
    int higher_dim, //z for 3D   
    int axis_dim)                
{
    // reinterpret to avoid constructor of complex<float>() + compiler warning
    __shared__ char shr[BDIM_X * BDIM_Y * sizeof(DTYPE)];
    auto shared_data = reinterpret_cast<DTYPE *>(shr);

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int tz = threadIdx.z; // only 0 here

    unsigned int ix = tx + blockIdx.x * BDIM_X;
    unsigned int iy = ty;
    unsigned int iz = tz + blockIdx.z * blockDim.z;

    // offset pointers for z dimension
    input += iz * axis_dim * lower_dim;
    output += iz * axis_dim * lower_dim;

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
                DTYPE plus1;
                if (ty < BDIM_Y - 1 && iy < axis_dim - 1) // we have a next element in shared data
                {
                    plus1 = shared_data[(ty + 1) * BDIM_X + tx];
                }
                else if (iy == axis_dim - 1) // end of axis - next same as current to get 0
                {
                    plus1 = shared_data[ty * BDIM_X + tx];
                }
                else // end of block, but nore input is there
                {
                    plus1 = input[(iy + 1) * lower_dim + ix];
                }
                output[iy * lower_dim + ix] = plus1 - shared_data[ty * BDIM_X + tx];
            }
            else
            {
                DTYPE minus1;
                if (ty > 0) // we have a previous element in shared
                {
                    minus1 = shared_data[(ty - 1) * BDIM_X + tx];
                }
                else if (iy == 0) // use same as next to get zero
                {
                    minus1 = shared_data[ty * BDIM_X + tx];
                }
                else // read previous input (ty == 0 but iy > 0)
                {
                    minus1 = input[(iy - 1) * lower_dim + ix];
                }
                output[iy * lower_dim + ix] = shared_data[ty * BDIM_X + tx] - minus1;
            }
        }
    }
}
