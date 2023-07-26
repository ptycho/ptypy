/** fill3D kernel.
 * 
 * Data types:
 * - IN_TYPE: the data type for the inputs
 * - OUT_TYPE: data type for outputs 
 */

#include "common.cuh"

extern "C" __global__ void fill3D(
    OUT_TYPE* A,
    const IN_TYPE* B,
    // final dimensions of A/B in [z, y, x]
    int A_Z,
    int A_Y,
    int A_X,
    int B_Z,
    int B_Y,
    int B_X,
    // offsets to start reading/writing
    int Ao_z,
    int Ao_y,
    int Ao_x,
    int Bo_z,
    int Bo_y,
    int Bo_x,
    // lengths to copy
    int len_z,
    int len_y,
    int len_x
    )
{
    // We use the following strategy:
    // - BlockIdx.z for the batch (first dims combined if 4D+)
    // - blockDim.z = 1
    // - multiple blocks are used across y and x dimensions
    // - we loop over z dimension within the thread block
    int batch = blockIdx.z;
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= len_x || iy >= len_y)
        return;

    // offset for current batch (4D+ dimension)
    A += batch * A_X * A_Y * A_Z;
    B += batch * B_X * B_Y * B_Z;

    // offset for start position in each dimension of the last 3
    A += Ao_z * A_Y * A_X + Ao_y * A_X + Ao_x;
    B += Bo_z * B_Y * B_X + Bo_y * B_X + Bo_x;

    // copy data
    for (int iz = 0; iz < len_z; ++iz) {
        A[iz * A_Y * A_X + iy * A_X + ix] = 
            B[iz * B_Y * B_X + iy * B_X + ix];
    }
}