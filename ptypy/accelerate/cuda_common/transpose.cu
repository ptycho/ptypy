/** Implementation taken from
 * https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
 *
 * Kernel optimised to ensure all global memory reads and writes are coalesced,
 * and shared memory access has no bank conflicts.
 */

/**
 * Data types:
 * - DTYPE - any pod type
 */

#include "common.cuh"

extern "C" __global__ void transpose(const DTYPE* idata,
                                     DTYPE* odata,
                                     int width,
                                     int height)
{
  __shared__ DTYPE block[BDIM][BDIM + 1];

  // read the matrix tile into shared memory
  // load one element per thread from device memory (idata) and
  // store it in transposed order in block[][]
  unsigned int xIndex = blockIdx.x * BDIM + threadIdx.x;
  unsigned int yIndex = blockIdx.y * BDIM + threadIdx.y;
  if (xIndex < width && yIndex < height)
  {
    unsigned int index_in = yIndex * width + xIndex;
    block[threadIdx.y][threadIdx.x] = idata[index_in];
  }

  // synchronise to ensure all writes to block[][] are complete
  __syncthreads();

  // write transposed matrix back to global memory (odata) in linear order
  xIndex = blockIdx.y * BDIM + threadIdx.x;
  yIndex = blockIdx.x * BDIM + threadIdx.y;
  if (xIndex < height && yIndex < width)
  {
    unsigned int index_out = yIndex * height + xIndex;
    odata[index_out] = block[threadIdx.x][threadIdx.y];
  }
}