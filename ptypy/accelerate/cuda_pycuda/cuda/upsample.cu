#include <thrust/complex.h>
using thrust::complex;

extern "C" __global__ void upsample(TYPE* A, 
                                    const TYPE* B, 
                                    int Arows, 
                                    int Acols, 
                                    int factor)
{
  // shared mem is BDIM/factor x BDIM/factor of TYPE, declared on the call-side
  extern __shared__ char sh_raw[];
  auto block = reinterpret_cast<TYPE*>(sh_raw);

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ax = threadIdx.x + blockIdx.x * blockDim.x;
  const int ay = threadIdx.y + blockIdx.y * blockDim.y;

  // offset A and B to the right point in first dimension
  A += blockIdx.z * Arows * Acols;
  B += blockIdx.z * Arows / factor * Acols / factor;

  // read the input data into the shared memory
  // we use a thread block per output size, so we only need a
  // subset of the threads to read the data in
  if (tx < blockDim.x / factor && ty < blockDim.y / factor && ax < Acols &&
      ay < Arows)
  {
    const int tbx = blockIdx.x * blockDim.x / factor + tx;
    const int tby = blockIdx.y * blockDim.y / factor + ty;
    block[ty * blockDim.x / factor + tx] =
        B[tby * Acols / factor + tbx] / (factor * factor);
  }
  __syncthreads();

  // now all the threads read from shared mem to output
  if (ax < Acols && ay < Arows)
  {
    A[ay * Acols + ax] = block[(ty / factor * blockDim.x + tx) / factor];
  }
}