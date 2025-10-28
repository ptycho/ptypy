/**
 * Data types:
 * - DTYPE (float/double/complex<float>/complex<double>)
 * - MATH_TYPE (float/double) - used for the convolution kernel itself
 * 
 * A symmetric convolution kernel is assumed here
 */

#include "common.cuh"

/** Implements reflect-mode index wrapping
 *
 * maps indexes like this (reflects on both ends):
 * Extension      | Input     | Extension
 *  5 6 6 5 4 3 2 | 2 3 4 5 6 | 6 5 4 3 2 2 3 4 5 6 6
 *
 */
 class IndexReflect
 {
 public:
   /** Create index range. maxX is not included in the valid range,
    * i.e., the range is [minX, maxX)
    */
   __device__ IndexReflect(int minX, int maxX) : maxX_(maxX), minX_(minX) {}
 
   /// Map given index to the valid range using reflect mode
   __device__ int operator()(int idx) const
   {
     if (idx < maxX_ && idx >= minX_)
       return idx;
     auto ddd = (idx - minX_) / (maxX_ - minX_);
     auto mmm = (idx - minX_) % (maxX_ - minX_);
     if (mmm < 0)
       mmm = -mmm - 1;
     // if odd it goes backwards from max
     // if even it goes upwards from min
     return ddd % 2 == 0 ? minX_ + mmm : maxX_ - mmm - 1;
   }
 
 private:
   int maxX_, minX_;
 };


/*
Row convolution kernel
*/
extern "C" __global__ void convolution_row(const DTYPE *__restrict__ input,
                                           DTYPE *output,
                                           int height,
                                           int width,
                                           const MATH_TYPE* kernel,
                                           int kernel_radius)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // offset for batch
    input  += width * height * blockIdx.z;
    output += width * height * blockIdx.z;

    // reinterpret to avoid compiler warning that
    // constructor of complex<float>() cannot be called if it's
    // shared memory - polluting the outputs
    extern __shared__ char shr[];
    auto shm = reinterpret_cast<DTYPE *>(shr);
                                       
    // Offset to block start of core area
    int gbx = bx * BDIM_X;
    int gby = by * BDIM_Y;
    int start = gbx * width + gby;
    input  += start;
    output += start;

    // width of shared memory
    int shwidth = BDIM_Y + 2 * kernel_radius;

    // only do this if row index is in range
    // (need to keep threads with us, so that synchthreads below doesn't deadlock)  
    if (gbx + tx < height)
    {
        // main data (center point for each thread) - reflecting as needed
        IndexReflect ind(-gby, width - gby);
        shm[tx * shwidth + (kernel_radius + ty)] = input[tx * width + ind(ty)];

        // left halo (kernel radius before)
        for (int i = ty - kernel_radius; i < 0; i += BDIM_Y)
        {
        shm[tx * shwidth + (i + kernel_radius)] = input[tx * width + ind(i)];
        }

        // right halo (kernel radius after)
        for (int i = ty + BDIM_Y; i < BDIM_Y + kernel_radius; i += BDIM_Y)
        {
        shm[tx * shwidth + (i + kernel_radius)] = input[tx * width + ind(i)];
        }
    }

    __syncthreads();

    // safe to return now, after syncing
    if (gby + ty >= width || gbx + tx >= height)
        return;

    // compute  - will be complex<double> if kernel is double
    auto sum = shm[tx * shwidth + (ty + kernel_radius)] * kernel[0];
    for (int i = 1; i <= kernel_radius; ++i)
    {
        sum += (shm[tx * shwidth + (ty + i + kernel_radius)] +
                shm[tx * shwidth + (ty - i + kernel_radius)]) *
                kernel[i];
    }

    output[tx * width + ty] = sum;
}


/*
Column convolution kernel
*/
extern "C" __global__ void convolution_col(const DTYPE *__restrict__ input,
                                           DTYPE *output,
                                           int height,
                                           int width,
                                           const MATH_TYPE* kernel,
                                           int kernel_radius)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // offset for batch
    input  += width * height * blockIdx.z;
    output += width * height * blockIdx.z;

    // reinterpret to avoid compiler warning that
    // constructor of complex<float>() cannot be called if it's
    // shared memory - polluting the outputs
    extern __shared__ char shr[];
    auto shm = reinterpret_cast<DTYPE *>(shr);

    // Offset to block start of core area
    int gbx = bx * BDIM_X;
    int gby = by * BDIM_Y;
    int start = gbx * width + gby;
    input  += start;
    output += start;

    // only do this if column index is in range
    // (need to keep threads with us, so that synchthreads below doesn't deadlock)
    if (gby + ty < width)
    {
        // main data (center point for each thread) - reflecting if needed
        IndexReflect ind(-gbx, height - gbx);
        shm[(kernel_radius + tx) * BDIM_Y + ty] = input[ind(tx) * width + ty];

        // upper halo (kernel radius before)
        for (int i = tx - kernel_radius; i < 0; i += BDIM_X)
        {
            shm[(i + kernel_radius) * BDIM_Y + ty] = input[ind(i) * width + ty];
        }

        // lower halo (kernel radius after)
        for (int i = tx + BDIM_X; i < BDIM_X + kernel_radius; i += BDIM_X)
        {
            shm[(i + kernel_radius) * BDIM_Y + ty] = input[ind(i) * width + ty];
        }
    }

    __syncthreads();

    // safe to return now, after syncing
    if (gby + ty >= width || gbx + tx >= height)
        return;

    // compute - will be complex<double> if kernel is double
    auto sum = shm[(tx + kernel_radius) * BDIM_Y + ty] * kernel[0];
    for (int i = 1; i <= kernel_radius; ++i)
    {
        sum += (shm[(tx + i + kernel_radius) * BDIM_Y + ty] +
                shm[(tx - i + kernel_radius) * BDIM_Y + ty]) *
                kernel[i];
    }

    output[tx * width + ty] = sum;
}
