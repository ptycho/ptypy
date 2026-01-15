/** clip_magnitudes.
 *
 */
 #include "common.cuh"
 
 extern "C" __global__ void clip_magnitudes(IN_TYPE *arr,
                                            float clip_min,
                                            float clip_max,
                                            int N)                                             
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id >= N)
    return;
  
  auto v = arr[id];
  auto mag = abs(v);
  auto theta = arg(v);

  if (mag > clip_max)
    mag = clip_max;
  if (mag < clip_min)
    mag = clip_min;

  v = thrust::polar(mag, theta);
  arr[id] = v;
}