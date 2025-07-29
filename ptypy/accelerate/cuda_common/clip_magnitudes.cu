/** clip_magnitudes.
 *
 */
 #include "common.cuh"
 
 extern "C" __global__ void clip_magnitudes(IN_TYPE *arr,
                                            float clip_min_mag,
                                            float clip_max_mag,
                                            float clip_min_phase,
                                            float clip_max_phase,                                            
                                            int N)                                             
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id >= N)
    return;
  
  auto v = arr[id];
  auto mag = abs(v);
  auto theta = arg(v);

  if (mag > clip_max_mag)
    mag = clip_max_mag;
  if (mag < clip_min_mag)
    mag = clip_min_mag;
  if (theta > clip_max_phase)
    theta = clip_max_phase;
  if (theta < clip_min_phase)
    theta = clip_min_phase;

  v = thrust::polar(mag, theta);
  arr[id] = v;
}