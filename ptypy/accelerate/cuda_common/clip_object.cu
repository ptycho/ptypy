/** clip_object.
 *
 */
 #include "common.cuh"
 
 extern "C" __global__ void clip_object(IN_TYPE *arr,
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

  if (isfinite(clip_max_mag) == 1 && mag > clip_max_mag)
    mag = clip_max_mag;
  if (isfinite(clip_min_mag) == 1 && mag < clip_min_mag)
    mag = clip_min_mag;
  if (isfinite(clip_max_phase) == 1 && theta > clip_max_phase)
    theta = clip_max_phase;
  if (isfinite(clip_min_phase) == 1 && theta < clip_min_phase)
    theta = clip_min_phase;



  v = thrust::polar(mag, theta);
  arr[id] = v;
}