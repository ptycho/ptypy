/** clip_magnitudes.
 *
 */

 #include <cassert>
 #include <cmath>
 #include <thrust/complex.h>
 using thrust::abs;
 using thrust::arg;
 using thrust::exp;
 using thrust::complex;
 
 extern "C" __global__ void __launch_bounds__(1024, 2)
     clip_magnitudes(IN_TYPE *arr,
                    float clip_min,
                    float clip_max,
                    int X,
                    int Y)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
 
  for (int iy = ty; iy < Y; iy += blockDim.y) {
    #pragma unroll(4)
    for (int ix = tx; ix < X; ix += blockDim.x) {
        float abs_arr = abs(arr[iy * X +ix]);
        auto phase_arr = IN_TYPE(arg(arr[iy * X +ix]));

        if (abs_arr > clip_max) {
            abs_arr = clip_max;
        }
        if (abs_arr < clip_min) {
            abs_arr = clip_min;
        }
        arr[iy * X + ix] = IN_TYPE(abs_arr * exp(phase_arr));
    }
  }
}