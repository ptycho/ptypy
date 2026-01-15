#pragma once
#include <iostream>

// allow testing this on CPU
#ifndef __NVCC__
#define __device__
#endif

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
