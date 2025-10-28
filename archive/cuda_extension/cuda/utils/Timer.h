#pragma once

#include <chrono>

/** Generic timer class, based on std::chrono .
 *
 * Use as:
 * @code
 *  Timer t;
 *  ... // do expensive stuff
 *  auto time = t.get_time();  // in ms
 * @encode
 */
class Timer
{
public:
  Timer() { start(); }
  void start() { start_ = std::chrono::high_resolution_clock::now(); }
  double get_time()
  {
    auto end = std::chrono::high_resolution_clock::now();
    return double(std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                        start_)
                      .count()) /
           1e3;
  }

private:
  std::chrono::high_resolution_clock::time_point start_;
};
