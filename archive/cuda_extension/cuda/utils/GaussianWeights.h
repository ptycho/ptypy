#pragma once

#include <cmath>
#include <vector>

// 1D Gaussian convolution kernel (symmetric)
inline std::vector<float> gaussian_kernel1d(float sigma, int radius)
{
  float factor = -0.5f / (sigma * sigma);
  std::vector<float> weights(radius + 1);
  float sum = 0.0f;
  for (auto i = 0; i < radius + 1; ++i)
  {
    weights[i] = std::exp(factor * (i * i));
    if (i == 0)
      sum += weights[i];
    else
      sum += 2.0f * weights[i];  // for symmetry
  }
  for (auto i = 0; i < radius + 1; ++i)
  {
    weights[i] /= sum;
  }
  return std::move(weights);
}
