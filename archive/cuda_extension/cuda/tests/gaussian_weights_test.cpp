#include "utils/GaussianWeights.h"
#include <gtest/gtest.h>

#include <numeric>

TEST(GaussianFilter, weightsCompareToPython)
{
  std::vector<float> expected = {1.9947465e-01f,
                                 1.7603576e-01f,
                                 1.2098749e-01f,
                                 6.4759940e-02f,
                                 2.6995959e-02f,
                                 8.7643042e-03f,
                                 2.2159631e-03f,
                                 4.3634902e-04f,
                                 6.6916291e-05f};
  auto actual = gaussian_kernel1d(2, 8);
  EXPECT_EQ(actual.size(), expected.size());

  for (int i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], actual[i], 1e-4f);
  }
  auto sum_right = std::accumulate(actual.begin() + 1, actual.end(), 0.0f);
  auto sum = 2 * sum_right + actual[0];
  EXPECT_NEAR(sum, 1.0f, 1e-4f);
}