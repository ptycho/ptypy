#include <cmath>
#include <complex>

// doing this on the CPU for now - this isn't needed anywhere on its own
// - this was used as a test for the cython interfacing
extern "C" void sqrt_abs_c(const float *in, float *out, int m, int n)
{
  auto cin = reinterpret_cast<const std::complex<float> *>(in);
  auto cout = reinterpret_cast<std::complex<float> *>(out);
  for (int i = 0; i < m * n; ++i)
  {
    cout[i] = std::sqrt(std::abs(cin[i]));
  }
}