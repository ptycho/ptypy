#pragma once
#include <sstream>
#include <string>

// some GCC versions don't provide the following functions, so we patch them in

namespace patch
{
template <typename T>
inline std::string to_string(const T& x)
{
  std::ostringstream sstr;
  sstr << x;
  return sstr.str();
}
}  // namespace patch

namespace std
{
using namespace ::patch;
}