#include "utils/CudaFunction.h"

#include <iostream>

CudaFunction::CudaFunction(const std::string& name) : name_(name) {}

void CudaFunction::printTimes() const
{
#if DO_GPU_TIMING
  std::cout << "\nTiming stats for " << name_ << ":\n";
  for (auto& t : times_)
  {
    std::cout << t.first << "=" << t.second << "ms\n";
  }
  std::cout.flush();
#endif
}

CudaFunction::~CudaFunction() { printTimes(); }