#pragma once

#include <string>
#include <utility>
#include <vector>

/** Used as base class for all CUDA functions.
 *
 * Mainly used for factoring out gpu timing events, using
 * ScopedTimer inside the child.
 */
class CudaFunction
{
protected:
  CudaFunction(const std::string &name);

public:
  void printTimes() const;
  /** Calls printTimes() if timing is enabled */
  virtual ~CudaFunction();

private:
  friend class ScopedTimer;
  std::string name_;
  std::vector<std::pair<std::string, double>> times_;
};