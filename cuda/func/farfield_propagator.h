#pragma once

#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

#include <cufft.h>
#include <map>
#include <tuple>

/** Class to manage FFT plans - re-using plans for same
 *  dimensions.
 *
 * Intended as static member of farfield_propagator
 *
 * Note that plans will stay allocated for the duration of the process.
 * To avoid that, this class could be included into the GpuManager, and
 * cleared together with the resetFunctionCache method.
 */
class FFTPlanManager
{
public:
  cufftHandle get_or_create_plan(int i, int m, int n);
  void clearCache();
  ~FFTPlanManager();

private:
  /// map key type
  struct key_t
  {
    int i, m, n;
  };

  struct comp_t
  {
    bool operator()(const key_t &a, const key_t &b) const
    {
      if (a.i != b.i)
        return a.i < b.i;
      if (a.m != b.m)
        return a.m < b.m;
      return a.n < b.n;
    }
  };

  /// map type
  typedef std::map<const key_t, cufftHandle, comp_t> map_t;

  /// our stored plans
  map_t plans_;
};

class FarfieldPropagator : public CudaFunction
{
public:
  FarfieldPropagator();
  void setParameters(size_t batch_size, size_t m, size_t n);

  // for setting external memory to be used
  // (can be null if internal should be used)
  void setDeviceBuffers(complex<float> *d_datain,
                        complex<float> *d_dataout,
                        complex<float> *d_prefilter,
                        complex<float> *d_postfilter);

  void allocate();
  void transfer_in(const complex<float> *data_to_be_transformed,
                   const complex<float> *prefilter,
                   const complex<float> *postfilter);
  void transfer_out(complex<float> *out);
  void run(bool doPreFilter, bool doPostFilter, bool isForward);
  static void clearPlanCache() { planManager_.clearCache(); }

private:
  size_t batch_size_ = 0, m_ = 0, n_ = 0;
  float sc_ = 1.0f;
  DevicePtrWrapper<complex<float>> d_datain_;
  DevicePtrWrapper<complex<float>> d_dataout_;
  DevicePtrWrapper<complex<float>> d_pre_;
  DevicePtrWrapper<complex<float>> d_post_;
  cufftHandle plan_ = 0;
  static FFTPlanManager planManager_;
};
