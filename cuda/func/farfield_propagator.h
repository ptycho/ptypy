#pragma once

#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

#include <cufft.h>
#include <tuple>
#include <unordered_map>

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
  ~FFTPlanManager();

private:
  /// map key type
  typedef std::tuple<int, int, int> key_t;

  /// operation to hash the key - combine hashes of individual values
  struct key_hash : public std::unary_function<key_t, std::size_t>
  {
    std::size_t hash_combiner(size_t left, size_t right) const
    {
      return left + 0x9e3779b9 + (right << 6) + (right >> 2);
    }
    std::size_t operator()(const key_t &k) const
    {
      return hash_combiner(hash_combiner(std::hash<int>()(std::get<0>(k)),
                                         std::hash<int>()(std::get<1>(k))),
                           std::hash<int>()(std::get<2>(k)));
    }
  };

  /// operation to compare keys
  struct key_equal : public std::binary_function<key_t, key_t, bool>
  {
    bool operator()(const key_t &a, const key_t &b) const
    {
      return std::get<0>(a) == std::get<0>(b) &&
             std::get<1>(a) == std::get<1>(b) &&
             std::get<2>(a) == std::get<2>(b);
    }
  };

  /// map type
  typedef std::unordered_map<const key_t, cufftHandle, key_hash, key_equal>
      map_t;

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
