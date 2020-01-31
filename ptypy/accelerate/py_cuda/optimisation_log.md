# PyCuda Optimisation Log

This log summarises the optimisations performed on the PyCuda engine,
including the attempted ones that did not lead to improvements,
for future reference.

- [Individual Kernels](#individual-kernels)
  - [Object and Probe Update](#object-and-probe-update)
    - [Tiled Version Optimisations](#tiled-version-optimisations)
    - [Atomic Version Optimisations](#atomic-version-optimisations)
  - [Build Exit Wave](#build-exit-wave)
  - [FFT](#fft)
  - [Error Reduce](#error-reduce)
  - [Fourier Error](#fourier-error)
- [Kernel Fusion](#kernel-fusion)
- [Streaming Engine](#streaming-engine)

## Individual Kernels

### Object and Probe Update

* The kernels for both are similar
  so the same optimisations apply for both
* 2 versions of the kernel:
    * A version using a thread block per line in the address array,
      with atomic adds to avoid race conditions in the output array -> [ob_update.cu](cuda/ob_update.cu). 
      This version was developed in the original CUDA Python module in 2018.
    * A version using a thread block per tile of the output array,
      iterating over all lines of the address array and only updating
      the part relevant to the current tile -> [ob_update2.cu](cuda/ob_update2.cu).
      This version is based on the original OpenCL version with PyOpenCL.
* Performance trade-offs are:
    * *Atomic Adds Version:*
        * No redudant loads of the address array (one line per thread block)
        * No checks if the update is relevant in the current tile 
          (unconditional application of the update)
        * BUT the overhead of atomics in global memory (requires a global 
          load, add, and store atomically every time)
    * *Tiled Version:*
        * No atomics - all updates are local
        * BUT conditional if the update is affecting the current   
          thread-block's tile.
        * Becomes more efficient if the probe array is larger so that the conditional becomes true for most cases and there are less misses
    * Initial tests showed that both versions are valid, depending on the size of the probe array -> the right version can be chosed based on the data sizes
    * These tests need to be repeated after all optimisations have been applied

#### Tiled Version Optimisations

1. Starting Point:
    * Every thread loads the full address array and iterates over it
    * Reads all the modes for the local thread into a local array (stored in registers)
    * Then updates this local array by iterating over all addresses
    * Writes back to global memory at the end
2. Shared Memory:
    * Avoids every thread loading the address array redudantly by 
      collaborating within a thread block for these loads
    * Lets every thread in a threadblock load a line of the address array
      into shared memory, then sync
    * Then the iteration can be done in shared memory, avoiding global memory access
    * Reduces redundant loads
    * **Speedup:** XXX
3. Coalesced Address Array Loads:
    * Transposes the address array before transfering to GPU,
      so that global memory accesses to the address lines are coalesced between threads
    * Reduces the global loads again as coalesced loads reduce global memory 
      access
    * **Speedup:** XXX
4. Compile-time Constants:
    * As PyCuda compiles kernels on the fly, we can set the number of modes
      and array sizes as constants before compilation
    * **Speedup:** XXX
5. Real-part Updates:
    * Only the real part of the denominator needs updating
    * The imaginary part is always zero, so it didn't need to be computed / added
    * This was modified
    * **Speedup:** XXX
6. Texture Caches:
    * The constant kernel parameters were put into texture caches using the
      `const X* __restrict__` modifiers
    * This accelerates the repeated loads and frees the L2/L1 caches for other data
    * **Speedup:** XXX
7. Loop unrolling:
    * Through experimentation it was found that the update loop could be unrolled by factor 4 for best performance
    * **Speedup:** XXX

#### Atomic Version Optimisations

1. Starting Point:
    * Version based on 2018 CUDA effort [extract_array_from_exit_wave.cu](../../../cuda/func/extract_array_from_exit_wave.cu)
2. Coalesced Access:
    * Swap loop order to iterate of the `threadIdx.x` dimension in the inner loop (this is the fast-running index between threads)
    * This makes sure that the global memory loads and stores are coalesced
    * **Speedup:** XXX
3. Texture Caches:
    * The constant kernel parameters were put into texture caches using the
      `const X* __restrict__` modifiers
    * This accelerates the repeated loads and frees the L2/L1 caches for other data
    * **Speedup:**: XXX
4. Loop Unrolling:
    * Experiments where made with different loop unrolling factors, but non of them made a difference
5. Real-part Updates:
    * Only the real part of the denominator needs updating
    * The imaginary part is always zero, so it didn't need to be computed / added
    * This was modified
    * **Speedup:** XXX

### Build Exit Wave

1. Starting Point
    * Version with atomic adds to update the exit wave array
2. Coalesced Access:
    * Swap loop order to iterate of the `threadIdx.x` dimension in the inner loop (this is the fast-running index between threads)
    * This makes sure that the global memory loads and stores are coalesced
    * **Speedup:** XXX
3. Remove Atomics
    * The exit wave output is actually not overlapping
    * Atomics are not needed
    * **Speedup:** XXX
4. Loop Unrolling
    * Unrolling innner loop by factor for gives a slight performance advantage
    * **Speedup:** 93ms -> 91ms

### FFT

* The Reikna version is used with pre-FFT and post-IFFT arrays built-in for scaling and shifting
* Comparisons showed large speedups of cuFFT if callback mechanism is used
* With separate kernels for pre- and post-filtering it's worse than Reikna
* Times:

```
For 100 calls of 256x256 with batch size 2000:
- Reikna with or without filters: 1,470ms
- cuFFT without filters         :   792ms
- cuFFT with separate filters   : 1,564ms
- cuFFT with callbacks          :   916ms

For 128x128 with batch size 2000:
- Reikna with or without filters:   389ms
- cuFFT without filters         :   194ms
- cuFFT with separate filters   :   388ms
- cuFFT with callbacks          :   223ms
```

* Put separate pybind11 module, compiled on-the-fly with cppimport, with hard-coded array sizes for greater efficiency (recompiled for different sizes)
* Implemented in cufft.py module

### Error Reduce

1. Coalesced Access:
    * Swap loop order to iterate of the `threadIdx.x` dimension in the inner loop (this is the fast-running index between threads)
    * This makesfmagess:
    * Swap loop order to iterate of the `threadIdx.x` dimension in the inner loop (this is the fast-running index between threads)
    * This makes sure that the global memory loads and stores are coalesced
    * **Speedup:** XXX
2. Texture Cache
  * Not beneficial on any of the constant inputs
3. Boolean Mask
  * Using a boolean expression with `m < 0.5 ? X : Y` is slightly slower than the floating point version (48.8ms -> 49.2ms)
4. Loop Unrolling
  * Make no difference

### Fourier Error

1. Coalesced Access:
    * Swap loop order to iterate of the `threadIdx.x` dimension in the inner loop (this is the fast-running index between threads)
    * This makes sure that the global memory loads and stores are coalesced
    * **Speedup:** XXX
2. Texture Cache
  * Not beneficial on any of the constant inputs
3. Store fdev in register
  * tries to avoid writing back to global memory and reading it back immediately
  * seems that compiler already does this optimisation -> no difference
4. Avoid absolute value calculation
  * The fdev value is squared afterwards and is real-valued, so there's no need for absolute value calculation
  * --> makes no noticable difference
5. Use Mask as Boolean
  * Chaning expression with the boolean ? operator to avoid unnecessary loads when mask is 0
  * Didn't change anything in the performance
6. Occupancy
   * Kernel only got only 50% occupancy, due to too many registers per block
   * Specifying `__launch_bounds__` on the kernel made compiler generate less registers --> 100% occupancy
   * Speedup: 35.8ms -> 31.5ms
7. Using one thread per mode + shared memory:
  * original lets every thread go in a loop over all modes to su
  * this version uses a thread per mode + shared memory + reduction instead
  * implemented in [fourier_error2.cu](cuda/fourier_error2.cu)
  * Test results on P100, minimal pre and run template for DM, 20 iterations:
    * original  : 35.80ms total (40 calls)
    * shared mem: 80.12ms

Further optimisations:
* why is abs(f)^2 calculated - what are "errors" here? OpenCL uses the real*real+imag*imag version
  


## Kernel Fusion

* fourier_error and fmag_all_update are joinable
  * In former, all modes are calculated by 1 block, while the latter looks at them individually
  * fourier_error2 does that in shared memory with same no. of threads than fmag_all_update, but it's >2x slower than the other fourier_error
* However, fourier_error2 and fmag_all_update are mergable
* This has been tried in fourier_update.cu, but it was > 2x slower than 
  individual kernels

## Streaming Engine