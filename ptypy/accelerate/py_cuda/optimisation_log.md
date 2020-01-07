# PyCuda Optimisation Log

This log summarises the optimisations performed on the PyCuda engine,
including the attempted ones that did not lead to improvements,
for future reference.

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
    * **Speedup:**: XXX
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


## Streaming Engine