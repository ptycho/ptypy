# PyCuda Optimisation Log

This log summarises the optimisations performed on the PyCuda engine,
including the attempted ones that did not lead to improvements,
during the December-February 2019/2020 project.
It should serve as a future reference to help understand the optimisations,
and to avoid re-attempting optimisations that were not beneficial.

**Contents**

- [DM Engine Kernel Optimisations](#dm-engine-kernel-optimisations)
  - [General Notes](#general-notes)
  - [Object and Probe Update](#object-and-probe-update)
    - [Tiled Version Optimisations](#tiled-version-optimisations)
    - [Atomic Version Optimisations](#atomic-version-optimisations)
  - [Build Exit Wave](#build-exit-wave)
  - [Error Reduce](#error-reduce)
  - [Fourier Error](#fourier-error)
  - [FFT](#fft)
    - [Analysis](#analysis)
    - [Implementation](#implementation)
  - [Kernel Fusion](#kernel-fusion)
- [ML Engine Kernels](#ml-engine-kernels)
  - [Finite Differences (Forward / Backward)](#finite-differences-forward--backward)
  - [Fill_b](#fill_b)
  - [Make_a012](#make_a012)
  - [Make_model](#make_model)
  - [Probe/Object Update](#probeobject-update)
  - [Dot Product](#dot-product)
- [MPI](#mpi)
- [Position Refinement](#position-refinement)
- [Streaming Engine](#streaming-engine)
  - [Data and Streaming Management Classes](#data-and-streaming-management-classes)
    - [GpuData](#gpudata)
    - [GpuDataManager](#gpudatamanager)
  - [Stream Management](#stream-management)
  - [Synchronisation](#synchronisation)
  - [Page-Locked Memory](#page-locked-memory)

## DM Engine Kernel Optimisations

### General Notes

- All kernels have been written / modified to make no assumptions about the
  input data sizes (for example being powers of two), or maximum number of modes,
  etc.
- This has been proven by successfully running the insanity test case using the
  PyCuda streaming engine
- The following optimisations were found to be most important (in order):
  - Coalescing global memory access
  - Reducing redundant global memory access by making use of shared memory.
    In once instance this required transposing the data on the CPU.
  - Tuning thread block sizes through experiments
  - Maximising occupancy, for example by giving hints to the compiler using `__launch_bounds__`
    so that it can optimise register usage to fit more blocks
  - Using texture caches
  - Math optimisations
  - loop unrolling
- Kernels not explicitly mentioned in the list below either follow the same optimisations,
  or were already found to be high performance

### Object and Probe Update

- The kernels for both are similar so the same optimisations apply for both
- 2 versions of the kernel exist:
    1. A version using a thread block per line in the address array,
       with atomic adds to avoid race conditions in the output array -> [ob_update.cu](cuda/ob_update.cu).
       This version was developed in the original CUDA Python module  in 2018.
    2. Another version using a thread block per tile of the output array,
       iterating over all lines of the address array and only updating
       the part relevant to the current tile -> [ob_update2.cu](cuda/ob_update2.cu).
       This version is based on the original OpenCL version with PyOpenCL.
- Performance trade-offs are:
  - *Atomic Adds Version:*
    - No redudant loads of the address array (one line per thread block)
    - No checks if the update is relevant in the current tile
      (unconditional application of the update)
    - BUT the overhead of atomics in global memory (requires a global load, add, and store atomically every time)
  - *Tiled Version:*
    - No atomics - all updates are local
    - BUT conditional if the update is affecting the current   thread-block's tile.
    - Becomes more efficient if the probe array is larger so  that the conditional becomes true for most cases and there are less misses
  - Initial tests showed that both versions are valid, depending on the size of the probe array -> the right version can be chosed based on the data sizes
  - These tests need to be repeated after all optimisations have been applied

#### Tiled Version Optimisations

1. Starting Point:
   - Every thread loads the full address array and iterates over it
   - Reads all the modes for the local thread into a local array (stored in registers)
   - Then updates this local array by iterating over all addresses
   - Writes back to global memory at the end
2. Shared Memory:
    - Avoids every thread loading the address array redudantly by
      collaborating within a thread block for these loads
    - Lets every thread in a threadblock load a line of the address array
      into shared memory, then sync
    - Then the iteration can be done in shared memory, avoiding global memory access
    - Reduces redundant loads
3. Coalesced Address Array Loads:
    - Transposes the address array before transfering to GPU,
      so that global memory accesses to the address lines are coalesced between threads
    - Reduces the global loads again as coalesced loads reduce global memory
      access
4. Compile-time Constants:
    - As PyCuda compiles kernels on the fly, we can set the number of modes
      and array sizes as constants before compilation, allowing better compiler optimisation
5. Real-part Updates:
    - Only the real part of the denominator needs updating
    - The imaginary part is always zero, so it didn't need to be computed / added
6. Texture Caches:
    - The constant kernel parameters were put into texture caches using the
      `const X* __restrict__` modifiers
    - This accelerates the repeated loads and frees the L2/L1 caches for other data
7. Loop unrolling:
    - Through experimentation it was found that the update loop could be unrolled by factor 4 for best performance
8. Real denominator:
    - Change obn/prn to real data type globally in ptypy, which benefits this kernel's performance (less data to load)

#### Atomic Version Optimisations

1. Starting Point:
    - Version based on 2018 CUDA effort [extract_array_from_exit_wave.cu](../../../archive/cuda_extension/cuda/func/extract_array_from_exit_wave.cu)
2. Coalesced Access:
    - Swap loop order to iterate of the `threadIdx.x` dimension in the inner loop (this is the fast-running index between threads)
    - This makes sure that the global memory loads and stores are coalesced
3. Texture Caches:
    - The constant kernel parameters were put into texture caches using the
      `const X* __restrict__` modifiers
    - This accelerates the repeated loads and frees the L2/L1 caches for other data
4. Loop Unrolling:
    - Experiments where made with different loop unrolling factors, but none of them made a difference
5. Real-part Updates:
    - Only the real part of the denominator needs updating
    - The imaginary part is always zero, so it didn't need to be computed / added
6. Real denominator:
    - Change obn/prn to real data type globally in ptypy, which benefits this kernel's performance (less data to load)

### Build Exit Wave

1. Starting Point
    - Version with atomic adds to update the exit wave array
2. Coalesced Access:
    - Swap loop order to iterate of the `threadIdx.x` dimension in the inner loop (this is the fast-running index between threads)
    - This makes sure that the global memory loads and stores are coalesced
3. Remove Atomics
    - The exit wave output is actually not overlapping, so atomics are not needed
4. Loop Unrolling
    - Unrolling innner loop by factor for gives a slight performance advantage

### Error Reduce

1. Coalesced Access:
    - Swap loop order to iterate of the `threadIdx.x` dimension in the inner loop (this is the fast-running index between threads)
    - This makes sure that the global memory loads and stores are coalesced
2. Texture Cache
   - Not beneficial on any of the constant inputs
3. Boolean Mask
   - Using a boolean expression with `m < 0.5 ? X : Y` is slightly slower than the floating point version (48.8ms -> 49.2ms), so we keep using it as a number
4. Loop Unrolling
   - Make no difference

### Fourier Error

1. Coalesced Access:
    - Swap loop order to iterate of the `threadIdx.x` dimension in the inner loop (this is the fast-running index between threads)
    - This makes sure that the global memory loads and stores are coalesced
2. Texture Cache
    - Not beneficial on any of the constant inputs
3. Store fdev in register
    - original code writes back to global memory and reading it back immediately
    - seems that compiler already does this optimisation -> no difference
4. Avoid absolute value calculation
   - The fdev value is squared and as it's real-valued, so there's to calc absolute value first
   - --> makes no noticable difference
5. Use Mask as Boolean
    - Changing expression with the boolean ? operator to avoid unnecessary loads when mask is 0
    - --> didn't affect performance
6. Occupancy
    - Kernel only achieved 50% occupancy, due to too many registers per block
    - Specifying `__launch_bounds__` on the kernel made compiler generate less registers --> 100% occupancy
    - Speedup: 35.8ms -> 31.5ms
7. Using one thread per mode + shared memory:
    - original lets every thread go in a loop over all modes to su
    - tried different version which uses a thread per mode + shared memory + reduction instead
    - implemented in [fourier_error2.cu](cuda/fourier_error2.cu)
    - Test results on P100, minimal prep and run template for DM, 20 iterations:
      - original  : 35.80ms total (40 calls)
      - shared mem: 80.12ms
    - --> it's far worse, so not using this
8. Future Optimisations (not done)
   - kernel calculates `real(abs(f)^2)`, which is mathematically the same as `real(f)*real(f) + imag(f)*imag(f)`
   - The second version is faster to compute
   - Code has a comment that with the second version, results differ from numpy
   - However, OpenCL uses the second version
   - This should be reconsidered and changed to the faster version

### FFT

#### Analysis

- The Reikna version is used with pre-FFT and post-IFFT arrays built-in for scaling and shifting
- Comparisons showed large speedups with cuFFT if built-in load/store callback mechanism is used
  for the scaling and shifting
- cuFFT separate kernels for pre- and post-filtering it's worse than Reikna, as shown below:

| Version                            | 128x128x2000 | 256x256x2000 |
|------------------------------------|--------------|--------------|
| Reikna with or without filters     |        389ms |      1,470ms |
| cuFFT without filters              |        194ms |        793ms |
| cuFFT with separate filter kernels |        388ms |      1,564ms |
| cuFFT with callbacks               |        223ms |        916ms |
| **=> cuFFT/callback vs Reikna**    |    **1.74x** |    **1.60x** |

#### Implementation

- cuFFT with callbacks only works from C++/CUDA, as callbacks are device functions
  which need to be compiled with device-side linking and then pointers to them need
  to be obtained and passed to the cuFFT calls.
- This mechanism is not supported in SciKit-CUDA or PyCUDA, hence we need a native
  compiled Python module for the task
- We don't want a compilation step at package-install time, prefer PyCUDA's runtime
  compilation approach
- Hence we used the [cppimport](https://github.com/tbenthompson/cppimport) to build a
  Python module with the [pybind11](https://pybind11.readthedocs.io/en/stable/) C++ library (header-only) for the Python bindings
- The headers can be install with pip conveniently
- CppImport compiles the code when it's imported on-the-fly (and caches it)
- We also used compile-time constant for the FFT sizes, so that the compiler is more efficient
  in the callbacks (see comments in the code)
- The [import_fft.py](import_fft.py) file mangaes the cppimport itself, setting the compiler
  to nvcc and setting up the compilation flags
- The [cufft.py](cufft.py) wraps the import and calls, providing the same interface as the
  [Reikna FFT class](fft.py)

### Kernel Fusion

- Of all kernels called in a sequence, the [fourier_error](cuda/fourier_error.cu),
  [error_reduce](cuda/error_reduce.cu) and [fmag_all_update](cuda/fmag_all_update.cu) kernels are joinable in principle
- In the former, all modes are calculated by 1 thread, while the latter flattens the data over
  the modes
- An attempt was made in [fourier_update.cu](cuda/fourier_update.cu):
  - Using shared memory reduction rather than one thread to add the modes, which unifies the thread pattern between both kernels
  - Then joining the fmag_all_update kernel
  - This was found to be more than twice slower than individual calls
  - Also there is a potential race condition, as the err_fmag array is accessed at a different
    address (using address book) than when the kernel writes to it. This should be fixable though, by working with the address book all along the fused kernel
- *Future optimisation*
  - Another attempt could be done by not using shared memory, but changing fmag_all_update
    to iterate over the modes in one thread as well
  - the race condition would also need to be fixed, using the address book (if possible)
  - This could potentially improve the performance of these kernel combined by up to 2x,
    but in the overall timeline, that's only in the single-digit percentages

## ML Engine Kernels

Several new kernels have been implemented for the ML engine. The ones with non-trivial
optimisations are given below.
Note that all these kernels have been written to support both float and double
inputs, using compile-time replacements.
It is recommended to modify the DM kernels in a similar fashion for flexibility.

### Finite Differences (Forward / Backward)

- These kernels are calculating the difference of 3D arrays with a shifted version of itself: [delx_mid.cu](cuda/delx_mid.cu) and [delx_last.cu](cuda/delx_last.cu)
- They use shared memory to avoid loading the current and next pixel in every thread (reduces global loads by 2x)
- The implementation uses one thread block to iterate along the difference axis,
  while the other dimensions are tiled using as many thread blocks as necessary
- Array dimenions are folded, i.e. any dimenions > 3 can be expressed as an array
  with 3 dimensions where the axes dimensions before and after the difference axis are multiplied together
- For < 3D, the other dimensions can be considered of size 1
- For coalescing the load operations, there are 2 kernels: one for the last axis
  and one for all previous axes. The difference is that the loads into shared
  memory are transposed.
- the code is commented in detail

### Fill_b

- the fill_b kernel is essentially some computations and then a reduction along the final 2 dimensions
- The math and first-stage of the reduction (within thread blocks) is in [fill_b](cuda/fill_b.cu)
- The final reduction stage across blocks is in [fill_b_reduce](cuda/fill_b_reduce.cu)
- We're reducing all 3 arrays in a single kernel, using shared memory, to avoid
  the call overheads
- the block sizes have been optimised via experiments

### Make_a012

- The kernel [make_a012](cuda/make_a012.cu) implements this functionality
- The summation across modes is done in a single thread, as this was found to be more effecient than shared memory (given only a small number of modes is typically used)

### Make_model

- The [make_model](cuda/make_model.cu) is similar, though simpler, to make_a012,
  and uses the same pattern (summation across modes in one thread)

### Probe/Object Update

- These kernels are the same as for DM, except that there is no denominator
- The same optimisations apply

### Dot Product

- The real and complex vector dot product has been implemented in [dot.cu](cuda/dot.cu), including blockwise reduction.
- This is followed by a full reduction over all blocks in [full_reduce.cu](cuda/full_reduce.cu)
- Note that for complex values, this is not mathematically computing the dot product - it calculates `real(a)*real(b) + imag(a)*imag(b)`

## MPI

- The MPI all-reduce calls are a major bottleneck, after all the GPU optimisations
- Measurements were taken on DLS cluster, using the [mpi_allreduce_bench.sh](../../../benchmark/mpi_allreduce_bench.sh) script, for single-node and multi-node MPI
- Results are in [this sheet](https://docs.google.com/spreadsheets/d/1OyIWTkFit-0EXKzdODkcw7WfqQC2ldjLXzdz05jTQHY/edit#gid=188734367)
- Key findings:
  - synchronisation time is quasi-linear with the number of nodes in DLS
  - OpenMPI is far faster than MPICH2 in DLS

## Position Refinement

- An optional position refinement algorithm was added to the PyCUDA engines
- So far, it mangles and shifts the addresses on the CPU, and uses the GPU
  to evaluate the errors
- It's working, but there is singificant data transfers between host and device
- Transposing the addresses for tiled version of ob/pr update is done on GPU though
- Suggested future optimisations:
  - Implement a GPU kernel to pick the indexes from the mangled and original arrays, i.e. for the `update_addr_and_error_state` method
    - this is straightforward, just reading a line for one of the two input arrays
      depending on a flag
    - This avoids a GPU->CPU-GPU turnaround
  - Implement a kernel for the `mangle_address` method, taking the deltas as
     an input for added flexibility.
    - This is mostly indexing operations, adding
      delta to the correct elements.
    - This avoids another GPU-CPU-GPU turaround, eliminating most of them
  - Perform deltas computation on GPU as well
    - for example, placing the random numbers on GPU for everything at the start
    - or generating them on the fly on the GPU - a simple integer XOR-shift based
      random generator should suffice, as statistical properties are less
      important here

## Streaming Engine

- implemented in [DM_pycuda_stream.py](../../engines/DM_pycuda_stream.py)
- Purpose: allow processing more blocks than fit in GPU memory, by transferring blocks as needed between GPU and CPU, in a chunked fashion
- Implemented using multiple CUDA streams, to overlap transfers with compute,
  and using page-locked CPU memory to make transfers asynchronous and fast
- Using separate class for these purposes enables to use more GPU memory blocks
  for the exit waves than for the ma/mag arrays, as the latter are faster to
  transfer.

### Data and Streaming Management Classes

- Data that needs to be cycled in/out of GPU is managed by 2 classes: GpuData and GpuDataManager
- GpuStreamData manages the streams and links it to the data classes
- All these have been documented in detail with Python doc strings

#### GpuData

- GpuData handles one block of memory for one array (e.g. exit_wave)
- It allocates a raw GPU buffer to hold any instance of the exit_wave,
  and uses a custom allocate function to return the same buffer every time for
  a new array
- It keeps track of which CPU array was transferred by means of an ID
- If the same object is used to transfer a new cpu array (different ID),
  it can copy the existing GPU data back to the held cpu reference before
  replacing it (if syncback=True)
- Some arrays aren't modified on GPU, so the syncback parameter may be false
- It also supports resizing, in case a second engine_prepare is called with different blocks
- Explicit free methods are also given, as we can't wait for the garbage collector
  to free GPU memory in case we need to reallocate immediately, to make sure
  that memory is available

#### GpuDataManager

- manages an array of GpuData objects for multiple blocks of the same array (e.g. exit_wave)
- The GpuData objects can be seen as a list of GPU memory blocks that are fixed
  size
- All interactions with GpuData objects are through this manager
- Supports looking up if a requested transfer to GPU is already present with
  the same ID, in which case it just returns that without transfer
- Otherwise, takes the oldest instance and replaces the memory with the new data
- Supports resizing, relaying calls to all internal GpuData instances

### Stream Management

- GpuDataStream manages one CUDA stream, the related computations and transfers
- It has references to the exit wave, ma, and mag GpuDataManagers
- Decoupling this from the data managment allows to re-use the same data on
  different streams, or have more streams than data blocks
- It provides events to synchronise data re-use for ptypy engines

### Synchronisation

- Even though we use streams to give an ordering of the computation,
  some synchronisation is still needed:
  - exit wave, ma, and mag memory blocks might still be in use when a new one
    is about to be transferred
  - the FFT plan scratch memory and aux blocks are re-used in all streams,
    so there can't be compute kernels of other streams
- for the data (exit wave etc.), this is managed in the GpuData class internally.
  An event is used in this class to mark when transfers are finished,
  and it is synchronised before the same memory block is used again for a new
  transfer to GPU. That means in the engine, we need to mark when we are done
  with the data, using `record_done`.
- for the compute overlaps (aux and FFT), an `end_compute` method is called
  when the compute is done, returning an event that was recorded.
  Then `start_compute` is called before starting to compute the next block,
  waiting for this event

### Page-Locked Memory

- For transfers to be truly asynchronous, the CPU-side memory needs to be
  page-locked (pinned).
- That is CPU memory that cannot be paged out to disk
- PyCUDA provides methods to reserve such memory and use it with numpy arrays
- The memory flag 0 (default) means it's regular pinned memory that is fast to
  read and write from CPU (used for the arrays that need to be MPI synced)
- The memory flag 4 means write-combined memory, which is fast to write on CPU
  and very fast to transfer from CPU to GPU, but its extremely slow to read on CPU.
  It is therefore used only for arrays that are not read/modified on CPU.
