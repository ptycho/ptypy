# PyCUDA to CuPy Porting Notes

This file collects notes for things to consider and issues that were fixed when 
porting the pycuda code to cupy.

## Simple Conversions

- `gpuarray.to_gpu` => `cp.asarray`
- `gpuarray.zeros`, etc, typically have cupy equivalents in `cp.`
- `gpuarray.get` generally works with `cp.get` as well, but cupy has a more descriptive `cp.asnumpy` as well
- all functions that don't have a direct numpy equivalent are in `cupyx` rather than `cupy`
  (for example for pinned arrays)
- raw data pointers to GPU arrays can be retrieved with `x.data.ptr`
- raw data pointers to streams: `stream.ptr`
- low-level APIs, are closer to the standard CUDA runtime calls and are in `cupy.cuda.runtime` module, for example `memcpyAsync`
- streams are not parameters, but rather contexts:

```python
stream = cp.cuda.Stream()
with stream:
  ... # kernel calls etc will go onto this stream

# alternative:
stream.use()
... # next kernel calls will use that stream
```


## Sticky Points

### Memory Pool

- cupy uses a device memory pool by default, which re-uses freed memory blocks
- the pool is empty at the start and new allocations are using the regular cudaAlloc functions
- once blocks are freed, they are not given back to the device with cudaFree, but are rather
  kept in a free list and re-used in further allocations
- therefore the flag for using device memory pool that some engines had made no sense
- this also affects are total available memory should be calculated - it is in fact the free 
  device memory + the free memory in the pool

### Page-locked Memory Pool

- cupy also uses a `PinnedMemoryPool` for obtaining page-locked blocks
- these will be kept in a free list when they are not required anymore
- it works similar to the `DeviceMemoryPool`

### Context Management

- cupy does not have explicit context creation or deletion of the context 
- everything runs in the CUDA runtime's default context (created on first use by default)
- no functions are available to pop the context (as in PyCuda), so need to be
  careful with cleanup


### Kernel Compilation

- cupy uses NVTRC, which is slightly different to NVCC. 
- the generated device code is not exactly the same for some reason
- Kernels might therefore perform a little bit different - faster or slower, but tests showed
  that they are largely equivalent in performance
