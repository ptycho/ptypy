#include <iostream>
#include <map>

// we have to define this in GpuManager.cu
// otherwise it gets destructed before the CudaFunctions
// are destroyed (potentially at least)
extern std::map<void*, size_t> alloc_map_;
extern size_t alloc_total;

void debug_addMemory(void* ptr, size_t size)
{
  alloc_map_[ptr] = size;
  alloc_total += size;
}

void debug_freeMemory(void* ptr)
{
  if (alloc_map_.find(ptr) != alloc_map_.end())
  {
    auto size = alloc_map_[ptr];
    alloc_total -= size;
    alloc_map_.erase(ptr);
  }
  else
  {
    std::cerr << "WARNING: freeing memory of ptr " << ptr
              << " that hasn't been registered before" << std::endl;
  }
}

size_t debug_getMemory() { return alloc_total; }