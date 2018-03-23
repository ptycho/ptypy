#include <map>
#include <iostream>

static std::map<void*, size_t> alloc_map_;
static size_t total = 0;

void debug_addMemory(void* ptr, size_t size)
{
    alloc_map_[ptr] = size;
    total += size;
}

void debug_freeMemory(void* ptr)
{
    if (alloc_map_.find(ptr) != alloc_map_.end()) {
        auto size = alloc_map_[ptr];
        total -= size;
        alloc_map_.erase(ptr);
    } else {
        std::cerr << "WARNING: freeing memory of ptr " << ptr << " that hasn't been registered before" << std::endl;
    }
}

size_t debug_getMemory()
{
    return total;
}