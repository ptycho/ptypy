#include "addr_info_helpers.h"

#include <map>


// finds which lines in the o addr array map to the same output 0 index
// --> return a map outputindex -> [addr_line_idx_0, addr_line_idx_1, ...]
static void remap_outaddr(const int *o, int N, std::map<int, std::vector<int>> &addr, int addr_stride)
{
  for (int i = 0; i < N; ++i)
  {
    int idx = i * addr_stride;
    auto o_0 = o[idx];
    addr[o_0].push_back(i);
  }
}

// re-encodes the map to 3 plain vectors as follows:
// outidx contains a list of all output indices that are written to
// start holds the corresponding starting index in the indices vector for the
// mapped list indices is a flattened list with all values from the map
static void make_addrarray(const std::map<int, std::vector<int>> &addr,
                    std::vector<int> &outidx,
                    std::vector<int> &start,
                    std::vector<int> &indices)
{
  for (auto &p : addr)
  {
    outidx.push_back(p.first);
    start.push_back(int(indices.size()));
    indices.insert(indices.end(), p.second.begin(), p.second.end());
  }
  start.push_back(int(indices.size()));
}


void flatten_out_addr(const int *out_addr,
                     int addr_len,
                     int addr_stride,
                     std::vector<int> &outidx,
                     std::vector<int> &startidx,
                     std::vector<int> &indices)
{
  std::map<int, std::vector<int>> addr;
  remap_outaddr(out_addr, addr_len, addr, addr_stride);
  make_addrarray(addr, outidx, startidx, indices);
}
