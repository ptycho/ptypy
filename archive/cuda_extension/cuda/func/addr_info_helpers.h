#pragma once

#include <vector>

/** Creates a datastructure to get the mapping of which indices in the addr_info
 * array (first dim) map to each output address (handling multiple occurrences).
 *
 *  This function is used by sum_to_buffer, to avoid data races on output
 * assignment.
 *
 *  @param out_addr_info pointer to addr_info start location containing the
 * output indices. Example, for da in full addr_info: addr_info + 9
 *  @param addr_len Number of lines in addr_info
 *  @param addr_stride Stride to go from one line to the next (it's 15 in full
 * addr_info, and 3 if data has been sliced as addr_info(:,3,:)
 *  @param outidx output array of indices in the output the data should be
 * written to (that's the unique values of out_addr_info[0] for every line)
 *  @param startidx output array of starting indices into the indices array, for
 * the corresponding element in outidx. The length is 1 longer than outidx, to
 * accomodate start + end indices at i and i+1
 *  @param indices Array of indices that map to the output index.
 *
 */
void flatten_out_addr(const int *out_addr_info,
                      int addr_len,
                      int addr_stride,
                      std::vector<int> &outidx,
                      std::vector<int> &startidx,
                      std::vector<int> &indices);
