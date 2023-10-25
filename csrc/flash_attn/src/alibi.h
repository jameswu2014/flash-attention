#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
inline __device__ void apply_alibi(Tensor<Engine, Layout> &tensor,
                                   const int col_idx_offset_,
                                   const int row_idx_offset,
                                   const int head_idx,
                                   const int num_heads,
                                   const float softmax_scale,
                                   const float *alibi_slopes,
                                   const int col_seqlen = 0,
                                   const int row_seqlen = 0,
                                   const int warp_row_stride = 0) {
    const float alibi_slope = alibi_slopes[head_idx];
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

    // printf("threadidx: %d, size<1, 1>(tensor):%d, size<1, 0>(tensor):%d, col_idx_offset_:%d, softmax_scale:%f, alibi_slope:%f, head_idx:%d \n",
    //                 threadIdx.x, size<1, 1>(tensor), size<1, 0>(tensor), col_idx_offset_, softmax_scale,alibi_slope, head_idx);
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    // tensor isn't scaled by softmax_scale so unscale alibi with softmax_scale
                    const float alibi = (alibi_slope * col_idx) / softmax_scale;
                    if (col_idx < col_seqlen && row_idx < row_seqlen) {
                        // Without the "make_coord" we get wrong results
                        tensor(make_coord(i, mi), make_coord(j, nj)) += alibi;
                    }   
                }
            }
        }
    }
}

}  // namespace flash