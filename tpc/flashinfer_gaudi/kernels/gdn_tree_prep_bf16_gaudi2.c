// SPDX-License-Identifier: Apache-2.0
// g2e: Tree-speculative GDN input preparation (width-4 causal conv along arbitrary tree ancestors + SiLU):
//   y[t, c] = silu(sum_{k=0..3} w[c, k] * seq[conv_idx[t, k], c])
// stored as X [128, 2N, 16] f32 (rows 0..N-1: k, N..2N-1: q) and V [C - 4096, N] f32.
// Tensors (dim 0 fastest):
//   0 seq      BF16 [C, N + 3]
//   1 w        BF16 [C, 4]
//   2 conv_idx I32  [4, N]   (graph input: 4 ancestor row indices in [0, N + 3) per tree node t)
//   3 X        F32  [128, 2N, 16]
//   4 V        F32  [C - 4096, N]
// Index space: [C / 128].
#define QK 2048

static float64_pair_t bf16_to_f32_linear(bfloat128 input)
{
    bfloat128_pair_t unpacked;
    unpacked.v1 = v_bf16_unpack_b(
        input, ((e_group_0) << 8) | ((e_every_second_element) << 9) | ((e_lower_half_group) << 10), unpacked.v1);
    unpacked.v2 = v_bf16_unpack_b(
        input, ((e_group_1) << 8) | ((e_every_second_element) << 9) | ((e_lower_half_group) << 10), unpacked.v2);
    const bfloat128 first_groups = unpacked.v1;
    unpacked.v1 = v_bf16_mov_dual_group_b(unpacked.v2, 0xFFFFFFFF, 0, 1, MkWr(1, 1), unpacked.v1);
    unpacked.v1 = v_bf16_mov_dual_group_b(first_groups, 0xFFFFFFFF, 1, 2, MkWr(1, 1), unpacked.v1);
    unpacked.v1 = v_bf16_mov_dual_group_b(unpacked.v2, 0xFFFFFFFF, 1, 3, MkWr(1, 1), unpacked.v1);
    unpacked.v2 = v_bf16_mov_dual_group_b(first_groups, 0xFFFFFFFF, 2, 0, MkWr(1, 1), unpacked.v2);
    unpacked.v2 = v_bf16_mov_dual_group_b(unpacked.v2, 0xFFFFFFFF, 2, 1, MkWr(1, 1), unpacked.v2);
    unpacked.v2 = v_bf16_mov_dual_group_b(first_groups, 0xFFFFFFFF, 3, 2, MkWr(1, 1), unpacked.v2);
    const float64_pair_t first = v_convert_bf16_to_f32_all_b(unpacked.v1);
    const float64_pair_t second = v_convert_bf16_to_f32_all_b(unpacked.v2);
    float64_pair_t result;
    result.v1 = first.v1;
    result.v2 = second.v1;
    return result;
}

void main(tensor seq, tensor w, tensor conv_idx, tensor X, tensor V)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int tokens = get_dim_size(seq, 1) - 3;

    for (int tile = start[0]; tile < end[0]; ++tile) {
        const int c = tile * 128;
        int5 wc = {c, 0, 0, 0, 0};
        const float64_pair_t w0 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(wc, w));
        wc[1] = 1;
        const float64_pair_t w1 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(wc, w));
        wc[1] = 2;
        const float64_pair_t w2 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(wc, w));
        wc[1] = 3;
        const float64_pair_t w3 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(wc, w));

        for (int t = 0; t < tokens; ++t) {
            int5 ic0 = {0, t, 0, 0, 0};
            int5 ic1 = {1, t, 0, 0, 0};
            int5 ic2 = {2, t, 0, 0, 0};
            int5 ic3 = {3, t, 0, 0, 0};
            const int r0 = s_i32_ld_g((__global__ int*)gen_addr(ic0, conv_idx));
            const int r1 = s_i32_ld_g((__global__ int*)gen_addr(ic1, conv_idx));
            const int r2 = s_i32_ld_g((__global__ int*)gen_addr(ic2, conv_idx));
            const int r3 = s_i32_ld_g((__global__ int*)gen_addr(ic3, conv_idx));

            int5 sc0 = {c, r0, 0, 0, 0};
            int5 sc1 = {c, r1, 0, 0, 0};
            int5 sc2 = {c, r2, 0, 0, 0};
            int5 sc3 = {c, r3, 0, 0, 0};
            const float64_pair_t s0 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(sc0, seq));
            const float64_pair_t s1 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(sc1, seq));
            const float64_pair_t s2 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(sc2, seq));
            const float64_pair_t s3 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(sc3, seq));

            float64_pair_t o;
            o.v1 = s0.v1 * w0.v1;
            o.v1 = v_f32_mac_b(s1.v1, w1.v1, o.v1);
            o.v1 = v_f32_mac_b(s2.v1, w2.v1, o.v1);
            o.v1 = v_f32_mac_b(s3.v1, w3.v1, o.v1);
            o.v2 = s0.v2 * w0.v2;
            o.v2 = v_f32_mac_b(s1.v2, w1.v2, o.v2);
            o.v2 = v_f32_mac_b(s2.v2, w2.v2, o.v2);
            o.v2 = v_f32_mac_b(s3.v2, w3.v2, o.v2);
            o.v1 = o.v1 * v_sigmoid_f32(o.v1);
            o.v2 = o.v2 * v_sigmoid_f32(o.v2);

            if (c < 2 * QK) {
                const int is_k = c >= QK;
                const int head = (c - (is_k ? QK : 0)) / 128;
                int5 xc = {0, is_k ? t : tokens + t, head, 0, 0};
                v_f32_st_tnsr(xc, X, o.v1);
                xc[0] = 64;
                v_f32_st_tnsr(xc, X, o.v2);
            } else {
                int5 vc = {c - 2 * QK, t, 0, 0, 0};
                v_f32_st_tnsr(vc, V, o.v1);
                vc[0] += 64;
                v_f32_st_tnsr(vc, V, o.v2);
            }
        }
    }
}
