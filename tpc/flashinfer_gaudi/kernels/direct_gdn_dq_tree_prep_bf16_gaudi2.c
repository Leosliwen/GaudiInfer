// SPDX-License-Identifier: Apache-2.0
// g2e Direct MME+SRAM Tree GDN prep: fused dequant(qkv) + st_seq stash + tree-ancestor width-4 conv + SiLU
// -> X_sram, V_sram, st_x. Runs in parallel with MME(z) + MME(ba) on all 24 TPC cores!
// Tensors (dim 0 fastest):
//   0 qkv_raw       BF16 [10240, N]     MME(qkv) raw output in SRAM
//   1 sa            F32  [1, N]         SRAM activation scale
//   2 swb           F32  [10240]        HBM pre-interleaved qkv weight scales [even 64, odd 64]
//   3 conv_w        BF16 [10240, 4]     HBM transposed conv1d weights
//   4 X_sram        F32  [128, 2*N, 16] SRAM QK output (rows 0..N-1 = K, N..2N-1 = Q)
//   5 V_sram        F32  [6144, N]      SRAM V output
//   6 st_x          F32  [128, 2*N, 16] HBM stashed X for tree commit
//   7 st_seq        BF16 [10240, N+3]   HBM stashed conv sequence [conv_in(0..2), dequant(qkv)(3..N+2)]
//   8 conv_in       BF16 [10240, 3]     HBM prior conv state
//   9 tree_conv_idx I32  [4, N]         HBM ancestor conv indices in [0..N+2] for each tree node
// Index space: flat 1D [0..80) (0..31 = QK tiles waking gram+bwr early, 32..79 = V tiles overlapping gram+bwr).
#define QK 2048
#define MAX_TREE_SEQ 67

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

static inline float64 fast_silu_f32(float64 x)
{
    const float64 xc = v_f32_min_b(v_f32_max_b(x, -87.0f), 87.0f);
    const float64 t = v_f32_mac_b(xc, -1.4426950408889634f, (float64)127.0f);
    const int64 n = v_convert_f32_to_i32_b(t, SW_RHNE);
    const float64 f = t - v_convert_i32_to_f32_b(n, 0);
    float64 p = 0.0096181291f;
    p = v_f32_mac_b(p, f, 0.0555041087f);
    p = v_f32_mac_b(p, f, 0.2402265070f);
    p = v_f32_mac_b(p, f, 0.6931471806f);
    p = v_f32_mac_b(p, f, 1.0f);
    const int64 bits = n << 23;
    const float64 d = 1.0f + p * as_float64(bits);
    float64 r = as_float64((int64)0x7ef311c3 - as_int64(d));
    r = r * (2.0f - d * r);
    r = r * (2.0f - d * r);
    r = r * (2.0f - d * r);
    return x * r;
}

__local__ bfloat128 seq_vlm[MAX_TREE_SEQ];

void main(tensor qkv_raw, tensor sa, tensor swb, tensor conv_w,
          tensor X_sram, tensor V_sram, tensor st_x, tensor st_seq,
          tensor conv_in, tensor tree_conv_idx)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int N = get_dim_size(qkv_raw, 1);

    // Preload tree_conv_idx [4, N] into SLM (at most 64 * 4 = 256 ints = 1 KiB -> load per node in loop or SLM)
    // Note: Gaudi2 SLM is 1 KiB, so read 4 scalar ints directly from tree_conv_idx (L0-cached) per node.
    for (int idx = start[0]; idx < end[0]; ++idx) {
        const int c = idx * 128;
        int5 wc = {c, 0, 0, 0, 0};
        const float64 sw1 = v_f32_ld_tnsr_b(wc, swb);
        wc[0] = c + 64;
        const float64 sw2 = v_f32_ld_tnsr_b(wc, swb);
        wc[0] = c;
        const float64_pair_t w0 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(wc, conv_w));
        wc[1] = 1;
        const float64_pair_t w1 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(wc, conv_w));
        wc[1] = 2;
        const float64_pair_t w2 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(wc, conv_w));
        wc[1] = 3;
        const float64_pair_t w3 = bf16_to_f32_linear(v_bf16_ld_tnsr_b(wc, conv_w));

        int5 hc = {c, 0, 0, 0, 0};
        const bfloat128 h0 = v_bf16_ld_tnsr_b(hc, conv_in);
        hc[1] = 1;
        const bfloat128 h1 = v_bf16_ld_tnsr_b(hc, conv_in);
        hc[1] = 2;
        const bfloat128 h2 = v_bf16_ld_tnsr_b(hc, conv_in);
        seq_vlm[0] = h0;
        seq_vlm[1] = h1;
        seq_vlm[2] = h2;
        int5 sqc = {c, 0, 0, 0, 0};
        v_bf16_st_tnsr(sqc, st_seq, h0);
        sqc[1] = 1;
        v_bf16_st_tnsr(sqc, st_seq, h1);
        sqc[1] = 2;
        v_bf16_st_tnsr(sqc, st_seq, h2);

        // Pass 1: dequantize qkv_raw[0..N-1] -> VLM seq_vlm[3..N+2] + HBM st_seq[3..N+2]
        #pragma loop_unroll(2)
        for (int t = 0; t < N; ++t) {
            int5 rc = {c, t, 0, 0, 0};
            int5 ac = {0, t, 0, 0, 0};
            const bfloat128 raw_v = v_bf16_ld_tnsr_b(rc, qkv_raw);
            const float64 a_now = v_broadcast_element_f32(v_f32_ld_tnsr_b(ac, sa), 0);
            float64_pair_t dq = v_convert_bf16_to_f32_all_b(raw_v);
            dq.v1 = dq.v1 * (sw1 * a_now);
            dq.v2 = dq.v2 * (sw2 * a_now);
            const bfloat128 dq_bf = v_convert_f32_to_bf16_all_b(dq);
            seq_vlm[3 + t] = dq_bf;
            sqc[1] = 3 + t;
            v_bf16_st_tnsr(sqc, st_seq, dq_bf);
        }

        // Pass 2: tree-ancestor causal conv1d + SiLU -> X_sram/st_x or V_sram
        const int is_qk = (c < 2 * QK);
        const int is_k = (c >= QK);
        const int head = (c - (is_k ? QK : 0)) >> 7;
        const int row_base = is_k ? 0 : N;
        const int v_col = c - 2 * QK;

        for (int t = 0; t < N; ++t) {
            int5 ic0 = {0, t, 0, 0, 0}, ic1 = {1, t, 0, 0, 0}, ic2 = {2, t, 0, 0, 0}, ic3 = {3, t, 0, 0, 0};
            const int i0 = s_i32_ld_g((__global__ int*)gen_addr(ic0, tree_conv_idx));
            const int i1 = s_i32_ld_g((__global__ int*)gen_addr(ic1, tree_conv_idx));
            const int i2 = s_i32_ld_g((__global__ int*)gen_addr(ic2, tree_conv_idx));
            const int i3 = s_i32_ld_g((__global__ int*)gen_addr(ic3, tree_conv_idx));
            const float64_pair_t s0 = bf16_to_f32_linear(seq_vlm[i0]);
            const float64_pair_t s1 = bf16_to_f32_linear(seq_vlm[i1]);
            const float64_pair_t s2 = bf16_to_f32_linear(seq_vlm[i2]);
            const float64_pair_t s3 = bf16_to_f32_linear(seq_vlm[i3]);
            float64_pair_t o;
            o.v1 = v_f32_mac_b(s3.v1, w3.v1, v_f32_mac_b(s2.v1, w2.v1, v_f32_mac_b(s1.v1, w1.v1, s0.v1 * w0.v1)));
            o.v2 = v_f32_mac_b(s3.v2, w3.v2, v_f32_mac_b(s2.v2, w2.v2, v_f32_mac_b(s1.v2, w1.v2, s0.v2 * w0.v2)));
            o.v1 = fast_silu_f32(o.v1);
            o.v2 = fast_silu_f32(o.v2);
            if (is_qk) {
                int5 xc = {0, row_base + t, head, 0, 0};
                v_f32_st_tnsr(xc, X_sram, o.v1);
                v_f32_st_tnsr(xc, st_x, o.v1);
                xc[0] = 64;
                v_f32_st_tnsr(xc, X_sram, o.v2);
                v_f32_st_tnsr(xc, st_x, o.v2);
            } else {
                int5 vc = {v_col, t, 0, 0, 0};
                v_f32_st_tnsr(vc, V_sram, o.v1);
                vc[0] = v_col + 64;
                v_f32_st_tnsr(vc, V_sram, o.v2);
            }
        }
    }
}
