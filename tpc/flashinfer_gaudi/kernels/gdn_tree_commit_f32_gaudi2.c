// SPDX-License-Identifier: Apache-2.0
// g2e: Ultra-fast fused GDN tree-speculative state commit along winning path path[0..a] (a < 16):
//   S_out[l, h] = (prod_{t=0..a} exp(g[l, path[t], h])) * S_in[l, h]
//               + sum_{t=0..a} ((prod_{i=t+1..a} exp(g[l, path[i], h])) * ds[l, path[t], h]) * k[l, h/3, path[t]]^T
// Tensors (dim 0 fastest):
//   0 ssm_in    F32 [128, 128, NG*48]
//   1 x_in      F32 [128, 2*N, 16, NG]  (rows 0..N-1 are raw k_t)
//   2 ds_in     F32 [128, 48, N, NG]    (N <= 64)
//   3 log_decay F32 [48, N, NG]
//   4 path_ctl  I32 [17]                (path_ctl[0] = a in [0..15], path_ctl[1+t] = node idx in [0..N))
//   5 ssm_out   F32 [128, 128, NG*48]   (may alias ssm_in)
// Index space: [NG * 48].
#define ROWS 8
#define MAX_A 16

__local__ float64 dsw_lo_buf[MAX_A];
__local__ float64 dsw_hi_buf[MAX_A];
__local__ float64 k_lo_buf[MAX_A];
__local__ float64 k_hi_buf[MAX_A];
__local__ float64 ratio_all_buf[MAX_A];

static inline float64 exp_vec_f32(float64 x)
{
    const float64 xc = v_f32_max_b(x, -87.0f);
    const float64 t = xc * 1.4426950408889634f;
    const int64 n = v_convert_f32_to_i32_b(t, SW_RHNE);
    const float64 f = t - v_convert_i32_to_f32_b(n, 0);
    float64 p = 0.0096181291f;
    p = v_f32_mac_b(p, f, 0.0555041087f);
    p = v_f32_mac_b(p, f, 0.2402265070f);
    p = v_f32_mac_b(p, f, 0.6931471806f);
    p = v_f32_mac_b(p, f, 1.0f);
    const int64 bits = (n + 127) << 23;
    const float64 r = p * (*((float64*)&bits));
    return v_f32_sel_leq_f32_b(x, -87.0f, 0.0f, r);
}

static inline uchar256 make_shuf_const(unsigned char idx)
{
    const uchar256 v = (uchar256)idx;
    return v;
}

#define RUN_SUB(PART, SUB, BUF)                                                                     \
    do {                                                                                            \
        const int r0 = ((PART) * 8 + (SUB)) * ROWS;                                                 \
        const int dg = (SUB) >> 1;                                                                  \
        const int is_g1 = (SUB) & 1;                                                                \
        const uchar256 sh0 = is_g1 ? sh1_0 : sh0_0;                                                 \
        const uchar256 sh1 = is_g1 ? sh1_1 : sh0_1;                                                 \
        const uchar256 sh2 = is_g1 ? sh1_2 : sh0_2;                                                 \
        const uchar256 sh3 = is_g1 ? sh1_3 : sh0_3;                                                 \
        const uchar256 sh4 = is_g1 ? sh1_4 : sh0_4;                                                 \
        const uchar256 sh5 = is_g1 ? sh1_5 : sh0_5;                                                 \
        const uchar256 sh6 = is_g1 ? sh1_6 : sh0_6;                                                 \
        const uchar256 sh7 = is_g1 ? sh1_7 : sh0_7;                                                 \
        float64 state_lo[ROWS];                                                                     \
        float64 state_hi[ROWS];                                                                     \
        _Pragma("unroll")                                                                           \
        for (int row = 0; row < ROWS; ++row) {                                                      \
            int5 sc = {0, r0 + row, h_idx, 0, 0};                                                   \
            state_lo[row] = v_f32_ld_tnsr_b(sc, ssm_in) * Gcur;                                     \
            sc[0] = 64;                                                                             \
            state_hi[row] = v_f32_ld_tnsr_b(sc, ssm_in) * Gcur;                                     \
        }                                                                                           \
        for (int t = 0; t <= a; ++t) {                                                              \
            const float64 dv = (BUF)[t];                                                            \
            const float64 k_lo = k_lo_buf[t];                                                       \
            const float64 k_hi = k_hi_buf[t];                                                       \
            const float64 rep = v_f32_mov_dual_group_all_b(dv, 0xFFFFFFFF, dg, dg, dg, dg, all_dg, dv); \
            const float64 d0 = v_f32_shuffle_b(rep, sh0, 0, rep);                                   \
            const float64 d1 = v_f32_shuffle_b(rep, sh1, 0, rep);                                   \
            const float64 d2 = v_f32_shuffle_b(rep, sh2, 0, rep);                                   \
            const float64 d3 = v_f32_shuffle_b(rep, sh3, 0, rep);                                   \
            const float64 d4 = v_f32_shuffle_b(rep, sh4, 0, rep);                                   \
            const float64 d5 = v_f32_shuffle_b(rep, sh5, 0, rep);                                   \
            const float64 d6 = v_f32_shuffle_b(rep, sh6, 0, rep);                                   \
            const float64 d7 = v_f32_shuffle_b(rep, sh7, 0, rep);                                   \
            state_lo[0] = v_f32_mac_b(k_lo, d0, state_lo[0]);                                       \
            state_hi[0] = v_f32_mac_b(k_hi, d0, state_hi[0]);                                       \
            state_lo[1] = v_f32_mac_b(k_lo, d1, state_lo[1]);                                       \
            state_hi[1] = v_f32_mac_b(k_hi, d1, state_hi[1]);                                       \
            state_lo[2] = v_f32_mac_b(k_lo, d2, state_lo[2]);                                       \
            state_hi[2] = v_f32_mac_b(k_hi, d2, state_hi[2]);                                       \
            state_lo[3] = v_f32_mac_b(k_lo, d3, state_lo[3]);                                       \
            state_hi[3] = v_f32_mac_b(k_hi, d3, state_hi[3]);                                       \
            state_lo[4] = v_f32_mac_b(k_lo, d4, state_lo[4]);                                       \
            state_hi[4] = v_f32_mac_b(k_hi, d4, state_hi[4]);                                       \
            state_lo[5] = v_f32_mac_b(k_lo, d5, state_lo[5]);                                       \
            state_hi[5] = v_f32_mac_b(k_hi, d5, state_hi[5]);                                       \
            state_lo[6] = v_f32_mac_b(k_lo, d6, state_lo[6]);                                       \
            state_hi[6] = v_f32_mac_b(k_hi, d6, state_hi[6]);                                       \
            state_lo[7] = v_f32_mac_b(k_lo, d7, state_lo[7]);                                       \
            state_hi[7] = v_f32_mac_b(k_hi, d7, state_hi[7]);                                       \
        }                                                                                           \
        _Pragma("unroll")                                                                           \
        for (int row = 0; row < ROWS; ++row) {                                                      \
            int5 sc = {0, r0 + row, h_idx, 0, 0};                                                   \
            v_f32_st_tnsr(sc, ssm_out, state_lo[row]);                                              \
            sc[0] = 64;                                                                             \
            v_f32_st_tnsr(sc, ssm_out, state_hi[row]);                                              \
        }                                                                                           \
    } while (0)

void main(tensor ssm_in, tensor x_in, tensor ds_in, tensor log_decay, tensor path_ctl, tensor ssm_out)
{
    const float64 zero = 0.0f;
    const int all_dg = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                       SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    const uchar256 sh0_0 = make_shuf_const(0x80);
    const uchar256 sh0_1 = make_shuf_const(0x81);
    const uchar256 sh0_2 = make_shuf_const(0x82);
    const uchar256 sh0_3 = make_shuf_const(0x83);
    const uchar256 sh0_4 = make_shuf_const(0x84);
    const uchar256 sh0_5 = make_shuf_const(0x85);
    const uchar256 sh0_6 = make_shuf_const(0x86);
    const uchar256 sh0_7 = make_shuf_const(0x87);
    const uchar256 sh1_0 = make_shuf_const(0xa0);
    const uchar256 sh1_1 = make_shuf_const(0xa1);
    const uchar256 sh1_2 = make_shuf_const(0xa2);
    const uchar256 sh1_3 = make_shuf_const(0xa3);
    const uchar256 sh1_4 = make_shuf_const(0xa4);
    const uchar256 sh1_5 = make_shuf_const(0xa5);
    const uchar256 sh1_6 = make_shuf_const(0xa6);
    const uchar256 sh1_7 = make_shuf_const(0xa7);

    const int5 begin = get_index_space_offset();
    const int5 end = get_index_space_size() + begin;
    int5 z0 = {0, 0, 0, 0, 0};
    int a = s_i32_ld_g((__global__ int*)gen_addr(z0, path_ctl));
    if (a < 0) a = 0;
    if (a >= MAX_A) a = MAX_A - 1;
    int path[MAX_A];
    for (int t = 0; t <= a; ++t) {
        int5 pc = {1 + t, 0, 0, 0, 0};
        path[t] = s_i32_ld_g((__global__ int*)gen_addr(pc, path_ctl));
    }

    int prev_layer = -1;
    int prev_kh = -1;
    float64 G_all = zero + 1.0f;

    for (int h_idx = begin[0]; h_idx < end[0]; ++h_idx) {
        const int layer = h_idx / 48;
        const int head = h_idx - layer * 48;
        const int kh = head / 3;

        if (layer != prev_layer) {
            float64 g_vec[MAX_A];
            for (int t = 0; t <= a; ++t) {
                int5 gc = {0, path[t], layer, 0, 0};
                g_vec[t] = exp_vec_f32(v_f32_ld_tnsr_b(gc, log_decay));
            }
            float64 suf = zero + 1.0f;
            for (int t = a; t >= 0; --t) {
                ratio_all_buf[t] = suf;
                suf = suf * g_vec[t];
            }
            G_all = suf;
            prev_layer = layer;
            prev_kh = -1;
        }

        if (kh != prev_kh) {
            for (int t = 0; t <= a; ++t) {
                int5 kc = {0, path[t], kh, layer, 0};
                k_lo_buf[t] = v_f32_ld_tnsr_b(kc, x_in);
                kc[0] = 64;
                k_hi_buf[t] = v_f32_ld_tnsr_b(kc, x_in);
            }
            prev_kh = kh;
        }

        const float64 Gcur = v_broadcast_element_f32(G_all, head);
        for (int t = 0; t <= a; ++t) {
            const float64 rw = v_broadcast_element_f32(ratio_all_buf[t], head);
            int5 dc = {0, head, path[t], layer, 0};
            dsw_lo_buf[t] = v_f32_ld_tnsr_b(dc, ds_in) * rw;
            dc[0] = 64;
            dsw_hi_buf[t] = v_f32_ld_tnsr_b(dc, ds_in) * rw;
        }

        RUN_SUB(0, 0, dsw_lo_buf);
        RUN_SUB(0, 1, dsw_lo_buf);
        RUN_SUB(0, 2, dsw_lo_buf);
        RUN_SUB(0, 3, dsw_lo_buf);
        RUN_SUB(0, 4, dsw_lo_buf);
        RUN_SUB(0, 5, dsw_lo_buf);
        RUN_SUB(0, 6, dsw_lo_buf);
        RUN_SUB(0, 7, dsw_lo_buf);
        RUN_SUB(1, 0, dsw_hi_buf);
        RUN_SUB(1, 1, dsw_hi_buf);
        RUN_SUB(1, 2, dsw_hi_buf);
        RUN_SUB(1, 3, dsw_hi_buf);
        RUN_SUB(1, 4, dsw_hi_buf);
        RUN_SUB(1, 5, dsw_hi_buf);
        RUN_SUB(1, 6, dsw_hi_buf);
        RUN_SUB(1, 7, dsw_hi_buf);
    }
}
