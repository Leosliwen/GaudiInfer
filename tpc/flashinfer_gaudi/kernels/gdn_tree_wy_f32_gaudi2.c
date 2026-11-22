// SPDX-License-Identifier: Apache-2.0
// g2e: Fast Exact Tree-WY GDN verification core for arbitrary tree topology (N <= 64, H=16, HV=48, K=V=128).
// Uses VLM pre-broadcast g_bcast[N] + direct 1-element SRAM gram loads -> ZERO GET_LANE in inner ancestor loops!
// Tensors (dim 0 fastest):
//   0 log_decay F32 [48, N]
//   1 beta      F32 [48, N]
//   2 rmat      F32 [384, 2N, 16]
//   3 gram      F32 [2N, 2N, 16]
//   4 vals      F32 [128, 48, N]
//   5 parent    I32 [N]            (host-uploaded: parent[0] = -1, 0 <= parent[t] < t)
//   6 output    F32 [128, 48, N]
//   7 ds_out    F32 [128, 48, N]
// Index space: [48] (dim 0 = value head -> 2 heads/TPC across 24 TPCs).

#define MAX_NODES 64

#define XOR8(v) v_f32_mov_group_b((v), 0xFFFFFFFF, SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN |           \
                                  SW_DUAL_GROUP1_EN | SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN, (v), 1, 0)
#define XOR16(v) v_f32_mov_dual_group_all_b((v), 0xFFFFFFFF, 1, 0, 3, 2, MkWrA(0b11, 0b11, 0b11, 0b11), (v))
#define XOR32(v) v_f32_mov_dual_group_all_b((v), 0xFFFFFFFF, 2, 3, 0, 1, MkWrA(0b11, 0b11, 0b11, 0b11), (v))
#define PRED(cond) v_u32_cmp_less_b(lanes, (uint64)((cond) ? 64 : 0))

#define GET_LANE(v, idx)                                                                                      \
    ({                                                                                                        \
        float64 _v = (v);                                                                                     \
        const int _i = (idx);                                                                                 \
        _v = v_f32_mov_vb(XOR32(_v), 0, _v, PRED(_i & 32));                                                   \
        _v = v_f32_mov_vb(XOR16(_v), 0, _v, PRED(_i & 16));                                                   \
        _v = v_f32_mov_vb(XOR8(_v), 0, _v, PRED(_i & 8));                                                     \
        _v = v_f32_mov_vb(v_f32_shuffle_b(_v, lut4, 0, _v), 0, _v, PRED(_i & 4));                             \
        _v = v_f32_mov_vb(v_f32_shuffle_b(_v, lut2, 0, _v), 0, _v, PRED(_i & 2));                             \
        _v = v_f32_mov_vb(v_f32_shuffle_b(_v, lut1, 0, _v), 0, _v, PRED(_i & 1));                             \
        v_broadcast_element_f32(_v, 0);                                                                       \
    })

static inline float64 rsqrt_vec_f32(float64 x)
{
    float64 y = as_float64((int64)0x5f3759df - (as_int64(x) >> 1));
    const float64 h = 0.5f * x;
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
    return y;
}

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

// 64 KiB VLM (fits cleanly in 80 KiB Gaudi2 VLM): ds_lo, ds_hi, g_bcast, G_bcast
__local__ float64 ds_lo_vlm[MAX_NODES];
__local__ float64 ds_hi_vlm[MAX_NODES];
__local__ float64 g_bcast[MAX_NODES];
__local__ float64 G_bcast[MAX_NODES];

void main(tensor log_decay, tensor beta, tensor rmat, tensor gram, tensor vals, tensor parent,
          tensor output, tensor ds_out)
{
    const uint64 lanes = read_lane_id_4b_b();
    const uint64 lanes7 = lanes & 7;
    const uint64 c4 = ((lanes7 ^ 4) | 0x80) * 0x01010101;
    const uint64 c2 = ((lanes7 ^ 2) | 0x80) * 0x01010101;
    const uint64 c1 = ((lanes7 ^ 1) | 0x80) * 0x01010101;
    const uchar256 lut4 = *((const uchar256*)&c4);
    const uchar256 lut2 = *((const uchar256*)&c2);
    const uchar256 lut1 = *((const uchar256*)&c1);

    const float64 zero = 0.0f;
    const float64 one = 1.0f;
    const float q_scale = 0.08838834764831845f;

    const int5 begin = get_index_space_offset();
    const int5 end = get_index_space_size() + begin;
    const int N = get_dim_size(vals, 2);

    int par[MAX_NODES];
    for (int i = 0; i < N; ++i) {
        int5 pc = {i, 0, 0, 0, 0};
        par[i] = s_i32_ld_g((__global__ int*)gen_addr(pc, parent));
    }

    for (int head = begin[0]; head < end[0]; ++head) {
        const int key_head = head / 3;
        const int rbase = (head % 3) * 128;

        // 1. Gather per-node scalars into 64-lane vectors (1-element SRAM loads)
        float64 kdiag = zero, qdiag = zero, ldv = zero, betav = zero;
        for (int t = 0; t < N; ++t) {
            int5 kc = {t, t, key_head, 0, 0};
            const float64 kd = v_broadcast_element_f32(v_f32_ld_tnsr_b(kc, gram), 0);
            int5 qc = {N + t, N + t, key_head, 0, 0};
            const float64 qd = v_broadcast_element_f32(v_f32_ld_tnsr_b(qc, gram), 0);
            int5 sc = {head, t, 0, 0, 0};
            const float64 ld = v_broadcast_element_f32(v_f32_ld_tnsr_b(sc, log_decay), 0);
            const float64 bt = v_broadcast_element_f32(v_f32_ld_tnsr_b(sc, beta), 0);
            const bool64 here = v_u32_cmp_eq_b(lanes, (uint64)t);
            kdiag = v_f32_mov_vb(kd, 0, kdiag, here);
            qdiag = v_f32_mov_vb(qd, 0, qdiag, here);
            ldv = v_f32_mov_vb(ld, 0, ldv, here);
            betav = v_f32_mov_vb(bt, 0, betav, here);
        }
        const float64 rkv = rsqrt_vec_f32(kdiag + 1e-6f);
        const float64 rqv = rsqrt_vec_f32(qdiag + 1e-6f) * q_scale;
        const float64 gv = exp_vec_f32(ldv);

        // 2. Pre-broadcast g[t] and cumulative G[t] into VLM (eliminates ALL GET_LANE from inner loops!)
        for (int t = 0; t < N; ++t) {
            const float64 gt = GET_LANE(gv, t);
            g_bcast[t] = gt;
            const int p = par[t];
            G_bcast[t] = (p < 0) ? gt : (G_bcast[p] * gt);
        }

        // 3. Exact Tree-WY forward pass walking only ancestors per node (ZERO GET_LANE inside ancestor loops!)
        for (int t = 0; t < N; ++t) {
            int5 vc0 = {0, head, t, 0, 0}, vc1 = {64, head, t, 0, 0};
            const float64 v0 = v_f32_ld_tnsr_b(vc0, vals);
            const float64 v1 = v_f32_ld_tnsr_b(vc1, vals);
            int5 rkc0 = {rbase, t, key_head, 0, 0}, rkc1 = {rbase + 64, t, key_head, 0, 0};
            const float64 rk0 = v_f32_ld_tnsr_b(rkc0, rmat);
            const float64 rk1 = v_f32_ld_tnsr_b(rkc1, rmat);
            int5 rqc0 = {rbase, N + t, key_head, 0, 0}, rqc1 = {rbase + 64, N + t, key_head, 0, 0};
            const float64 rq0 = v_f32_ld_tnsr_b(rqc0, rmat);
            const float64 rq1 = v_f32_ld_tnsr_b(rqc1, rmat);

            const float64 rk_t = GET_LANE(rkv, t);
            const float64 rq_t = GET_LANE(rqv, t);
            const float64 beta_t = GET_LANE(betav, t);
            const float64 g_t = g_bcast[t];
            const float64 G_t = G_bcast[t];
            const float64 neg_b_rk2 = -(beta_t * rk_t * rk_t);

            float64 ds0 = v_f32_mac_b(rk0, neg_b_rk2 * G_t, v0 * (rk_t * beta_t));
            float64 ds1 = v_f32_mac_b(rk1, neg_b_rk2 * G_t, v1 * (rk_t * beta_t));

            float64 ratio = g_t;
            for (int j = par[t]; j >= 0; j = par[j]) {
                int5 gk = {j, t, key_head, 0, 0};
                const float64 kj = v_broadcast_element_f32(v_f32_ld_tnsr_b(gk, gram), 0);
                const float64 coef = (ratio * kj) * neg_b_rk2;
                ds0 = v_f32_mac_b(ds_lo_vlm[j], coef, ds0);
                ds1 = v_f32_mac_b(ds_hi_vlm[j], coef, ds1);
                ratio = ratio * g_bcast[j];
            }
            ds_lo_vlm[t] = ds0;
            ds_hi_vlm[t] = ds1;

            int5 dc0 = {0, head, t, 0, 0}, dc1 = {64, head, t, 0, 0};
            v_f32_st_tnsr(dc0, ds_out, ds0);
            v_f32_st_tnsr(dc1, ds_out, ds1);

            float64 o0 = rq0 * (G_t * rq_t);
            float64 o1 = rq1 * (G_t * rq_t);
            ratio = one;
            for (int j = t; j >= 0; j = par[j]) {
                int5 gq = {j, N + t, key_head, 0, 0};
                const float64 qj = v_broadcast_element_f32(v_f32_ld_tnsr_b(gq, gram), 0);
                const float64 coef = (ratio * qj) * rq_t;
                o0 = v_f32_mac_b(ds_lo_vlm[j], coef, o0);
                o1 = v_f32_mac_b(ds_hi_vlm[j], coef, o1);
                ratio = ratio * g_bcast[j];
            }
            int5 oc0 = {0, head, t, 0, 0}, oc1 = {64, head, t, 0, 0};
            v_f32_st_tnsr(oc0, output, o0);
            v_f32_st_tnsr(oc1, output, o1);
        }
    }
}
