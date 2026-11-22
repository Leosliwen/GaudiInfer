// SPDX-License-Identifier: Apache-2.0
// g2e Direct SCAL Tree KV commit: gathers (a+1) winning rows per KV head from kv_scratch into 4 attention layers'
// k_cache/v_cache at `pos` (works for both FP8 [D_bf16=128] and BF16 [D_bf16=256] via BF16 bit-copy!).
// Tensors (11 <= 16):
//   0  kv_scratch BF16 [D_bf16, N, 4*2*4]  (slice of 4 attention layers: layer_local * 8 + kv * 4 + h)
//   1  path_ctl   I32  [17]                {a, win_nodes[0..a]}
//   2  pos_buf    I32  [1]                 destination start row `pos` in KV cache
//   3..6  k0..k3  BF16 [D_bf16, cap, 4]    4 layers' K caches (strided dim 2 = cap * D_bf16)
//   7..10 v0..v3  BF16 [D_bf16, cap, 4]    4 layers' V caches (strided dim 2 = cap * D_bf16)
// Index space: dim 0 = item in [0..32) (item = layer_local * 8 + kv * 4 + h), split across 24 TPCs.
void main(tensor kv_scratch, tensor path_ctl, tensor pos_buf,
          tensor k0, tensor k1, tensor k2, tensor k3,
          tensor v0, tensor v1, tensor v2, tensor v3)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int d_bf16 = get_dim_size(kv_scratch, 0);  // 128 (FP8) or 256 (BF16)
    int5 c0 = {0, 0, 0, 0, 0};
    const int a = s_i32_ld_g((__global__ int*)gen_addr(c0, path_ctl));
    const int pos = s_i32_ld_g((__global__ int*)gen_addr(c0, pos_buf));
    const int n_win = a + 1;
    int win[16];
    for (int t = 0; t < n_win; ++t) {
        int5 wc = {t + 1, 0, 0, 0, 0};
        win[t] = s_i32_ld_g((__global__ int*)gen_addr(wc, path_ctl));
    }

    for (int item = start[0]; item < end[0]; ++item) {
        const int lay = item >> 3;
        const int kv = (item >> 2) & 1;
        const int h = item & 3;
        for (int t = 0; t < n_win; ++t) {
            const int src_row = win[t];
            const int dst_row = pos + t;
            for (int col = 0; col < d_bf16; col += 128) {
                int5 sc = {col, src_row, item, 0, 0};
                const bfloat128 x = v_bf16_ld_tnsr_b(sc, kv_scratch);
                int5 dc = {col, dst_row, h, 0, 0};
                if (kv == 0) {
                    if (lay == 0) v_bf16_st_tnsr(dc, k0, x);
                    else if (lay == 1) v_bf16_st_tnsr(dc, k1, x);
                    else if (lay == 2) v_bf16_st_tnsr(dc, k2, x);
                    else v_bf16_st_tnsr(dc, k3, x);
                } else {
                    if (lay == 0) v_bf16_st_tnsr(dc, v0, x);
                    else if (lay == 1) v_bf16_st_tnsr(dc, v1, x);
                    else if (lay == 2) v_bf16_st_tnsr(dc, v2, x);
                    else v_bf16_st_tnsr(dc, v3, x);
                }
            }
        }
    }
}
