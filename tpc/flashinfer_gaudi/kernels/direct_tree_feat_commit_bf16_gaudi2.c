// SPDX-License-Identifier: Apache-2.0
// g2e Direct SCAL Tree drafter feature commit: gathers (a+1) winning rows [H] for each captured layer
// from tree_feats [H, N, K] -> feats [H, 16, K].
// Index space: dim 0 = k * 16 + t (k in [0..K), t in [0..16)), split across 24 TPCs.
void main(tensor tree_feats, tensor path_ctl, tensor feats_out)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int H = get_dim_size(tree_feats, 0);
    int5 c0 = {0, 0, 0, 0, 0};
    const int a = s_i32_ld_g((__global__ int*)gen_addr(c0, path_ctl));
    const int n_win = a + 1;

    for (int idx = start[0]; idx < end[0]; ++idx) {
        const int k = idx >> 4;
        const int t = idx & 15;
        if (t >= n_win) continue;
        int5 wc = {t + 1, 0, 0, 0, 0};
        const int src_row = s_i32_ld_g((__global__ int*)gen_addr(wc, path_ctl));
        for (int col = 0; col < H; col += 128) {
            int5 sc = {col, src_row, k, 0, 0};
            int5 dc = {col, t, k, 0, 0};
            v_bf16_st_tnsr(dc, feats_out, v_bf16_ld_tnsr_b(sc, tree_feats));
        }
    }
}
