// Direct Gaudi2 MME + 24-TPC + 48 MiB SRAM engine for Tree Speculative Verification (N=32 or N=48).
#pragma once

#include "g2e/direct_gdn.hpp"
#include "g2e/direct_mme.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace g2e {

// Additional persistent SRAM buffers inside the existing 6 MiB persistent zone (< kDirectSramWorkOff = 0x600000):
//   [0x000000..0x0a0000) Resid      BF16 [5120, N]
//   [0x140000..0x200000) Gate       BF16 [6144, N]
//   [0x2c0000..0x380000) Q_prefix   F8/BF16 [256, 6*N, 4]
//   [0x380000..0x440000) Q_bf       BF16 [256, 6*N, 4]
//   [0x440000..0x500000) Att        BF16 [6144, N]
//   [0x520000..0x540000) K_tree_bf  BF16 [256, N, 4]
//   [0x540000..0x580000) V_tree_f32 F32  [256, N, 4]
constexpr uint64_t kDirectTreeSramQBfOff = 0x380000ull;
constexpr uint64_t kDirectTreeSramKTOff  = 0x520000ull;
constexpr uint64_t kDirectTreeSramVTOff  = 0x540000ull;

struct DirectTreeAssets {
    uint32_t N = 48;
    // GDN + MLP + AttnQKV MME templates for R = N
    MmeTemplate qkv, z, ba, gram, bwr, out, guh, dn, attn_qkv;
    // Attention QK / PV MME templates for RHead = 6 * N
    MmeTemplate qk_f8_1024, pv_f8_1024;
    MmeTemplate qk_f8_2048, pv_f8_2048;
    MmeTemplate qk_f8_4096, pv_f8_4096;
    MmeTemplate qk_bf16_1024, pv_bf16_1024;
    MmeTemplate qk_bf16_2048, pv_bf16_2048;
    MmeTemplate qk_bf16_4096, pv_bf16_4096;
    // TPC ISAs
    uint64_t isa_embed_nq = 0;
    uint64_t isa_anq_dual = 0;
    uint64_t isa_anq = 0;
    uint64_t isa_finalize = 0;
    uint64_t isa_tree_prep = 0;
    uint64_t isa_ba_siluz = 0;
    uint64_t isa_tree_wy = 0;
    uint64_t isa_norm_mq = 0;
    uint64_t isa_dq_silu = 0;
    uint64_t isa_dq_mq = 0;
    uint64_t isa_sig_mq = 0;
    uint64_t isa_apre_tree_bf16 = 0;
    uint64_t isa_apre_tree_f8 = 0;
    uint64_t isa_tile_step_bf16 = 0;
    uint64_t isa_tile_step_f8 = 0;
    uint64_t isa_tree_state_commit = 0;
    uint64_t isa_conv_commit = 0;
    uint64_t isa_kv_commit = 0;
    uint64_t isa_feat_commit = 0;

    static DirectTreeAssets load(DirectRuntime& rt, const std::string& desc_dir, const std::string& tpc_obj_dir,
                                 uint32_t N);
};

struct DirectTreeApreWeights {
    uint64_t in_norm1p = 0;    // BF16 [H]
    uint64_t qkv_w = 0;        // F8   [14336, H]
    uint64_t qkv_s_int = 0;    // F32  [14336] pre-interleaved
    uint64_t q_norm = 0;       // BF16 [256]
    uint64_t k_norm = 0;       // BF16 [256]
    uint64_t rope_pos = 0;     // I32  [N] HBM (tree_direct_ctl_)
    uint64_t cos_tab = 0;      // F32  [32, tab_rows] HBM
    uint64_t sin_tab = 0;      // F32  [32, tab_rows] HBM
    uint32_t rope_tab_rows = 0;
    uint64_t k_hbm = 0;        // F8/BF16 [256, N, 4] HBM kv_scratch K slice
    uint64_t v_hbm = 0;        // F8/BF16 [256, N, 4] HBM kv_scratch V slice
    bool kv8 = true;
};

struct DirectTreeBlockArgs {
    uint64_t embed_ids = 0;      // optional Span-0 direct embed gather: I32 [N]
    uint64_t embed_table = 0;    // optional Span-0 direct embed gather: BF16 [H, V]
    uint32_t embed_vocab = 0;
    uint64_t tree_parent = 0;    // HBM I32 [N]
    uint64_t tree_conv_idx = 0;  // HBM I32 [4, N]
    uint64_t gate_in = 0;        // SRAM persistent gate (when apost != nullptr)
    uint64_t att_in = 0;         // SRAM persistent att  (when apost != nullptr)
    uint64_t resid_in = 0;       // SRAM persistent resid (when apost != nullptr)
    uint64_t res_out = 0;        // HBM final res_out  (final span only)
    uint64_t delta_out = 0;      // HBM final delta_out (final span only)
    // Per GDN layer j in [0..NL-1]:
    std::vector<uint64_t> conv_in;  // [NL] HBM BF16 [10240, 3]
    std::vector<uint64_t> ssm_in;   // [NL] HBM F32  [128, 384, 16]
    std::vector<uint64_t> st_x;     // [NL] HBM F32  [128, 2*N, 16]
    std::vector<uint64_t> st_v;     // [NL] HBM F32  [128, 48, N]
    std::vector<uint64_t> st_g;     // [NL] HBM F32  [48, N]
    std::vector<uint64_t> st_seq;   // [NL] HBM BF16 [10240, N+3]
};

class DirectTreeBlock {
public:
    DirectTreeBlock(DirectRuntime& rt, const DirectTreeAssets& assets,
                    const std::vector<DirectGdnLayerWeights>& layers,
                    const DirectTreeBlockArgs& init_args,
                    uint32_t tpc_cmd = 0xa,
                    const DirectApostLayerWeights* apost = nullptr,
                    const DirectTreeApreWeights* apre = nullptr,
                    uint64_t sram_offset = kDirectSramWorkOff);
    ~DirectTreeBlock();

    void update_args(const DirectTreeBlockArgs& args);
    double run();
    DirectPlan* plan() { return plan_.get(); }
    uint64_t sram_used_bytes() const { return sram_used_; }

private:
    uint32_t nl_ = 0;
    uint32_t N_ = 48;
    uint64_t sram_used_ = 0;
    std::unique_ptr<DirectPlan> plan_;
    DirectTreeBlockArgs cur_{};
};

struct DirectTreeAttnArgs {
    uint64_t q_prefix = 0;    // SRAM F8/BF16 [256, 6*N, 4]
    uint64_t q_bf = 0;        // SRAM BF16    [256, 6*N, 4]
    uint64_t k_tree_bf = 0;   // SRAM BF16    [256, N, 4]
    uint64_t v_tree_f32 = 0;  // SRAM F32     [256, N, 4]
    uint64_t att_out = 0;     // SRAM BF16    [6144, N]
    uint64_t pos_buf = 0;     // HBM  I32     [1]
    uint64_t parent = 0;      // HBM  I32     [N]
    std::array<uint64_t, 4> k_cache{};  // [kv_head] HBM base addr of [cap, 256]
    std::array<uint64_t, 4> v_cache{};  // [kv_head] HBM base addr of [cap, 256]
};

class DirectTreeAttnCore {
public:
    // `prefix_rows`: multiple of 1024 >= max(pos, 1).
    DirectTreeAttnCore(DirectRuntime& rt, const DirectTreeAssets& assets, bool kv8,
                       uint32_t prefix_rows, const DirectTreeAttnArgs& init_args,
                       uint64_t sram_offset = kDirectSramWorkOff, uint32_t tpc_cmd = 0xa);
    ~DirectTreeAttnCore();

    void update_args(const DirectTreeAttnArgs& args);
    double run();
    void collect_plans(std::vector<DirectPlan*>& out) const;
    uint64_t sram_used_bytes() const { return sram_used_; }

private:
    struct TileDesc {
        uint32_t col_start = 0;
        uint32_t width = 0;
        bool is_first = false;
        bool is_last = false;
        uint32_t plan_idx = 0;
    };

    bool kv8_ = true;
    uint32_t N_ = 48;
    uint64_t sram_used_ = 0;
    std::vector<TileDesc> tiles_;
    std::vector<std::unique_ptr<DirectPlan>> plans_;
    DirectTreeAttnArgs cur_{};
};

struct DirectTreeCommitArgs {
    uint64_t ssm = 0;          // IN-PLACE F32 [NG*48, 128, 128]
    uint64_t conv_out = 0;     // BF16 [NG, 3, 10240]
    uint64_t x = 0;            // F32  [NG, 16, 2*N, 128]
    uint64_t v = 0;            // F32  [NG, N, 48, 128]
    uint64_t g = 0;            // F32  [NG, N, 48]
    uint64_t seq = 0;          // BF16 [NG, N+3, 10240]
    uint64_t path_ctl = 0;     // I32  [17] {a, win_nodes[0..a]}
    uint64_t conv_rows = 0;    // I32  [3]
    uint64_t pos_buf = 0;      // I32  [1]
    uint64_t kv_scratch = 0;   // BF16 [NA*8, N, D_bf16]
    std::vector<uint64_t> k_cache;  // [NA] base addr of each attn layer's K cache
    std::vector<uint64_t> v_cache;  // [NA] base addr of each attn layer's V cache
    uint32_t kv_cap = 0;
    uint64_t feat_in = 0;      // HBM feats_ [cap_n, stride_in]
    uint64_t feat_out = 0;     // HBM tree_feats_ [cap_n, stride_out]
    uint32_t feat_layers = 0;
    uint32_t feat_in_stride_elems = 0;
    uint32_t feat_out_stride_elems = 0;
};

class DirectTreeCommit {
public:
    DirectTreeCommit(DirectRuntime& rt, const DirectTreeAssets& assets, uint32_t ng, uint32_t na, bool kv8,
                     const DirectTreeCommitArgs& init_args, uint32_t tpc_cmd = 0xa);
    ~DirectTreeCommit();

    void update_args(const DirectTreeCommitArgs& args);
    double run();
    DirectPlan* plan() { return plan_.get(); }
    uint32_t built_cap() const { return cap_; }
    uint32_t built_feat_layers() const { return feat_layers_; }

private:
    uint32_t na_ = 0;
    uint32_t cap_ = 0;
    uint32_t feat_layers_ = 0;
    std::unique_ptr<DirectPlan> plan_;
    DirectTreeCommitArgs cur_{};
};

}  // namespace g2e
