#include "g2e/direct_tree.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace g2e {

namespace {

constexpr uint32_t kH = 5120;
constexpr uint32_t kF = 17408;
constexpr uint32_t kHK = 16;
constexpr uint32_t kHV = 48;
constexpr uint32_t kDK = 128;
constexpr uint32_t kDV = 128;
constexpr uint32_t kC = 2 * kHK * kDK + kHV * kDV;  // 10240
constexpr uint32_t kZ = kHV * kDV;                  // 6144
constexpr uint32_t kKV = 4;
constexpr uint32_t kG = 6;
constexpr uint32_t kD = 256;
constexpr uint32_t kNHD = kKV * kG * kD;            // 6144
constexpr uint32_t kTpcs = 24;
constexpr uint32_t kMaxTilesPerPlan = 32;
constexpr float kEps = 1e-6f;

uint64_t align256(uint64_t x) { return (x + 255ull) & ~255ull; }
uint32_t f32_bits(float v) { uint32_t u = 0; std::memcpy(&u, &v, 4); return u; }

enum BlockTags : int {
    kTagResOut = 1,
    kTagDeltaOut = 2,
};
int tag_conv_in(uint32_t j) { return 100 + int(j) * 8 + 0; }
int tag_ssm_in(uint32_t j)  { return 100 + int(j) * 8 + 1; }
int tag_st_x(uint32_t j)    { return 100 + int(j) * 8 + 2; }
int tag_st_v(uint32_t j)    { return 100 + int(j) * 8 + 3; }
int tag_st_g(uint32_t j)    { return 100 + int(j) * 8 + 4; }
int tag_st_seq(uint32_t j)  { return 100 + int(j) * 8 + 5; }

int tag_attn_k(uint32_t tidx, uint32_t h) { return 1000 + int(tidx) * 16 + int(h); }
int tag_attn_v(uint32_t tidx, uint32_t h) { return 1000 + int(tidx) * 16 + 4 + int(h); }

}  // namespace

DirectTreeAssets DirectTreeAssets::load(DirectRuntime& rt, const std::string& desc_dir,
                                        const std::string& tpc_obj_dir, uint32_t N) {
    if (N != 24 && N != 32 && N != 48) throw std::invalid_argument("direct_tree: N must be 24, 32, or 48");
    const std::string rtag = "_R" + std::to_string(N);
    DirectTreeAssets a;
    a.N = N;
    a.qkv = MmeTemplate::load_dir(desc_dir + "/qkv" + rtag + "/mme_debug");
    a.z = MmeTemplate::load_dir(desc_dir + "/z" + rtag + "/mme_debug");
    a.ba = MmeTemplate::load_dir(desc_dir + "/ba" + rtag + "/mme_debug");
    a.gram = MmeTemplate::load_dir(desc_dir + "/gram" + rtag + "/mme_debug");
    a.bwr = MmeTemplate::load_dir(desc_dir + "/bwr" + rtag + "/mme_debug");
    a.out = MmeTemplate::load_dir(desc_dir + "/out" + rtag + "/mme_debug");
    a.guh = MmeTemplate::load_dir(desc_dir + "/guh" + rtag + "/mme_debug");
    a.dn = MmeTemplate::load_dir(desc_dir + "/dn" + rtag + "/mme_debug");
    a.attn_qkv = MmeTemplate::load_dir(desc_dir + "/attn_qkv" + rtag + "/mme_debug");

    a.qk_f8_1024 = MmeTemplate::load_dir(desc_dir + "/attn_qk_f8_1024/mme_debug");
    a.pv_f8_1024 = MmeTemplate::load_dir(desc_dir + "/attn_pv_f8_1024/mme_debug");
    a.qk_f8_2048 = MmeTemplate::load_dir(desc_dir + "/attn_qk_f8_2048/mme_debug");
    a.pv_f8_2048 = MmeTemplate::load_dir(desc_dir + "/attn_pv_f8_2048/mme_debug");
    a.qk_f8_4096 = MmeTemplate::load_dir(desc_dir + "/attn_qk_f8_4096/mme_debug");
    a.pv_f8_4096 = MmeTemplate::load_dir(desc_dir + "/attn_pv_f8_4096/mme_debug");
    a.qk_bf16_1024 = MmeTemplate::load_dir(desc_dir + "/attn_qk_bf16_1024/mme_debug");
    a.pv_bf16_1024 = MmeTemplate::load_dir(desc_dir + "/attn_pv_bf16_1024/mme_debug");
    a.qk_bf16_2048 = MmeTemplate::load_dir(desc_dir + "/attn_qk_bf16_2048/mme_debug");
    a.pv_bf16_2048 = MmeTemplate::load_dir(desc_dir + "/attn_pv_bf16_2048/mme_debug");
    a.qk_bf16_4096 = MmeTemplate::load_dir(desc_dir + "/attn_qk_bf16_4096/mme_debug");
    a.pv_bf16_4096 = MmeTemplate::load_dir(desc_dir + "/attn_pv_bf16_4096/mme_debug");

    a.isa_embed_nq = rt.load_tpc_elf(tpc_obj_dir + "/direct_embed_norm_quant_bf16_gaudi2.o");
    a.isa_anq_dual = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_add_rmsnorm_quant_dual_bf16_gaudi2.o");
    a.isa_anq = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_add_rmsnorm_quant_bf16_gaudi2.o");
    a.isa_finalize = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_finalize_bf16_gaudi2.o");
    a.isa_tree_prep = rt.load_tpc_elf(tpc_obj_dir + "/direct_gdn_dq_tree_prep_bf16_gaudi2.o");
    a.isa_ba_siluz = rt.load_tpc_elf(tpc_obj_dir + "/direct_gdn_ba_siluz_bf16_gaudi2.o");
    a.isa_tree_wy = rt.load_tpc_elf(tpc_obj_dir + "/gdn_tree_wy_f32_gaudi2.o");
    a.isa_norm_mq = rt.load_tpc_elf(tpc_obj_dir + "/direct_gdn_norm_mul_quant_bf16_gaudi2.o");
    a.isa_dq_silu = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_silu_bf16_gaudi2.o");
    a.isa_dq_mq = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_mul_quant_bf16_gaudi2.o");
    a.isa_sig_mq = rt.load_tpc_elf(tpc_obj_dir + "/direct_sigmoid_mul_quant_bf16_gaudi2.o");
    a.isa_apre_tree_bf16 = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_attn_tree_prep_bf16_gaudi2.o");
    a.isa_apre_tree_f8 = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_attn_tree_prep_f8_gaudi2.o");
    a.isa_tile_step_bf16 = rt.load_tpc_elf(tpc_obj_dir + "/direct_attn_tree_tile_step_bf16_gaudi2.o");
    a.isa_tile_step_f8 = rt.load_tpc_elf(tpc_obj_dir + "/direct_attn_tree_tile_step_f8_gaudi2.o");
    a.isa_tree_state_commit = rt.load_tpc_elf(tpc_obj_dir + "/gdn_tree_commit_f32_gaudi2.o");
    a.isa_conv_commit = rt.load_tpc_elf(tpc_obj_dir + "/direct_conv_commit_bf16_gaudi2.o");
    a.isa_kv_commit = rt.load_tpc_elf(tpc_obj_dir + "/direct_tree_kv_commit_bf16_gaudi2.o");
    a.isa_feat_commit = rt.load_tpc_elf(tpc_obj_dir + "/direct_tree_feat_commit_bf16_gaudi2.o");
    return a;
}

// ==============================================================================
// DirectTreeBlock
// ==============================================================================

DirectTreeBlock::DirectTreeBlock(DirectRuntime& rt, const DirectTreeAssets& assets,
                                 const std::vector<DirectGdnLayerWeights>& layers,
                                 const DirectTreeBlockArgs& init_args,
                                 uint32_t tpc_cmd,
                                 const DirectApostLayerWeights* apost,
                                 const DirectTreeApreWeights* apre,
                                 uint64_t sram_offset)
    : nl_(uint32_t(layers.size())), N_(assets.N), plan_(std::make_unique<DirectPlan>(rt, 0)), cur_(init_args) {
    const bool has_apost = (apost != nullptr);
    const bool has_apre = (apre != nullptr);
    if (nl_ == 0 && !has_apost) throw std::invalid_argument("direct_tree: empty block");

    const uint32_t R = N_;
    const uint32_t sig_r = std::min(kTpcs, R);

    uint64_t sp = align256(rt.sram_base() + sram_offset);
    auto alloc = [&](uint64_t bytes) -> uint64_t {
        const uint64_t addr = sp;
        sp = align256(sp + bytes);
        return addr;
    };

    const uint64_t s_res0 = alloc(uint64_t(R) * kH * 2);
    const uint64_t s_res1 = alloc(uint64_t(R) * kH * 2);
    const uint64_t s_xnorm = alloc(uint64_t(R) * kH * 2);
    const uint64_t s_x8 = alloc(uint64_t(R) * kH);
    const uint64_t s_xs = alloc(uint64_t(R) * 4);
    const uint64_t scratch_base = sp;

    // Phase A (Tree-GDN mixer)
    const uint64_t s_qkv = alloc(uint64_t(R) * kC * 2);
    const uint64_t s_z = alloc(uint64_t(R) * kZ * 2);
    const uint64_t s_ba = alloc(uint64_t(R) * (2 * kHV) * 4);
    const uint64_t s_g = alloc(uint64_t(R) * kHV * 4);
    const uint64_t s_beta = alloc(uint64_t(R) * kHV * 4);
    const uint64_t s_X = alloc(uint64_t(kHK) * (2 * R) * kDK * 4);
    const uint64_t s_V = alloc(uint64_t(R) * kZ * 4);
    const uint64_t s_gram = alloc(uint64_t(kHK) * (2 * R) * (2 * R) * 4);
    const uint64_t s_rmat = alloc(uint64_t(kHK) * (2 * R) * (3 * kDV) * 4);
    const uint64_t s_core = alloc(uint64_t(R) * kHV * kDV * 4);
    const uint64_t s_o8 = alloc(uint64_t(R) * kZ);
    const uint64_t s_os = alloc(uint64_t(R) * 4);
    const uint64_t s_out = alloc(uint64_t(R) * kH * 2);
    const uint64_t peak_a = sp;

    // Phase B (MLP, reuses scratch_base)
    sp = scratch_base;
    const uint64_t s_gate = alloc(uint64_t(R) * kF * 2);
    const uint64_t s_up = alloc(uint64_t(R) * kF * 2);
    const uint64_t s_m8 = alloc(uint64_t(R) * kF);
    const uint64_t s_ms = alloc(uint64_t(R) * 4);
    const uint64_t s_dn = alloc(uint64_t(R) * kH * 2);
    const uint64_t peak_b = sp;

    // Phase C (optional DirectTreeApre, reuses scratch_base)
    sp = scratch_base;
    const uint64_t s_attn_qkv = alloc(uint64_t(R) * 14336ull * 2ull);
    const uint64_t peak_c = sp;

    sram_used_ = std::max({peak_a, peak_b, peak_c}) - rt.sram_base();
    if (sram_used_ > rt.sram_size()) throw std::runtime_error("direct_tree: block SRAM overflow");

    const uint32_t anq_srfs[4] = {f32_bits(kEps), f32_bits(1.0f / float(kH)), f32_bits(1.0f / 240.0f), 0};
    const uint32_t eps_srf[1] = {f32_bits(kEps)};

    uint32_t s0 = 0, s1 = 0, s2 = 0;

    auto emit_mlp_and_transition = [&](uint64_t gate_w, uint64_t up_w, uint64_t gate_s_int, uint64_t up_s_int,
                                       uint64_t dn_w, uint64_t dn_s_int, uint64_t feat_out,
                                       bool is_final, uint64_t next_in_norm1p, uint64_t next_res_dst) {
        plan_->add_mme(assets.guh, s_x8, gate_w, s_gate, 0, 2, s2, 0);
        s0 += assets.guh.signal_count;
        plan_->add_mme(assets.guh, s_x8, up_w, s_up, 0, -1, 0, 1);
        s1 += assets.guh.signal_count;

        {
            const DirectTpcTensor t[3] = {
                {s_gate, DType::BF16, 2, {kF, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {gate_s_int, DType::F32, 1, {kF, 1, 1, 1, 1}},
            };
            const uint32_t silu_items = 3u * R;
            plan_->add_tpc_grid(assets.isa_dq_silu, t, {}, {silu_items, 1, 1, 1, 1}, 0, kTpcs, 0, s0, 1, tpc_cmd);
            s1 += std::min(kTpcs, silu_items);
        }
        {
            const DirectTpcTensor t[6] = {
                {s_gate, DType::BF16, 2, {kF, R, 1, 1, 1}},
                {s_up, DType::BF16, 2, {kF, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {up_s_int, DType::F32, 1, {kF, 1, 1, 1, 1}},
                {s_m8, DType::F8, 2, {kF, R, 1, 1, 1}},
                {s_ms, DType::F32, 2, {1, R, 1, 1, 1}},
            };
            plan_->add_tpc(assets.isa_dq_mq, t, {}, R, kTpcs, 1, s1, 2, tpc_cmd);
            s2 += sig_r;
        }

        plan_->add_mme(assets.dn, s_m8, dn_w, s_dn, 0, 2, s2, 0);
        s0 += assets.dn.signal_count;

        if (!is_final) {
            const uint32_t srfs[4] = {anq_srfs[0], anq_srfs[1], anq_srfs[2], feat_out ? 1u : 0u};
            const DirectTpcTensor t[10] = {
                {s_dn, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {s_ms, DType::F32, 2, {1, R, 1, 1, 1}},
                {dn_s_int, DType::F32, 1, {kH, 1, 1, 1, 1}},
                {s_res1, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {next_in_norm1p, DType::BF16, 1, {kH, 1, 1, 1, 1}},
                {s_x8, DType::F8, 2, {kH, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {next_res_dst, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {s_xnorm, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {feat_out ? feat_out : next_res_dst, DType::BF16, 2, {kH, R, 1, 1, 1}},
            };
            plan_->add_tpc(assets.isa_anq_dual, t, srfs, R, kTpcs, 0, s0, 2, tpc_cmd);
            s2 += sig_r;
        } else {
            const uint32_t srfs[1] = {feat_out ? 1u : 0u};
            const DirectTpcTensor t[7] = {
                {s_dn, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {s_ms, DType::F32, 2, {1, R, 1, 1, 1}},
                {dn_s_int, DType::F32, 1, {kH, 1, 1, 1, 1}},
                {s_res1, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {init_args.res_out, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {init_args.delta_out, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {feat_out ? feat_out : init_args.res_out, DType::BF16, 2, {kH, R, 1, 1, 1}},
            };
            const int tags[7] = {-1, -1, -1, -1, kTagResOut, kTagDeltaOut, -1};
            plan_->add_tpc(assets.isa_finalize, t, srfs, R, kTpcs, 0, s0, -1, tpc_cmd, tags);
        }
    };

    if (has_apost) {
        {
            const DirectTpcTensor t[4] = {
                {init_args.gate_in, DType::BF16, 2, {kZ, R, 1, 1, 1}},
                {init_args.att_in, DType::BF16, 2, {kZ, R, 1, 1, 1}},
                {s_o8, DType::F8, 2, {kZ, R, 1, 1, 1}},
                {s_os, DType::F32, 2, {1, R, 1, 1, 1}},
            };
            plan_->add_tpc(assets.isa_sig_mq, t, {}, R, kTpcs, -1, 0, 2, tpc_cmd);
            s2 += sig_r;
        }
        plan_->add_mme(assets.out, s_o8, apost->o_w, s_out, 0, 2, s2, 0);
        s0 += assets.out.signal_count;
        {
            const DirectTpcTensor t[8] = {
                {s_out, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {s_os, DType::F32, 2, {1, R, 1, 1, 1}},
                {apost->o_s_int, DType::F32, 1, {kH, 1, 1, 1, 1}},
                {init_args.resid_in, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {apost->post_norm1p, DType::BF16, 1, {kH, 1, 1, 1, 1}},
                {s_x8, DType::F8, 2, {kH, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {s_res1, DType::BF16, 2, {kH, R, 1, 1, 1}},
            };
            plan_->add_tpc(assets.isa_anq, t, std::span<const uint32_t>(anq_srfs, 3), R, kTpcs, 0, s0, 2, tpc_cmd);
            s2 += sig_r;
        }
        const bool apost_to_apre = (nl_ == 0 && has_apre);
        emit_mlp_and_transition(apost->gate_w, apost->up_w, apost->gate_s_int, apost->up_s_int,
                                apost->dn_w, apost->dn_s_int, apost->feat_out,
                                nl_ == 0 && !has_apre,
                                nl_ ? layers[0].in_norm1p : (apost_to_apre ? apre->in_norm1p : 0),
                                apost_to_apre ? (rt.sram_base() + kDirectSramResidOff) : s_res0);
    } else {
        // Span 0: direct embed gather + L0 RMSNorm + FP8 quant into SRAM
        const DirectTpcTensor t[7] = {
            {init_args.embed_ids, DType::I32, 1, {R, 1, 1, 1, 1}},
            {init_args.embed_table, DType::BF16, 2, {kH, init_args.embed_vocab, 1, 1, 1}},
            {layers[0].in_norm1p, DType::BF16, 1, {kH, 1, 1, 1, 1}},
            {s_x8, DType::F8, 2, {kH, R, 1, 1, 1}},
            {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
            {s_res0, DType::BF16, 2, {kH, R, 1, 1, 1}},
            {s_xnorm, DType::BF16, 2, {kH, R, 1, 1, 1}},
        };
        plan_->add_tpc(assets.isa_embed_nq, t, std::span<const uint32_t>(anq_srfs, 3), R, kTpcs, -1, 0, 2, tpc_cmd);
        s2 += sig_r;
    }

    for (uint32_t j = 0; j < nl_; ++j) {
        const auto& W = layers[j];
        const bool last = (j + 1 == nl_);

        // 1) Coalesced MME(qkv) + MME(z) + MME(ba)
        plan_->add_mme(assets.qkv, s_x8, W.qkv_w, s_qkv, 0, 2, s2, 0);
        s0 += assets.qkv.signal_count;
        plan_->add_mme(assets.z, s_x8, W.z_w, s_z, 0, -1, 0, 1);
        s1 += assets.z.signal_count;
        plan_->add_mme(assets.ba, s_xnorm, W.ba_w, s_ba, 0, -1, 0, 1);
        s1 += assets.ba.signal_count;

        // 2) Concurrent 24-TPC(direct_gdn_dq_tree_prep): QK (32 items -> wakes gram+bwr!) + V (48 items -> overlaps!)
        {
            const DirectTpcTensor t[10] = {
                {s_qkv, DType::BF16, 2, {kC, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {W.qkv_s_int, DType::F32, 1, {kC, 1, 1, 1, 1}},
                {W.conv_t, DType::BF16, 2, {kC, 4, 1, 1, 1}},
                {s_X, DType::F32, 3, {kDK, 2 * R, kHK, 1, 1}},
                {s_V, DType::F32, 2, {kZ, R, 1, 1, 1}},
                {init_args.st_x[j], DType::F32, 3, {kDK, 2 * R, kHK, 1, 1}},
                {init_args.st_seq[j], DType::BF16, 2, {kC, R + 3, 1, 1, 1}},
                {init_args.conv_in[j], DType::BF16, 2, {kC, 3, 1, 1, 1}},
                {init_args.tree_conv_idx, DType::I32, 2, {4, R, 1, 1, 1}},
            };
            const int tags[10] = {-1, -1, -1, -1, -1, -1, tag_st_x(j), tag_st_seq(j), tag_conv_in(j), -1};
            constexpr uint32_t kQkItems = 2 * kHK;               // 32
            constexpr uint32_t kVItems = (kC / 128) - 2 * kHK;   // 48
            plan_->add_tpc_grid(assets.isa_tree_prep, t, {}, {kQkItems, 1, 1, 1, 1}, 0, kTpcs,
                                0, s0, 2, tpc_cmd, tags, 0);
            s2 += std::min(kTpcs, kQkItems);
            plan_->add_tpc_grid(assets.isa_tree_prep, t, {}, {kVItems, 1, 1, 1, 1}, 0, kTpcs,
                                -1, 0, 0, tpc_cmd, tags, kQkItems);
            s0 += std::min(kTpcs, kVItems);
        }

        // 3a) Coalesced MME(gram) + MME(bwr)
        plan_->add_mme(assets.gram, s_X, s_X, s_gram, 0, 2, s2, 0);
        s0 += assets.gram.signal_count;
        plan_->add_mme(assets.bwr, s_X, init_args.ssm_in[j], s_rmat, 0, -1, 0, 0, {-1, tag_ssm_in(j), -1, -1});
        s0 += assets.bwr.signal_count;

        // 3b) Concurrent TPC(direct_gdn_ba_siluz): waits s1 (z + ba), runs in parallel with gram + bwr!
        {
            const DirectTpcTensor t[9] = {
                {s_ba, DType::F32, 2, {2 * kHV, R, 1, 1, 1}},
                {W.dt_bias, DType::BF16, 1, {kHV, 1, 1, 1, 1}},
                {W.negexpA, DType::F32, 1, {kHV, 1, 1, 1, 1}},
                {s_g, DType::F32, 3, {kHV, R, 1, 1, 1}},
                {s_beta, DType::F32, 3, {kHV, R, 1, 1, 1}},
                {init_args.st_g[j], DType::F32, 3, {kHV, R, 1, 1, 1}},
                {s_z, DType::BF16, 2, {kZ, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {W.z_s_int, DType::F32, 1, {kZ, 1, 1, 1, 1}},
            };
            const int tags[9] = {-1, -1, -1, -1, -1, tag_st_g(j), -1, -1, -1};
            plan_->add_tpc(assets.isa_ba_siluz, t, {}, R, kTpcs, 1, s1, 0, tpc_cmd, tags);
            s0 += sig_r;
        }

        // 4) 24-TPC(gdn_tree_wy): waits s0 (gram + bwr + prep_v + ba_siluz), 48 heads -> 2/TPC!
        {
            const DirectTpcTensor t[8] = {
                {s_g, DType::F32, 2, {kHV, R, 1, 1, 1}},
                {s_beta, DType::F32, 2, {kHV, R, 1, 1, 1}},
                {s_rmat, DType::F32, 3, {3 * kDV, 2 * R, kHK, 1, 1}},
                {s_gram, DType::F32, 3, {2 * R, 2 * R, kHK, 1, 1}},
                {s_V, DType::F32, 3, {kDV, kHV, R, 1, 1}},
                {init_args.tree_parent, DType::I32, 1, {R, 1, 1, 1, 1}},
                {s_core, DType::F32, 3, {kDV, kHV, R, 1, 1}},
                {init_args.st_v[j], DType::F32, 3, {kDV, kHV, R, 1, 1}},
            };
            const int tags[8] = {-1, -1, -1, -1, -1, -1, -1, tag_st_v(j)};
            plan_->add_tpc_grid(assets.isa_tree_wy, t, {}, {kHV, 1, 1, 1, 1}, 0, kTpcs,
                                0, s0, 2, tpc_cmd, tags);
            s2 += std::min(kTpcs, kHV);
        }

        // 5) TPC(direct_gdn_norm_mul_quant)
        {
            const DirectTpcTensor t[5] = {
                {s_core, DType::F32, 3, {kDV, kHV, R, 1, 1}},
                {s_z, DType::BF16, 2, {kZ, R, 1, 1, 1}},
                {W.gdn_norm, DType::BF16, 1, {kDV, 1, 1, 1, 1}},
                {s_o8, DType::F8, 2, {kZ, R, 1, 1, 1}},
                {s_os, DType::F32, 2, {1, R, 1, 1, 1}},
            };
            plan_->add_tpc(assets.isa_norm_mq, t, eps_srf, R, kTpcs, 2, s2, 2, tpc_cmd);
            s2 += sig_r;
        }

        // 6) MME(out)
        plan_->add_mme(assets.out, s_o8, W.out_w, s_out, 0, 2, s2, 0);
        s0 += assets.out.signal_count;

        // 7) TPC(post_norm anq)
        {
            const DirectTpcTensor t[8] = {
                {s_out, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {s_os, DType::F32, 2, {1, R, 1, 1, 1}},
                {W.out_s_int, DType::F32, 1, {kH, 1, 1, 1, 1}},
                {s_res0, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {W.post_norm1p, DType::BF16, 1, {kH, 1, 1, 1, 1}},
                {s_x8, DType::F8, 2, {kH, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {s_res1, DType::BF16, 2, {kH, R, 1, 1, 1}},
            };
            plan_->add_tpc(assets.isa_anq, t, std::span<const uint32_t>(anq_srfs, 3), R, kTpcs, 0, s0, 2, tpc_cmd);
            s2 += sig_r;
        }

        // 8-9) Overlapped MLP & transition
        const bool to_apre = (last && has_apre);
        emit_mlp_and_transition(W.gate_w, W.up_w, W.gate_s_int, W.up_s_int,
                                W.dn_w, W.dn_s_int, W.feat_out,
                                last && !has_apre,
                                !last ? layers[j + 1].in_norm1p : (to_apre ? apre->in_norm1p : 0),
                                to_apre ? (rt.sram_base() + kDirectSramResidOff) : s_res0);
    }

    if (has_apre) {
        const DType kv_dt = apre->kv8 ? DType::F8 : DType::BF16;
        const uint64_t sb = rt.sram_base();
        const uint32_t r_head = kG * R;

        plan_->add_mme(assets.attn_qkv, s_x8, apre->qkv_w, s_attn_qkv, 0, 2, s2, 0);
        s0 += assets.attn_qkv.signal_count;

        const DirectTpcTensor t[15] = {
            {s_attn_qkv, DType::BF16, 2, {14336, R, 1, 1, 1}},
            {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
            {apre->qkv_s_int, DType::F32, 1, {14336, 1, 1, 1, 1}},
            {apre->q_norm, DType::BF16, 1, {kD, 1, 1, 1, 1}},
            {apre->k_norm, DType::BF16, 1, {kD, 1, 1, 1, 1}},
            {apre->rope_pos, DType::I32, 1, {R, 1, 1, 1, 1}},
            {apre->cos_tab, DType::F32, 2, {32, apre->rope_tab_rows, 1, 1, 1}},
            {apre->sin_tab, DType::F32, 2, {32, apre->rope_tab_rows, 1, 1, 1}},
            {sb + kDirectSramQOff, kv_dt, 3, {kD, r_head, kKV, 1, 1}},
            {sb + kDirectTreeSramQBfOff, DType::BF16, 3, {kD, r_head, kKV, 1, 1}},
            {sb + kDirectSramGateOff, DType::BF16, 2, {kNHD, R, 1, 1, 1}},
            {sb + kDirectTreeSramKTOff, DType::BF16, 3, {kD, R, kKV, 1, 1}},
            {sb + kDirectTreeSramVTOff, DType::F32, 3, {kD, R, kKV, 1, 1}},
            {apre->k_hbm, kv_dt, 3, {kD, R, kKV, 1, 1}},
            {apre->v_hbm, kv_dt, 3, {kD, R, kKV, 1, 1}},
        };
        const uint64_t isa_prep = apre->kv8 ? assets.isa_apre_tree_f8 : assets.isa_apre_tree_bf16;
        plan_->add_tpc(isa_prep, t, eps_srf, 3 * R, kTpcs, 0, s0, -1, tpc_cmd);
    }

    plan_->prepare();
}

DirectTreeBlock::~DirectTreeBlock() = default;

void DirectTreeBlock::update_args(const DirectTreeBlockArgs& args) {
    if (args.res_out != cur_.res_out) { plan_->patch_addr(kTagResOut, args.res_out); cur_.res_out = args.res_out; }
    if (args.delta_out != cur_.delta_out) { plan_->patch_addr(kTagDeltaOut, args.delta_out); cur_.delta_out = args.delta_out; }
    for (uint32_t j = 0; j < nl_; ++j) {
        if (args.conv_in[j] != cur_.conv_in[j]) { plan_->patch_addr(tag_conv_in(j), args.conv_in[j]); cur_.conv_in[j] = args.conv_in[j]; }
        if (args.ssm_in[j] != cur_.ssm_in[j]) { plan_->patch_addr(tag_ssm_in(j), args.ssm_in[j]); cur_.ssm_in[j] = args.ssm_in[j]; }
        if (args.st_x[j] != cur_.st_x[j]) { plan_->patch_addr(tag_st_x(j), args.st_x[j]); cur_.st_x[j] = args.st_x[j]; }
        if (args.st_v[j] != cur_.st_v[j]) { plan_->patch_addr(tag_st_v(j), args.st_v[j]); cur_.st_v[j] = args.st_v[j]; }
        if (args.st_g[j] != cur_.st_g[j]) { plan_->patch_addr(tag_st_g(j), args.st_g[j]); cur_.st_g[j] = args.st_g[j]; }
        if (args.st_seq[j] != cur_.st_seq[j]) { plan_->patch_addr(tag_st_seq(j), args.st_seq[j]); cur_.st_seq[j] = args.st_seq[j]; }
    }
}

double DirectTreeBlock::run() { return plan_->run(); }

// ==============================================================================
// DirectTreeAttnCore
// ==============================================================================

DirectTreeAttnCore::DirectTreeAttnCore(DirectRuntime& rt, const DirectTreeAssets& assets, bool kv8,
                                       uint32_t prefix_rows, const DirectTreeAttnArgs& init_args,
                                       uint64_t sram_offset, uint32_t tpc_cmd)
    : kv8_(kv8), N_(assets.N), cur_(init_args) {
    if (prefix_rows == 0 || (prefix_rows % 1024u) != 0)
        throw std::invalid_argument("direct_tree_attn: prefix_rows must be a positive multiple of 1024");
    const uint32_t r_head = kG * N_;
    const uint32_t rows_all = kKV * r_head;
    const uint32_t env_tw = [] { const char* e = std::getenv("G2E_TREE_MAX_TILE_W"); return e ? uint32_t(std::atoi(e)) : 0u; }();
    const uint32_t max_tile_w = (env_tw == 1024u || env_tw == 2048u || env_tw == 4096u) ? env_tw : ((!kv8 && N_ > 32u) ? 2048u : 4096u);
    const bool pv_pair = (r_head <= 256u);

    // 1. Decompose prefix_rows into {max_tile_w, 2048, 1024} tiles
    uint32_t col = 0;
    bool first = true;
    while (col < prefix_rows) {
        const uint32_t rem = prefix_rows - col;
        const uint32_t w = (rem >= max_tile_w) ? max_tile_w : (rem >= 2048u ? 2048u : 1024u);
        tiles_.push_back({col, w, first, col + w == prefix_rows, 0});
        col += w;
        first = false;
    }
    const uint32_t total_tiles = uint32_t(tiles_.size());

    // 2. Allocate SRAM arena starting at rt.sram_base() + sram_offset
    const uint64_t kv_es = kv8 ? 1ull : 2ull;
    const DType kv_dt = kv8 ? DType::F8 : DType::BF16;
    uint64_t sp = align256(rt.sram_base() + sram_offset);
    auto alloc = [&](uint64_t bytes) -> uint64_t {
        const uint64_t addr = sp;
        sp = align256(sp + bytes);
        return addr;
    };

    const uint64_t q_head_bytes = uint64_t(r_head) * kD * kv_es;
    const uint64_t s_S[2] = {
        alloc(uint64_t(rows_all) * max_tile_w * 2ull),
        alloc(uint64_t(rows_all) * max_tile_w * 2ull),
    };
    const uint64_t s_P[2] = {
        alloc(uint64_t(rows_all) * max_tile_w * kv_es),
        alloc(uint64_t(rows_all) * max_tile_w * kv_es),
    };
    const uint64_t s_PV[2] = {
        alloc(uint64_t(rows_all) * kD * 4ull),
        alloc(uint64_t(rows_all) * kD * 4ull),
    };
    const uint64_t s_O = alloc(uint64_t(rows_all) * kD * 4ull);
    const uint64_t s_m = alloc(uint64_t(rows_all) * 4ull);
    const uint64_t s_l = alloc(uint64_t(rows_all) * 4ull);
    uint64_t s_res[4];
    for (uint32_t r = 0; r < 4; ++r) s_res[r] = alloc(uint64_t(rows_all) * 4ull);

    sram_used_ = sp - (rt.sram_base() + sram_offset);
    if (sp > rt.sram_base() + rt.sram_size()) throw std::runtime_error("direct_tree_attn: SRAM overflow");

    const uint64_t isa_step = kv8 ? assets.isa_tile_step_f8 : assets.isa_tile_step_bf16;
    const uint32_t kl_bits = f32_bits(float((1.0 / std::sqrt(double(kD))) * 1.4426950408889634));

    auto qk_tmpl = [&](uint32_t w) -> const MmeTemplate& {
        if (kv8) return w == 4096 ? assets.qk_f8_4096 : (w == 2048 ? assets.qk_f8_2048 : assets.qk_f8_1024);
        return w == 4096 ? assets.qk_bf16_4096 : (w == 2048 ? assets.qk_bf16_2048 : assets.qk_bf16_1024);
    };
    auto pv_tmpl = [&](uint32_t w) -> const MmeTemplate& {
        if (kv8) return w == 4096 ? assets.pv_f8_4096 : (w == 2048 ? assets.pv_f8_2048 : assets.pv_f8_1024);
        return w == 4096 ? assets.pv_bf16_4096 : (w == 2048 ? assets.pv_bf16_2048 : assets.pv_bf16_1024);
    };

    // 3. Build double-buffered sub-plans
    for (uint32_t base = 0; base < total_tiles; base += kMaxTilesPerPlan) {
        const uint32_t count = std::min(kMaxTilesPerPlan, total_tiles - base);
        const uint32_t plan_idx = uint32_t(plans_.size());
        for (uint32_t k = 0; k < count; ++k) tiles_[base + k].plan_idx = plan_idx;

        auto plan = std::make_unique<DirectPlan>(rt, 0);
        uint32_t s_mme[2] = {0, 0};
        uint32_t s_tpc[2] = {0, 0};

        auto emit_qk = [&](uint32_t local_k, uint32_t parity, int wait_sob, uint32_t wait_val) {
            const uint32_t tidx = base + local_k;
            const auto& td = tiles_[tidx];
            const auto& tmpl = qk_tmpl(td.width);
            const uint64_t s_head_bytes = uint64_t(r_head) * td.width * 2ull;
            const int sig_sob = int(parity);
            for (uint32_t h = 0; h < kKV; ++h) {
                const uint64_t q_addr = init_args.q_prefix + uint64_t(h) * q_head_bytes;
                const uint64_t k_addr = init_args.k_cache[h] + uint64_t(td.col_start) * kD * kv_es;
                const uint64_t s_addr = s_S[parity] + uint64_t(h) * s_head_bytes;
                plan->add_mme(tmpl, q_addr, k_addr, s_addr, 0,
                              h == 0 ? wait_sob : -1, h == 0 ? wait_val : 0, sig_sob,
                              {-1, tag_attn_k(tidx, h), -1, -1});
                s_mme[parity] += tmpl.signal_count;
            }
        };

        auto emit_pv = [&](uint32_t local_k, uint32_t parity, int wait_sob, uint32_t wait_val) {
            const uint32_t tidx = base + local_k;
            const auto& td = tiles_[tidx];
            const auto& tmpl = pv_tmpl(td.width);
            const uint64_t p_head_bytes = uint64_t(r_head) * td.width * kv_es;
            const uint64_t pv_head_bytes = uint64_t(r_head) * kD * 4ull;
            const int sig_sob = int(parity);
            if (pv_pair) {
                for (uint32_t hp = 0; hp < kKV; hp += 2) {
                    const uint32_t h0 = hp, h1 = hp + 1;
                    plan->add_mme_pair(tmpl,
                                       s_P[parity] + uint64_t(h0) * p_head_bytes,
                                       init_args.v_cache[h0] + uint64_t(td.col_start) * kD * kv_es,
                                       s_PV[parity] + uint64_t(h0) * pv_head_bytes,
                                       s_P[parity] + uint64_t(h1) * p_head_bytes,
                                       init_args.v_cache[h1] + uint64_t(td.col_start) * kD * kv_es,
                                       s_PV[parity] + uint64_t(h1) * pv_head_bytes,
                                       hp == 0 ? wait_sob : -1, hp == 0 ? wait_val : 0, sig_sob,
                                       {-1, tag_attn_v(tidx, h0), -1, -1},
                                       {-1, tag_attn_v(tidx, h1), -1, -1});
                    s_mme[parity] += 2u;
                }
            } else {
                for (uint32_t h = 0; h < kKV; ++h) {
                    plan->add_mme(tmpl,
                                  s_P[parity] + uint64_t(h) * p_head_bytes,
                                  init_args.v_cache[h] + uint64_t(td.col_start) * kD * kv_es,
                                  s_PV[parity] + uint64_t(h) * pv_head_bytes, 0,
                                  h == 0 ? wait_sob : -1, h == 0 ? wait_val : 0, sig_sob,
                                  {-1, tag_attn_v(tidx, h), -1, -1});
                    s_mme[parity] += tmpl.signal_count;
                }
            }
        };

        auto emit_tpc_step = [&](int sm_local_k, int acc_local_k, uint32_t parity,
                                 int wait_sob, uint32_t wait_val, int sig_sob) {
            const bool do_sm = sm_local_k >= 0;
            const bool do_acc = acc_local_k >= 0;
            const uint32_t sm_tidx = do_sm ? (base + uint32_t(sm_local_k)) : base;
            const uint32_t acc_tidx = do_acc ? (base + uint32_t(acc_local_k)) : base;
            const auto& sm_td = tiles_[sm_tidx];
            const auto& acc_td = tiles_[acc_tidx];

            const uint32_t tile_w = do_sm ? sm_td.width : 1024u;
            const uint32_t flags_sm = do_sm ? (1u | (sm_td.is_first ? 2u : 0u)) : 0u;
            const uint32_t flags_acc = do_acc ? (1u | (acc_td.is_first ? 2u : 0u) | (acc_td.is_last ? 4u : 0u)) : 0u;

            const DirectTpcTensor t[15] = {
                {s_S[parity], DType::BF16, 2, {tile_w, rows_all, 1, 1, 1}},
                {s_P[parity], kv_dt, 2, {tile_w, rows_all, 1, 1, 1}},
                {s_m, DType::F32, 2, {1, rows_all, 1, 1, 1}},
                {s_l, DType::F32, 2, {1, rows_all, 1, 1, 1}},
                {s_res[sm_tidx & 3u], DType::F32, 2, {1, rows_all, 1, 1, 1}},
                {s_PV[parity], DType::F32, 2, {kD, rows_all, 1, 1, 1}},
                {s_O, DType::F32, 2, {kD, rows_all, 1, 1, 1}},
                {s_res[acc_tidx & 3u], DType::F32, 2, {1, rows_all, 1, 1, 1}},
                {s_l, DType::F32, 2, {1, rows_all, 1, 1, 1}},
                {init_args.att_out, DType::BF16, 2, {kNHD, N_, 1, 1, 1}},
                {init_args.pos_buf, DType::I32, 1, {1, 1, 1, 1, 1}},
                {init_args.q_bf, DType::BF16, 3, {kD, r_head, kKV, 1, 1}},
                {init_args.k_tree_bf, DType::BF16, 3, {kD, N_, kKV, 1, 1}},
                {init_args.v_tree_f32, DType::F32, 3, {kD, N_, kKV, 1, 1}},
                {init_args.parent, DType::I32, 1, {N_, 1, 1, 1, 1}},
            };
            const uint32_t srfs[5] = {kl_bits, do_sm ? sm_td.col_start : 0u, N_, flags_sm, flags_acc};
            plan->add_tpc(isa_step, t, srfs, rows_all, kTpcs, wait_sob, wait_val, sig_sob, tpc_cmd);
            if (sig_sob == 2) s_tpc[0] += kTpcs;
            else if (sig_sob == 3) s_tpc[1] += kTpcs;
        };

        if (count == 1) {
            emit_qk(0, 0, -1, 0);
            const uint32_t qk0_done = s_mme[0];
            emit_tpc_step(0, -1, 0, 0, qk0_done, 2);
            emit_pv(0, 0, 2, s_tpc[0]);
            const uint32_t pv0_done = s_mme[0];
            emit_tpc_step(-1, 0, 0, 0, pv0_done, -1);
        } else {
            std::vector<uint32_t> mme_ready(count + 2, 0);
            emit_qk(0, 0, -1, 0);
            mme_ready[0] = s_mme[0];
            emit_qk(1, 1, -1, 0);
            mme_ready[1] = s_mme[1];
            for (uint32_t k = 0; k < count; ++k) {
                const uint32_t p = k & 1u;
                emit_tpc_step(int(k), k >= 2 ? int(k - 2) : -1, p, int(p), mme_ready[k], int(2u + p));
                emit_pv(k, p, int(2u + p), s_tpc[p]);
                if (k + 2 < count) emit_qk(k + 2, p, -1, 0);
                mme_ready[k + 2] = s_mme[p];
            }
            const uint32_t pe0 = (count - 2) & 1u;
            const uint32_t pe1 = (count - 1) & 1u;
            emit_tpc_step(-1, int(count - 2), pe0, int(pe0), mme_ready[count], int(2u + pe0));
            emit_tpc_step(-1, int(count - 1), pe1, int(pe1), mme_ready[count + 1], -1);
        }

        plan->prepare();
        plans_.push_back(std::move(plan));
    }
}

DirectTreeAttnCore::~DirectTreeAttnCore() = default;

void DirectTreeAttnCore::update_args(const DirectTreeAttnArgs& args) {
    const uint64_t kv_es = kv8_ ? 1ull : 2ull;
    bool diff = false;
    for (uint32_t h = 0; h < kKV; ++h)
        if (args.k_cache[h] != cur_.k_cache[h] || args.v_cache[h] != cur_.v_cache[h]) diff = true;
    if (!diff) return;
    for (uint32_t tidx = 0; tidx < tiles_.size(); ++tidx) {
        const auto& td = tiles_[tidx];
        auto& pl = *plans_[td.plan_idx];
        for (uint32_t h = 0; h < kKV; ++h) {
            if (args.k_cache[h] != cur_.k_cache[h])
                pl.patch_addr(tag_attn_k(tidx, h), args.k_cache[h] + uint64_t(td.col_start) * kD * kv_es);
            if (args.v_cache[h] != cur_.v_cache[h])
                pl.patch_addr(tag_attn_v(tidx, h), args.v_cache[h] + uint64_t(td.col_start) * kD * kv_es);
        }
    }
    cur_.k_cache = args.k_cache;
    cur_.v_cache = args.v_cache;
}

double DirectTreeAttnCore::run() {
    double total_us = 0.0;
    for (auto& pl : plans_) total_us += pl->run();
    return total_us;
}

void DirectTreeAttnCore::collect_plans(std::vector<DirectPlan*>& out) const {
    for (const auto& pl : plans_) out.push_back(pl.get());
}

// ==============================================================================
// DirectTreeCommit
// ==============================================================================

DirectTreeCommit::DirectTreeCommit(DirectRuntime& rt, const DirectTreeAssets& assets, uint32_t ng, uint32_t na, bool kv8,
                                   const DirectTreeCommitArgs& init_args, uint32_t tpc_cmd)
    : na_(na), cap_(init_args.kv_cap), feat_layers_(init_args.feat_layers),
      plan_(std::make_unique<DirectPlan>(rt, 0)), cur_(init_args) {
    const uint32_t N = assets.N;
    const uint32_t d_bf16 = kv8 ? (kD / 2u) : kD;
    const uint64_t kv_es = kv8 ? 1ull : 2ull;

    // 1. In-place GDN tree SSM state commit
    {
        const DirectTpcTensor t[6] = {
            {init_args.ssm, DType::F32, 3, {kDK, kDV, ng * kHV, 1, 1}},
            {init_args.x, DType::F32, 4, {kDK, 2 * N, kHK, ng, 1}},
            {init_args.v, DType::F32, 4, {kDV, kHV, N, ng, 1}},
            {init_args.g, DType::F32, 3, {kHV, N, ng, 1, 1}},
            {init_args.path_ctl, DType::I32, 1, {17, 1, 1, 1, 1}},
            {init_args.ssm, DType::F32, 3, {kDK, kDV, ng * kHV, 1, 1}},
        };
        const int tags[6] = {1, 3, 4, 5, -1, 1};
        plan_->add_tpc_grid(assets.isa_tree_state_commit, t, {}, {ng * kHV, 1, 1, 1, 1}, 0, kTpcs,
                            -1, 0, -1, tpc_cmd, tags);
    }

    // 2. GDN conv state commit
    {
        const DirectTpcTensor t[3] = {
            {init_args.seq, DType::BF16, 3, {kC, N + 3, ng, 1, 1}},
            {init_args.conv_rows, DType::I32, 1, {3, 1, 1, 1, 1}},
            {init_args.conv_out, DType::BF16, 3, {kC, 3, ng, 1, 1}},
        };
        const int tags[3] = {6, -1, 2};
        plan_->add_tpc_grid(assets.isa_conv_commit, t, {}, {1, ng, 1, 1, 1}, 1, kTpcs,
                            -1, 0, -1, tpc_cmd, tags);
    }

    // 3. Attention KV cache commit (4 layers per TPC launch, ZERO inter-group barriers!)
    const uint32_t h_stride = cap_ * d_bf16;
    for (uint32_t lay0 = 0; lay0 < na; lay0 += 4) {
        const uint32_t cnt = std::min<uint32_t>(4u, na - lay0);
        const uint64_t scr_base = init_args.kv_scratch + uint64_t(lay0) * 8ull * N * kD * kv_es;
        DirectTpcTensor t[11];
        t[0] = {scr_base, DType::BF16, 3, {d_bf16, N, cnt * 8u, 1, 1}};
        t[1] = {init_args.path_ctl, DType::I32, 1, {17, 1, 1, 1, 1}};
        t[2] = {init_args.pos_buf, DType::I32, 1, {1, 1, 1, 1, 1}};
        int tags[11] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        for (uint32_t k = 0; k < 4; ++k) {
            const uint32_t s = (k < cnt) ? (lay0 + k) : lay0;
            t[3 + k] = {init_args.k_cache[s], DType::BF16, 3, {d_bf16, cap_, kKV, 1, 1}, {0, 0, h_stride, 0, 0}};
            t[7 + k] = {init_args.v_cache[s], DType::BF16, 3, {d_bf16, cap_, kKV, 1, 1}, {0, 0, h_stride, 0, 0}};
            if (k < cnt) {
                tags[3 + k] = 100 + int(s);
                tags[7 + k] = 200 + int(s);
            }
        }
        plan_->add_tpc_grid(assets.isa_kv_commit, t, {}, {cnt * 8u, 1, 1, 1, 1}, 0, kTpcs,
                            -1, 0, -1, tpc_cmd, tags);
    }

    // 4. Optional drafter feature commit
    if (feat_layers_ > 0 && init_args.feat_in != 0 && init_args.feat_out != 0) {
        const DirectTpcTensor t[3] = {
            {init_args.feat_in, DType::BF16, 3, {kH, N, feat_layers_, 1, 1},
             {0, 0, init_args.feat_in_stride_elems, 0, 0}},
            {init_args.path_ctl, DType::I32, 1, {17, 1, 1, 1, 1}},
            {init_args.feat_out, DType::BF16, 3, {kH, 16, feat_layers_, 1, 1},
             {0, 0, init_args.feat_out_stride_elems, 0, 0}},
        };
        const int tags[3] = {8, -1, 9};
        plan_->add_tpc_grid(assets.isa_feat_commit, t, {}, {feat_layers_ * 16u, 1, 1, 1, 1}, 0, kTpcs,
                            -1, 0, -1, tpc_cmd, tags);
    }

    plan_->prepare();
}

DirectTreeCommit::~DirectTreeCommit() = default;

void DirectTreeCommit::update_args(const DirectTreeCommitArgs& args) {
    if (args.ssm != cur_.ssm) { plan_->patch_addr(1, args.ssm); cur_.ssm = args.ssm; }
    if (args.conv_out != cur_.conv_out) { plan_->patch_addr(2, args.conv_out); cur_.conv_out = args.conv_out; }
    if (args.x != cur_.x) { plan_->patch_addr(3, args.x); cur_.x = args.x; }
    if (args.v != cur_.v) { plan_->patch_addr(4, args.v); cur_.v = args.v; }
    if (args.g != cur_.g) { plan_->patch_addr(5, args.g); cur_.g = args.g; }
    if (args.seq != cur_.seq) { plan_->patch_addr(6, args.seq); cur_.seq = args.seq; }
    if (args.feat_in != cur_.feat_in && feat_layers_ > 0) { plan_->patch_addr(8, args.feat_in); cur_.feat_in = args.feat_in; }
    if (args.feat_out != cur_.feat_out && feat_layers_ > 0) { plan_->patch_addr(9, args.feat_out); cur_.feat_out = args.feat_out; }
    for (uint32_t s = 0; s < na_; ++s) {
        if (args.k_cache[s] != cur_.k_cache[s]) { plan_->patch_addr(100 + int(s), args.k_cache[s]); cur_.k_cache[s] = args.k_cache[s]; }
        if (args.v_cache[s] != cur_.v_cache[s]) { plan_->patch_addr(200 + int(s), args.v_cache[s]); cur_.v_cache[s] = args.v_cache[s]; }
    }
}

double DirectTreeCommit::run() { return plan_->run(); }

}  // namespace g2e
