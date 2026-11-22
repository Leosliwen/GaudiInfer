// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

struct GdnSpecPrepBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_spec_prep_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct GdnMtpWyoF32Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_mtp_wyo_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

// the same core for T = 8 (batched verification at width 8)
struct GdnMtpWyo8F32Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_mtp_wyo8_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

// output-only WY core that also outputs ds_t [B, T, 48, 128] for gdn_state_commit
struct GdnMtpWyodF32Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_mtp_wyod_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct GdnMtpWyod8F32Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_mtp_wyod8_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct GdnStateCommitF32Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_state_commit_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct RowSoftmaxStatsF32Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_row_softmax_stats_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct RowSoftmaxStatsCmaxF32Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_row_softmax_stats_cmax_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct GdnSpecPrepBatchBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_spec_prep_batch_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct AttnQkvPrepBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_attn_qkv_prep_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct AttnQkvPrepF8Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_attn_qkv_prep_f8_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct GdnPrefillConvPrepBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_prefill_conv_prep_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct GdnPrefillGateQuantBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_prefill_gate_quant_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct GdnPrefillDecayPrepBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_prefill_decay_prep_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct DSpineBridgeNormBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_dspine_bridge_norm_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct DSpineBridgeCodesBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_dspine_bridge_codes_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct DSpineBridgeGateBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_dspine_bridge_gate_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct DSpineBridgePostBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_dspine_bridge_post_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct DSpineQkvPrepBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_dspine_qkv_prep_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct GdnTreePrepBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_tree_prep_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct GdnTreeWyF32Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_tree_wy_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct GdnTreeCommitF32Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_tree_commit_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};
