// SPDX-License-Identifier: Apache-2.0
// Glue for kernels/gdn_spec_prep_bf16_gaudi2.c (conv + layout for the WY verification core) and for the output-only
// WY core kernels/gdn_mtp_wyo_f32_gaudi2.c.
#include "gdn_spec_prep_bf16_gaudi2.hpp"

#include <cstring>
#include <initializer_list>

extern unsigned char _binary_gdn_spec_prep_bf16_gaudi2_o_start;
extern unsigned char _binary_gdn_spec_prep_bf16_gaudi2_o_end;
extern unsigned char _binary_gdn_mtp_wyo_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_mtp_wyo_f32_gaudi2_o_end;
extern unsigned char _binary_gdn_mtp_wyo8_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_mtp_wyo8_f32_gaudi2_o_end;
extern unsigned char _binary_gdn_mtp_wyod_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_mtp_wyod_f32_gaudi2_o_end;
extern unsigned char _binary_gdn_mtp_wyod8_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_mtp_wyod8_f32_gaudi2_o_end;
extern unsigned char _binary_gdn_state_commit_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_state_commit_f32_gaudi2_o_end;
extern unsigned char _binary_row_softmax_stats_f32_gaudi2_o_start;
extern unsigned char _binary_row_softmax_stats_f32_gaudi2_o_end;
extern unsigned char _binary_row_softmax_stats_cmax_f32_gaudi2_o_start;
extern unsigned char _binary_row_softmax_stats_cmax_f32_gaudi2_o_end;
extern unsigned char _binary_gdn_spec_prep_batch_bf16_gaudi2_o_start;
extern unsigned char _binary_gdn_spec_prep_batch_bf16_gaudi2_o_end;
extern unsigned char _binary_attn_qkv_prep_bf16_gaudi2_o_start;
extern unsigned char _binary_attn_qkv_prep_bf16_gaudi2_o_end;
extern unsigned char _binary_attn_qkv_prep_f8_gaudi2_o_start;
extern unsigned char _binary_attn_qkv_prep_f8_gaudi2_o_end;
extern unsigned char _binary_gdn_prefill_conv_prep_bf16_gaudi2_o_start;
extern unsigned char _binary_gdn_prefill_conv_prep_bf16_gaudi2_o_end;
extern unsigned char _binary_gdn_prefill_gate_quant_bf16_gaudi2_o_start;
extern unsigned char _binary_gdn_prefill_gate_quant_bf16_gaudi2_o_end;
extern unsigned char _binary_gdn_prefill_decay_prep_bf16_gaudi2_o_start;
extern unsigned char _binary_gdn_prefill_decay_prep_bf16_gaudi2_o_end;
extern unsigned char _binary_dspine_bridge_norm_bf16_gaudi2_o_start;
extern unsigned char _binary_dspine_bridge_norm_bf16_gaudi2_o_end;
extern unsigned char _binary_dspine_bridge_codes_bf16_gaudi2_o_start;
extern unsigned char _binary_dspine_bridge_codes_bf16_gaudi2_o_end;
extern unsigned char _binary_dspine_bridge_gate_bf16_gaudi2_o_start;
extern unsigned char _binary_dspine_bridge_gate_bf16_gaudi2_o_end;
extern unsigned char _binary_dspine_bridge_post_bf16_gaudi2_o_start;
extern unsigned char _binary_dspine_bridge_post_bf16_gaudi2_o_end;
extern unsigned char _binary_dspine_qkv_prep_bf16_gaudi2_o_start;
extern unsigned char _binary_dspine_qkv_prep_bf16_gaudi2_o_end;
extern unsigned char _binary_gdn_tree_prep_bf16_gaudi2_o_start;
extern unsigned char _binary_gdn_tree_prep_bf16_gaudi2_o_end;
extern unsigned char _binary_gdn_tree_wy_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_tree_wy_f32_gaudi2_o_end;
extern unsigned char _binary_gdn_tree_commit_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_tree_commit_f32_gaudi2_o_end;

namespace {
using namespace tpc_lib_api;
bool Is(const Tensor& t, TensorDataType type, std::initializer_list<uint64_t> sizes)
{
    if (t.geometry.dataType != type || t.geometry.dims != sizes.size()) return false;
    unsigned d = 0;
    for (uint64_t s : sizes)
        if (t.geometry.maxSizes[d++] != s) return false;
    return true;
}
void Map(TensorAccessPattern& pattern, unsigned dim, unsigned indexDim, int a, int first, int last)
{
    pattern.mapping[dim].indexSpaceDim = indexDim;
    pattern.mapping[dim].a = a;
    pattern.mapping[dim].start_b = first;
    pattern.mapping[dim].end_b = last;
}
GlueCodeReturn Elf(HabanaKernelInstantiation* out, unsigned char* begin, unsigned char* end)
{
    const unsigned size = static_cast<unsigned>(end - begin);
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, size);
    return GLUE_SUCCESS;
}
}  // namespace

GlueCodeReturn GdnSpecPrepBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& seq = in->inputTensors[0].geometry;
    const uint64_t C = seq.maxSizes[0], rows = seq.maxSizes[1];
    if (seq.dataType != DATA_BF16 || seq.dims != 2 || C % 128 || C <= 4096 || rows < 4)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t T = rows - 3;
    if (!Is(in->inputTensors[1], DATA_BF16, {C, 4})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_F32, {128, 2 * T, 16}) || !Is(in->outputTensors[1], DATA_F32, {C - 4096, T}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = C / 128;
    for (unsigned i = 0; i < 2; ++i) {
        out->inputTensorAccessPattern[i].allRequired = true;
        out->outputTensorAccessPattern[i].allRequired = true;
    }
    out->kernel.paramsNr = 0;
    return Elf(out, &_binary_gdn_spec_prep_bf16_gaudi2_o_start, &_binary_gdn_spec_prep_bf16_gaudi2_o_end);
}

// Inputs: 0 log_decay F32 [48, T, B]; 1 beta F32 [48, T, B]; 2 rmat F32 [384, 2T, 16, B] (key head h: rows t:
// S0_v k_t, rows T + t: S0_v q_t for its 3 value heads side by side); 3 gram F32 [2T, 2T, 16, B]; 4 vals F32
// [128, 48, T, B]. Output: F32 [128, 48, T, B]. T = 16 (wyo) or 8 (wyo8).
static GlueCodeReturn WyoDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out, uint64_t T,
                                     unsigned char* elf, unsigned char* elf_end, bool with_ds = false)
{
    const unsigned nout = with_ds ? 2 : 1;
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != nout) { in->outputTensorNr = nout; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* t = in->inputTensors;
    const uint64_t B = t[0].geometry.maxSizes[2];
    if (B == 0 || B > 16 || !Is(t[0], DATA_F32, {48, T, B}) || !Is(t[1], DATA_F32, {48, T, B}) ||
        !Is(t[2], DATA_F32, {384, 2 * T, 16, B}) || !Is(t[3], DATA_F32, {2 * T, 2 * T, 16, B}) ||
        !Is(t[4], DATA_F32, {128, 48, T, B}))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_F32, {128, 48, T, B})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (with_ds && !Is(in->outputTensors[1], DATA_F32, {128, 48, T, B})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 4;
    out->indexSpaceGeometry[0] = 1;
    out->indexSpaceGeometry[1] = 1;
    out->indexSpaceGeometry[2] = 48;
    out->indexSpaceGeometry[3] = B;
    for (unsigned i = 0; i < 5; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    for (unsigned o = 0; o < nout; ++o) {
        Map(out->outputTensorAccessPattern[o], 0, 0, 0, 0, 127);
        Map(out->outputTensorAccessPattern[o], 1, 2, 1, 0, 0);
        Map(out->outputTensorAccessPattern[o], 2, 0, 0, 0, static_cast<int>(T) - 1);
        Map(out->outputTensorAccessPattern[o], 3, 3, 1, 0, 0);
    }
    out->kernel.paramsNr = 0;
    return Elf(out, elf, elf_end);
}

GlueCodeReturn GdnMtpWyoF32Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    return WyoDefinitions(in, out, 16, &_binary_gdn_mtp_wyo_f32_gaudi2_o_start, &_binary_gdn_mtp_wyo_f32_gaudi2_o_end);
}

GlueCodeReturn GdnMtpWyo8F32Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    return WyoDefinitions(in, out, 8, &_binary_gdn_mtp_wyo8_f32_gaudi2_o_start, &_binary_gdn_mtp_wyo8_f32_gaudi2_o_end);
}

GlueCodeReturn GdnMtpWyodF32Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    return WyoDefinitions(in, out, 16, &_binary_gdn_mtp_wyod_f32_gaudi2_o_start, &_binary_gdn_mtp_wyod_f32_gaudi2_o_end, true);
}

GlueCodeReturn GdnMtpWyod8F32Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    return WyoDefinitions(in, out, 8, &_binary_gdn_mtp_wyod8_f32_gaudi2_o_start, &_binary_gdn_mtp_wyod8_f32_gaudi2_o_end, true);
}

GlueCodeReturn GdnStateCommitF32Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* t = in->inputTensors;
    const auto& sg = t[0].geometry;
    const auto& xg = t[1].geometry;
    const auto& dg = t[2].geometry;
    if (sg.dataType != DATA_F32 || sg.dims != 3 || sg.maxSizes[0] != 128 || sg.maxSizes[1] != 128 ||
        sg.maxSizes[2] == 0 || sg.maxSizes[2] % 48 != 0)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t NG = sg.maxSizes[2] / 48;
    if (dg.dataType != DATA_F32 || dg.dims != 4 || dg.maxSizes[0] != 128 || dg.maxSizes[1] != 48 ||
        dg.maxSizes[2] == 0 || dg.maxSizes[2] > 16 || dg.maxSizes[3] != NG)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t T = dg.maxSizes[2];
    if (xg.dataType != DATA_F32 || xg.dims != 4 || xg.maxSizes[0] != 128 ||
        (xg.maxSizes[1] != T && xg.maxSizes[1] != 2 * T) || xg.maxSizes[2] != 16 || xg.maxSizes[3] != NG)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(t[3], DATA_F32, {48, T, NG}) || !Is(t[4], DATA_I32, {1}))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_F32, {128, 128, NG * 48}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = NG * 48;
    Map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, 127);
    Map(out->inputTensorAccessPattern[0], 1, 0, 0, 0, 127);
    Map(out->inputTensorAccessPattern[0], 2, 0, 1, 0, 0);
    for (unsigned i = 1; i < 5; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    Map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, 127);
    Map(out->outputTensorAccessPattern[0], 1, 0, 0, 0, 127);
    Map(out->outputTensorAccessPattern[0], 2, 0, 1, 0, 0);
    out->kernel.paramsNr = 0;
    return Elf(out, &_binary_gdn_state_commit_f32_gaudi2_o_start, &_binary_gdn_state_commit_f32_gaudi2_o_end);
}

GlueCodeReturn RowSoftmaxStatsF32Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& lg = in->inputTensors[0].geometry;
    const auto& it = in->inputTensors[1].geometry;
    if (lg.dataType != DATA_F32 || lg.dims != 2 || lg.maxSizes[0] == 0 || lg.maxSizes[0] % 64 != 0 || lg.maxSizes[1] == 0)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t V = lg.maxSizes[0], R = lg.maxSizes[1];
    const bool it_ok = it.dataType == DATA_F32 &&
                       ((it.dims == 1 && it.maxSizes[0] == 1) ||
                        (it.dims == 2 && it.maxSizes[0] == 1 && (it.maxSizes[1] == 1 || it.maxSizes[1] == R)));
    if (!it_ok) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_F32, {1, R}) || !Is(in->outputTensors[1], DATA_F32, {1, R}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = R;
    Map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, static_cast<int>(V) - 1);
    Map(out->inputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    out->inputTensorAccessPattern[1].allRequired = true;
    for (unsigned o = 0; o < 2; ++o) {
        Map(out->outputTensorAccessPattern[o], 0, 0, 0, 0, 0);
        Map(out->outputTensorAccessPattern[o], 1, 0, 1, 0, 0);
    }
    out->kernel.paramsNr = 0;
    return Elf(out, &_binary_row_softmax_stats_f32_gaudi2_o_start, &_binary_row_softmax_stats_f32_gaudi2_o_end);
}

GlueCodeReturn RowSoftmaxStatsCmaxF32Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 3) { in->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& lg = in->inputTensors[0].geometry;
    const auto& it = in->inputTensors[1].geometry;
    if (lg.dataType != DATA_F32 || lg.dims != 2 || lg.maxSizes[0] < 384 * 128 || (lg.maxSizes[0] % 128) != 0 || lg.maxSizes[1] == 0)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t V = lg.maxSizes[0], R = lg.maxSizes[1], nc = V / 128;
    const bool it_ok = it.dataType == DATA_F32 &&
                       ((it.dims == 1 && it.maxSizes[0] == 1) ||
                        (it.dims == 2 && it.maxSizes[0] == 1 && (it.maxSizes[1] == 1 || it.maxSizes[1] == R)));
    if (!it_ok) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_F32, {64, R, 3}) ||
        !Is(in->outputTensors[1], DATA_F32, {64, R, 3}) ||
        !Is(in->outputTensors[2], DATA_F32, {nc, R}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = R * 3;
    for (unsigned i = 0; i < 2; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    for (unsigned o = 0; o < 3; ++o) out->outputTensorAccessPattern[o].allRequired = true;
    out->kernel.paramsNr = 0;
    return Elf(out, &_binary_row_softmax_stats_cmax_f32_gaudi2_o_start, &_binary_row_softmax_stats_cmax_f32_gaudi2_o_end);
}

GlueCodeReturn GdnSpecPrepBatchBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& seq = in->inputTensors[0].geometry;
    const uint64_t C = seq.maxSizes[0], rows = seq.maxSizes[1], B = seq.maxSizes[2];
    if (seq.dataType != DATA_BF16 || seq.dims != 3 || C % 128 || C <= 4096 || rows < 4 || B == 0 || B > 16)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t T = rows - 3;
    if (!Is(in->inputTensors[1], DATA_BF16, {C, 4})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_F32, {128, 2 * T, 16, B}) ||
        !Is(in->outputTensors[1], DATA_F32, {C - 4096, T, B}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = C / 128;
    out->indexSpaceGeometry[1] = B;
    for (unsigned i = 0; i < 2; ++i) {
        out->inputTensorAccessPattern[i].allRequired = true;
        out->outputTensorAccessPattern[i].allRequired = true;
    }
    out->kernel.paramsNr = 0;
    return Elf(out, &_binary_gdn_spec_prep_batch_bf16_gaudi2_o_start, &_binary_gdn_spec_prep_batch_bf16_gaudi2_o_end);
}

static GlueCodeReturn AttnQkvPrepDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out,
                                             TensorDataType kv_type, unsigned char* elf, unsigned char* elf_end)
{
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 4) { in->outputTensorNr = 4; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& qkv = in->inputTensors[0].geometry;
    const auto& qn = in->inputTensors[1].geometry;
    const auto& kn = in->inputTensors[2].geometry;
    const auto& cs = in->inputTensors[3].geometry;
    const auto& sn = in->inputTensors[4].geometry;
    const auto& qo = in->outputTensors[0].geometry;
    const auto& go = in->outputTensors[1].geometry;
    const auto& ko = in->outputTensors[2].geometry;
    const auto& vo = in->outputTensors[3].geometry;
    if (qkv.dataType != DATA_BF16 || qn.dataType != DATA_BF16 || kn.dataType != DATA_BF16 ||
        cs.dataType != DATA_F32 || sn.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (qo.dataType != DATA_BF16 || go.dataType != DATA_BF16 || ko.dataType != kv_type || vo.dataType != kv_type)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (qkv.dims != 2 || qo.dims != 3 || ko.dims != 3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t W = qkv.maxSizes[0], R = qkv.maxSizes[1];
    const uint64_t NH = qo.maxSizes[1], KV = ko.maxSizes[2];
    if (!R || !NH || !KV || W != NH * 512 + KV * 512) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->inputTensors[1], DATA_BF16, {256}) || !Is(in->inputTensors[2], DATA_BF16, {256}) ||
        !Is(in->inputTensors[3], DATA_F32, {32, R}) || !Is(in->inputTensors[4], DATA_F32, {32, R}))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_BF16, {256, NH, R}) ||
        !Is(in->outputTensors[1], DATA_BF16, {NH * 256, R}) ||
        !Is(in->outputTensors[2], kv_type, {256, R, KV}) ||
        !Is(in->outputTensors[3], kv_type, {256, R, KV}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    float eps = 1e-6f;
    if (in->nodeParams.nodeParams && in->nodeParams.nodeParamsSize == sizeof(float))
        std::memcpy(&eps, in->nodeParams.nodeParams, sizeof(float));
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = R;
    Map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, static_cast<int>(W) - 1);
    Map(out->inputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    out->inputTensorAccessPattern[1].allRequired = true;
    out->inputTensorAccessPattern[2].allRequired = true;
    Map(out->inputTensorAccessPattern[3], 0, 0, 0, 0, 31);
    Map(out->inputTensorAccessPattern[3], 1, 0, 1, 0, 0);
    Map(out->inputTensorAccessPattern[4], 0, 0, 0, 0, 31);
    Map(out->inputTensorAccessPattern[4], 1, 0, 1, 0, 0);
    Map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, 255);
    Map(out->outputTensorAccessPattern[0], 1, 0, 0, 0, static_cast<int>(NH) - 1);
    Map(out->outputTensorAccessPattern[0], 2, 0, 1, 0, 0);
    Map(out->outputTensorAccessPattern[1], 0, 0, 0, 0, static_cast<int>(NH * 256) - 1);
    Map(out->outputTensorAccessPattern[1], 1, 0, 1, 0, 0);
    for (unsigned o = 2; o < 4; ++o) {
        Map(out->outputTensorAccessPattern[o], 0, 0, 0, 0, 255);
        Map(out->outputTensorAccessPattern[o], 1, 0, 1, 0, 0);
        Map(out->outputTensorAccessPattern[o], 2, 0, 0, 0, static_cast<int>(KV) - 1);
    }
    out->kernel.paramsNr = 1;
    std::memcpy(out->kernel.scalarParams, &eps, sizeof(float));
    return Elf(out, elf, elf_end);
}

GlueCodeReturn AttnQkvPrepBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    return AttnQkvPrepDefinitions(in, out, DATA_BF16, &_binary_attn_qkv_prep_bf16_gaudi2_o_start,
                                  &_binary_attn_qkv_prep_bf16_gaudi2_o_end);
}

GlueCodeReturn AttnQkvPrepF8Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    return AttnQkvPrepDefinitions(in, out, DATA_F8_143, &_binary_attn_qkv_prep_f8_gaudi2_o_start,
                                  &_binary_attn_qkv_prep_f8_gaudi2_o_end);
}

GlueCodeReturn GdnPrefillConvPrepBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 3) { in->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& seq = in->inputTensors[0].geometry;
    const auto& qc = in->outputTensors[0].geometry;
    const auto& vc = in->outputTensors[2].geometry;
    if (seq.dataType != DATA_BF16 || seq.dims != 2 || seq.maxSizes[1] < 4) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t C_tot = seq.maxSizes[0], T = seq.maxSizes[1] - 3;
    if (!Is(in->inputTensors[1], DATA_BF16, {C_tot, 4})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (qc.dataType != DATA_BF16 || qc.dims != 3 || qc.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    const uint64_t C_chunk = qc.maxSizes[1];
    if (!C_chunk || T % C_chunk) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    const uint64_t N = T / C_chunk;
    if (qc.maxSizes[2] % N || vc.maxSizes[2] % N) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    const uint64_t HK = qc.maxSizes[2] / N, HV = vc.maxSizes[2] / N;
    if ((2 * HK + HV) * 128 != C_tot) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (!Is(in->outputTensors[1], DATA_BF16, {128, C_chunk, N * HK}) ||
        !Is(in->outputTensors[2], DATA_BF16, {128, C_chunk, N * HV}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (C_chunk % 4) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = N * (2 * HK + HV);
    for (unsigned i = 0; i < 2; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    for (unsigned o = 0; o < 3; ++o) out->outputTensorAccessPattern[o].allRequired = true;
    out->kernel.paramsNr = 0;
    return Elf(out, &_binary_gdn_prefill_conv_prep_bf16_gaudi2_o_start, &_binary_gdn_prefill_conv_prep_bf16_gaudi2_o_end);
}

GlueCodeReturn GdnPrefillGateQuantBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 3) { in->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize < sizeof(float)) return GLUE_MISSING_PRIVATE_STRUCTURE;
    const auto& out_c = in->inputTensors[0].geometry;
    const auto& z = in->inputTensors[1].geometry;
    if (out_c.dataType != DATA_BF16 || out_c.dims != 3 || out_c.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (z.dataType != DATA_BF16 || z.dims != 2 || z.maxSizes[0] % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t C_chunk = out_c.maxSizes[1], BV = out_c.maxSizes[2];
    const uint64_t HV = z.maxSizes[0] / 128, T = z.maxSizes[1];
    if (!HV || HV > 64 || (HV & 3) || !C_chunk || T % C_chunk || (T / C_chunk) * HV != BV)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->inputTensors[2], DATA_BF16, {128})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_F8_143, {HV * 128, T}) ||
        !Is(in->outputTensors[1], DATA_F32, {1, T}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = T;
    out->inputTensorAccessPattern[0].allRequired = true;
    Map(out->inputTensorAccessPattern[1], 0, 0, 0, 0, static_cast<int>(HV * 128 - 1));
    Map(out->inputTensorAccessPattern[1], 1, 0, 1, 0, 0);
    out->inputTensorAccessPattern[2].allRequired = true;
    Map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, static_cast<int>(HV * 128 - 1));
    Map(out->outputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    Map(out->outputTensorAccessPattern[1], 0, 0, 0, 0, 0);
    Map(out->outputTensorAccessPattern[1], 1, 0, 1, 0, 0);
    out->kernel.paramsNr = 1;
    std::memcpy(&out->kernel.scalarParams[0], in->nodeParams.nodeParams, sizeof(float));
    return Elf(out, &_binary_gdn_prefill_gate_quant_bf16_gaudi2_o_start, &_binary_gdn_prefill_gate_quant_bf16_gaudi2_o_end);
}

GlueCodeReturn GdnPrefillDecayPrepBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 4) { in->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 9) { in->outputTensorNr = 9; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& kk = in->inputTensors[0].geometry;
    const auto& cum = in->inputTensors[2].geometry;
    if (kk.dataType != DATA_BF16 || kk.dims != 3 || kk.maxSizes[0] != 128 || kk.maxSizes[1] != 32)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (cum.dataType != DATA_F32 || cum.dims != 2 || cum.maxSizes[0] != 64) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t BV = cum.maxSizes[1], BK = kk.maxSizes[2];
    if (!BK || !BV || (BV % BK) != 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->inputTensors[1], DATA_BF16, {128, 32, BK}) ||
        !Is(in->inputTensors[3], DATA_F32, {64, BV}))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned o = 0; o < 4; ++o)
        if (!Is(in->outputTensors[o], DATA_BF16, {128, 32, BV})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    for (unsigned o = 4; o < 8; ++o)
        if (!Is(in->outputTensors[o], DATA_BF16, {64, BV})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (!Is(in->outputTensors[8], DATA_F32, {1, 1, BV})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = BV;
    out->inputTensorAccessPattern[0].allRequired = true;
    out->inputTensorAccessPattern[1].allRequired = true;
    for (unsigned i : {2u, 3u}) {
        Map(out->inputTensorAccessPattern[i], 0, 0, 0, 0, 63);
        Map(out->inputTensorAccessPattern[i], 1, 0, 1, 0, 0);
    }
    for (unsigned o = 0; o < 4; ++o) {
        Map(out->outputTensorAccessPattern[o], 0, 0, 0, 0, 127);
        Map(out->outputTensorAccessPattern[o], 1, 0, 0, 0, 31);
        Map(out->outputTensorAccessPattern[o], 2, 0, 1, 0, 0);
    }
    for (unsigned o = 4; o < 8; ++o) {
        Map(out->outputTensorAccessPattern[o], 0, 0, 0, 0, 63);
        Map(out->outputTensorAccessPattern[o], 1, 0, 1, 0, 0);
    }
    Map(out->outputTensorAccessPattern[8], 0, 0, 0, 0, 0);
    Map(out->outputTensorAccessPattern[8], 1, 0, 0, 0, 0);
    Map(out->outputTensorAccessPattern[8], 2, 0, 1, 0, 0);
    out->kernel.paramsNr = 0;
    return Elf(out, &_binary_gdn_prefill_decay_prep_bf16_gaudi2_o_start, &_binary_gdn_prefill_decay_prep_bf16_gaudi2_o_end);
}

GlueCodeReturn DSpineBridgeNormBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 4) { in->outputTensorNr = 4; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize < 2 * sizeof(float)) return GLUE_MISSING_PRIVATE_STRUCTURE;
    const auto& g0 = in->inputTensors[0].geometry;
    if (g0.dataType != DATA_BF16 || g0.dims != 2 || g0.maxSizes[0] == 0 || (g0.maxSizes[0] % 512) != 0 || g0.maxSizes[0] > 8192)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t H = g0.maxSizes[0], BT = g0.maxSizes[1];
    if (!BT || !Is(in->inputTensors[1], DATA_BF16, {H, BT})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_BF16, {H, BT}) ||
        !Is(in->outputTensors[1], DATA_BF16, {H, BT}) ||
        !Is(in->outputTensors[2], DATA_F32, {1, BT}) ||
        !Is(in->outputTensors[3], DATA_F32, {1, BT}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = BT;
    for (unsigned i = 0; i < 2; ++i) {
        Map(out->inputTensorAccessPattern[i], 0, 0, 0, 0, static_cast<int>(H - 1));
        Map(out->inputTensorAccessPattern[i], 1, 0, 1, 0, 0);
    }
    for (unsigned o = 0; o < 2; ++o) {
        Map(out->outputTensorAccessPattern[o], 0, 0, 0, 0, static_cast<int>(H - 1));
        Map(out->outputTensorAccessPattern[o], 1, 0, 1, 0, 0);
    }
    for (unsigned o = 2; o < 4; ++o) {
        Map(out->outputTensorAccessPattern[o], 0, 0, 0, 0, 0);
        Map(out->outputTensorAccessPattern[o], 1, 0, 1, 0, 0);
    }
    out->kernel.paramsNr = 2;
    std::memcpy(&out->kernel.scalarParams[0], in->nodeParams.nodeParams, 2 * sizeof(float));
    return Elf(out, &_binary_dspine_bridge_norm_bf16_gaudi2_o_start, &_binary_dspine_bridge_norm_bf16_gaudi2_o_end);
}

GlueCodeReturn DSpineBridgeCodesBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize < 2 * sizeof(float)) return GLUE_MISSING_PRIVATE_STRUCTURE;
    const auto& g0 = in->inputTensors[0].geometry;
    if (g0.dataType != DATA_BF16 || g0.dims != 2 || g0.maxSizes[0] == 0 || (g0.maxSizes[0] % 512) != 0 || g0.maxSizes[0] > 2048)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t C = g0.maxSizes[0], BT = g0.maxSizes[1];
    if (!BT || (BT % 16) != 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t B = BT / 16;
    if (!Is(in->inputTensors[1], DATA_BF16, {C, B})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_BF16, {C, BT})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = BT;
    Map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, static_cast<int>(C - 1));
    Map(out->inputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    out->inputTensorAccessPattern[1].allRequired = true;
    Map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, static_cast<int>(C - 1));
    Map(out->outputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    out->kernel.paramsNr = 2;
    std::memcpy(&out->kernel.scalarParams[0], in->nodeParams.nodeParams, 2 * sizeof(float));
    return Elf(out, &_binary_dspine_bridge_codes_bf16_gaudi2_o_start, &_binary_dspine_bridge_codes_bf16_gaudi2_o_end);
}

GlueCodeReturn DSpineBridgeGateBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& g0 = in->inputTensors[0].geometry;
    const auto& g1 = in->inputTensors[1].geometry;
    if (g0.dataType != DATA_BF16 || g0.dims != 2 || g1.dataType != DATA_BF16 || g1.dims != 1)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t R = g1.maxSizes[0], BT = g0.maxSizes[1];
    if (!R || (R % 256) != 0 || !BT || (BT % 16) != 0 || g0.maxSizes[0] != 3 * R)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_BF16, {R, BT})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = BT;
    out->inputTensorAccessPattern[0].allRequired = true;
    out->inputTensorAccessPattern[1].allRequired = true;
    Map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, static_cast<int>(R - 1));
    Map(out->outputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    out->kernel.paramsNr = 0;
    return Elf(out, &_binary_dspine_bridge_gate_bf16_gaudi2_o_start, &_binary_dspine_bridge_gate_bf16_gaudi2_o_end);
}

GlueCodeReturn DSpineBridgePostBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 4) { in->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize < 2 * sizeof(float)) return GLUE_MISSING_PRIVATE_STRUCTURE;
    const auto& g0 = in->inputTensors[0].geometry;
    if (g0.dataType != DATA_BF16 || g0.dims != 2 || g0.maxSizes[0] == 0 || (g0.maxSizes[0] % 512) != 0 || g0.maxSizes[0] > 8192)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t H = g0.maxSizes[0], BT = g0.maxSizes[1];
    if (!BT || (BT % 16) != 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->inputTensors[1], DATA_BF16, {H, BT}) ||
        !Is(in->inputTensors[2], DATA_F32, {1, BT}) ||
        !Is(in->inputTensors[3], DATA_BF16, {H}))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_BF16, {H, BT}) ||
        !Is(in->outputTensors[1], DATA_BF16, {H, BT}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = BT;
    for (unsigned i = 0; i < 2; ++i) {
        Map(out->inputTensorAccessPattern[i], 0, 0, 0, 0, static_cast<int>(H - 1));
        Map(out->inputTensorAccessPattern[i], 1, 0, 1, 0, 0);
    }
    Map(out->inputTensorAccessPattern[2], 0, 0, 0, 0, 0);
    Map(out->inputTensorAccessPattern[2], 1, 0, 1, 0, 0);
    out->inputTensorAccessPattern[3].allRequired = true;
    for (unsigned o = 0; o < 2; ++o) {
        Map(out->outputTensorAccessPattern[o], 0, 0, 0, 0, static_cast<int>(H - 1));
        Map(out->outputTensorAccessPattern[o], 1, 0, 1, 0, 0);
    }
    out->kernel.paramsNr = 2;
    std::memcpy(&out->kernel.scalarParams[0], in->nodeParams.nodeParams, 2 * sizeof(float));
    return Elf(out, &_binary_dspine_bridge_post_bf16_gaudi2_o_start, &_binary_dspine_bridge_post_bf16_gaudi2_o_end);
}

GlueCodeReturn DSpineQkvPrepBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 3) { in->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize < 2 * sizeof(float)) return GLUE_MISSING_PRIVATE_STRUCTURE;
    const auto& gqkv = in->inputTensors[0].geometry;
    const auto& gkb = in->outputTensors[1].geometry;
    const auto& gqsc = in->outputTensors[0].geometry;
    if (gqkv.dataType != DATA_BF16 || gqkv.dims != 2 || gqkv.maxSizes[1] == 0 || (gqkv.maxSizes[1] % 16) != 0)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t BT = gqkv.maxSizes[1], B = BT / 16;
    if (gkb.dataType != DATA_BF16 || gkb.dims != 3 || gkb.maxSizes[0] != 128 || gkb.maxSizes[1] != 16 ||
        gkb.maxSizes[2] == 0 || (gkb.maxSizes[2] % B) != 0)
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    const uint64_t BKV = gkb.maxSizes[2], KV = BKV / B;
    if (gqsc.dataType != DATA_BF16 || gqsc.dims != 3 || gqsc.maxSizes[0] != 128 || gqsc.maxSizes[2] != BKV ||
        gqsc.maxSizes[1] == 0 || (gqsc.maxSizes[1] % 16) != 0)
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    const uint64_t G = gqsc.maxSizes[1] / 16, NH = KV * G;
    if (gqkv.maxSizes[0] != (NH + 2 * KV) * 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->inputTensors[1], DATA_BF16, {128}) ||
        !Is(in->inputTensors[2], DATA_BF16, {128}) ||
        !Is(in->inputTensors[3], DATA_F32, {64, BT}) ||
        !Is(in->inputTensors[4], DATA_F32, {64, BT}))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[2], DATA_BF16, {128, 16, BKV})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = BT * KV;
    for (unsigned i = 0; i < 5; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    for (unsigned o = 0; o < 3; ++o) out->outputTensorAccessPattern[o].allRequired = true;
    out->kernel.paramsNr = 2;
    std::memcpy(&out->kernel.scalarParams[0], in->nodeParams.nodeParams, 2 * sizeof(float));
    return Elf(out, &_binary_dspine_qkv_prep_bf16_gaudi2_o_start, &_binary_dspine_qkv_prep_bf16_gaudi2_o_end);
}

GlueCodeReturn GdnTreePrepBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 3) { in->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& seq = in->inputTensors[0].geometry;
    const uint64_t C = seq.maxSizes[0], rows = seq.maxSizes[1];
    if (seq.dataType != DATA_BF16 || seq.dims != 2 || C % 128 || C <= 4096 || rows < 4)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t N = rows - 3;
    if (!Is(in->inputTensors[1], DATA_BF16, {C, 4}) || !Is(in->inputTensors[2], DATA_I32, {4, N}))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_F32, {128, 2 * N, 16}) || !Is(in->outputTensors[1], DATA_F32, {C - 4096, N}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = C / 128;
    for (unsigned i = 0; i < 3; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    for (unsigned o = 0; o < 2; ++o) out->outputTensorAccessPattern[o].allRequired = true;
    out->kernel.paramsNr = 0;
    return Elf(out, &_binary_gdn_tree_prep_bf16_gaudi2_o_start, &_binary_gdn_tree_prep_bf16_gaudi2_o_end);
}

GlueCodeReturn GdnTreeWyF32Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 6) { in->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* t = in->inputTensors;
    const auto& vg = t[4].geometry;
    if (vg.dataType != DATA_F32 || vg.dims != 3 || vg.maxSizes[0] != 128 || vg.maxSizes[1] != 48 ||
        vg.maxSizes[2] == 0 || vg.maxSizes[2] > 64)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t N = vg.maxSizes[2];
    if (!Is(t[0], DATA_F32, {48, N}) || !Is(t[1], DATA_F32, {48, N}) ||
        !Is(t[2], DATA_F32, {384, 2 * N, 16}) || !Is(t[3], DATA_F32, {2 * N, 2 * N, 16}) ||
        !Is(t[5], DATA_I32, {N}))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_F32, {128, 48, N}) ||
        !Is(in->outputTensors[1], DATA_F32, {128, 48, N}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = 48;
    for (unsigned i = 0; i < 6; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    for (unsigned o = 0; o < 2; ++o) {
        Map(out->outputTensorAccessPattern[o], 0, 0, 0, 0, 127);
        Map(out->outputTensorAccessPattern[o], 1, 0, 1, 0, 0);
        Map(out->outputTensorAccessPattern[o], 2, 0, 0, 0, static_cast<int>(N - 1));
    }
    out->kernel.paramsNr = 0;
    return Elf(out, &_binary_gdn_tree_wy_f32_gaudi2_o_start, &_binary_gdn_tree_wy_f32_gaudi2_o_end);
}

GlueCodeReturn GdnTreeCommitF32Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* t = in->inputTensors;
    const auto& sg = t[0].geometry;
    const auto& xg = t[1].geometry;
    const auto& dg = t[2].geometry;
    if (sg.dataType != DATA_F32 || sg.dims != 3 || sg.maxSizes[0] != 128 || sg.maxSizes[1] != 128 ||
        sg.maxSizes[2] == 0 || sg.maxSizes[2] % 48 != 0)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t NG = sg.maxSizes[2] / 48;
    if (dg.dataType != DATA_F32 || dg.dims != 4 || dg.maxSizes[0] != 128 || dg.maxSizes[1] != 48 ||
        dg.maxSizes[2] == 0 || dg.maxSizes[2] > 64 || dg.maxSizes[3] != NG)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t N = dg.maxSizes[2];
    if (xg.dataType != DATA_F32 || xg.dims != 4 || xg.maxSizes[0] != 128 ||
        (xg.maxSizes[1] != N && xg.maxSizes[1] != 2 * N) || xg.maxSizes[2] != 16 || xg.maxSizes[3] != NG)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(t[3], DATA_F32, {48, N, NG}) || !Is(t[4], DATA_I32, {17}))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_F32, {128, 128, NG * 48}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = NG * 48;
    Map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, 127);
    Map(out->inputTensorAccessPattern[0], 1, 0, 0, 0, 127);
    Map(out->inputTensorAccessPattern[0], 2, 0, 1, 0, 0);
    for (unsigned i = 1; i < 5; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    Map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, 127);
    Map(out->outputTensorAccessPattern[0], 1, 0, 0, 0, 127);
    Map(out->outputTensorAccessPattern[0], 2, 0, 1, 0, 0);
    out->kernel.paramsNr = 0;
    return Elf(out, &_binary_gdn_tree_commit_f32_gaudi2_o_start, &_binary_gdn_tree_commit_f32_gaudi2_o_end);
}
