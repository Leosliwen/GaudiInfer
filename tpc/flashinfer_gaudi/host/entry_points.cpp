// SPDX-License-Identifier: Apache-2.0
// Kernel database of the GaudiInfer TPC library: the GUIDs the engine launches and their glue.
#include <cstring>

#include "add_rmsnorm_quant_bf16_gaudi2.hpp"
#include "gated_norm_quant_bf16_gaudi2.hpp"
#include "gdn_packed_decode_f32_gaudi2.hpp"
#include "gdn_spec_prep_bf16_gaudi2.hpp"
#include "masked_softmax_bf16_gaudi2.hpp"
#include "silu_and_mul_bf16_gaudi2.hpp"
#include "silu_mul_quant_bf16_gaudi2.hpp"
#include "spec_conv_bf16_gaudi2.hpp"

namespace {
template <class K>
tpc_lib_api::GlueCodeReturn glue(tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance)
{
    return K{}.GetGcDefinitions(params, instance);
}
template <class K>
void static_name(char* out) { std::strcpy(out, K::name); }
template <class K>
void call_name(char* out) { K{}.GetKernelName(out); }

struct Entry {
    void (*name)(char*);
    tpc_lib_api::GlueCodeReturn (*glue)(tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
};

constexpr Entry kKernels[] = {
    {&static_name<AddRmsNormQuantBf16Gaudi2>, &glue<AddRmsNormQuantBf16Gaudi2>},
    {&static_name<AttnQkvPrepBf16Gaudi2>, &glue<AttnQkvPrepBf16Gaudi2>},
    {&static_name<AttnQkvPrepF8Gaudi2>, &glue<AttnQkvPrepF8Gaudi2>},
    {&static_name<DSpineBridgeCodesBf16Gaudi2>, &glue<DSpineBridgeCodesBf16Gaudi2>},
    {&static_name<DSpineBridgeGateBf16Gaudi2>, &glue<DSpineBridgeGateBf16Gaudi2>},
    {&static_name<DSpineBridgeNormBf16Gaudi2>, &glue<DSpineBridgeNormBf16Gaudi2>},
    {&static_name<DSpineBridgePostBf16Gaudi2>, &glue<DSpineBridgePostBf16Gaudi2>},
    {&static_name<DSpineQkvPrepBf16Gaudi2>, &glue<DSpineQkvPrepBf16Gaudi2>},
    {&static_name<GatedNormQuantBf16Gaudi2>, &glue<GatedNormQuantBf16Gaudi2>},
    {&call_name<GdnPackedDecodeF32Gaudi2>, &glue<GdnPackedDecodeF32Gaudi2>},
    {&static_name<GdnMtpWyo8F32Gaudi2>, &glue<GdnMtpWyo8F32Gaudi2>},
    {&static_name<GdnMtpWyoF32Gaudi2>, &glue<GdnMtpWyoF32Gaudi2>},
    {&static_name<GdnMtpWyod8F32Gaudi2>, &glue<GdnMtpWyod8F32Gaudi2>},
    {&static_name<GdnMtpWyodF32Gaudi2>, &glue<GdnMtpWyodF32Gaudi2>},
    {&static_name<GdnPrefillConvPrepBf16Gaudi2>, &glue<GdnPrefillConvPrepBf16Gaudi2>},
    {&static_name<GdnPrefillDecayPrepBf16Gaudi2>, &glue<GdnPrefillDecayPrepBf16Gaudi2>},
    {&static_name<GdnPrefillGateQuantBf16Gaudi2>, &glue<GdnPrefillGateQuantBf16Gaudi2>},
    {&static_name<GdnSpecPrepBatchBf16Gaudi2>, &glue<GdnSpecPrepBatchBf16Gaudi2>},
    {&static_name<GdnSpecPrepBf16Gaudi2>, &glue<GdnSpecPrepBf16Gaudi2>},
    {&static_name<GdnStateCommitF32Gaudi2>, &glue<GdnStateCommitF32Gaudi2>},
    {&static_name<GdnTreeCommitF32Gaudi2>, &glue<GdnTreeCommitF32Gaudi2>},
    {&static_name<GdnTreePrepBf16Gaudi2>, &glue<GdnTreePrepBf16Gaudi2>},
    {&static_name<GdnTreeWyF32Gaudi2>, &glue<GdnTreeWyF32Gaudi2>},
    {&static_name<MaskedSoftmaxBf16F8Gaudi2>, &glue<MaskedSoftmaxBf16F8Gaudi2>},
    {&static_name<MaskedSoftmaxBf16Gaudi2>, &glue<MaskedSoftmaxBf16Gaudi2>},
    {&static_name<MaskedSoftmaxF16Gaudi2>, &glue<MaskedSoftmaxF16Gaudi2>},
    {&static_name<RowQuantBf16Gaudi2>, &glue<RowQuantBf16Gaudi2>},
    {&static_name<RowSoftmaxStatsCmaxF32Gaudi2>, &glue<RowSoftmaxStatsCmaxF32Gaudi2>},
    {&static_name<SiluAndMulBf16Gaudi2>, &glue<SiluAndMulBf16Gaudi2>},
    {&static_name<SiluMulQuantBf16Gaudi2>, &glue<SiluMulQuantBf16Gaudi2>},
    {&static_name<SiluMulQuantDqBf16Gaudi2>, &glue<SiluMulQuantDqBf16Gaudi2>},
    {&static_name<SpecConvBf16Gaudi2>, &glue<SpecConvBf16Gaudi2>},
};
constexpr uint32_t kCount = sizeof(kKernels) / sizeof(kKernels[0]);
}  // namespace

extern "C" {
tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId deviceId, uint32_t* kernelCount,
                                           tpc_lib_api::GuidInfo* guids)
{
    if (kernelCount == nullptr) return tpc_lib_api::GLUE_FAILED;
    if (deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
        *kernelCount = 0;
        return tpc_lib_api::GLUE_SUCCESS;
    }
    const uint32_t capacity = *kernelCount;
    *kernelCount = kCount;
    if (guids == nullptr || capacity == 0) return tpc_lib_api::GLUE_SUCCESS;
    if (capacity < kCount) return tpc_lib_api::GLUE_FAILED;
    std::memset(guids, 0, kCount * sizeof(tpc_lib_api::GuidInfo));
    for (uint32_t i = 0; i < kCount; ++i) kKernels[i].name(guids[i].name);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(tpc_lib_api::HabanaKernelParams* params,
                                                 tpc_lib_api::HabanaKernelInstantiation* instance)
{
    if (!params || !instance) return tpc_lib_api::GLUE_FAILED;
    char name[tpc_lib_api::MAX_NODE_NAME];
    for (const auto& k : kKernels) {
        k.name(name);
        if (std::strcmp(params->guid.name, name) == 0) return k.glue(params, instance);
    }
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId, tpc_lib_api::ShapeInferenceParams*,
                                              tpc_lib_api::ShapeInferenceOutput*)
{
    return tpc_lib_api::GLUE_SUCCESS;
}
}  // extern "C"
