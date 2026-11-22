# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(flashinfer_gaudi_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(TPC_COMPILER tpc-clang REQUIRED)
find_program(ELF_OBJCOPY objcopy REQUIRED)
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE PATH "TPC compiler include directory")
set(HABANA_INCLUDE_DIR "/usr/include/habanalabs" CACHE PATH "Gaudi host API include directory")

set(TPC_EMBEDDED_OBJECTS)
set(WY_EXTRA "" CACHE STRING "extra tpc-clang flags for experiments")
separate_arguments(WY_EXTRA)
foreach(KERNEL
        add_rmsnorm_quant_bf16_gaudi2
        add_rmsnorm_quant_bf16_gaudi2_dq
        add_rmsnorm_quant_bf16_gaudi2_no_norm
        add_rmsnorm_quant_bf16_gaudi2_small
        attn_qkv_prep_bf16_gaudi2
        attn_qkv_prep_f8_gaudi2
        direct_attn_q_pack_bf16_gaudi2
        direct_attn_q_pack_f8_gaudi2
        direct_attn_tile_step_bf16_gaudi2
        direct_attn_tile_step_f8_gaudi2
        direct_attn_tree_tile_step_bf16_gaudi2
        direct_attn_tree_tile_step_f8_gaudi2
        direct_dq_attn_tree_prep_bf16_gaudi2
        direct_dq_attn_tree_prep_f8_gaudi2
        direct_gdn_dq_tree_prep_bf16_gaudi2
        direct_tree_feat_commit_bf16_gaudi2
        direct_tree_kv_commit_bf16_gaudi2
        direct_conv_commit_bf16_gaudi2
        direct_dq_add_rmsnorm_quant_bf16_gaudi2
        direct_dq_add_rmsnorm_quant_dual_bf16_gaudi2
        direct_dq_attn_qkv_prep_bf16_gaudi2
        direct_dq_attn_qkv_prep_f8_gaudi2
        direct_dq_finalize_bf16_gaudi2
        direct_dq_mul_quant_bf16_gaudi2
        direct_dq_silu_bf16_gaudi2
        direct_embed_norm_quant_bf16_gaudi2
        direct_gdn_ba_siluz_bf16_gaudi2
        direct_gdn_dq_spec_prep_bf16_gaudi2
        direct_gdn_norm_mul_quant_bf16_gaudi2
        direct_kv_scatter_bf16_gaudi2
        direct_kv_scatter_f8_gaudi2
        direct_sigmoid_mul_quant_bf16_gaudi2
        dspine_bridge_codes_bf16_gaudi2
        dspine_bridge_gate_bf16_gaudi2
        dspine_bridge_norm_bf16_gaudi2
        dspine_bridge_post_bf16_gaudi2
        dspine_qkv_prep_bf16_gaudi2
        gated_norm_quant_bf16_gaudi2
        gdn_mtp_wyo8_f32_gaudi2
        gdn_mtp_wyo_f32_gaudi2
        gdn_mtp_wyod8_f32_gaudi2
        gdn_mtp_wyod_f32_gaudi2
        gdn_packed_decode_f32_gaudi2
        gdn_prefill_conv_prep_bf16_gaudi2
        gdn_prefill_decay_prep_bf16_gaudi2
        gdn_prefill_gate_quant_bf16_gaudi2
        gdn_spec_prep_batch_bf16_gaudi2
        gdn_spec_prep_bf16_gaudi2
        gdn_state_commit_f32_gaudi2
        gdn_tree_commit_f32_gaudi2
        gdn_tree_prep_bf16_gaudi2
        gdn_tree_wy_f32_gaudi2
        masked_softmax_bf16_f8_gaudi2
        masked_softmax_bf16_gaudi2
        masked_softmax_f16_gaudi2
        row_quant_bf16_gaudi2
        row_softmax_stats_cmax_f32_gaudi2
        row_softmax_stats_f32_gaudi2
        silu_and_mul_bf16_gaudi2
        silu_mul_quant_bf16_gaudi2
        silu_mul_quant_dq_bf16_gaudi2
        spec_conv_bf16_gaudi2)
    set(TPC_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${KERNEL}.c")
    set(TPC_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}.o")
    set(TPC_EMBEDDED_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}_embedded.o")
    set(TPC_EXTRA_DEPENDENCIES)
    set(TPC_EXTRA_FLAGS)
    if(KERNEL MATCHES "^add_rmsnorm_quant_bf16_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/norm_math_gaudi2.h")
    endif()
    if(KERNEL STREQUAL "gdn_mtp_wy8_f32_gaudi2" OR KERNEL STREQUAL "gdn_mtp_wyo_f32_gaudi2" OR
       KERNEL STREQUAL "gdn_mtp_wyo8_f32_gaudi2" OR KERNEL STREQUAL "gdn_mtp_wyod_f32_gaudi2" OR
       KERNEL STREQUAL "gdn_mtp_wyod8_f32_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_mtp_wy_f32_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "masked_softmax_bf16m_gaudi2" OR KERNEL STREQUAL "masked_softmax_bf16_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/masked_softmax_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "gdn_spec_prep_batch_bf16_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_spec_prep_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "attn_qkv_prep_bf16_gaudi2" OR KERNEL STREQUAL "attn_qkv_prep_f8_gaudi2" OR
       KERNEL STREQUAL "direct_embed_norm_quant_bf16_gaudi2" OR
       KERNEL STREQUAL "direct_dq_add_rmsnorm_quant_bf16_gaudi2" OR
       KERNEL STREQUAL "direct_dq_add_rmsnorm_quant_dual_bf16_gaudi2" OR
       KERNEL STREQUAL "dspine_bridge_norm_bf16_gaudi2" OR
       KERNEL STREQUAL "dspine_bridge_codes_bf16_gaudi2" OR
       KERNEL STREQUAL "dspine_bridge_post_bf16_gaudi2" OR
       KERNEL STREQUAL "dspine_qkv_prep_bf16_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/norm_math_gaudi2.h")
    endif()
    if(KERNEL STREQUAL "attn_qkv_prep_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attn_qkv_prep_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "direct_attn_tile_step_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/direct_attn_tile_step_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "direct_attn_q_pack_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/direct_attn_q_pack_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "direct_dq_attn_qkv_prep_bf16_gaudi2" OR KERNEL STREQUAL "direct_dq_attn_qkv_prep_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/norm_math_gaudi2.h")
    endif()
    if(KERNEL STREQUAL "direct_dq_attn_qkv_prep_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/direct_dq_attn_qkv_prep_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "direct_kv_scatter_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/direct_kv_scatter_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "add_rmsnorm_quant_bf16_gaudi2_small" OR
       KERNEL STREQUAL "add_rmsnorm_quant_bf16_gaudi2_no_norm" OR
       KERNEL STREQUAL "add_rmsnorm_quant_bf16_gaudi2_dq")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_rmsnorm_quant_bf16_gaudi2.c")
    endif()
    add_custom_command(
        OUTPUT "${TPC_OBJECT}"
        COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2 ${TPC_EXTRA_FLAGS} ${WY_EXTRA}
                -I"${TPC_INCLUDE_DIR}" "${TPC_SOURCE}" -c -o "${TPC_OBJECT}"
        DEPENDS "${TPC_SOURCE}" ${TPC_EXTRA_DEPENDENCIES}
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${TPC_EMBEDDED_OBJECT}"
        COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
                "${KERNEL}.o" "${KERNEL}_embedded.o"
        DEPENDS "${TPC_OBJECT}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        VERBATIM
    )
    set_source_files_properties("${TPC_EMBEDDED_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    list(APPEND TPC_EMBEDDED_OBJECTS "${TPC_EMBEDDED_OBJECT}")
endforeach()

add_library(
    flashinfer_gaudi_kernels
    SHARED
    host/entry_points.cpp
    host/add_rmsnorm_quant_bf16_gaudi2.cpp
    host/gated_norm_quant_bf16_gaudi2.cpp
    host/gdn_packed_decode_f32_gaudi2.cpp
    host/gdn_spec_prep_bf16_gaudi2.cpp
    host/masked_softmax_bf16_gaudi2.cpp
    host/silu_and_mul_bf16_gaudi2.cpp
    host/silu_mul_quant_bf16_gaudi2.cpp
    host/spec_conv_bf16_gaudi2.cpp
    ${TPC_EMBEDDED_OBJECTS}
)
target_include_directories(
    flashinfer_gaudi_kernels
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/host"
    "${TPC_INCLUDE_DIR}"
    "${HABANA_INCLUDE_DIR}"
)
target_compile_options(flashinfer_gaudi_kernels PRIVATE -Wall -Wextra -Werror -fno-strict-aliasing)
target_link_options(flashinfer_gaudi_kernels PRIVATE -Wl,-z,noexecstack)
