#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

struct dw_silu_case {
    int64_t channels;
    int64_t frames;
    ggml_type weight_type;
    const char * variant;
    bool fusable;
    bool cpu_reference;
};

static std::vector<dw_silu_case> dw_silu_cases() {
    std::vector<dw_silu_case> cases;
    for (int64_t frames : {1, 2, 3, 4, 5, 8, 9, 17, 31, 32, 33, 171, 375}) {
        cases.push_back({1024, frames, GGML_TYPE_F32, "plain", true, true});
    }
    for (int64_t channels : {31, 32, 33, 255, 256, 257}) {
        cases.push_back({channels, 33, GGML_TYPE_F32, "plain", true, true});
    }
    for (const char * variant : {"offset", "consumer", "signed", "wide"}) {
        cases.push_back({1024, 33, GGML_TYPE_F32, variant, true, true});
    }
    for (int64_t frames : {1, 33, 171}) {
        cases.push_back({1024, frames, GGML_TYPE_F16, "plain", false, true});
    }
    for (const char * variant : {"relu", "product_output", "permuted_output", "reshape_output", "sum_output",
                                 "reuse_product", "reuse_permuted", "reuse_sum",
                                 "inplace_output", "input_alias", "partial_input_alias", "weight_alias",
                                 "bias_alias", "wrong_reshape", "bias_full", "batch", "stride", "dilation",
                                 "kernel5", "no_reshape", "strided_bias", "strided_output"}) {
        cases.push_back({1024, 33, GGML_TYPE_F32, variant, false, true});
    }
    // The CPU channel-fastest convolution reads packed rows without consulting the input strides.
    cases.push_back({1024, 33, GGML_TYPE_F32, "strided_input", false, false});
    cases.push_back({1, 33, GGML_TYPE_F32, "plain", false, true});
    for (const char * variant : {"mapped_input", "mapped_partial_input", "mapped_kernel", "mapped_partial_kernel",
                                 "mapped_bias", "mapped_partial_bias"}) {
        cases.push_back({1024, 33, GGML_TYPE_F32, variant, false, false});
    }
    cases.push_back({1024, 33, GGML_TYPE_F32, "mapped_disjoint", true, false});
    for (const char * variant : {"mapped_conv_bias_same", "mapped_partial_conv_bias_same",
                                 "mapped_conv_bias_cross", "mapped_partial_conv_bias_cross"}) {
        cases.push_back({1024, 33, GGML_TYPE_F32, variant, false, false});
    }
    return cases;
}

/** The actual convolution, permutation, reshape, bias and activation chain, with explicit rejection variants. */
struct dw_silu_graph {
    ggml_tensor * input_storage;
    ggml_tensor * weight_storage;
    ggml_tensor * bias_storage;
    ggml_tensor * input;
    ggml_tensor * kernel;
    ggml_tensor * bias;
    ggml_tensor * conv;
    ggml_tensor * permuted;
    ggml_tensor * reshaped;
    ggml_tensor * sum;
    ggml_tensor * activation;
    ggml_tensor * out;
};

static dw_silu_graph make_dw_silu_graph(ggml_context * ctx, const dw_silu_case & c) {
    const std::string variant(c.variant);
    const int64_t channels = c.channels, frames = c.frames;
    const int64_t taps = variant == "kernel5" ? 5 : 9;
    const int64_t batch = variant == "batch" ? 2 : 1;
    const int stride = variant == "stride" ? 2 : 1;
    const int dilation = variant == "dilation" ? 2 : 1;
    const int padding = (int) (taps - 1)*dilation/2;
    const bool offset = variant == "offset";
    const int64_t input_channels = channels + (variant == "strided_input" ? 4 : 0);
    const int64_t input_frames = frames + (offset || variant == "partial_input_alias" ? 1 : 0);
    dw_silu_graph g{};
    g.input_storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, input_channels, input_frames, 1, batch);
    auto * input = ggml_view_4d(ctx, g.input_storage, channels, frames, 1, batch,
                              g.input_storage->nb[1], g.input_storage->nb[2], g.input_storage->nb[3],
                              offset ? channels*sizeof(float) : 0);
    g.input = ggml_permute(ctx, input, 2, 0, 1, 3);
    const int64_t stored_taps = variant == "weight_alias" ? std::max(taps, frames + 1) : taps + offset;
    g.weight_storage = ggml_new_tensor_4d(ctx, c.weight_type, channels, 1, stored_taps, 1);
    auto * packed = ggml_view_4d(ctx, g.weight_storage, channels, 1, taps, 1,
                               g.weight_storage->nb[1], g.weight_storage->nb[2], taps*channels*ggml_type_size(c.weight_type),
                               offset ? channels*ggml_type_size(c.weight_type) : 0);
    g.kernel = ggml_permute(ctx, packed, 3, 2, 0, 1);
    g.conv = ggml_conv_2d_dw_direct(ctx, g.kernel, g.input, stride, 1, padding, 0, dilation, 1);
    g.permuted = ggml_permute(ctx, g.conv, 1, 2, 0, 3);
    const int64_t columns = variant == "wrong_reshape" ? g.conv->ne[0] : channels;
    const int64_t rows = ggml_nelements(g.conv)/columns;
    g.reshaped = variant == "no_reshape" ? g.permuted : ggml_reshape_2d(ctx, g.permuted, columns, rows);
    const int64_t bias_count = variant == "bias_alias" ? columns*rows : columns + offset;
    g.bias_storage = variant == "bias_full" || variant == "strided_bias" ?
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns + (variant == "strided_bias" ? 4 : 0), rows) :
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bias_count);
    g.bias = variant == "bias_full" ? g.bias_storage : variant == "strided_bias" ?
        ggml_view_2d(ctx, g.bias_storage, columns, rows, g.bias_storage->nb[1], 0) :
        ggml_view_1d(ctx, g.bias_storage, columns, offset ? sizeof(float) : 0);
    g.sum = ggml_add(ctx, g.reshaped, g.bias);
    g.activation = variant == "inplace_output" ? ggml_silu_inplace(ctx, g.sum) :
                   variant == "relu" ? ggml_relu(ctx, g.sum) : ggml_silu(ctx, g.sum);
    if (variant == "input_alias" || variant == "partial_input_alias") {
        g.activation->view_src = g.input_storage;
        g.activation->view_offs = variant == "partial_input_alias" ? sizeof(float) : 0;
    } else if (variant == "weight_alias") {
        g.activation->view_src = g.weight_storage;
        g.activation->view_offs = 0;
    } else if (variant == "bias_alias") {
        g.activation->view_src = g.bias_storage;
        g.activation->view_offs = 0;
    } else if (variant == "strided_output") {
        g.activation->view_src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns + 4, rows);
        g.activation->view_offs = 0;
        g.activation->nb[1] = (columns + 4)*sizeof(float);
        g.activation->nb[2] = g.activation->nb[3] = rows*g.activation->nb[1];
    }
    g.out = g.activation;
    if (variant == "consumer") g.out = ggml_scale(ctx, g.out, 0.5f);
    if (variant == "reuse_product") g.out = ggml_add(ctx, g.out, g.reshaped);
    if (variant == "reuse_permuted") g.out = ggml_add(ctx, g.out, g.permuted);
    if (variant == "reuse_sum") g.out = ggml_add(ctx, g.out, g.sum);
    if (variant == "product_output") ggml_set_output(g.conv);
    if (variant == "permuted_output") ggml_set_output(g.permuted);
    if (variant == "reshape_output") ggml_set_output(g.reshaped);
    if (variant == "sum_output") ggml_set_output(g.sum);
    ggml_set_output(g.out);
    return g;
}

static void initialize_dw_silu_tensor(ggml_tensor * tensor, uint32_t seed, float scale, float offset) {
    std::vector<float> data((size_t) ggml_nelements(tensor));
    for (float & value : data) {
        seed = 1664525u*seed + 1013904223u;
        value = offset + scale*((int32_t) (seed >> 16) - 32768)/32768.0f;
    }
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    const size_t size = ggml_quantize_chunk(tensor->type, data.data(), bytes.data(), 0,
                                          ggml_nrows(tensor), tensor->ne[0], nullptr);
    GGML_ASSERT(size == bytes.size());
    ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
}

/** Positive alias inputs differ from every expected activation, exposing a missing store after restoration. */
static void initialize_dw_silu_graph(const dw_silu_graph & g, const dw_silu_case & c) {
    const std::string variant(c.variant);
    initialize_dw_silu_tensor(g.input_storage, 123456789u, variant == "wide" ? 35000.0f : 0.5f,
                             variant == "wide" ? 105000.0f : 1.5f);
    initialize_dw_silu_tensor(g.weight_storage, 362436069u, variant == "wide" ? 0.00000005f :
                             variant == "signed" ? 0.5f : 0.0005f,
                             variant == "wide" ? 0.00000015f : variant == "signed" ? 0.0f : 0.0015f);
    // Mapped partial aliases also need every neighboring bias to exceed every expected activation.
    initialize_dw_silu_tensor(g.bias_storage, 521288629u, variant.find("mapped_") == 0 ? 0.03125f : 0.125f, 0.375f);
}
