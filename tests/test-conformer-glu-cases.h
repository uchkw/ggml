#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <string>
#include <vector>

struct conformer_glu_case {
    ggml_type type;
    int64_t width, frames;
    bool swap;
    const char * variant;
    bool fusable;
};

static std::vector<conformer_glu_case> conformer_glu_cases() {
    std::vector<conformer_glu_case> cases;
    for (bool swap : {false, true}) {
        for (int64_t frames : {1, 31, 32, 33, 171, 172, 375}) {
            for (const char * variant : {"plain", "large"}) {
                cases.push_back({GGML_TYPE_F32, 1024, frames, swap, variant, true});
            }
        }
        for (const char * variant : {"offset", "consumer", "alias_a", "alias_gate", "same_input"}) {
            cases.push_back({GGML_TYPE_F32, 1024, 33, swap, variant, true});
        }
        for (const char * variant : {"output", "reuse", "stride_gate", "stride_a", "batch", "broadcast", "silu",
                                    "partial_gate", "alias_unary", "unary_alias_a"}) {
            cases.push_back({GGML_TYPE_F32, 1024, 33, swap, variant, false});
        }
        cases.push_back({GGML_TYPE_F16, 1024, 33, swap, "plain", false});
        for (int64_t width : {1023, 1025}) cases.push_back({GGML_TYPE_F32, width, 33, swap, "plain", false});
    }
    return cases;
}

struct conformer_glu_graph {
    ggml_tensor * gate_base;
    ggml_tensor * a_base;
    ggml_tensor * gate;
    ggml_tensor * a;
    ggml_tensor * unary;
    ggml_tensor * mul;
    ggml_tensor * out;
};

/** The graph keeps two independent projections; no SiLU factor or projection packing is introduced. */
static conformer_glu_graph conformer_glu_build(ggml_context * ctx, const conformer_glu_case & c) {
    const std::string variant(c.variant);
    const int64_t batch = variant == "batch" ? 2 : 1;
    auto * gate_base = ggml_new_tensor_3d(ctx, c.type, c.width + (variant == "stride_gate" ? 4 : 0),
                                        c.frames + (variant == "offset" || variant == "partial_gate"), batch);
    auto * a_base = ggml_new_tensor_3d(ctx, c.type, c.width + (variant == "stride_a" ? 4 : 0),
                                     variant == "broadcast" ? 1 : c.frames + (variant == "offset"), batch);
    ggml_set_name(gate_base, "glu_gate_base");
    ggml_set_name(a_base, "glu_a_base");
    auto * gate = gate_base;
    auto * a = a_base;
    if (variant == "offset") {
        gate = ggml_view_2d(ctx, gate_base, c.width, c.frames, gate_base->nb[1], gate_base->nb[1]);
        a = ggml_view_2d(ctx, a_base, c.width, c.frames, a_base->nb[1], a_base->nb[1]);
    } else if (variant == "stride_gate" || variant == "partial_gate") {
        gate = ggml_view_2d(ctx, gate_base, c.width, c.frames, gate_base->nb[1], 0);
    } else if (variant == "stride_a") {
        a = ggml_view_2d(ctx, a_base, c.width, c.frames, a_base->nb[1], 0);
    }
    if (variant == "same_input") a = gate;
    auto * unary = ggml_unary(ctx, gate, variant == "silu" ? GGML_UNARY_OP_SILU : GGML_UNARY_OP_SIGMOID);
    if (variant == "unary_alias_a") {
        unary->view_src = a_base;
        unary->view_offs = 0;
    }
    ggml_set_name(unary, "glu_sigmoid");
    if (variant == "output") ggml_set_output(unary);
    // Broadcasting is valid only in src1, so this negative control always places the sigmoid first.
    const bool second = c.swap && variant != "broadcast";
    auto * mul = second ? ggml_mul(ctx, a, unary) : ggml_mul(ctx, unary, a);
    if (variant == "alias_gate" || variant == "partial_gate") {
        mul->view_src = gate_base;
        mul->view_offs = variant == "partial_gate" ? ggml_type_size(c.type) : 0;
    } else if (variant == "alias_a") {
        mul->view_src = a_base;
        mul->view_offs = 0;
    } else if (variant == "alias_unary") {
        mul->view_src = unary;
        mul->view_offs = 0;
    }
    ggml_set_name(mul, "glu_product");
    auto * out = variant == "consumer" ? ggml_scale(ctx, mul, 0.25f) :
                variant == "reuse" ? ggml_add(ctx, mul, unary) : mul;
    ggml_set_name(out, "glu_out");
    return {gate_base, a_base, gate, a, unary, mul, out};
}

/** Distinct inputs expose swapped roles and a missing sigmoid; large cases also exercise saturation and signed zero. */
static std::vector<float> conformer_glu_values(const ggml_tensor * tensor, bool gate, bool large) {
    std::vector<float> values((size_t) ggml_nelements(tensor));
    const float gates[] = {-1000, -89, -32, -0.001f, -0.0f, 0, 0.001f, 1, 16, 89, 1000};
    const float scales[] = {-1e10f, -7, -1e-10f, -0.0f, 0, 1e-10f, 3, 1e10f};
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = large ? (gate ? gates[i % 11] : scales[(i * 3 + 1) % 8]) :
            (gate ? std::sin((float) (i + 7) * 0.031f) * 6.0f : std::cos((float) (i + 37) * 0.017f) * 3.0f + 0.25f);
    }
    return values;
}

static void conformer_glu_write(ggml_tensor * tensor, const std::vector<float> & values, size_t offset = 0) {
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), offset, values.size()*sizeof(float));
    } else {
        GGML_ASSERT(tensor->type == GGML_TYPE_F16);
        std::vector<ggml_fp16_t> half(values.size());
        ggml_fp32_to_fp16_row(values.data(), half.data(), (int64_t) half.size());
        ggml_backend_tensor_set(tensor, half.data(), offset, half.size()*sizeof(ggml_fp16_t));
    }
}

static void conformer_glu_initialize(const conformer_glu_graph & graph, const conformer_glu_case & c) {
    const bool large = std::string(c.variant) == "large";
    conformer_glu_write(graph.gate_base, conformer_glu_values(graph.gate_base, true, large));
    conformer_glu_write(graph.a_base, conformer_glu_values(graph.a_base, false, large));
}
