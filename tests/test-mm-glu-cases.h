#pragma once

// Shared case table, graph fixture and deterministic initialization for the planned Metal
// gate-projection epilogue: MUL_MAT -> UNARY(SIGMOID) -> MUL computes a * sigmoid(W_gate*x)
// inside the gate mat-mul kernel on simdgroup matrix hardware. The factor stays independent
// (never SiLU, never a concatenated weight, never a fused pair of projections), and sigmoid
// must round as F32 before the separate multiplication.
//
// Both test-backend-ops.cpp (cross-backend numerical comparison of the public gate product,
// sigmoid and final output) and the Metal-only test-metal-mm-glu.cpp (exact fusion-count and
// bit-exactness expectations) build their graphs from this single description.

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

// The positive gate: F32/F16/Q8_0/Q4_K weights, width == K == 1024, 17 <= frames <= 384,
// contiguous rank-2 F32 activation/factor/intermediate/output, both mul operand orders.
struct mm_glu_case {
    ggml_type type;     // gate weight type
    int64_t width;      // gate projection rows (M)
    int64_t k;          // contraction dimension
    int64_t frames;     // activation columns (T)
    bool swap;          // false: mul(sigmoid, a); true: mul(a, sigmoid)
    const char * variant;
    bool fusable;       // whether the planned MUL_MAT+UNARY+MUL fusion may apply
};

static std::vector<mm_glu_case> mm_glu_cases() {
    std::vector<mm_glu_case> cases;
    const std::vector<ggml_type> types = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K};
    const std::vector<int64_t> positive_frames = {17, 31, 32, 33, 63, 64, 65, 171, 172, 375, 376, 384};
    const std::vector<int64_t> negative_frames = {1, 8, 16, 385};
    for (ggml_type type : types) {
        for (bool swap : {false, true}) {
            for (int64_t frames : positive_frames) cases.push_back({type, 1024, 1024, frames, swap, "plain", true});
            for (const char * variant : {"large", "offset", "consumer", "alias_factor"}) {
                cases.push_back({type, 1024, 1024, 33, swap, variant, true});
            }
            for (const char * variant : {"stride_x", "transpose_x", "stride_a", "batch", "silu", "gate_output",
                                         "gate_reuse", "sigmoid_output", "sigmoid_reuse", "a_is_gate", "inplace_public",
                                         "gate_overwrites_a", "sigmoid_overwrites_a", "alias_x", "alias_x_partial"}) {
                cases.push_back({type, 1024, 1024, 33, swap, variant, false});
            }
            for (int64_t frames : negative_frames) cases.push_back({type, 1024, 1024, frames, swap, "plain", false});
            cases.push_back({type, 1023, 1024, 33, swap, "plain", false});
            cases.push_back({type, 1025, 1024, 33, swap, "plain", false});
            cases.push_back({type, 1024, 1280, 33, swap, "plain", false});
        }
        // Broadcasting is valid only in src1, so the broadcast factor always follows the sigmoid.
        cases.push_back({type, 1024, 1024, 33, false, "broadcast", false});
        // The encoder traversal order (factor projection first, then the candidate triple) fixes
        // the mul operand order; a swapped two-projection graph interleves the factor projection
        // between the sigmoid and the mul and is covered by the ordinary swapped cases.
        cases.push_back({type, 1024, 1024, 33, false, "twoproj", true});
        if (type == GGML_TYPE_F32) {
            // Safe, valid unfused F32 views: only an F32 weight can back an F32 output alias.
            for (bool swap : {false, true}) {
                for (const char * variant : {"alias_w", "alias_w_offset"}) {
                    cases.push_back({GGML_TYPE_F32, 1024, 1024, 33, swap, variant, false});
                }
                // Explicit F32 source-operand precision (src[1] must stay F32): an ordinary
                // graph with the usual comparison bound, but the planned kernel gate accepts
                // only the default (undefined) source precision, so nothing may fuse.
                cases.push_back({GGML_TYPE_F32, 1024, 1024, 33, swap, "src_f32", false});
            }
        }
    }
    // BF16 weights are outside the planned gate.
    for (bool swap : {false, true}) cases.push_back({GGML_TYPE_BF16, 1024, 1024, 33, swap, "plain", false});
    return cases;
}

/** Cross-backend cases keep the ordinary in-memory semantics; raw output aliases, buffers
 *  overwritten through view_src and the Metal mapped wrappers stay in the Metal runner,
 *  which controls allocation explicitly. */
static bool mm_glu_case_is_cross_backend(const mm_glu_case & c) {
    const std::string variant(c.variant);
    return variant.rfind("alias", 0) != 0 && variant.rfind("mapped_", 0) != 0 &&
           variant != "gate_overwrites_a" && variant != "sigmoid_overwrites_a";
}

struct mm_glu_graph {
    ggml_tensor * w_base;   // gate weight (k x width, case type)
    ggml_tensor * wa;       // factor projection weight, twoproj only
    ggml_tensor * x_base;   // activation storage (F32, k-major)
    ggml_tensor * a_base;   // independent factor storage (F32), null for twoproj
    ggml_tensor * x;        // activation consumed by the mat-muls (view or transpose)
    ggml_tensor * a;        // factor consumed by the final mul (view for strided factors)
    ggml_tensor * a_mm;     // factor projection product, twoproj only
    ggml_tensor * mm;       // gate projection product
    ggml_tensor * sigmoid;  // unary on the gate product
    ggml_tensor * mul;      // a * sigmoid(gate product)
    ggml_tensor * out;      // graph output: the mul or a later consumer of it
};

/** The graph keeps the encoder.cpp traversal order: an optional factor projection first, then
 *  W_gate*x -> sigmoid -> mul as three consecutive nodes. The factor is an independent input
 *  (or the Wa projection), never packed into the gate weight. */
static mm_glu_graph mm_glu_build(ggml_context * ctx, const mm_glu_case & c) {
    const std::string variant(c.variant);
    const int64_t batch = variant == "batch" ? 2 : 1;
    ggml_tensor * x_base;
    ggml_tensor * x;
    if (variant == "transpose_x") {
        x_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.frames, c.k);
        x = ggml_transpose(ctx, x_base);
    } else {
        // one spare row keeps every later view (offset, alias_x_partial) inside the allocation
        const int64_t spare = variant == "offset" || variant == "alias_x_partial" ? 1 : 0;
        x_base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k + (variant == "stride_x" ? 4 : 0),
                                    c.frames + spare, batch);
        x = x_base;
        if (variant == "offset" || variant == "alias_x_partial") {
            x = ggml_view_2d(ctx, x_base, c.k, c.frames, x_base->nb[1],
                             variant == "offset" ? x_base->nb[1] : 0);
        } else if (variant == "stride_x") {
            x = ggml_view_2d(ctx, x_base, c.k, c.frames, x_base->nb[1], 0);
        }
    }
    ggml_set_name(x_base, "glu_x_base");
    ggml_set_name(x, "glu_x");

    auto * w_base = ggml_new_tensor_2d(ctx, c.type, c.k, c.width);
    ggml_set_name(w_base, "glu_w_base");

    ggml_tensor * wa = nullptr;
    ggml_tensor * a_base = nullptr;
    ggml_tensor * a = nullptr;
    if (variant == "twoproj") {
        wa = ggml_new_tensor_2d(ctx, c.type, c.k, c.width);
        ggml_set_name(wa, "glu_wa");
    } else {
        a_base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.width + (variant == "stride_a" ? 4 : 0),
                                    variant == "broadcast" ? 1 : c.frames + (variant == "offset"), batch);
        ggml_set_name(a_base, "glu_a_base");
        a = a_base;
        if (variant == "offset") {
            a = ggml_view_2d(ctx, a_base, c.width, c.frames, a_base->nb[1], a_base->nb[1]);
        } else if (variant == "stride_a") {
            a = ggml_view_2d(ctx, a_base, c.width, c.frames, a_base->nb[1], 0);
        }
        if (a != a_base) {
            ggml_set_name(a, "glu_a");
        }
    }

    // the real family requests F32 accumulation for both projections; the source precision
    // stays at its default so the planned epilogue gate (default source precision) applies
    auto * a_mm = variant == "twoproj" ? ggml_mul_mat(ctx, wa, x) : nullptr;
    if (a_mm) {
        ggml_prec_set_acc(a_mm, GGML_PREC_F32);
    }
    auto * mm = ggml_mul_mat(ctx, w_base, x);
    ggml_set_name(mm, "glu_gate_mm");
    ggml_prec_set_acc(mm, GGML_PREC_F32);
    if (variant == "src_f32") {
        ggml_prec_set_src(mm, GGML_PREC_F32, 1);
    }
    auto * sigmoid = variant == "inplace_public" ? ggml_sigmoid_inplace(ctx, mm) :
                     variant == "silu" ? ggml_silu(ctx, mm) : ggml_sigmoid(ctx, mm);
    ggml_set_name(sigmoid, "glu_sigmoid");
    if (variant == "gate_output") ggml_set_output(mm);
    if (variant == "sigmoid_output" || variant == "inplace_public") ggml_set_output(sigmoid);

    auto * factor = variant == "twoproj" ? a_mm : variant == "a_is_gate" ? mm : a;
    auto * mul = c.swap ? ggml_mul(ctx, factor, sigmoid) : ggml_mul(ctx, sigmoid, factor);
    ggml_set_name(mul, "glu_mul");
    if (variant == "alias_factor" || variant == "alias_x" || variant == "alias_x_partial" ||
        variant == "alias_w" || variant == "alias_w_offset") {
        // The final output may alias an input; ggml's fusion checks only skip the view source
        // of the primary output, so these stay deliberate expected/non-fusing controls.
        mul->view_src = variant == "alias_factor" ? a_base : variant == "alias_w" || variant == "alias_w_offset" ? w_base : x_base;
        mul->view_offs = variant == "alias_x_partial" || variant == "alias_w_offset" ? 4*sizeof(float) : 0;
    } else if (variant == "gate_overwrites_a") {
        mm->view_src = a_base;
        mm->view_offs = 0;
    } else if (variant == "sigmoid_overwrites_a") {
        sigmoid->view_src = a_base;
        sigmoid->view_offs = 0;
    }

    auto * out = variant == "consumer" ? ggml_scale(ctx, mul, 0.5f) :
                 variant == "gate_reuse" ? ggml_add(ctx, mul, mm) :
                 variant == "sigmoid_reuse" ? ggml_add(ctx, mul, sigmoid) : mul;
    ggml_set_name(out, "glu_out");
    return {w_base, wa, x_base, a_base, x, a, a_mm, mm, sigmoid, mul, out};
}

// ---- deterministic initialization -----------------------------------------

// Ordinary values keep every matrix product finite for all weight types (no F16 operand
// overflow); the large fixture saturates the sigmoid through a sparse diagonal gate weight
// (so dense sums of huge values never form) and gives the factor finite large magnitudes
// and signed zeros.
static float mm_glu_weight_value(int64_t row, int64_t col, bool large) {
    if (large) {
        return row == col ? (row % 2 ? -8.0f : 8.0f) : 0.0f;
    }
    return (float) (((row * 73 + col * 127) % 25) - 12) / 48.0f;
}

static float mm_glu_x_value(int64_t index, bool large) {
    if (large) {
        const float table[] = {-1000.0f, -89.0f, -32.0f, -0.001f, -0.0f, 0.0f, 0.001f, 1.0f, 16.0f, 89.0f, 1000.0f};
        return table[index % 11];
    }
    return std::sin((float) index * 0.031f + 0.7f) * 1.5f;
}

static float mm_glu_a_value(int64_t index, bool large) {
    if (large) {
        const float table[] = {-1e10f, -7.0f, -1e-10f, -0.0f, 0.0f, 1e-10f, 3.0f, 1e10f};
        return table[index % 8];
    }
    return std::cos((float) index * 0.017f + 1.3f) * 2.0f + 0.25f;
}

/** Writes a full weight tensor (all rows) from deterministic F32 values through the type's
 *  exact storage: raw F32, fp16/bf16 row conversion, or ggml_quantize_chunk. */
static void mm_glu_write_weight(ggml_tensor * w, bool large, int64_t phase) {
    const int64_t rows = w->ne[1];
    const int64_t cols = w->ne[0];
    std::vector<float> values((size_t) (rows * cols));
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            values[(size_t) (row * cols + col)] = mm_glu_weight_value(row + phase, col, large);
        }
    }
    if (w->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(w, values.data(), 0, values.size() * sizeof(float));
    } else if (w->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(values.size());
        ggml_fp32_to_fp16_row(values.data(), half.data(), (int64_t) half.size());
        ggml_backend_tensor_set(w, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
    } else if (w->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> bf16(values.size());
        ggml_fp32_to_bf16_row(values.data(), bf16.data(), (int64_t) bf16.size());
        ggml_backend_tensor_set(w, bf16.data(), 0, bf16.size() * sizeof(ggml_bf16_t));
    } else {
        std::vector<uint8_t> packed(ggml_nbytes(w));
        const size_t written = ggml_quantize_chunk(w->type, values.data(), packed.data(), 0, rows, cols, nullptr);
        GGML_ASSERT(written == packed.size());
        ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    }
}

/** Distinct deterministic inputs for one case; identical in both runners and on every run,
 *  so disabled/enabled comparisons and cross-build binaries share one input set. */
static void mm_glu_initialize(const mm_glu_graph & g, const mm_glu_case & c) {
    const bool large = std::string(c.variant) == "large";
    mm_glu_write_weight(g.w_base, large, 0);
    if (g.wa) {
        mm_glu_write_weight(g.wa, false, 511);
    }
    std::vector<float> x_values((size_t) ggml_nelements(g.x_base));
    for (size_t i = 0; i < x_values.size(); ++i) x_values[i] = mm_glu_x_value((int64_t) i, large);
    ggml_backend_tensor_set(g.x_base, x_values.data(), 0, x_values.size() * sizeof(float));
    if (g.a_base) {
        std::vector<float> a_values((size_t) ggml_nelements(g.a_base));
        for (size_t i = 0; i < a_values.size(); ++i) a_values[i] = mm_glu_a_value((int64_t) i, large);
        ggml_backend_tensor_set(g.a_base, a_values.data(), 0, a_values.size() * sizeof(float));
    }
}
