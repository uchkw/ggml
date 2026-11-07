#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

struct head_store_case {
    int64_t frames;
    ggml_type type;
    const char * variant;
    bool fusable;
    bool cpu_reference;
};

static std::vector<head_store_case> head_store_cases() {
    std::vector<head_store_case> cases;
    for (int64_t frames : {64, 65, 95, 96, 127, 128, 129, 171, 375}) {
        cases.push_back({frames, GGML_TYPE_F32, "plain", true, true});
    }
    for (const char * variant : {"offset", "consumer", "cont_consumer", "cont_output", "no_reshape"}) {
        cases.push_back({171, GGML_TYPE_F32, variant, true, true});
    }
    for (int64_t frames : {1, 8, 9, 16, 17, 31, 32, 33, 63}) {
        cases.push_back({frames, GGML_TYPE_F32, "plain", false, true});
    }
    for (const char * variant : {"heads7", "width127", "width129", "k_ne_n", "batch", "broadcast",
                                 "strided_a", "strided_b", "wrong_permute", "f32_src", "mm_output",
                                 "perm_output", "reuse_mm", "reuse_perm", "no_cont", "strided_cont",
                                 "alias_a", "partial_alias_a", "alias_b", "partial_alias_b"}) {
        cases.push_back({171, GGML_TYPE_F32, variant, false, true});
    }
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K}) {
        cases.push_back({256, type, "plain", false, true});
    }
    cases.push_back({171, GGML_TYPE_F32, "padded_head_a", false, true});
    for (const char * variant : {"mapped_a", "mapped_partial_a", "mapped_b", "mapped_partial_b", "mapped_disjoint"}) {
        cases.push_back({171, GGML_TYPE_F32, variant, false, false});
    }
    return cases;
}

struct head_store_graph {
    ggml_tensor * a_storage;
    ggml_tensor * b_storage;
    ggml_tensor * a;
    ggml_tensor * b;
    ggml_tensor * mm;
    ggml_tensor * perm;
    ggml_tensor * cont;
    ggml_tensor * out;
};

/** The attention product and head permutation keep the subsequent reshape as an ordinary view. */
static head_store_graph make_head_store_graph(ggml_context * ctx, const head_store_case & c) {
    const std::string variant(c.variant);
    const int64_t d = variant == "width127" ? 127 : variant == "width129" ? 129 : 128;
    const int64_t h = variant == "heads7" ? 7 : 8;
    const int64_t batch = variant == "batch" ? 2 : 1;
    const int64_t ah = variant == "broadcast" ? 1 : h;
    const int64_t k = c.frames + (variant == "k_ne_n" ? 1 : 0);
    const int64_t ak = k + (variant == "strided_a" ? 4 : 0);
    const int64_t bk = k + (variant == "strided_b" ? 4 : 0);
    const int64_t a_head = ak*d + (variant == "padded_head_a" ? 4 : 0);
    const size_t offset = variant == "offset" ? 4*sizeof(float) : 0;
    const int64_t extra_a = ggml_blck_size(c.type) > 1 ? ggml_blck_size(c.type) : 8;
    head_store_graph g{};
    g.a_storage = ggml_new_tensor_1d(ctx, c.type, a_head*ah*batch + extra_a);
    g.b_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bk*c.frames*h*batch + 8);
    g.a = ggml_view_4d(ctx, g.a_storage, k, d, ah, batch, ggml_row_size(c.type, ak),
                     ggml_row_size(c.type, a_head), ggml_row_size(c.type, a_head)*ah, offset);
    g.b = ggml_view_4d(ctx, g.b_storage, k, c.frames, h, batch, bk*sizeof(float),
                      bk*c.frames*sizeof(float), bk*c.frames*h*sizeof(float), offset);
    g.mm = ggml_mul_mat(ctx, g.a, g.b);
    ggml_prec_set_acc(g.mm, GGML_PREC_F32);
    if (variant == "f32_src") ggml_prec_set_src(g.mm, GGML_PREC_F32, 1);
    g.perm = variant == "wrong_permute" ? ggml_permute(ctx, g.mm, 1, 0, 2, 3) :
                                         ggml_permute(ctx, g.mm, 0, 2, 1, 3);
    if (variant == "no_cont") {
        g.out = g.perm;
    } else {
        ggml_tensor * cont_storage = nullptr;
        if (variant == "offset") {
            cont_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(g.perm) + 8);
        } else if (variant == "strided_cont") {
            cont_storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d + 4, h, c.frames, batch);
        }
        g.cont = ggml_cont(ctx, g.perm);
        if (variant == "alias_a" || variant == "partial_alias_a") {
            g.cont->view_src = g.a_storage;
            g.cont->view_offs = variant == "alias_a" ? 0 : 4*sizeof(float);
        } else if (variant == "alias_b" || variant == "partial_alias_b") {
            g.cont->view_src = g.b_storage;
            g.cont->view_offs = variant == "alias_b" ? 0 : 4*sizeof(float);
        } else if (variant == "offset") {
            g.cont->view_src = cont_storage;
            g.cont->view_offs = offset;
        } else if (variant == "strided_cont") {
            g.cont->view_src = cont_storage;
            g.cont->view_offs = 0;
            g.cont->nb[1] = (d + 4)*sizeof(float);
            g.cont->nb[2] = g.cont->nb[1]*h;
            g.cont->nb[3] = g.cont->nb[2]*c.frames;
        }
        g.out = variant == "strided_cont" || variant == "no_reshape" ? g.cont :
            ggml_reshape_2d(ctx, g.cont, d*h, c.frames*batch);
        if (variant == "consumer") g.out = ggml_scale(ctx, g.out, 0.5f);
        if (variant == "cont_consumer") g.out = ggml_add(ctx, g.out, ggml_reshape_2d(ctx, g.cont, d*h, c.frames*batch));
        if (variant == "reuse_mm") g.out = ggml_add(ctx, g.out, ggml_reshape_2d(ctx, g.mm, d*h, c.frames*batch));
        if (variant == "reuse_perm") g.out = ggml_add(ctx, g.out, ggml_reshape_2d(ctx, ggml_cont(ctx, g.perm), d*h, c.frames*batch));
    }
    if (variant == "mm_output") ggml_set_output(g.mm);
    if (variant == "perm_output") ggml_set_output(g.perm);
    if (variant == "cont_output") ggml_set_output(g.cont);
    ggml_set_output(g.out);
    return g;
}

static void initialize_head_store_tensor(ggml_tensor * tensor, uint32_t seed, float scale, float offset) {
    std::vector<float> data((size_t) ggml_nelements(tensor));
    for (float & value : data) {
        seed = 1664525u*seed + 1013904223u;
        value = offset + scale*((int32_t) (seed >> 16) - 32768)/32768.0f;
    }
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    const size_t size = ggml_quantize_chunk(tensor->type, data.data(), bytes.data(), 0, 1, tensor->ne[0], nullptr);
    GGML_ASSERT(size == bytes.size());
    ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
}

static void initialize_head_store_graph(const head_store_graph & g) {
    initialize_head_store_tensor(g.a_storage, 42, 0.125f, 0.0f);
    initialize_head_store_tensor(g.b_storage, 137, 0.00390625f, 0.0078125f);
}
