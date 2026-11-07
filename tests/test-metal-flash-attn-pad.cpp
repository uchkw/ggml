// Checks the last FlashAttention block in Metal's scratch buffer, with all source strides allocated.
// Usage: test-metal-flash-attn-pad

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include "../src/ggml-metal/ggml-metal-impl.h"

#include <cstdio>
#include <cstring>
#include <vector>

enum class layout { contiguous, interleaved, strided };

struct pad_case {
    ggml_type type;
    layout memory_layout;
    int dk;
    int dv;
    int keys;
    int queries;
    int heads;
    int batches;
    bool mask;
};

struct kv_tensor {
    ggml_tensor * storage;
    ggml_tensor * view;
    std::vector<uint8_t> values;
};

static kv_tensor make_kv(ggml_context * ctx, const pad_case & c, int d, const char * name) {
    // A spare time step (and a padded row for the strided layout) keeps every full-stride
    // source read inside the allocated array, even if a kernel copies whole row strides.
    const size_t row = ggml_row_size(c.type, d + (c.memory_layout == layout::strided ? 32 : 0));
    const bool interleaved = c.memory_layout == layout::interleaved;
    // The stride between key rows spans all heads when the layout is interleaved.
    const size_t nb1 = interleaved ? row*c.heads : row;
    // The stride between heads is a single row when the layout is interleaved.
    const size_t nb2 = interleaved ? row : row*(c.keys + 1);
    // The stride between batches always spans every head and key row.
    const size_t nb3 = row*c.heads*(c.keys + 1);
    ggml_tensor * storage = ggml_new_tensor_1d(ctx, c.type, nb3*c.batches/ggml_type_size(c.type));
    ggml_tensor * view = ggml_view_4d(ctx, storage, d, c.keys, c.heads, c.batches, nb1, nb2, nb3, 0);
    ggml_set_name(view, name);
    // Finite sentinel bytes fill the gaps between the logical rows.
    return {storage, view, std::vector<uint8_t>(ggml_nbytes(storage), 0x5a)};
}

static void initialize_kv(kv_tensor & t, int shift) {
    for (int64_t b = 0; b < t.view->ne[3]; ++b) {
        for (int64_t h = 0; h < t.view->ne[2]; ++h) {
            for (int64_t k = 0; k < t.view->ne[1]; ++k) {
                for (int64_t d = 0; d < t.view->ne[0]; ++d) {
                    const float value = ((d + 3*h + 5*k + 7*b + shift)%17 - 8)/8.0f;
                    const size_t offset = d*t.view->nb[0] + k*t.view->nb[1] + h*t.view->nb[2] + b*t.view->nb[3];
                    if (t.view->type == GGML_TYPE_F32) {
                        memcpy(t.values.data() + offset, &value, sizeof(value));
                    } else {
                        const ggml_fp16_t half = ggml_fp32_to_fp16(value);
                        memcpy(t.values.data() + offset, &half, sizeof(half));
                    }
                }
            }
        }
    }
    ggml_backend_tensor_set(t.storage, t.values.data(), 0, t.values.size());
}

static void initialize_mask(ggml_tensor * mask, std::vector<uint8_t> & values) {
    values.assign(ggml_nbytes(mask), 0x5a);
    for (int64_t b = 0; b < mask->ne[3]; ++b) {
        for (int64_t q = 0; q < mask->ne[1]; ++q) {
            for (int64_t k = 0; k < mask->ne[0]; ++k) {
                const float value = ((k + 3*q + 5*b)%13 - 6)/6.0f;
                const ggml_fp16_t half = ggml_fp32_to_fp16(value);
                memcpy(values.data() + (k + q*mask->ne[0] + b*mask->ne[0]*mask->ne[1])*sizeof(half), &half, sizeof(half));
            }
        }
    }
    ggml_backend_tensor_set(mask, values.data(), 0, values.size());
}

// The pad kernel writes, per key/head row of the last chunk, one destination row as wide as
// the source row stride: first the K rows, then the V rows, then the padded mask. Only the
// logical bytes of each row may be copied, the rest of the row and the invalid rows must be
// zero. Without a tail the whole scratch must be left untouched.
static size_t check_kv(const uint8_t * actual, const kv_tensor & t, int chunk, bool has_tail) {
    const size_t pitch = t.view->nb[1];
    std::vector<uint8_t> expected(pitch*chunk*t.view->ne[2]*t.view->ne[3], has_tail ? 0 : 0xa5);
    if (has_tail) {
        const int64_t first = t.view->ne[1] - t.view->ne[1]%chunk;
        const size_t row = ggml_row_size(t.view->type, t.view->ne[0]);
        for (int64_t b = 0; b < t.view->ne[3]; ++b) {
            for (int64_t h = 0; h < t.view->ne[2]; ++h) {
                for (int64_t k = first; k < t.view->ne[1]; ++k) {
                    const size_t source = k*t.view->nb[1] + h*t.view->nb[2] + b*t.view->nb[3];
                    const size_t target = ((b*t.view->ne[2] + h)*chunk + k - first)*pitch;
                    memcpy(expected.data() + target, t.values.data() + source, row);
                }
            }
        }
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        mismatches += actual[i] != expected[i];
    }
    return mismatches;
}

static size_t check_mask(const uint8_t * actual, const std::vector<uint8_t> & mask_values, const ggml_tensor * mask, int chunk, bool has_tail) {
    const int64_t keys    = mask->ne[0];
    const int64_t queries = mask->ne[1];
    const int64_t batches = mask->ne[3];
    std::vector<uint8_t> expected(sizeof(ggml_fp16_t)*chunk*queries*batches, 0xa5);
    if (has_tail) {
        const int64_t icp   = keys%chunk;
        const int64_t first = keys - icp;
        for (int64_t b = 0; b < batches; ++b) {
            for (int64_t q = 0; q < queries; ++q) {
                for (int64_t k = 0; k < chunk; ++k) {
                    // The value -MAXHALF masks out the invalid keys.
                    ggml_fp16_t value = ggml_fp32_to_fp16(-65504.0f);
                    if (k < icp) {
                        memcpy(&value, mask_values.data() + (first + k + q*keys + b*keys*queries)*sizeof(value), sizeof(value));
                    }
                    memcpy(expected.data() + ((b*queries + q)*chunk + k)*sizeof(value), &value, sizeof(value));
                }
            }
        }
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        mismatches += actual[i] != expected[i];
    }
    return mismatches;
}

static bool run_case(ggml_backend_t backend, const pad_case & c) {
    ggml_init_params params = {ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.dk, c.queries, c.heads, c.batches);
    kv_tensor k = make_kv(ctx, c, c.dk, "k");
    kv_tensor v = make_kv(ctx, c, c.dv, "v");
    ggml_tensor * mask = c.mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.keys, c.queries, 1, c.batches) : nullptr;
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k.view, v.view, mask, 1.0f, 0.0f, 0.0f);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    // Mirror ggml_metal_op_flash_attn_ext_use_vec: with a head size divisible by 32, batches
    // below 20 queries run the vec kernel (chunks of 32), larger ones the half8x8 kernel
    // (chunks of 64).
    const int chunk = c.queries < 20 ? OP_FLASH_ATTN_EXT_VEC_NCPSG : OP_FLASH_ATTN_EXT_NCPSG;
    const bool has_tail = c.keys%chunk != 0;
    const size_t k_bytes = k.view->nb[1]*chunk*c.heads*c.batches;
    const size_t v_bytes = v.view->nb[1]*chunk*c.heads*c.batches;
    const size_t mask_bytes = c.mask ? sizeof(ggml_fp16_t)*chunk*c.queries*c.batches : 0;
    const size_t pad_bytes = k_bytes + v_bytes + mask_bytes;

    const char * error = nullptr;
    bool computed = false;
    std::vector<uint8_t> mask_values;
    std::vector<uint8_t> pad;
    size_t k_errors = 0;
    size_t v_errors = 0;
    size_t mask_errors = 0;
    ggml_backend_buffer_t buffer = nullptr;

    do {
        if (!ggml_backend_supports_op(backend, out)) {
            error = "the case is not supported";
            break;
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) {
            error = "the allocation failed";
            break;
        }
        // Metal reserves the scratch of the FlashAttention output inside its own allocation.
        if (ggml_backend_buffer_get_alloc_size(buffer, out) < ggml_nbytes(out) + pad_bytes) {
            error = "the allocated output does not cover the padding scratch";
            break;
        }
        // The readback tensor is created after the allocation, so it does not get storage of
        // its own, and is bound over that scratch. Metal buffers are not reported as host
        // memory, so the bytes are fetched through the backend, never through out->data.
        ggml_tensor * readback = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, pad_bytes);
        if (ggml_backend_tensor_alloc(out->buffer, readback, (char *) out->data + ggml_nbytes(out)) != GGML_STATUS_SUCCESS) {
            error = "binding the readback tensor failed";
            break;
        }

        ggml_backend_buffer_clear(buffer, 0xa5);
        ggml_backend_tensor_memset(q, 0, 0, ggml_nbytes(q));
        initialize_kv(k, 0);
        initialize_kv(v, 2);
        if (mask) {
            initialize_mask(mask, mask_values);
        }
        computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);

        pad.resize(pad_bytes);
        ggml_backend_tensor_get(readback, pad.data(), 0, pad_bytes);
        k_errors = check_kv(pad.data(), k, chunk, has_tail);
        v_errors = check_kv(pad.data() + k_bytes, v, chunk, has_tail);
        mask_errors = mask ? check_mask(pad.data() + k_bytes + v_bytes, mask_values, mask, chunk, has_tail) : 0;
    } while (false);

    const bool ok = error == nullptr && computed && k_errors == 0 && v_errors == 0 && mask_errors == 0;
    const char * name = c.memory_layout == layout::interleaved ? "interleaved" :
                        c.memory_layout == layout::strided ? "strided" : "contiguous";
    printf("%s %s dk=%d dv=%d keys=%d queries=%d heads=%d batches=%d mask=%d: "
           "K=%zu V=%zu mask=%zu mismatched bytes%s%s%s: %s\n",
           ggml_type_name(c.type), name, c.dk, c.dv, c.keys, c.queries, c.heads, c.batches, (int) c.mask,
           k_errors, v_errors, mask_errors,
           error ? " (" : "", error ? error : "", error ? ")" : "",
           ok ? "OK" : "FAIL");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(void) {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name("MTL0", nullptr);
    if (!backend) {
        fprintf(stderr, "The Metal backend is unavailable.\n");
        return 1;
    }
    int failed = 0;
    int count = 0;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (int queries : {1, 20}) {
            for (int keys : {64, 65}) {
                failed += !run_case(backend, {type, layout::interleaved, 128, 128, keys, queries, 8, 1, false});
                ++count;
            }
            failed += !run_case(backend, {type, layout::interleaved, 128, 96, 33, queries, 2, 2, true});
            failed += !run_case(backend, {type, layout::strided, 128, 96, 65, queries, 2, 2, true});
            failed += !run_case(backend, {type, layout::contiguous, 128, 96, 65, queries, 2, 2, true});
            count += 3;
        }
    }
    printf("%d cases, %d failed\n", count, failed);
    ggml_backend_free(backend);
    return failed == 0 ? 0 : 1;
}
