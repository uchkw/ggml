// Compares the gate projection MUL_MAT -> SIGMOID -> MUL (a * sigmoid(W_gate*x)) with Metal
// generic fusion disabled and enabled on identical inputs and the same allocated buffers.
// Usage: test-metal-mm-glu [expect-fusion: 0|1] [output.bin]. JSON Lines go to stdout.
// expect-fusion selects which build the counts are checked against: 0 for the baseline (the
// MUL_MAT+UNARY+MUL epilogue does not exist yet), 1 for the kernel branch. The CLI and every
// check are identical for both; the enabled run is only diagnostic when fusion is absent.
// The optional binary holds the enabled-run output of every case, length-delimited, so the
// original, fusion-disabled and fusion-enabled builds can be compared byte for byte.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../src/ggml-backend-impl.h"
#include "test-mm-glu-cases.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <vector>

using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

// Counts only the planned gate-projection epilogue; every other generic fusion label is
// ignored so existing patterns cannot disturb the expectation.
struct Fusion {
    using Get = void * (*)(ggml_backend_dev_t);
    using Initialize = void (*)(void *);
    using Reset = void (*)(void *);
    using Stats = int (*)(void *, const char **, uint64_t *, int);
    using Enable = void (*)(void *, bool);
    void * info;
    Reset reset;
    Stats stats;
    Enable enable;

    explicit Fusion(ggml_backend_t backend) {
        const auto device = ggml_backend_get_device(backend);
        const auto reg = ggml_backend_dev_backend_reg(device);
        const auto get = reinterpret_cast<Get>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_get"));
        const auto init = reinterpret_cast<Initialize>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_init"));
        reset = reinterpret_cast<Reset>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_reset"));
        stats = reinterpret_cast<Stats>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_get"));
        enable = reinterpret_cast<Enable>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_set_enabled"));
        require(get && init && reset && stats && enable, "Metal fusion statistics are unavailable");
        info = get(device);
        init(info);
    }

    uint64_t count() const {
        const char * labels[64] = {};
        uint64_t counts[64] = {};
        const int n = stats(info, labels, counts, 64);
        uint64_t result = 0;
        for (int i = 0; i < n; ++i) {
            if (std::strcmp(labels[i], "MUL_MAT+UNARY+MUL") == 0) result += counts[i];
        }
        return result;
    }
};

static std::vector<uint8_t> bytes(const ggml_tensor * tensor) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    return result;
}

static bool finite_f32(const std::vector<uint8_t> & data) {
    for (size_t offset = 0; offset + sizeof(float) <= data.size(); offset += sizeof(float)) {
        float value;
        std::memcpy(&value, data.data() + offset, sizeof(value));
        if (!std::isfinite(value)) return false;
    }
    return true;
}

static uint64_t hash_bytes(const void * data, size_t size) {
    uint64_t hash = 14695981039346656037ULL;
    const auto * raw = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) hash = (hash ^ raw[i]) * 1099511628211ULL;
    return hash;
}

/** Poisons every output byte outside the active input ranges (weights, activation, factor),
 *  including only the non-overlapping tail of a partially aliased output. Any byte a correct
 *  kernel must write but a faulty one skips (or writes to the wrong destination) stays NaN. */
static size_t poison(const std::vector<ggml_tensor *> & outputs, const std::vector<ggml_tensor *> & inputs) {
    size_t protected_values = 0;
    for (ggml_tensor * out : outputs) {
        require(out->type == GGML_TYPE_F32 && ggml_is_contiguous(out), "output poison requires a contiguous F32 tensor");
        const size_t element = sizeof(float);
        const auto begin = reinterpret_cast<uintptr_t>(out->data);
        size_t run = 0;
        for (size_t offset = 0; offset <= ggml_nbytes(out); offset += element) {
            bool protect = offset == ggml_nbytes(out);
            if (!protect) {
                for (const ggml_tensor * input : inputs) {
                    const auto first = reinterpret_cast<uintptr_t>(input->data);
                    if (begin + offset < first + ggml_nbytes(input) && first < begin + offset + element) protect = true;
                }
            }
            if (!protect) continue;
            if (offset > run) {
                const std::vector<float> nan_run((offset - run)/element, std::numeric_limits<float>::quiet_NaN());
                ggml_backend_tensor_set(out, nan_run.data(), run, nan_run.size()*element);
            }
            if (offset < ggml_nbytes(out)) ++protected_values;
            run = offset + element;
        }
    }
    return protected_values;
}

/** Runs the Metal graph optimization and counts the allocation dependencies it adds; the
 *  planned epilogue reuses the existing buffers, so any new dependency is a failure. */
static size_t optimize(ggml_backend_t backend, ggml_cgraph * graph) {
    require(backend->iface.graph_optimize, "Metal graph optimization is unavailable");
    size_t dependencies = 0;
    ggml_backend_graph_optimize_params params{
        [](void * data, ggml_tensor *, ggml_tensor *) { ++*static_cast<size_t *>(data); }, &dependencies};
    backend->iface.graph_optimize(backend, graph, &params);
    return dependencies;
}

/** Two host-copy faults must fail the comparison: a store to the wrong tensor and a store
 *  that skips the last element. Without poison the stale output would pass, and the poison
 *  itself must never touch the active inputs. */
static void poison_guard(ggml_backend_t backend) {
    Context ctx(ggml_init({ggml_tensor_overhead()*4, nullptr, true}), ggml_free);
    require(bool(ctx), "cannot allocate the poison guard context");
    auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    auto * wrong = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    auto * out = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(buffer), "cannot allocate the poison guard tensors");
    const std::vector<float> input_values{2, 3, 5, 7};
    const std::vector<float> expected{11, 13, 17, 19};
    const auto expected_bits = [](const std::vector<float> & values) {
        const auto * raw = reinterpret_cast<const uint8_t *>(values.data());
        return std::vector<uint8_t>(raw, raw + values.size()*sizeof(float));
    };
    const auto expected_bytes = expected_bits(expected);
    ggml_backend_tensor_set(input, input_values.data(), 0, 4*sizeof(float));
    const auto before = bytes(input);
    for (bool wrong_destination : {true, false}) {
        ggml_backend_tensor_set(out, expected.data(), 0, 4*sizeof(float));
        const auto faulty_store = [&]() {
            if (wrong_destination) {
                ggml_backend_tensor_set(wrong, expected.data(), 0, 4*sizeof(float));
            } else {
                ggml_backend_tensor_set(out, expected.data(), 0, 3*sizeof(float));
            }
        };
        faulty_store();
        const bool stale_pass = bytes(out) == expected_bytes;
        poison({wrong, out}, {input});
        faulty_store();
        const bool detected = !finite_f32(bytes(out));
        const bool inputs_unchanged = bytes(input) == before;
        const bool ok = stale_pass && detected && inputs_unchanged;
        std::printf("{\"kind\":\"poison_guard\",\"fault\":\"%s\",\"gpu_dispatch\":false,"
                    "\"old_stale_output_would_pass\":%s,\"poisoned_output_detected\":%s,"
                    "\"inputs_unchanged\":%s,\"passed\":%s}\n",
                    wrong_destination ? "wrong-destination" : "skip-last-store",
                    stale_pass ? "true" : "false", detected ? "true" : "false",
                    inputs_unchanged ? "true" : "false", ok ? "true" : "false");
        require(ok, "output poison does not detect an incomplete store");
    }
}

/** Two Metal mappings of one host arena: tensors bound through distinct buffer objects hide
 *  their raw overlap from every buffer-identity check, so these unsafe graphs must stay
 *  unfused and still match the ordinary run bit for bit. */
struct MappedArena {
    std::unique_ptr<void, decltype(&std::free)> host{nullptr, std::free};
    Buffer first{nullptr, ggml_backend_buffer_free};
    Buffer second{nullptr, ggml_backend_buffer_free};

    void allocate(ggml_backend_t backend, const mm_glu_graph & t, const std::string & variant) {
        const long page = sysconf(_SC_PAGESIZE);
        require(page > 0, "cannot obtain the host page size");
        const size_t stride = ((ggml_nbytes(t.w_base) + 16 + (size_t) page - 1)/(size_t) page)*(size_t) page;
        const size_t size = 6*stride;
        void * data = nullptr;
        require(posix_memalign(&data, (size_t) page, size) == 0, "cannot allocate the mapped GLU arena");
        host.reset(data);
        const auto device = ggml_backend_get_device(backend);
        first.reset(ggml_backend_dev_buffer_from_host_ptr(device, data, size, ggml_nbytes(t.w_base)));
        second.reset(ggml_backend_dev_buffer_from_host_ptr(device, data, size, ggml_nbytes(t.w_base)));
        require(first && second && first->context && second->context && first->context != second->context,
                "cannot create two distinct Metal mappings of the GLU arena");
        const std::vector<ggml_tensor *> all{t.w_base, t.x_base, t.a_base, t.mm, t.sigmoid, t.mul};
        // slots: weight, activation, factor, gate product, sigmoid, final product; the wrapped
        // tensor lands 16 bytes into the activation (or factor) slot through the second mapping
        const size_t wrapped = variant == "mapped_out_x" ? 5 : variant == "mapped_sigmoid_x" ? 4 : 3;
        const size_t wrapped_offset = (wrapped == 3 ? 2 : 1)*stride + 16;
        for (size_t i = 0; i < all.size(); ++i) {
            const size_t offset = i == wrapped ? wrapped_offset : i*stride;
            require(ggml_backend_tensor_alloc(i == wrapped ? second.get() : first.get(), all[i],
                        static_cast<char *>(data) + offset) == GGML_STATUS_SUCCESS, "cannot bind the mapped GLU tensor");
        }
        const auto * overlapped = wrapped == 3 ? t.a_base : t.x_base;
        require(all[wrapped]->data == static_cast<char *>(overlapped->data) + 16 &&
                    all[wrapped]->buffer != overlapped->buffer,
                "the mapped GLU tensors do not overlap across wrappers");
    }
};

static bool run_case(ggml_backend_t backend, Fusion & fusion, const mm_glu_case & c, bool expected, FILE * dump) {
    constexpr int nodes = 32;
    Context ctx(ggml_init({ggml_tensor_overhead()*nodes + ggml_graph_overhead_custom(nodes, false), nullptr, true}), ggml_free);
    require(bool(ctx), "cannot allocate the GLU context");
    const std::string variant(c.variant);
    const mm_glu_graph t = mm_glu_build(ctx.get(), c);
    ggml_set_output(t.out);
    auto * graph = ggml_new_graph_custom(ctx.get(), nodes, false);
    ggml_build_forward_expand(graph, t.out);
    const size_t dependencies = optimize(backend, graph);

    const bool mapped = variant.rfind("mapped_", 0) == 0;
    MappedArena arena;
    Buffer buffer(nullptr, ggml_backend_buffer_free);
    if (mapped) {
        arena.allocate(backend, t, variant);
    } else {
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        require(bool(buffer), "cannot allocate the GLU tensors");
    }

    std::vector<ggml_tensor *> inputs{t.w_base};
    if (t.wa) inputs.push_back(t.wa);
    inputs.push_back(t.x_base);
    if (t.a_base) inputs.push_back(t.a_base);
    std::vector<ggml_tensor *> outputs;
    if (t.a_mm) outputs.push_back(t.a_mm);
    outputs.push_back(t.mm);
    outputs.push_back(t.sigmoid);
    outputs.push_back(t.mul);
    if (t.out != t.mul) outputs.push_back(t.out);

    const bool gate_public = (t.mm->flags & GGML_TENSOR_FLAG_OUTPUT) != 0;
    const bool sigmoid_public = (t.sigmoid->flags & GGML_TENSOR_FLAG_OUTPUT) != 0;
    size_t protected_values = 0;
    const auto prepare = [&]() {
        mm_glu_initialize(t, c);
        std::vector<std::vector<uint8_t>> before;
        before.reserve(inputs.size());
        for (const ggml_tensor * input : inputs) before.push_back(bytes(input));
        protected_values = poison(outputs, inputs);
        for (size_t i = 0; i < inputs.size(); ++i) {
            require(bytes(inputs[i]) == before[i], "output poison changes an active input");
        }
    };
    const auto compute = [&](bool enabled) {
        fusion.enable(fusion.info, enabled);
        fusion.reset(fusion.info);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Metal GLU computation fails");
        ggml_backend_synchronize(backend);
    };

    prepare();
    compute(false);
    const auto ordinary = bytes(t.out);
    const auto saved_mm = gate_public ? bytes(t.mm) : std::vector<uint8_t>{};
    const auto saved_sigmoid = sigmoid_public ? bytes(t.sigmoid) : std::vector<uint8_t>{};
    const uint64_t disabled = fusion.count();
    prepare();
    compute(true);
    const auto selected = bytes(t.out);
    const uint64_t enabled = fusion.count();
    const uint64_t want = expected && c.fusable ? 1 : 0;

    const bool equal_bits = ordinary == selected;
    const bool finite = finite_f32(ordinary) && finite_f32(selected);
    const bool mm_equal = saved_mm.empty() || (bytes(t.mm) == saved_mm && finite_f32(saved_mm));
    const bool sigmoid_equal = saved_sigmoid.empty() || (bytes(t.sigmoid) == saved_sigmoid && finite_f32(saved_sigmoid));
    const bool ok = equal_bits && finite && mm_equal && sigmoid_equal &&
                    disabled == 0 && enabled == want && dependencies == 0;
    if (dump) {
        const uint64_t size = selected.size();
        require(std::fwrite(&size, sizeof(size), 1, dump) == 1 &&
                    std::fwrite(selected.data(), 1, selected.size(), dump) == selected.size(),
                "cannot write GLU comparison bytes");
    }
    std::printf("{\"kind\":\"comparison\",\"type\":\"%s\",\"width\":%lld,\"k\":%lld,\"frames\":%lld,"
                "\"swap\":%s,\"variant\":\"%s\",\"fusable\":%s,\"equal_bits\":%s,\"same_input_buffers\":true,"
                "\"mapped_wrappers\":%d,\"poison_protected_input_values\":%zu,\"allocation_dependencies\":%zu,"
                "\"gate_mm_marked_output\":%s,\"sigmoid_marked_output\":%s,"
                "\"output_disabled_fnv1a64\":\"%016llx\",\"output_enabled_fnv1a64\":\"%016llx\","
                "\"protected_gate_mm_equal\":%s,\"protected_sigmoid_equal\":%s,"
                "\"disabled_fusions\":%llu,\"enabled_fusions\":%llu,\"expected_fusions\":%llu,"
                "\"finite\":%s,\"passed\":%s}\n",
                ggml_type_name(c.type), (long long) c.width, (long long) c.k, (long long) c.frames,
                c.swap ? "true" : "false", c.variant, c.fusable ? "true" : "false",
                equal_bits ? "true" : "false", mapped ? 2 : 0, protected_values, dependencies,
                gate_public ? "true" : "false", sigmoid_public ? "true" : "false",
                (unsigned long long) hash_bytes(ordinary.data(), ordinary.size()),
                (unsigned long long) hash_bytes(selected.data(), selected.size()),
                mm_equal ? "true" : "false", sigmoid_equal ? "true" : "false",
                (unsigned long long) disabled, (unsigned long long) enabled, (unsigned long long) want,
                finite ? "true" : "false", ok ? "true" : "false");
    std::fflush(stdout);
    return ok;
}

int main(int argc, char ** argv) {
    try {
        const int expected = argc > 1 ? std::stoi(argv[1]) : 0;
        require(argc <= 3 && (expected == 0 || expected == 1), "usage: test-metal-mm-glu [expect-fusion: 0|1] [output.bin]");
        std::unique_ptr<FILE, decltype(&std::fclose)> dump(argc > 2 ? ggml_fopen(argv[2], "wb") : nullptr, std::fclose);
        require(argc <= 2 || bool(dump), "cannot open the GLU result file");
        ggml_backend_load_all();
        Backend backend(ggml_backend_init_by_name("MTL0", nullptr), ggml_backend_free);
        require(bool(backend), "cannot initialize MTL0");
        Fusion fusion(backend.get());
        poison_guard(backend.get());
        auto cases = mm_glu_cases();
        for (bool swap : {false, true}) {
            for (const char * variant : {"mapped_out_x", "mapped_sigmoid_x", "mapped_mm_a"}) {
                cases.push_back({GGML_TYPE_F32, 1024, 1024, 33, swap, variant, false});
            }
        }
        size_t failed = 0;
        for (const auto & c : cases) failed += !run_case(backend.get(), fusion, c, expected != 0, dump.get());
        fusion.enable(fusion.info, true);
        std::printf("{\"kind\":\"summary\",\"cases\":%zu,\"failed\":%zu,\"poison_guard_cases\":2}\n", cases.size(), failed);
        return failed ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
}
