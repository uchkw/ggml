// Compares MUL_MAT followed by SiLU with Metal fusion disabled and enabled on the same buffers, bit for bit.
// The fusion counter must match the explicit expectation, including rejected graph patterns.
// Usage: test-metal-mm-silu [expect-fusion: 0|1]. JSON Lines go to stdout.

#include "ggml.h"
#include "ggml-backend.h"
#include "../src/ggml-backend-impl.h"
#include "test-mm-silu-cases.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;

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
        auto device = ggml_backend_get_device(backend);
        auto reg = ggml_backend_dev_backend_reg(device);
        auto get = reinterpret_cast<Get>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_get"));
        auto init = reinterpret_cast<Initialize>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_init"));
        reset = reinterpret_cast<Reset>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_reset"));
        stats = reinterpret_cast<Stats>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_get"));
        enable = reinterpret_cast<Enable>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_set_enabled"));
        if (!get || !init || !reset || !stats || !enable) throw std::runtime_error("Metal fusion statistics are unavailable");
        info = get(device);
        init(info);
    }

    uint64_t count() const {
        const char * labels[64] = {};
        uint64_t counts[64] = {};
        const int n = stats(info, labels, counts, 64);
        uint64_t total = 0;
        for (int i = 0; i < n; i++) {
            if (std::strcmp(labels[i], "MUL_MAT+UNARY") == 0) total += counts[i];
        }
        return total;
    }
};

static std::vector<float> values(size_t count, uint32_t seed) {
    std::vector<float> result(count);
    for (float & value : result) {
        seed = 1664525u * seed + 1013904223u;
        value = (float) ((int32_t) (seed >> 16) - 32768) / 32768.0f;
    }
    return result;
}

static uint64_t hash_bytes(const void * data, size_t size) {
    uint64_t hash = 14695981039346656037ULL;
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
    return hash;
}

static std::vector<float> read(ggml_tensor * t) {
    std::vector<float> result((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, result.data(), 0, ggml_nbytes(t));
    return result;
}

static bool overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const auto first = reinterpret_cast<uintptr_t>(a->data);
    const auto second = reinterpret_cast<uintptr_t>(b->data);
    return first < second + ggml_nbytes(b) && second < first + ggml_nbytes(a);
}

static void poison_outputs(const std::vector<ggml_tensor *> & outputs, const ggml_tensor * a,
                           const ggml_tensor * b, bool alias) {
    for (ggml_tensor * out : outputs) {
        if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || overlaps(out, a)) {
            throw std::runtime_error("cannot poison a non-contiguous F32 output or an output overlapping the weight input");
        }
        size_t offset = 0;
        if (overlaps(out, b)) {
            if (!alias || out->data != b->data || ggml_nbytes(out) <= ggml_nbytes(b)) {
                throw std::runtime_error("unexpected output overlap with the activation input");
            }
            // The explicit alias control preserves the active input prefix and poisons the remaining output.
            offset = ggml_nbytes(b);
        }
        const std::vector<float> poison((ggml_nbytes(out) - offset)/sizeof(float), std::numeric_limits<float>::quiet_NaN());
        ggml_backend_tensor_set(out, poison.data(), offset, poison.size()*sizeof(float));
    }
}

static size_t optimize(ggml_backend_t backend, ggml_cgraph * graph) {
    if (!backend->iface.graph_optimize) throw std::runtime_error("Metal graph optimization is unavailable");
    size_t dependencies = 0;
    ggml_backend_graph_optimize_params params{
        [](void * data, ggml_tensor *, ggml_tensor *) { ++*static_cast<size_t *>(data); }, &dependencies};
    backend->iface.graph_optimize(backend, graph, &params);
    return dependencies;
}

static void check_poison_guard(ggml_backend_t backend) {
    ggml_init_params params{ggml_tensor_overhead()*8, nullptr, true};
    Context context(ggml_init(params), ggml_free);
    if (!context) throw std::runtime_error("cannot allocate the poison guard context");
    auto * a = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    auto * b = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    auto * mm = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    auto * activation = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    auto * out = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    if (!buffer) throw std::runtime_error("cannot allocate the poison guard tensors");
    const std::vector<float> input{2, 3, 5, 7};
    const std::vector<float> expected{11, 13, 17, 19};
    ggml_backend_tensor_set(a, input.data(), 0, 4*sizeof(float));
    ggml_backend_tensor_set(b, input.data(), 0, 4*sizeof(float));
    for (bool wrong_destination : {true, false}) {
        ggml_backend_tensor_set(activation, expected.data(), 0, 4*sizeof(float));
        ggml_backend_tensor_set(out, expected.data(), 0, 4*sizeof(float));
        const auto faulty_store = [&]() {
            ggml_backend_tensor_set(wrong_destination ? mm : out, expected.data(), 0,
                                    (wrong_destination ? 4 : 3)*sizeof(float));
        };
        faulty_store();
        const bool stale_pass = read(out) == expected;
        poison_outputs({mm, activation, out}, a, b, false);
        faulty_store();
        const auto poisoned = read(out);
        const bool detected = std::any_of(poisoned.begin(), poisoned.end(), [](float value) { return !std::isfinite(value); });
        const bool inputs_equal = read(a) == input && read(b) == input;
        const bool ok = stale_pass && detected && inputs_equal;
        std::printf("{\"kind\":\"poison_guard\",\"fault\":\"%s\",\"gpu_dispatch\":false,"
                    "\"old_stale_output_would_pass\":%s,\"poisoned_output_detected\":%s,\"inputs_unchanged\":%s,\"passed\":%s}\n",
                    wrong_destination ? "wrong-destination" : "skip-last-store", stale_pass ? "true" : "false",
                    detected ? "true" : "false", inputs_equal ? "true" : "false", ok ? "true" : "false");
        if (!ok) throw std::runtime_error("output poison does not detect an incomplete store");
    }
}

static void compute(ggml_backend_t backend, ggml_cgraph * graph, const Fusion & fusion, bool enabled) {
    fusion.enable(fusion.info, enabled);
    fusion.reset(fusion.info);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Metal graph computation failed");
    ggml_backend_synchronize(backend);
}

static bool run_case(ggml_backend_t backend, const Fusion & fusion, const mm_silu_case & c, bool expected) {
    constexpr int graph_size = 32;
    ggml_init_params params{ggml_tensor_overhead()*32 + ggml_graph_overhead_custom(graph_size, false), nullptr, true};
    Context context(ggml_init(params), ggml_free);
    if (!context) throw std::runtime_error("cannot allocate the test context");
    const std::string variant(c.variant);
    const int64_t batch = variant == "batch" ? 2 : 1;
    auto * abase = ggml_new_tensor_3d(context.get(), c.type, c.k, c.m + (variant == "offset"), batch);
    auto * bbase = variant == "transpose" ? ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, c.n, c.k) :
        ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, c.k + (variant == "stride" ? 4 : 0),
                           variant == "alias" ? c.n*c.m/c.k : c.n + (variant == "offset"), batch);
    auto * a = abase;
    auto * b = bbase;
    if (variant == "offset") {
        a = ggml_view_2d(context.get(), a, c.k, c.m, a->nb[1], a->nb[1]);
        b = ggml_view_2d(context.get(), b, c.k, c.n, b->nb[1], b->nb[1]);
    } else if (variant == "stride") {
        b = ggml_view_2d(context.get(), b, c.k, c.n, b->nb[1], 0);
    } else if (variant == "transpose") {
        b = ggml_transpose(context.get(), b);
    } else if (variant == "alias") {
        b = ggml_view_2d(context.get(), b, c.k, c.n, c.k*sizeof(float), 0);
    }
    auto * mm = ggml_mul_mat(context.get(), a, b);
    ggml_prec_set_acc(mm, GGML_PREC_F32);
    if (variant == "output") ggml_set_output(mm);
    auto * activation = variant == "inplace" ? ggml_silu_inplace(context.get(), mm) :
                        variant == "relu" ? ggml_relu(context.get(), mm) : ggml_silu(context.get(), mm);
    if (variant == "alias") {
        activation->view_src = bbase;
        activation->view_offs = 0;
    }
    auto * out = activation;
    if (variant == "reuse") out = ggml_add(context.get(), out, mm);
    if (variant == "consumer") out = ggml_scale(context.get(), out, 0.5f);
    ggml_set_output(out);
    auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);
    ggml_build_forward_expand(graph, out);
    const size_t allocation_dependencies = optimize(backend, graph);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    if (!buffer) throw std::runtime_error("cannot allocate the test tensors");
    const auto av = values((size_t) ggml_nelements(abase), 123456789u);
    const auto bv = values((size_t) ggml_nelements(bbase), 362436069u);
    std::vector<uint8_t> packed(ggml_nbytes(abase));
    const size_t written = ggml_quantize_chunk(c.type, av.data(), packed.data(), 0, ggml_nrows(abase), c.k, nullptr);
    if (written != packed.size()) throw std::runtime_error("quantized data size differs from the tensor allocation");
    const size_t input_b_bytes = variant == "alias" ? ggml_nbytes(b) : ggml_nbytes(bbase);
    const auto prepare = [&]() {
        ggml_backend_tensor_set(abase, packed.data(), 0, packed.size());
        ggml_backend_tensor_set(bbase, bv.data(), 0, input_b_bytes);
        poison_outputs({mm, activation, out}, abase, variant == "alias" ? b : bbase, variant == "alias");
    };

    prepare();
    compute(backend, graph, fusion, false);
    const auto baseline = read(out);
    const auto product = variant == "output" ? read(mm) : std::vector<float>{};
    const uint64_t disabled_count = fusion.count();
    prepare();
    compute(backend, graph, fusion, true);
    const auto candidate = read(out);
    const uint64_t enabled_count = fusion.count();
    const auto product_after = variant == "output" ? read(mm) : std::vector<float>{};
    size_t different = 0;
    double largest = 0;
    bool finite = true;
    for (size_t i = 0; i < baseline.size(); i++) {
        different += std::memcmp(&baseline[i], &candidate[i], sizeof(float)) != 0;
        finite = finite && std::isfinite(baseline[i]) && std::isfinite(candidate[i]);
        largest = std::max(largest, std::abs((double) baseline[i] - candidate[i]));
    }
    for (float value : product) finite = finite && std::isfinite(value);
    for (float value : product_after) finite = finite && std::isfinite(value);
    const bool product_equal = product.empty() || std::memcmp(product.data(), product_after.data(), product.size()*sizeof(float)) == 0;
    const uint64_t want = expected && c.fusable ? 1 : 0;
    const bool ok = finite && different == 0 && product_equal && disabled_count == 0 && enabled_count == want && allocation_dependencies == 0;
    std::printf("{\"kind\":\"comparison\",\"type\":\"%s\",\"m\":%lld,\"n\":%lld,\"k\":%lld,\"variant\":\"%s\","
                "\"same_input_buffers\":true,\"outputs_poisoned_before_each_run\":true,\"poison_preserves_input_alias\":%s,"
                "\"graph_optimized\":true,\"allocation_dependencies\":%zu,"
                "\"input_a_fnv1a64\":\"%016llx\",\"input_b_fnv1a64\":\"%016llx\","
                "\"output_disabled_fnv1a64\":\"%016llx\",\"output_enabled_fnv1a64\":\"%016llx\","
                "\"disabled_fusions\":%llu,\"enabled_fusions\":%llu,\"expected_fusions\":%llu,"
                "\"different_values\":%zu,\"max_abs\":%.17g,\"finite\":%s,\"protected_product_equal\":%s,\"passed\":%s}\n",
                ggml_type_name(c.type), (long long) c.m, (long long) c.n, (long long) c.k, c.variant,
                variant == "alias" ? "true" : "false", allocation_dependencies,
                (unsigned long long) hash_bytes(packed.data(), packed.size()), (unsigned long long) hash_bytes(bv.data(), input_b_bytes),
                (unsigned long long) hash_bytes(baseline.data(), baseline.size()*sizeof(float)),
                (unsigned long long) hash_bytes(candidate.data(), candidate.size()*sizeof(float)),
                (unsigned long long) disabled_count, (unsigned long long) enabled_count, (unsigned long long) want,
                different, largest, finite ? "true" : "false", product_equal ? "true" : "false", ok ? "true" : "false");
    std::fflush(stdout);
    return ok;
}

int main(int argc, char ** argv) {
    try {
        const int expected = argc > 1 ? std::stoi(argv[1]) : 0;
        if (argc > 2 || (expected != 0 && expected != 1)) {
            throw std::runtime_error("usage: test-metal-mm-silu [expect-fusion: 0|1]");
        }
        ggml_backend_load_all();
        Backend backend(ggml_backend_init_by_name("MTL0", nullptr), ggml_backend_free);
        if (!backend) throw std::runtime_error("cannot initialize MTL0");
        const Fusion fusion(backend.get());
        check_poison_guard(backend.get());
        const auto cases = mm_silu_cases();
        size_t failed = 0;
        for (const auto & c : cases) failed += !run_case(backend.get(), fusion, c, expected != 0);
        fusion.enable(fusion.info, true);
        std::printf("{\"kind\":\"summary\",\"poison_guard_cases\":2,\"cases\":%zu,\"failed\":%zu}\n", cases.size(), failed);
        return failed ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
}
