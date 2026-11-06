// Compares the attention head-layout store with Metal fusion disabled and enabled, bit for bit.
// Restored inputs and poisoned outputs expose stale stores, including output aliases of external inputs.
// Usage: test-metal-head-store [expect-fusion: 0|1] [output.bin]. JSON Lines go to stdout.
// The optional binary stores each enabled output in logical element order for exact comparisons between builds.

#include "ggml.h"
#include "ggml-backend.h"
#include "../src/ggml-backend-impl.h"
#include "test-head-store-cases.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unistd.h>

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
            if (std::strcmp(labels[i], "MUL_MAT+CONT") == 0) total += counts[i];
        }
        return total;
    }
};

static std::vector<uint8_t> bytes(const ggml_tensor * tensor) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    return result;
}

static uint64_t hash(const std::vector<uint8_t> & data) {
    uint64_t result = 14695981039346656037ULL;
    for (uint8_t value : data) result = (result ^ value)*1099511628211ULL;
    return result;
}

static std::vector<float> read(const ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32) throw std::runtime_error("the output is not F32");
    const auto data = bytes(tensor);
    std::vector<float> result;
    result.reserve((size_t) ggml_nelements(tensor));
    for (int64_t i3 = 0; i3 < tensor->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; i0++) {
                    float value;
                    std::memcpy(&value, data.data() + i0*tensor->nb[0] + i1*tensor->nb[1] +
                                i2*tensor->nb[2] + i3*tensor->nb[3], sizeof(value));
                    result.push_back(value);
                }
            }
        }
    }
    return result;
}

static bool equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

static bool finite(const std::vector<float> & data) {
    return std::all_of(data.begin(), data.end(), [](float value) { return std::isfinite(value); });
}

static bool poison(const std::vector<ggml_tensor *> & outputs, const std::vector<ggml_tensor *> & inputs) {
    std::vector<std::vector<uint8_t>> before;
    for (const auto * input : inputs) before.push_back(bytes(input));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (auto * output : outputs) {
        if (output->type != GGML_TYPE_F32) throw std::runtime_error("cannot poison a non-F32 output");
        auto data = bytes(output);
        for (size_t offset = 0; offset + sizeof(float) <= data.size(); offset += sizeof(float)) {
            const auto address = reinterpret_cast<uintptr_t>(output->data) + offset;
            bool overlaps_input = false;
            for (const auto * input : inputs) {
                const auto begin = reinterpret_cast<uintptr_t>(input->data);
                overlaps_input |= address < begin + ggml_nbytes(input) && begin < address + sizeof(float);
            }
            if (!overlaps_input) std::memcpy(data.data() + offset, &nan, sizeof(nan));
        }
        ggml_backend_tensor_set(output, data.data(), 0, data.size());
    }
    for (size_t i = 0; i < inputs.size(); i++) {
        if (before[i] != bytes(inputs[i])) return false;
    }
    return true;
}

static size_t optimize(ggml_backend_t backend, ggml_cgraph * graph) {
    if (!backend->iface.graph_optimize) throw std::runtime_error("Metal graph optimization is unavailable");
    size_t dependencies = 0;
    ggml_backend_graph_optimize_params params{
        [](void * data, ggml_tensor *, ggml_tensor *) { ++*static_cast<size_t *>(data); }, &dependencies};
    backend->iface.graph_optimize(backend, graph, &params);
    return dependencies;
}

static void compute(ggml_backend_t backend, ggml_cgraph * graph, const Fusion & fusion, bool enabled) {
    fusion.enable(fusion.info, enabled);
    fusion.reset(fusion.info);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Metal graph computation failed");
    ggml_backend_synchronize(backend);
}

static void check_poison_guard(ggml_backend_t backend) {
    ggml_init_params params{ggml_tensor_overhead()*4, nullptr, true};
    Context context(ggml_init(params), ggml_free);
    if (!context) throw std::runtime_error("cannot allocate the poison guard context");
    auto * input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    auto * wrong = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    auto * out = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    if (!buffer) throw std::runtime_error("cannot allocate the poison guard tensors");
    const std::vector<float> source{2, 3, 5, 7}, expected{11, 13, 17, 19};
    ggml_backend_tensor_set(input, source.data(), 0, source.size()*sizeof(float));
    for (bool wrong_destination : {true, false}) {
        ggml_backend_tensor_set(out, expected.data(), 0, expected.size()*sizeof(float));
        const auto faulty_store = [&]() {
            ggml_backend_tensor_set(wrong_destination ? wrong : out, expected.data(), 0,
                                    (wrong_destination ? 4 : 3)*sizeof(float));
        };
        faulty_store();
        const bool stale_pass = equal(read(out), expected);
        const bool inputs_unchanged = poison({wrong, out}, {input});
        faulty_store();
        const bool detected = !finite(read(out));
        const bool ok = stale_pass && detected && inputs_unchanged && equal(read(input), source);
        std::printf("{\"kind\":\"poison_guard\",\"fault\":\"%s\",\"gpu_dispatch\":false,"
                    "\"old_stale_output_would_pass\":%s,\"poisoned_output_detected\":%s,\"inputs_unchanged\":%s,\"passed\":%s}\n",
                    wrong_destination ? "wrong-destination" : "skip-last-store", stale_pass ? "true" : "false",
                    detected ? "true" : "false", inputs_unchanged ? "true" : "false", ok ? "true" : "false");
        if (!ok) throw std::runtime_error("output poison does not detect an incomplete store");
    }
}

/** Separate Metal mappings expose aliases that a same-buffer gate must reject. */
struct MappedArena {
    std::unique_ptr<void, decltype(&std::free)> host{nullptr, std::free};
    Buffer first{nullptr, ggml_backend_buffer_free};
    Buffer second{nullptr, ggml_backend_buffer_free};

    void allocate(ggml_backend_t backend, const head_store_graph & t, const std::string & variant) {
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0) throw std::runtime_error("cannot obtain the host page size");
        const size_t largest = std::max({ggml_nbytes(t.a_storage), ggml_nbytes(t.b_storage), ggml_nbytes(t.cont)});
        const size_t stride = ((largest + 16 + page - 1)/page)*page;
        const size_t size = 3*stride;
        void * data = nullptr;
        if (posix_memalign(&data, (size_t) page, size) != 0) throw std::runtime_error("cannot allocate the mapped head-store arena");
        host.reset(data);
        const auto device = ggml_backend_get_device(backend);
        first.reset(ggml_backend_dev_buffer_from_host_ptr(device, data, size, largest));
        second.reset(ggml_backend_dev_buffer_from_host_ptr(device, data, size, largest));
        if (!first || !second || !first->context || !second->context || first->context == second->context) {
            throw std::runtime_error("cannot create two distinct Metal mappings of the head-store arena");
        }
        const std::vector<ggml_tensor *> roots{t.a_storage, t.b_storage, t.cont};
        for (size_t i = 0; i < roots.size(); i++) {
            size_t offset = i*stride;
            if (i == 2) {
                if (variant == "mapped_a" || variant == "mapped_partial_a") offset = 0;
                if (variant == "mapped_b" || variant == "mapped_partial_b") offset = stride;
                if (variant.find("partial") != std::string::npos) offset += 4*sizeof(float);
            }
            if (ggml_backend_tensor_alloc(i == 2 ? second.get() : first.get(), roots[i],
                    static_cast<char *>(data) + offset) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("cannot bind the mapped head-store tensor");
            }
        }
    }
};

static bool run_case(ggml_backend_t backend, const Fusion & fusion, const head_store_case & c, bool expected, FILE * dump) {
    constexpr int graph_size = 64;
    ggml_init_params params{ggml_tensor_overhead()*64 + ggml_graph_overhead_custom(graph_size, false), nullptr, true};
    Context context(ggml_init(params), ggml_free);
    if (!context) throw std::runtime_error("cannot allocate the test context");
    const auto t = make_head_store_graph(context.get(), c);
    auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);
    ggml_build_forward_expand(graph, t.out);
    const size_t allocation_dependencies = optimize(backend, graph);
    const bool mapped = std::string(c.variant).find("mapped_") == 0;
    MappedArena arena;
    if (mapped) arena.allocate(backend, t, c.variant);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    if (!buffer) throw std::runtime_error("cannot allocate the test tensors");
    if (mapped && (t.cont->buffer == t.a->buffer || t.cont->buffer == t.b->buffer)) {
        throw std::runtime_error("the mapped output does not use a separate buffer wrapper");
    }
    std::vector<ggml_tensor *> protected_tensors;
    for (auto * tensor : {t.mm, t.cont}) {
        if (tensor && (tensor->flags & GGML_TENSOR_FLAG_OUTPUT)) protected_tensors.push_back(tensor);
    }
    std::vector<ggml_tensor *> outputs{t.mm, t.out};
    if (t.cont) outputs.push_back(t.cont);
    const auto prepare = [&]() {
        initialize_head_store_graph(t);
        return poison(outputs, {t.a, t.b});
    };
    bool inputs_unchanged = prepare();
    const uint64_t a_hash = hash(bytes(t.a)), b_hash = hash(bytes(t.b));
    compute(backend, graph, fusion, false);
    const auto baseline = read(t.out);
    const uint64_t baseline_hash = hash(bytes(t.out));
    std::vector<std::vector<float>> protected_values;
    for (const auto * tensor : protected_tensors) protected_values.push_back(read(tensor));
    const uint64_t disabled_count = fusion.count();
    inputs_unchanged = prepare() && inputs_unchanged;
    inputs_unchanged = inputs_unchanged && a_hash == hash(bytes(t.a)) && b_hash == hash(bytes(t.b));
    const auto before_candidate = read(t.out);
    bool all_values_cleared = true;
    for (size_t i = 0; i < baseline.size(); i++) {
        all_values_cleared = all_values_cleared && std::memcmp(&baseline[i], &before_candidate[i], sizeof(float)) != 0;
    }
    compute(backend, graph, fusion, true);
    const auto candidate = read(t.out);
    if (dump && std::fwrite(candidate.data(), sizeof(float), candidate.size(), dump) != candidate.size()) {
        throw std::runtime_error("cannot write the enabled output dump");
    }
    const uint64_t enabled_count = fusion.count();
    bool protected_equal = true, values_finite = finite(baseline) && finite(candidate);
    for (size_t i = 0; i < protected_tensors.size(); i++) {
        const auto after = read(protected_tensors[i]);
        protected_equal = protected_equal && equal(protected_values[i], after);
        values_finite = values_finite && finite(protected_values[i]) && finite(after);
    }
    size_t different = 0;
    double largest = 0;
    for (size_t i = 0; i < baseline.size(); i++) {
        different += std::memcmp(&baseline[i], &candidate[i], sizeof(float)) != 0;
        largest = std::max(largest, std::abs((double) baseline[i] - candidate[i]));
    }
    const uint64_t want = expected && c.fusable ? 1 : 0;
    const bool ok = values_finite && different == 0 && protected_equal && inputs_unchanged && all_values_cleared &&
                    disabled_count == 0 && enabled_count == want && allocation_dependencies == 0;
    std::printf("{\"kind\":\"comparison\",\"type\":\"%s\",\"frames\":%lld,\"variant\":\"%s\","
                "\"cpu_reference_case\":%s,\"mapped_arena\":%s,\"same_input_buffers\":true,\"outputs_poisoned_before_each_run\":true,"
                "\"inputs_unchanged\":%s,\"all_output_values_cleared_before_candidate\":%s,"
                "\"graph_optimized\":true,\"allocation_dependencies\":%zu,"
                "\"a_fnv1a64\":\"%016llx\",\"b_fnv1a64\":\"%016llx\","
                "\"output_disabled_fnv1a64\":\"%016llx\",\"output_enabled_fnv1a64\":\"%016llx\","
                "\"disabled_fusions\":%llu,\"enabled_fusions\":%llu,\"expected_fusions\":%llu,"
                "\"different_values\":%zu,\"max_abs\":%.17g,\"finite\":%s,\"protected_intermediates_equal\":%s,\"passed\":%s}\n",
                ggml_type_name(c.type), (long long) c.frames, c.variant,
                c.cpu_reference ? "true" : "false", mapped ? "true" : "false", inputs_unchanged ? "true" : "false",
                all_values_cleared ? "true" : "false", allocation_dependencies,
                (unsigned long long) a_hash, (unsigned long long) b_hash, (unsigned long long) baseline_hash,
                (unsigned long long) hash(bytes(t.out)), (unsigned long long) disabled_count,
                (unsigned long long) enabled_count, (unsigned long long) want,
                different, largest, values_finite ? "true" : "false", protected_equal ? "true" : "false", ok ? "true" : "false");
    std::fflush(stdout);
    return ok;
}

int main(int argc, char ** argv) {
    try {
        const int expected = argc > 1 ? std::stoi(argv[1]) : 0;
        if (argc > 3 || (expected != 0 && expected != 1)) {
            throw std::runtime_error("usage: test-metal-head-store [expect-fusion: 0|1] [output.bin]");
        }
        std::unique_ptr<FILE, decltype(&std::fclose)> dump(argc > 2 ? ggml_fopen(argv[2], "wb") : nullptr, std::fclose);
        if (argc > 2 && !dump) throw std::runtime_error("cannot open the enabled output dump");
        ggml_backend_load_all();
        Backend backend(ggml_backend_init_by_name("MTL0", nullptr), ggml_backend_free);
        if (!backend) throw std::runtime_error("cannot initialize MTL0");
        const Fusion fusion(backend.get());
        check_poison_guard(backend.get());
        const auto cases = head_store_cases();
        size_t failed = 0;
        for (const auto & c : cases) failed += !run_case(backend.get(), fusion, c, expected != 0, dump.get());
        if (dump && std::fflush(dump.get()) != 0) throw std::runtime_error("cannot flush the enabled output dump");
        fusion.enable(fusion.info, true);
        std::printf("{\"kind\":\"summary\",\"poison_guard_cases\":2,\"cases\":%zu,\"failed\":%zu}\n", cases.size(), failed);
        return failed ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
}
