// Compares Conformer's sigmoid(gate) times a with Metal fusion disabled and enabled on identical inputs.
// Usage: test-metal-conformer-glu [expect-fusion: 0|1] [output.bin]. JSON Lines describe each case.
// The optional binary holds every enabled result for strict comparisons between original, env=0 and env=1 builds.

#include "ggml.h"
#include "ggml-backend.h"
#include "../src/ggml-backend-impl.h"
#include "test-conformer-glu-cases.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

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
        for (int i = 0; i < n; ++i) if (std::strcmp(labels[i], "UNARY+MUL") == 0) result += counts[i];
        return result;
    }
};

static std::vector<uint8_t> bytes(const ggml_tensor * tensor) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    return result;
}

static bool finite(const ggml_tensor * tensor, const std::vector<uint8_t> & data) {
    for (size_t offset = 0; offset < data.size(); offset += ggml_type_size(tensor->type)) {
        float value;
        if (tensor->type == GGML_TYPE_F32) std::memcpy(&value, data.data() + offset, sizeof(value));
        else {
            ggml_fp16_t half;
            std::memcpy(&half, data.data() + offset, sizeof(half));
            value = ggml_fp16_to_fp32(half);
        }
        if (!std::isfinite(value)) return false;
    }
    return true;
}

/** Poisons every output byte outside the active input ranges, including the tail of a partial alias. */
static size_t poison(const std::vector<ggml_tensor *> & outputs, const ggml_tensor * gate, const ggml_tensor * a) {
    size_t protected_values = 0;
    for (auto * out : outputs) {
        require(ggml_is_contiguous(out), "output poison requires a contiguous tensor");
        const size_t element = ggml_type_size(out->type);
        const auto begin = reinterpret_cast<uintptr_t>(out->data);
        size_t run = 0;
        for (size_t offset = 0; offset <= ggml_nbytes(out); offset += element) {
            bool protect = offset == ggml_nbytes(out);
            if (!protect) {
                for (const auto * input : {gate, a}) {
                    const auto first = reinterpret_cast<uintptr_t>(input->data);
                    if (begin + offset < first + ggml_nbytes(input) && first < begin + offset + element) protect = true;
                }
            }
            if (!protect) continue;
            if (offset > run) {
                conformer_glu_write(out, std::vector<float>((offset - run)/element, std::numeric_limits<float>::quiet_NaN()), run);
            }
            if (offset < ggml_nbytes(out)) ++protected_values;
            run = offset + element;
        }
    }
    return protected_values;
}

static size_t optimize(ggml_backend_t backend, ggml_cgraph * graph) {
    require(backend->iface.graph_optimize, "Metal graph optimization is unavailable");
    size_t dependencies = 0;
    ggml_backend_graph_optimize_params params{
        [](void * data, ggml_tensor *, ggml_tensor *) { ++*static_cast<size_t *>(data); }, &dependencies};
    backend->iface.graph_optimize(backend, graph, &params);
    return dependencies;
}

static void poison_guard(ggml_backend_t backend) {
    Context ctx(ggml_init({ggml_tensor_overhead()*4, nullptr, true}), ggml_free);
    require(bool(ctx), "cannot allocate poison guard context");
    auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    auto * wrong = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    auto * out = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(buffer), "cannot allocate poison guard buffer");
    const std::vector<float> expected{2, 3, 5, 7};
    conformer_glu_write(input, expected);
    const auto before = bytes(input);
    for (bool wrong_destination : {false, true}) {
        conformer_glu_write(out, expected);
        const auto faulty_store = [&]() {
            conformer_glu_write(wrong_destination ? wrong : out, wrong_destination ? expected : std::vector<float>{2, 3, 5});
        };
        faulty_store();
        const auto stale = bytes(out);
        poison({wrong, out}, input, input);
        faulty_store();
        const bool ok = stale == before && !finite(out, bytes(out)) && bytes(input) == before;
        std::printf("{\"kind\":\"poison_guard\",\"fault\":\"%s\",\"gpu_dispatch\":false,\"passed\":%s}\n",
                    wrong_destination ? "wrong-destination" : "skip-last-store", ok ? "true" : "false");
        require(ok, "poison guard does not detect an incomplete store");
    }
}

static bool run_case(ggml_backend_t backend, const Fusion & fusion, const conformer_glu_case & c, bool expected, FILE * dump) {
    constexpr int nodes = 32;
    Context ctx(ggml_init({ggml_tensor_overhead()*nodes + ggml_graph_overhead_custom(nodes, false), nullptr, true}), ggml_free);
    require(bool(ctx), "cannot allocate the GLU context");
    const auto tensors = conformer_glu_build(ctx.get(), c);
    ggml_set_output(tensors.out);
    auto * graph = ggml_new_graph_custom(ctx.get(), nodes, false);
    ggml_build_forward_expand(graph, tensors.out);
    const size_t dependencies = optimize(backend, graph);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(buffer), "cannot allocate GLU tensors");
    size_t protected_values = 0;
    const auto prepare = [&]() {
        conformer_glu_initialize(tensors, c);
        const auto gate_before = bytes(tensors.gate), a_before = bytes(tensors.a);
        protected_values = poison({tensors.unary, tensors.mul, tensors.out}, tensors.gate, tensors.a);
        require(bytes(tensors.gate) == gate_before && bytes(tensors.a) == a_before, "output poison changes an active input");
    };
    const auto compute = [&](bool enabled) {
        fusion.enable(fusion.info, enabled);
        fusion.reset(fusion.info);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Metal GLU computation fails");
        ggml_backend_synchronize(backend);
    };
    prepare(); compute(false);
    const auto ordinary = bytes(tensors.out);
    const auto saved_unary = std::string(c.variant) == "output" ? bytes(tensors.unary) : std::vector<uint8_t>{};
    const uint64_t disabled = fusion.count();
    prepare(); compute(true);
    const auto selected = bytes(tensors.out);
    const uint64_t enabled = fusion.count(), want = expected && c.fusable ? 1 : 0;
    const bool protected_equal = saved_unary.empty() || (bytes(tensors.unary) == saved_unary && finite(tensors.unary, saved_unary));
    const bool ok = ordinary == selected && finite(tensors.out, ordinary) && finite(tensors.out, selected) &&
                    disabled == 0 && enabled == want && dependencies == 0 && protected_equal;
    if (dump) {
        const uint64_t size = selected.size();
        require(std::fwrite(&size, sizeof(size), 1, dump) == 1 &&
                    std::fwrite(selected.data(), 1, selected.size(), dump) == selected.size(), "cannot write GLU comparison bytes");
    }
    std::printf("{\"kind\":\"comparison\",\"type\":\"%s\",\"width\":%lld,\"frames\":%lld,\"swap\":%s,\"variant\":\"%s\","
                "\"equal_bits\":%s,\"same_input_buffers\":true,\"poison_protected_input_values\":%zu,\"allocation_dependencies\":%zu,"
                "\"sigmoid_marked_output\":%s,\"disabled_fusions\":%llu,\"enabled_fusions\":%llu,\"expected_fusions\":%llu,\"passed\":%s}\n",
                ggml_type_name(c.type), (long long) c.width, (long long) c.frames, c.swap ? "true" : "false", c.variant,
                ordinary == selected ? "true" : "false", protected_values, dependencies,
                (tensors.unary->flags & GGML_TENSOR_FLAG_OUTPUT) ? "true" : "false",
                (unsigned long long) disabled, (unsigned long long) enabled, (unsigned long long) want, ok ? "true" : "false");
    std::fflush(stdout);
    return ok;
}

int main(int argc, char ** argv) {
    try {
        const int expected = argc > 1 ? std::stoi(argv[1]) : 0;
        require(argc <= 3 && (expected == 0 || expected == 1), "usage: test-metal-conformer-glu [expect-fusion: 0|1] [output.bin]");
        std::unique_ptr<FILE, decltype(&std::fclose)> dump(argc > 2 ? ggml_fopen(argv[2], "wb") : nullptr, std::fclose);
        require(argc <= 2 || bool(dump), "cannot open GLU result file");
        ggml_backend_load_all();
        Backend backend(ggml_backend_init_by_name("MTL0", nullptr), ggml_backend_free);
        require(bool(backend), "cannot initialize MTL0");
        const Fusion fusion(backend.get());
        poison_guard(backend.get());
        const auto cases = conformer_glu_cases();
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
