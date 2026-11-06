// Compares Metal's experimental 32-column product with a candidate tile on the same tensor buffers, then times
// them in alternating order. Usage: test-metal-mm-tile [test|perf] [rounds] [16|64]. JSON Lines go to stdout.

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;

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

static void compute(ggml_backend_t backend, ggml_cgraph * graph, int tile) {
    if (setenv("GGML_METAL_MM_TILE_N", std::to_string(tile).c_str(), 1) != 0) {
        throw std::runtime_error("cannot set GGML_METAL_MM_TILE_N");
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Metal graph computation failed");
    }
    ggml_backend_synchronize(backend);
}

static bool run_case(ggml_backend_t backend, ggml_type type, int64_t m, int64_t n, int64_t k, bool perf, int rounds, int candidate) {
    constexpr int repeats = 64;
    constexpr int graph_size = repeats + 16;
    ggml_init_params params{ggml_tensor_overhead() * 16 + 2 * ggml_graph_overhead_custom(graph_size, false), nullptr, true};
    Context context(ggml_init(params), ggml_free);
    if (!context) throw std::runtime_error("cannot allocate a test context");
    auto * a = ggml_new_tensor_2d(context.get(), type, k, m);
    auto * b = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, k, n);
    auto * out = ggml_mul_mat(context.get(), a, b);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    auto * graph = ggml_new_graph_custom(context.get(), graph_size, false);
    ggml_build_forward_expand(graph, out);
    auto * repeated = ggml_new_graph_custom(context.get(), graph_size, false);
    ggml_build_forward_expand(repeated, out);
    for (int i = 1; i < repeats; ++i) ggml_graph_add_node(repeated, out);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    if (!buffer) throw std::runtime_error("cannot allocate test tensors");

    const auto av = values((size_t) (m * k), 123456789u);
    const auto bv = values((size_t) (n * k), 362436069u);
    std::vector<uint8_t> packed(ggml_nbytes(a));
    const size_t written = ggml_quantize_chunk(type, av.data(), packed.data(), 0, m, k, nullptr);
    if (written != packed.size()) throw std::runtime_error("quantized tensor size differs from its allocation");
    ggml_backend_tensor_set(a, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(b, bv.data(), 0, ggml_nbytes(b));

    std::vector<float> first((size_t) (m * n)), second(first.size());
    compute(backend, graph, 32);
    ggml_backend_tensor_get(out, first.data(), 0, ggml_nbytes(out));
    compute(backend, graph, candidate);
    ggml_backend_tensor_get(out, second.data(), 0, ggml_nbytes(out));
    size_t different = 0;
    double max_abs = 0;
    bool finite = true;
    for (size_t i = 0; i < first.size(); ++i) {
        different += std::memcmp(&first[i], &second[i], sizeof(float)) != 0;
        finite = finite && std::isfinite(first[i]) && std::isfinite(second[i]);
        max_abs = std::max(max_abs, std::abs((double) first[i] - second[i]));
    }
    const bool equal = different == 0 && finite;
    const bool eligible = type != GGML_TYPE_Q4_K && n > 16 && (candidate != 16 || m == 1024);
    std::printf("{\"kind\":\"comparison\",\"type\":\"%s\",\"m\":%lld,\"n\":%lld,\"k\":%lld,"
                "\"baseline_tile\":32,\"candidate_tile\":%d,\"eligible_for_candidate\":%s,"
                "\"same_input_buffers\":true,\"input_a_fnv1a64\":\"%016llx\",\"input_b_fnv1a64\":\"%016llx\","
                "\"output_baseline_fnv1a64\":\"%016llx\",\"output_candidate_fnv1a64\":\"%016llx\","
                "\"bitwise_equal\":%s,\"finite\":%s,\"different_values\":%zu,\"max_abs\":%.17g}\n",
                ggml_type_name(type), (long long) m, (long long) n, (long long) k,
                candidate, eligible ? "true" : "false",
                (unsigned long long) hash_bytes(packed.data(), packed.size()),
                (unsigned long long) hash_bytes(bv.data(), ggml_nbytes(b)),
                (unsigned long long) hash_bytes(first.data(), ggml_nbytes(out)),
                (unsigned long long) hash_bytes(second.data(), ggml_nbytes(out)),
                equal ? "true" : "false", finite ? "true" : "false", different, max_abs);
    std::fflush(stdout);
    if (!equal || !perf) return equal;

    compute(backend, repeated, 32);
    compute(backend, repeated, candidate);
    for (int round = 0; round < rounds; ++round) {
        for (int choice = 0; choice < 2; ++choice) {
            const int tile = ((round + choice) % 2 == 0) ? 32 : candidate;
            if (setenv("GGML_METAL_MM_TILE_N", std::to_string(tile).c_str(), 1) != 0) {
                throw std::runtime_error("cannot set GGML_METAL_MM_TILE_N");
            }
            ggml_backend_synchronize(backend);
            const int64_t start = ggml_time_us();
            if (ggml_backend_graph_compute(backend, repeated) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("timed Metal graph computation failed");
            }
            ggml_backend_synchronize(backend);
            const double elapsed_ms = (ggml_time_us() - start) / 1000.0;
            std::printf("{\"kind\":\"timing\",\"type\":\"%s\",\"m\":%lld,\"n\":%lld,\"k\":%lld,"
                        "\"tile\":%d,\"round\":%d,\"order\":%d,\"repeats\":%d,\"graph_ms\":%.6f,\"op_ms\":%.9f}\n",
                        ggml_type_name(type), (long long) m, (long long) n, (long long) k,
                        tile, round, choice, repeats, elapsed_ms, elapsed_ms / repeats);
            std::fflush(stdout);
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "test";
        const int rounds = argc > 2 ? std::stoi(argv[2]) : 3;
        const int candidate = argc > 3 ? std::stoi(argv[3]) : 16;
        if (argc > 4 || (mode != "test" && mode != "perf") || rounds < 3 || (candidate != 16 && candidate != 64)) {
            throw std::runtime_error("usage: test-metal-mm-tile [test|perf] [rounds >= 3] [16|64]");
        }
        ggml_backend_load_all();
        Backend backend(ggml_backend_init_by_name("MTL0", nullptr), ggml_backend_free);
        if (!backend) throw std::runtime_error("cannot initialize MTL0");
        const std::vector<int> ns = mode == "perf" ? std::vector<int>{38, 75, 125, 172, 250, 375, 749} :
            std::vector<int>{15, 16, 17, 31, 32, 33, 38, 47, 48, 49, 63, 64, 65, 75, 96, 97, 125, 127, 128, 129, 171, 172, 250, 341, 375, 749};
        size_t cases = 0, failed = 0;
        for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K}) {
            for (const auto & mk : {std::array<int64_t, 2>{4096, 1024}, {1024, 4096}, {1024, 1024}}) {
                for (int n : ns) {
                    if (type == GGML_TYPE_Q4_K && n != 38 && n != 375) continue;
                    failed += !run_case(backend.get(), type, mk[0], n, mk[1], mode == "perf", rounds, candidate);
                    ++cases;
                }
            }
        }
        std::printf("{\"kind\":\"summary\",\"cases\":%zu,\"failed\":%zu}\n", cases, failed);
        return failed == 0 ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
}
