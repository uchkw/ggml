#pragma once

#include "ggml.h"

#include <vector>

// The positive cases cover Haas's two FFN-up products per encoder layer, including both sides of tile tails.
struct mm_silu_case {
    ggml_type type;
    int64_t m, n, k;
    const char * variant;
    bool fusable;
};

static std::vector<mm_silu_case> mm_silu_cases() {
    std::vector<mm_silu_case> cases;
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K}) {
        const std::vector<int64_t> columns = {17, 31, 32, 33, 38, 63, 64, 65, 75, 125, 171, 172, 250, 375};
        for (int64_t n : columns) cases.push_back({type, 4096, n, 1024, "plain", true});
        for (const char * variant : {"offset", "inplace", "consumer"}) {
            cases.push_back({type, 4096, 33, 1024, variant, true});
        }
        for (const char * variant : {"reuse", "output", "stride", "transpose", "batch", "relu"}) {
            cases.push_back({type, 4096, 33, 1024, variant, false});
        }
        for (int64_t n : {1, 8, 16}) cases.push_back({type, 4096, n, 1024, "plain", false});
        cases.push_back({type, 4095, 33, 1024, "plain", false});
        cases.push_back({type, 4096, 33, 1280, "plain", false});
    }
    cases.push_back({GGML_TYPE_F32, 4096, 33, 1024, "plain", false});
    cases.push_back({GGML_TYPE_BF16, 4096, 33, 1024, "plain", false});
    return cases;
}
