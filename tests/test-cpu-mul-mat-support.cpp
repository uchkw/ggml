// test-cpu-mul-mat-support
//
// This test checks, from the tensor metadata alone, when the CPU device
// reports support for GGML_OP_MUL_MAT. An RHS (src1) whose rows are not
// packed, that is src1->nb[0] != ggml_type_size(src1->type), is refused,
// while packed rows stay supported whether or not their row stride is padded.
// The dtype and Hadamard hint rules keep their reach.
//
// The test allocates no tensor data (no_alloc) and builds no graph; it only
// queries ggml_backend_dev_supports_op() on the CPU device.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>

static int g_failures = 0;

static void check_support(ggml_backend_dev_t dev, const char * name, const struct ggml_tensor * op, bool expected) {
    const bool supported = ggml_backend_dev_supports_op(dev, op);

    const struct ggml_tensor * rhs = op->src[1];

    printf("%-58s : src1 %s, ne0 = %lld, nb0 = %zu (type size %zu), expected %s, got %s\n",
        name,
        ggml_type_name(rhs->type),
        (long long) rhs->ne[0],
        rhs->nb[0],
        ggml_type_size(rhs->type),
        expected ? "true" : "false",
        supported ? "true" : "false");

    if (supported != expected) {
        printf("-> FAIL: %s\n", name);
        g_failures++;
    }
}

int main(void) {
    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == NULL) {
        fprintf(stderr, "test-cpu-mul-mat-support: no CPU backend device found\n");
        return 1;
    }

    // The seven cases below create approximately 25 tensors, which fit comfortably in 64 descriptors.
    constexpr size_t max_tensors = 64;

    // no_alloc is set, so the context stores only tensor metadata.
    const struct ggml_init_params init_params = {
        max_tensors * ggml_tensor_overhead(),
        NULL,
        true,
    };

    struct ggml_context * ctx = ggml_init(init_params);
    if (ctx == NULL) {
        fprintf(stderr, "test-cpu-mul-mat-support: ggml_init failed\n");
        return 1;
    }

    {
        struct ggml_tensor * a  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);
        struct ggml_tensor * b  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);
        struct ggml_tensor * mm = ggml_mul_mat(ctx, a, b);

        check_support(dev, "packed F32 RHS", mm, true);
    }

    // The view pads the row stride (nb1 > ne0 * type size) while the rows stay packed.
    {
        struct ggml_tensor * a    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);
        struct ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 20, 4);
        struct ggml_tensor * b    = ggml_view_2d(ctx, rows, 16, 4, 20 * sizeof(float), 0);
        struct ggml_tensor * mm   = ggml_mul_mat(ctx, a, b);

        check_support(dev, "F32 RHS view, padded row stride, packed rows", mm, true);
    }

    // The transposed RHS has nb0 = 132, the source row stride, so its rows are not packed.
    {
        struct ggml_tensor * a    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 4);
        struct ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 33, 1024);
        struct ggml_tensor * b    = ggml_transpose(ctx, rows);
        struct ggml_tensor * mm   = ggml_mul_mat(ctx, a, b);

        check_support(dev, "transposed F32 RHS [1024, 33] from [33, 1024]", mm, false);
    }

    // A permuted RHS whose inner stride is not packed stays refused.
    {
        struct ggml_tensor * a    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 7, 3);
        struct ggml_tensor * rows = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 7, 3);
        struct ggml_tensor * b    = ggml_permute(ctx, rows, 2, 0, 1, 3);
        struct ggml_tensor * mm   = ggml_mul_mat(ctx, a, b);

        check_support(dev, "permuted F32 RHS, inner stride not packed, nb0 < nb1", mm, false);
    }

    // Even for K = 1, the transposed RHS has nb0 > sizeof(float), so its rows are not packed.
    {
        struct ggml_tensor * a    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2);
        struct ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
        struct ggml_tensor * b    = ggml_transpose(ctx, rows);
        struct ggml_tensor * mm   = ggml_mul_mat(ctx, a, b);

        check_support(dev, "transposed F32 RHS with K = 1, nb0 > sizeof(float)", mm, false);
    }

    // A packed F16 RHS with F32 weights and no hint stays refused by the dtype rule.
    {
        struct ggml_tensor * a  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);
        struct ggml_tensor * b  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 16, 4);
        struct ggml_tensor * mm = ggml_mul_mat(ctx, a, b);

        check_support(dev, "packed F16 RHS, F32 weights, no hint", mm, false);
    }

    // The same packed F16 RHS is supported with the Hadamard hint and square power-of-two weights.
    {
        struct ggml_tensor * a  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        struct ggml_tensor * b  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 4);
        struct ggml_tensor * mm = ggml_mul_mat(ctx, a, b);

        ggml_mul_mat_set_hint(mm, GGML_HINT_SRC0_IS_HADAMARD);

        check_support(dev, "packed F16 RHS, Hadamard hint, square weights", mm, true);
    }

    ggml_free(ctx);

    if (g_failures > 0) {
        fprintf(stderr, "test-cpu-mul-mat-support: %d case(s) failed\n", g_failures);
        return 1;
    }

    printf("test-cpu-mul-mat-support: all cases passed\n");

    return 0;
}
