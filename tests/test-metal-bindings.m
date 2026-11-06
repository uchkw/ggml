// Checks Metal buffer binding transitions through the production encoder wrappers.
// Usage: test-metal-bindings [output.bin]. The optional file records every output for original/off/on comparison.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include "ggml-metal-device.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { output_words = 32*64, stride_bytes = 256 };

static void require(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "test-metal-bindings: %s\n", message);
        exit(1);
    }
}

struct fixture {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> a;
    id<MTLBuffer> b;
    id<MTLBuffer> mapped_a;
    id<MTLBuffer> mapped_b;
    struct ggml_metal_pipeline_with_params plain;
    struct ggml_metal_pipeline_with_params other;
    struct ggml_metal_pipeline_with_params high;
};

struct execution {
    ggml_metal_encoder_t encoder;
    id<MTLBuffer> output;
    uint32_t expected[output_words];
    int count;
};

struct result {
    uint32_t output[output_words];
    int dispatches;
};

static void bind(ggml_metal_encoder_t enc, id<MTLBuffer> buffer, size_t offset, int index) {
    ggml_metal_encoder_set_buffer(enc, (struct ggml_metal_buffer_id) { buffer, offset }, index);
}

/** Each store has an aligned destination and untouched guard words around it. The barrier
    after every dispatch orders a dispatch that reads what an earlier one wrote, as the Metal
    backend orders dependent nodes on an encoder of either dispatch type. */
static void dispatch_value(struct execution * e, const struct fixture * f, uint32_t argument,
                           uint32_t expected, int mode) {
    require(e->count < 32, "too many dispatches in a case");
    const bool high = mode == 2;
    ggml_metal_encoder_set_pipeline(e->encoder, high ? f->high : mode == 1 ? f->other : f->plain);
    bind(e->encoder, e->output, e->count*stride_bytes, high ? 29 : 1);
    ggml_metal_encoder_set_bytes(e->encoder, &argument, sizeof(argument), high ? 28 : 2);
    ggml_metal_encoder_dispatch_threadgroups(e->encoder, 1, 1, 1, 1, 1, 1);
    ggml_metal_encoder_memory_barrier(e->encoder);
    e->expected[e->count*64] = expected;
    e->count++;
}

static void read_buffer(struct execution * e, const struct fixture * f, id<MTLBuffer> buffer,
                        size_t offset, uint32_t argument, uint32_t expected, int mode) {
    bind(e->encoder, buffer, offset, mode == 2 ? 30 : 0);
    dispatch_value(e, f, argument, expected, mode);
}

static const char * case_names[] = {
    "same-pair", "offsets", "different-buffer", "bytes-then-buffer", "changed-stack-bytes",
    "nil-then-buffer", "pipeline-change", "highest-slot", "dependent-dispatches", "mapped-identity",
};

static struct result run_case(const struct fixture * f, int variant, bool concurrent) {
    @autoreleasepool {
        id<MTLCommandBuffer> command = [f->queue commandBuffer];
        require(command != nil, "cannot create a command buffer");
        struct execution e = {0};
        memset(e.expected, 0xa5, sizeof(e.expected));
        e.output = [f->device newBufferWithBytes:e.expected length:sizeof(e.expected)
                                           options:MTLResourceStorageModeShared];
        require(e.output != nil, "cannot create an output buffer");
        e.encoder = ggml_metal_encoder_init(command, concurrent);
        require(e.encoder != NULL, "cannot create an encoder wrapper");
        switch (variant) {
            case 0:
                bind(e.encoder, f->a, 0, 0);
                bind(e.encoder, f->a, 0, 0);
                read_buffer(&e, f, f->a, 0, 3, 14, 0);
                read_buffer(&e, f, f->a, 0, 7, 18, 0);
                break;
            case 1:
                read_buffer(&e, f, f->a, 0,   3, 14, 0);
                read_buffer(&e, f, f->a, 256, 3, 40, 0);
                read_buffer(&e, f, f->a, 512, 3, 76, 0);
                read_buffer(&e, f, f->a, 0,   3, 14, 0);
                break;
            case 2:
                read_buffer(&e, f, f->a, 0, 3, 14, 0);
                read_buffer(&e, f, f->b, 0, 3, 104, 0);
                read_buffer(&e, f, f->a, 0, 3, 14, 0);
                break;
            case 3: {
                read_buffer(&e, f, f->a, 0, 3, 14, 0);
                uint32_t value = 29;
                ggml_metal_encoder_set_bytes(e.encoder, &value, sizeof(value), 0);
                dispatch_value(&e, f, 3, 32, 0);
                read_buffer(&e, f, f->a, 0, 3, 14, 0);
                break;
            }
            case 4: {
                uint32_t value = 13;
                ggml_metal_encoder_set_bytes(e.encoder, &value, sizeof(value), 0);
                dispatch_value(&e, f, 5, 18, 0);
                value = 47;
                ggml_metal_encoder_set_bytes(e.encoder, &value, sizeof(value), 0);
                dispatch_value(&e, f, 9, 56, 0);
                read_buffer(&e, f, f->a, 0, 3, 14, 0);
                break;
            }
            case 5:
                bind(e.encoder, nil, 0, 0);
                read_buffer(&e, f, f->a, 0, 3, 14, 0);
                bind(e.encoder, nil, 0, 0);
                bind(e.encoder, nil, 0, 0);
                read_buffer(&e, f, f->a, 0, 7, 18, 0);
                break;
            case 6:
                read_buffer(&e, f, f->a, 0, 3, 14, 0);
                read_buffer(&e, f, f->a, 0, 3, 8, 1);
                read_buffer(&e, f, f->a, 0, 7, 18, 0);
                break;
            case 7: {
                read_buffer(&e, f, f->a, 0, 3, 14, 2);
                read_buffer(&e, f, f->a, 256, 3, 40, 2);
                uint32_t value = 59;
                ggml_metal_encoder_set_bytes(e.encoder, &value, sizeof(value), 30);
                dispatch_value(&e, f, 3, 62, 2);
                read_buffer(&e, f, f->a, 256, 3, 40, 2);
                break;
            }
            case 8:
                read_buffer(&e, f, f->a, 0, 3, 14, 0);
                read_buffer(&e, f, e.output, 0, 7, 21, 0);
                read_buffer(&e, f, e.output, 256, 5, 26, 0);
                break;
            case 9:
                // The two wrappers are distinct MTLBuffer objects over one host page, and the
                // bindings below transition between them. Their storage is the same, so equal
                // results cannot alone show that each transition re-bound the slot; a later
                // count of the encoder's setBuffer calls must confirm it.
                read_buffer(&e, f, f->mapped_a, 0, 3, 214, 0);
                read_buffer(&e, f, f->mapped_b, 0, 7, 218, 0);
                read_buffer(&e, f, f->mapped_b, 256, 3, 260, 0);
                read_buffer(&e, f, f->mapped_a, 0, 3, 214, 0);
                break;
            default:
                require(false, "unknown binding case");
        }
        ggml_metal_encoder_end_encoding(e.encoder);
        ggml_metal_encoder_free(e.encoder);
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "Metal: %s\n", [[command.error description] UTF8String]);
            require(false, "command buffer failed");
        }
        require(e.count > 0, "case did not dispatch a kernel");
        require(memcmp(e.output.contents, e.expected, sizeof(e.expected)) == 0,
                "output or surrounding guard words differ");
        struct result result;
        memcpy(result.output, e.output.contents, sizeof(result.output));
        result.dispatches = e.count;
        [e.output release];
        return result;
    }
}

static void record(FILE * file, const char * name, bool concurrent, const struct result * result) {
    printf("{\"case\":\"%s\",\"concurrent\":%s,\"dispatches\":%d,\"guards_exact\":true,\"passed\":true}\n",
           name, concurrent ? "true" : "false", result->dispatches);
    if (file) require(fwrite(result->output, sizeof(result->output), 1, file) == 1, "cannot write output file");
}

int main(int argc, char ** argv) {
    require(argc <= 2, "usage: test-metal-bindings [output.bin]");
    @autoreleasepool {
        ggml_metal_device_t dev = ggml_metal_device_init(0, 1);
        require(dev != NULL, "cannot initialize the Metal device");
        const char * source =
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void read_add(constant uint * x [[buffer(0)]], device uint * y [[buffer(1)]], "
            "constant uint & a [[buffer(2)]]) { y[0] = x[0] + a; }\n"
            "kernel void read_xor(constant uint * x [[buffer(0)]], device uint * y [[buffer(1)]], "
            "constant uint & a [[buffer(2)]]) { y[0] = x[0] ^ a; }\n"
            "kernel void read_high(constant uint * x [[buffer(30)]], device uint * y [[buffer(29)]], "
            "constant uint & a [[buffer(28)]]) { y[0] = x[0] + a; }\n";
        ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, source, true);
        require(lib != NULL, "cannot compile binding check kernels");
        struct fixture f = {0};
        f.device = ggml_metal_device_get_obj(dev);
        f.queue = [f.device newCommandQueue];
        require(f.queue != nil, "cannot create a Metal queue");
        f.plain = ggml_metal_library_compile_pipeline(lib, "read_add", "read_add", NULL);
        f.other = ggml_metal_library_compile_pipeline(lib, "read_xor", "read_xor", NULL);
        f.high = ggml_metal_library_compile_pipeline(lib, "read_high", "read_high", NULL);
        require(f.plain.pipeline && f.other.pipeline && f.high.pipeline, "cannot compile check pipelines");
        uint32_t a[256] = {11}, b[256] = {101};
        a[64] = 37;
        a[128] = 73;
        f.a = [f.device newBufferWithBytes:a length:sizeof(a) options:MTLResourceStorageModeShared];
        f.b = [f.device newBufferWithBytes:b length:sizeof(b) options:MTLResourceStorageModeShared];
        require(f.a && f.b, "cannot allocate input buffers");
        const long page = sysconf(_SC_PAGESIZE);
        require(page >= 1024, "cannot obtain the host page size");
        void * host = NULL;
        require(posix_memalign(&host, (size_t) page, (size_t) page) == 0, "cannot allocate mapped storage");
        memset(host, 0, (size_t) page);
        ((uint32_t *) host)[0] = 211;
        ((uint32_t *) host)[64] = 257;
        f.mapped_a = [f.device newBufferWithBytesNoCopy:host length:(size_t) page
                                              options:MTLResourceStorageModeShared deallocator:nil];
        f.mapped_b = [f.device newBufferWithBytesNoCopy:host length:(size_t) page
                                              options:MTLResourceStorageModeShared deallocator:nil];
        require(f.mapped_a && f.mapped_b && f.mapped_a != f.mapped_b &&
                f.mapped_a.contents == host && f.mapped_b.contents == host, "distinct mappings are unavailable");
        FILE * file = argc == 2 ? ggml_fopen(argv[1], "wb") : NULL;
        require(argc != 2 || file != NULL, "cannot open output file");
        int cases = 0;
        for (int concurrent = 0; concurrent <= 1; ++concurrent) {
            for (int variant = 0; variant < 10; ++variant) {
                struct result result = run_case(&f, variant, concurrent);
                record(file, case_names[variant], concurrent, &result);
                ++cases;
            }
            // A new encoder must bind from nothing again. Every case builds its own encoder,
            // command buffer and buffers, so no case shows reuse of one encoder or one
            // allocator across cases; that needs a later comparison of binding counts.
            struct result repeated = run_case(&f, 0, concurrent);
            record(file, "fresh-encoder", concurrent, &repeated);
            ++cases;
        }
        struct result * pair = calloc(2, sizeof(struct result));
        require(pair != NULL, "cannot allocate parallel results");
        const struct fixture * parallel_fixture = &f;
        dispatch_apply(2, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            pair[i] = run_case(parallel_fixture, i == 0 ? 1 : 3, true);
        });
        record(file, "parallel-offsets", true, &pair[0]);
        record(file, "parallel-bytes", true, &pair[1]);
        cases += 2;
        free(pair);
        if (file) require(fclose(file) == 0, "cannot close output file");
        printf("{\"kind\":\"summary\",\"cases\":%d,\"failed\":0}\n", cases);
        [f.mapped_b release];
        [f.mapped_a release];
        free(host);
        [f.b release];
        [f.a release];
        [f.queue release];
        ggml_metal_library_free(lib);
        ggml_metal_device_free(dev);
        return 0;
    }
}
