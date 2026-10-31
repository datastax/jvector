/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "jvector_ash_fp16.h"
#include <oneapi/dnnl/dnnl.hpp>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

// Search callers own parallelism. This library must never start a nested pool.
static_assert(DNNL_CPU_RUNTIME == DNNL_RUNTIME_SEQ,
              "ASH FP16 requires the sequential oneDNN CPU runtime");

static_assert(offsetof(jvector_ash_fp16_result, message) == sizeof(int32_t));
static_assert(sizeof(jvector_ash_fp16_result) == sizeof(int32_t) + 256);

namespace {
using namespace dnnl;
std::atomic<uint64_t> next_id{1};
int set_error(jvector_ash_fp16_result* result, int status, const char* message) noexcept {
    result->status = status;
    std::snprintf(result->message, sizeof(result->message), "%s", message);
    return status;
}

bool representable(const float* values, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i]) || std::fabs(values[i]) > 65504.0f) return false;
    }
    return true;
}

struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };

struct Context {
    const uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
    const int rows, columns;
    engine cpu{engine::kind::cpu, 0};
    matmul::primitive_desc descriptor;
    matmul operation;
    memory weights;
    uint64_t weight_bytes = 0;

    Context(const float* matrix, int r, int c) : rows(r), columns(c) {
        // A is row-major [rows][columns]. Its transpose is the [columns][rows]
        // right operand, represented by strides instead of a second FP32 copy.
        const memory::dims shape{columns, rows}, strides{1, columns};
        auto source = memory(memory::desc(shape, memory::data_type::f32, strides),
                             cpu, const_cast<float*>(matrix));
        auto half = memory(memory::desc(shape, memory::data_type::f16, strides), cpu);
        stream setup(cpu);
        reorder(source, half).execute(setup, source, half);
        setup.wait();

        primitive_attr attributes;
        attributes.set_scratchpad_mode(scratchpad_mode::user);
        attributes.set_fpmath_mode(fpmath_mode::strict);
        attributes.set_accumulation_mode(accumulation_mode::strict);
        descriptor = matmul::primitive_desc(cpu,
            memory::desc({1, columns}, memory::data_type::f16, memory::format_tag::ab),
            memory::desc(shape, memory::data_type::f16, memory::format_tag::any),
            memory::desc({1, rows}, memory::data_type::f32, memory::format_tag::ab),
            attributes);
        if (std::strncmp(descriptor.impl_info_str(), "ref", 3) == 0)
            throw Unsupported("No optimized oneDNN FP16 kernel supports this model");
        operation = matmul(descriptor);
        if (descriptor.weights_desc() == half.get_desc()) weights = std::move(half);
        else {
            weights = memory(descriptor.weights_desc(), cpu);
            reorder(half, weights).execute(setup, half, weights);
            setup.wait();
        }
        weight_bytes = descriptor.weights_desc().get_size();
        // Only the prepared representation survives construction. In particular,
        // an additional unneeded copy of the FP16 matrix is not retained.
    }
};

// Calling-thread state owns scratch and library handles, never a Context or its
// weights. Switching models replaces this state; it cannot keep a model alive.
struct Workspace {
    uint64_t context_id = 0;
    std::unique_ptr<stream> execution;
    memory source_f32, source_f16, output, scratch;
    reorder conversion;

    void initialize(const Context& context, const float* input, float* out) {
        // Invalidate old state first: a failed allocation while switching models
        // must not leave the previous id paired with partially replaced buffers.
        context_id = 0;
        execution = std::make_unique<stream>(context.cpu);
        source_f32 = memory(memory::desc({1, context.columns}, memory::data_type::f32,
                             memory::format_tag::ab), context.cpu, const_cast<float*>(input));
        source_f16 = memory(context.descriptor.src_desc(), context.cpu);
        output = memory(context.descriptor.dst_desc(), context.cpu, out);
        scratch = memory(context.descriptor.scratchpad_desc(), context.cpu);
        conversion = reorder(source_f32, source_f16);
        context_id = context.id; // Publish only after every allocation succeeds.
    }
};
thread_local Workspace workspace;
}

extern "C" {
int jvector_ash_fp16_abi(void) { return 1; }

void* jvector_ash_fp16_create(const float* matrix, int rows, int columns, jvector_ash_fp16_result* result) {
    if (!result) return nullptr;
    result->message[0] = '\0';
    result->status = JV_ASH_FP16_INVALID_ARGUMENT;
    if (!matrix || rows <= 0 || columns <= 0
        || uint64_t(rows) * uint64_t(columns) > std::numeric_limits<size_t>::max() / sizeof(float)) {
        set_error(result, result->status, "Invalid matrix pointer or dimensions"); return nullptr;
    }
    if (!representable(matrix, size_t(rows) * size_t(columns))) {
        set_error(result, JV_ASH_FP16_RANGE, "Projection matrix is outside the finite FP16 range");
        return nullptr;
    }
    try {
        auto prepared = std::make_unique<Context>(matrix, rows, columns);
        result->status = JV_ASH_FP16_OK;
        return prepared.release();
    } catch (const Unsupported& e) {
        set_error(result, JV_ASH_FP16_UNSUPPORTED, e.what());
    } catch (const dnnl::error& e) {
        set_error(result, e.status == dnnl_unimplemented ? JV_ASH_FP16_UNSUPPORTED
                      : e.status == dnnl_out_of_memory ? JV_ASH_FP16_NO_MEMORY
                                                     : JV_ASH_FP16_ERROR, e.what());
    } catch (const std::bad_alloc& e) {
        set_error(result, JV_ASH_FP16_NO_MEMORY, e.what());
    } catch (const std::exception& e) {
        set_error(result, JV_ASH_FP16_ERROR, e.what());
    } catch (...) { set_error(result, JV_ASH_FP16_ERROR, "Unknown projection initialization failure"); }
    return nullptr;
}

void jvector_ash_fp16_destroy(void* pointer) { delete static_cast<Context*>(pointer); }

uint64_t jvector_ash_fp16_bytes(const void* pointer) {
    return pointer ? static_cast<const Context*>(pointer)->weight_bytes : 0;
}

int jvector_ash_fp16_project(void* pointer, const float* input, float* output,
                             jvector_ash_fp16_result* result) {
    if (!result) return JV_ASH_FP16_INVALID_ARGUMENT;
    result->message[0] = '\0';
    result->status = JV_ASH_FP16_OK;
    if (!pointer || !input || !output || input == output)
        return set_error(result, JV_ASH_FP16_INVALID_ARGUMENT, "Invalid or aliased projection buffers");
    auto& context = *static_cast<Context*>(pointer);
    if (!representable(input, context.columns))
        return set_error(result, JV_ASH_FP16_RANGE, "Query is outside the finite FP16 range");
    try {
        auto& buffers = workspace;
        if (buffers.context_id != context.id) buffers.initialize(context, input, output);
        buffers.source_f32.set_data_handle(const_cast<float*>(input));
        buffers.output.set_data_handle(output);
        // Stack argument arrays avoid the allocating C++ map overloads.
        const dnnl_exec_arg_t conversion[] = {
            {DNNL_ARG_FROM, buffers.source_f32.get()}, {DNNL_ARG_TO, buffers.source_f16.get()}};
        auto status = dnnl_primitive_execute(buffers.conversion.get(),
            buffers.execution->get(), 2, conversion);
        if (status != dnnl_success) return set_error(result, status == dnnl_out_of_memory
            ? JV_ASH_FP16_NO_MEMORY : JV_ASH_FP16_ERROR, "FP16 input conversion failed");
        const dnnl_exec_arg_t arguments[] = {{DNNL_ARG_SRC, buffers.source_f16.get()},
            {DNNL_ARG_WEIGHTS, context.weights.get()}, {DNNL_ARG_DST, buffers.output.get()},
            {DNNL_ARG_SCRATCHPAD, buffers.scratch.get()}};
        status = dnnl_primitive_execute(context.operation.get(), buffers.execution->get(), 4, arguments);
        if (status != dnnl_success) return set_error(result, status == dnnl_out_of_memory
            ? JV_ASH_FP16_NO_MEMORY : JV_ASH_FP16_ERROR, "FP16 projection execution failed");
        buffers.execution->wait();
        return JV_ASH_FP16_OK;
    } catch (const dnnl::error& e) { return set_error(result, e.status == dnnl_out_of_memory
        ? JV_ASH_FP16_NO_MEMORY : JV_ASH_FP16_ERROR, e.what()); }
    catch (const std::bad_alloc& e) { return set_error(result, JV_ASH_FP16_NO_MEMORY, e.what()); }
    catch (const std::exception& e) { return set_error(result, JV_ASH_FP16_ERROR, e.what()); }
    catch (...) { return set_error(result, JV_ASH_FP16_ERROR, "Unknown projection execution failure"); }
}

}
