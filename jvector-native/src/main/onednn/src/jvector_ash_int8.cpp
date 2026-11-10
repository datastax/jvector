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

// Query-only INT8 representation; FP32/FP16 implementations remain independent.
#include "jvector_ash_fp16.h"
#include <oneapi/dnnl/dnnl.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <vector>

static_assert(DNNL_CPU_RUNTIME == DNNL_RUNTIME_SEQ, "Calling workers own parallelism");
static_assert(offsetof(jvector_ash_fp16_result, message) == sizeof(int32_t));
static_assert(sizeof(jvector_ash_fp16_result) == sizeof(int32_t) + 256);

namespace {
using namespace dnnl;
constexpr int kBatchTile = 8; // Matches the existing FP16 native execution tile, not an API limit.
constexpr int kMaxColumns = 65535; // 255 * 127 * D remains within signed int32.
std::atomic<uint64_t> next_id{1};
struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };

int set_error(jvector_ash_fp16_result* result, int status, const char* message) noexcept {
    result->status = status;
    std::snprintf(result->message, sizeof(result->message), "%s", message);
    return status;
}

// Translate every native failure before it reaches FFM, using the existing ABI-1 statuses.
int current_error(jvector_ash_fp16_result* result) noexcept {
    try { throw; }
    catch (const Unsupported& e) { return set_error(result, JV_ASH_FP16_UNSUPPORTED, e.what()); }
    catch (const dnnl::error& e) {
        return set_error(result, e.status == dnnl_unimplemented ? JV_ASH_FP16_UNSUPPORTED
                : e.status == dnnl_out_of_memory ? JV_ASH_FP16_NO_MEMORY
                : e.status == dnnl_invalid_arguments ? JV_ASH_FP16_INVALID_ARGUMENT
                : JV_ASH_FP16_ERROR, e.what());
    }
    catch (const std::bad_alloc& e) { return set_error(result, JV_ASH_FP16_NO_MEMORY, e.what()); }
    catch (const std::invalid_argument& e) { return set_error(result, JV_ASH_FP16_INVALID_ARGUMENT, e.what()); }
    catch (const std::out_of_range& e) { return set_error(result, JV_ASH_FP16_RANGE, e.what()); }
    catch (const std::exception& e) { return set_error(result, JV_ASH_FP16_ERROR, e.what()); }
    catch (...) { return set_error(result, JV_ASH_FP16_ERROR, "Unknown INT8 projection failure"); }
}

bool integer_dot_available() {
#if defined(__x86_64__) || defined(_M_X64)
    // Honor oneDNN's effective ISA, including its documented runtime ISA cap.
    switch (get_effective_cpu_isa()) {
        case cpu_isa::avx2_vnni:
        case cpu_isa::avx2_vnni_2:
        case cpu_isa::avx512_core_vnni:
        case cpu_isa::avx512_core_bf16:
        case cpu_isa::avx512_core_fp16:
        case cpu_isa::avx512_core_amx:
        case cpu_isa::avx512_core_amx_fp16: return true;
        default: return false;
    }
#else
    return false; // Arm backends require separate qualification of their exact-dot primitive.
#endif
}

// Every qualified x86 VNNI/AMX backend includes AVX2. Keep the entry-point
// hardware check generic; specialize only these INT8 computation functions.
// This preserves direct FP32 rounding instructions in the portable library.
#if defined(__x86_64__) || defined(_M_X64)
#define JV_ASH_INT8_TARGET __attribute__((target("avx2")))
#else
#define JV_ASH_INT8_TARGET
#endif

// Share the prototype's FP32 quantization between matrix/query preparation.
// Reject a nonzero operand whose scale rounds to zero, before division and
// integer conversion. Do not change the ordinary arithmetic or rounding.
template <typename Code>
JV_ASH_INT8_TARGET float quantize(const float* input, int count, Code* output, int offset, int32_t* sum = nullptr) {
    float maxabs = 0;
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(input[i])) throw std::invalid_argument("Nonfinite INT8 operand");
        maxabs = std::max(maxabs, std::fabs(input[i]));
    }
    float scale = maxabs == 0 ? 1.0f : maxabs / 127.0f;
    if (scale == 0) throw std::invalid_argument("INT8 operand scale underflows FP32; rescale input");
    int32_t total = 0;
    for (int i = 0; i < count; ++i) {
        int q = std::clamp(int(std::nearbyint(input[i] / scale)), -127, 127);
        output[i] = Code(q + offset);
        total += q;
    }
    if (sum) *sum = total;
    return scale;
}

struct Kernel {
    matmul::primitive_desc descriptor;
    matmul operation;
    memory weights;
    Kernel(const engine& cpu, int rows, int columns, int count) {
        primitive_attr attributes;
        attributes.set_scratchpad_mode(scratchpad_mode::user);
        descriptor = matmul::primitive_desc(cpu,
                memory::desc({count, columns}, memory::data_type::u8, memory::format_tag::ab),
                memory::desc({columns, rows}, memory::data_type::s8, memory::format_tag::any),
                memory::desc({count, rows}, memory::data_type::s32, memory::format_tag::ab), attributes);
        // The pinned oneDNN exposes the selected implementation through this
        // public descriptor query. Accept its VNNI AND AMX integer kernels;
        // legacy u8*s8 kernels can saturate at a 16-bit intermediate and must not
        // silently compute a different dot product. Do not gate on CPU vendor.
        const char* implementation = descriptor.impl_info_str();
        if (!std::strstr(implementation, "vnni") && !std::strstr(implementation, "amx"))
            throw Unsupported("No qualified full-range VNNI/AMX INT8 projection kernel");
        operation = matmul(descriptor);
    }
};

struct Context {
    const uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
    const int rows, columns;
    engine cpu{engine::kind::cpu, 0};
    std::vector<float> scales;
    std::vector<int32_t> sums;
    // Retained once so additional oneDNN layouts never require another FP32
    // matrix copy or requantization. Compatible batch kernels share packed weights.
    memory plain;
    std::array<std::unique_ptr<Kernel>, kBatchTile + 1> kernels{};
    std::array<std::atomic<Kernel*>, kBatchTile + 1> ready{};
    std::mutex mutex;
    std::atomic<uint64_t> retained_bytes{0};

    Context(const float* matrix, int r, int c) : rows(r), columns(c), scales(r), sums(r) {
        for (auto& entry : ready) entry.store(nullptr, std::memory_order_relaxed);
        auto descriptor = memory::desc({columns, rows}, memory::data_type::s8, memory::dims{1, columns});
        plain = memory(descriptor, cpu);
        auto* target = static_cast<int8_t*>(plain.get_data_handle());
        for (int row = 0; row < rows; ++row)
            scales[row] = quantize(matrix + size_t(row) * columns, columns,
                    target + size_t(row) * columns, 0, &sums[row]);
        retained_bytes.store(descriptor.get_size() + sizeof(Context)
                + size_t(rows) * (sizeof(float) + sizeof(int32_t)), std::memory_order_relaxed);
        kernel(1); // Check required hardware/primitive during model initialization.
    }

    const Kernel& kernel(int count) {
        auto* prepared = ready[count].load(std::memory_order_acquire);
        if (!prepared) {
            std::lock_guard<std::mutex> guard(mutex);
            prepared = ready[count].load(std::memory_order_relaxed);
            if (!prepared) {
                auto next = std::make_unique<Kernel>(cpu, rows, columns, count);
                uint64_t added = sizeof(Kernel);
                for (const auto& other : kernels)
                    if (other && other->weights.get_desc() == next->descriptor.weights_desc()) {
                        next->weights = other->weights;
                        break;
                    }
                if (!next->weights) {
                    next->weights = memory(next->descriptor.weights_desc(), cpu);
                    stream setup(cpu);
                    reorder(plain, next->weights).execute(setup, plain, next->weights);
                    setup.wait();
                    added += next->descriptor.weights_desc().get_size();
                }
                kernels[count] = std::move(next);
                retained_bytes.fetch_add(added, std::memory_order_release);
                prepared = kernels[count].get();
                ready[count].store(prepared, std::memory_order_release);
            }
        }
        return *prepared;
    }
};

struct Workspace {
    uint64_t context_id = 0;
    std::unique_ptr<stream> execution;
    memory input, output, scratch;
    std::array<float, kBatchTile> scales{};
    void initialize(const Context& context, const Kernel& kernel) {
        context_id = 0; // Never publish partially replaced buffers after allocation failure.
        execution = std::make_unique<stream>(context.cpu);
        input = memory(kernel.descriptor.src_desc(), context.cpu);
        output = memory(kernel.descriptor.dst_desc(), context.cpu);
        scratch = memory(kernel.descriptor.scratchpad_desc(), context.cpu);
        context_id = context.id;
    }
};
// Bounded per-thread slots hold no Context or packed weights.
thread_local std::array<Workspace, kBatchTile + 1> workspaces;
}

extern "C" {
void* jvector_ash_int8_create(const float* matrix, int rows, int columns, jvector_ash_fp16_result* result) {
    if (!result) return nullptr;
    result->status = JV_ASH_FP16_OK;
    result->message[0] = 0;
    try {
        if (!matrix || rows <= 0 || columns <= 0
                || uint64_t(rows) * uint64_t(columns) > std::numeric_limits<size_t>::max() / sizeof(float))
            throw std::invalid_argument("Invalid INT8 matrix pointer or dimensions");
        if (columns > kMaxColumns) throw std::out_of_range("INT8 accumulator dimension exceeds 65535");
        if (!integer_dot_available()) throw Unsupported("INT8 projection requires an enabled x86 VNNI/AMX backend");
        return new Context(matrix, rows, columns);
    } catch (...) { current_error(result); return nullptr; }
}

void jvector_ash_int8_destroy(void* pointer) { delete static_cast<Context*>(pointer); }

uint64_t jvector_ash_int8_bytes(const void* pointer) {
    return pointer ? static_cast<const Context*>(pointer)->retained_bytes.load(std::memory_order_acquire) : 0;
}

JV_ASH_INT8_TARGET int jvector_ash_int8_project_batch(void* pointer, const float* input, int count,
        float* output, jvector_ash_fp16_result* result) {
    if (!result) return JV_ASH_FP16_INVALID_ARGUMENT;
    result->status = JV_ASH_FP16_OK;
    result->message[0] = 0;
    try {
        if (!pointer || !input || !output || count <= 0 || input == output)
            throw std::invalid_argument("Invalid INT8 projection buffers or count");
        auto& context = *static_cast<Context*>(pointer);
        if (uint64_t(count) > std::numeric_limits<size_t>::max() / sizeof(float)
                / uint64_t(std::max(context.rows, context.columns)))
            throw std::invalid_argument("INT8 batch size overflow");
        for (int first = 0; first < count;) {
            int size = std::min(kBatchTile, count - first);
            const auto& kernel = context.kernel(size);
            auto& buffers = workspaces[size];
            if (buffers.context_id != context.id) buffers.initialize(context, kernel);
            auto* quantized = static_cast<uint8_t*>(buffers.input.get_data_handle());
            for (int i = 0; i < size; ++i)
                buffers.scales[i] = quantize(input + size_t(first + i) * context.columns,
                        context.columns, quantized + size_t(i) * context.columns, 128);
            const dnnl_exec_arg_t args[] = {{DNNL_ARG_SRC, buffers.input.get()},
                    {DNNL_ARG_WEIGHTS, kernel.weights.get()}, {DNNL_ARG_DST, buffers.output.get()},
                    {DNNL_ARG_SCRATCHPAD, buffers.scratch.get()}};
            error::wrap_c_api(dnnl_primitive_execute(kernel.operation.get(), buffers.execution->get(), 4, args),
                    "INT8 projection execution failed");
            buffers.execution->wait();
            const auto* sums = static_cast<const int32_t*>(buffers.output.get_data_handle());
            for (int i = 0; i < size; ++i)
                for (int row = 0; row < context.rows; ++row) {
                    int32_t dot = sums[size_t(i) * context.rows + row] - 128 * context.sums[row];
                    // Preserve the prototype's FP32 rescaling and evaluation order.
                    output[size_t(first + i) * context.rows + row] =
                            float(dot) * buffers.scales[i] * context.scales[row];
                }
            first += size; // Cannot overflow even when count approaches INT_MAX.
        }
        return JV_ASH_FP16_OK;
    } catch (...) { return current_error(result); }
}
}
