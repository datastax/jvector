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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Preferred oneDNN FP32 GEMM for bulk base-vector projection, with Highway fallback.
 * Owns no second matrix: the immutable row-major Highway matrix is shared.
 * Single-vector projection and query precision selection stay with the delegate.
 * Library scratch and JIT preparation are included in the caller's first use.
 */
final class NativeASHFP32BatchProjection implements VectorUtilSupport.AshProjection {
    private static final class Bindings {
        static final MethodHandle BATCH = load();
        private static MethodHandle load() {
            // The optional package already embeds sequential oneDNN for FP16
            // queries. Older packages simply lack the new FP32 entry point.
            if (!LibraryLoader.loadAshFp16()) return null;
            try {
                SymbolLookup lookup = SymbolLookup.loaderLookup();
                MethodHandle abi = Linker.nativeLinker().downcallHandle(
                        lookup.find("jvector_ash_fp16_abi")
                                .orElseThrow(() -> new UnsatisfiedLinkError("Missing ASH add-on ABI")),
                        FunctionDescriptor.of(JAVA_INT));
                if ((int) abi.invokeExact() != 1)
                    throw new UnsatisfiedLinkError("Unsupported ASH add-on ABI");
                return lookup.find("jvector_ash_fp32_project_batch")
                        .map(symbol -> Linker.nativeLinker().downcallHandle(symbol,
                                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS,
                                        JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS)))
                        .orElse(null);
            } catch (UnsatisfiedLinkError | IllegalCallerException | SecurityException
                     | UnsupportedOperationException unavailable) {
                return null;
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new AssertionError("Unexpected ASH FP32 binding failure", failure);
            }
        }
    }

    private final NativeASHProjection fallback;

    static VectorUtilSupport.AshProjection prepare(NativeASHProjection fallback) {
        return Bindings.BATCH == null ? fallback : new NativeASHFP32BatchProjection(fallback);
    }

    private NativeASHFP32BatchProjection(NativeASHProjection fallback) {
        this.fallback = fallback;
    }

    @Override
    public long ramBytesUsed() { return fallback.ramBytesUsed(); }

    @Override
    public void project(float[] input, float[] output) { fallback.project(input, output); }

    @Override
    public VectorUtilSupport.AshProjection forQueryScoring() { return fallback.forQueryScoring(); }

    @Override
    public void projectBatch(float[][] inputs, float[][] outputs, int count) {
        VectorUtilSupport.AshProjection.validateBatch(inputs, outputs, count);
        if (count == 0) return;
        final int rows = fallback.rows(), columns = fallback.columns();
        if (inputs[0].length != columns || outputs[0].length != rows)
            throw new IllegalArgumentException("ASH projection batch shape mismatch");
        long inputRowBytes = (long) columns * Float.BYTES;
        long outputRowBytes = (long) rows * Float.BYTES;
        NativeASHProjection.Buffers buffers = NativeASHProjection.buffers(
                Math.multiplyExact(count, inputRowBytes), Math.multiplyExact(count, outputRowBytes));
        for (int i = 0; i < count; i++)
            MemorySegment.copy(MemorySegment.ofArray(inputs[i]), 0, buffers.input,
                    i * inputRowBytes, inputRowBytes);
        final int status;
        try {
            status = (int) Bindings.BATCH.invokeExact(fallback.matrix(), buffers.input,
                    rows, columns, count, buffers.output);
        } catch (RuntimeException | Error error) {
            throw error;
        } catch (Throwable error) {
            throw new AssertionError("Unexpected exception from oneDNN FP32 projection", error);
        }
        // oneDNN status values are translated by the C wrapper. Unsupported
        // kernels retain Highway; allocation/runtime failures are never hidden.
        if (status == 3) {
            fallback.projectBatch(inputs, outputs, count);
            return;
        }
        if (status == 1) throw new OutOfMemoryError("oneDNN FP32 projection allocation failed");
        if (status == 2) throw new IllegalArgumentException("Invalid oneDNN FP32 projection arguments");
        if (status != 0) throw new IllegalStateException("oneDNN FP32 projection failed: " + status);
        for (int i = 0; i < count; i++)
            MemorySegment.copy(buffers.output, i * outputRowBytes,
                    MemorySegment.ofArray(outputs[i]), 0, outputRowBytes);
    }
}
