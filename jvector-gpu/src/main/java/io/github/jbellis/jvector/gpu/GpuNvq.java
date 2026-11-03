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
package io.github.jbellis.jvector.gpu;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.disk.ByteBufferReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.NVQVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * NVQ encoding on the GPU: {@link #encodeAll} returns JVector's own {@link NVQVectors}, encoded as
 * {@link NVQuantization#encodeAll} would (8 bits a dimension, the same growth-rate search), one work-group per
 * (vector, sub-vector). It uses JVector's CPU code when TornadoVM isn't present or the quantizer isn't 8-bit.
 */
public final class GpuNvq {
    /** Rows encoded per GPU batch. */
    static final int BATCH = 32_768;

    private GpuNvq() {
    }

    /** Encodes every vector with {@code nvq} on the GPU. */
    public static NVQVectors encodeAll(NVQuantization nvq, RandomAccessVectorValues ravv) {
        if (nvq.bitsPerDimension != NVQuantization.BitsPerDimension.EIGHT || !GpuProductQuantization.runtimeAvailable()) {
            return nvq.encodeAll(ravv, PhysicalCoreExecutor.pool());
        }
        int n = ravv.size();
        int dim = ravv.dimension();
        int subs = nvq.subvectorSizesAndOffsets.length;
        int batch = Math.min(BATCH, n);
        IntArray sizes = new IntArray(subs);
        IntArray offsets = new IntArray(subs);
        for (int s = 0; s < subs; s++) {
            sizes.set(s, nvq.subvectorSizesAndOffsets[s][0]);
            offsets.set(s, nvq.subvectorSizesAndOffsets[s][1]);
        }
        FloatArray mean = new FloatArray(dim);
        for (int d = 0; d < dim; d++) {
            mean.set(d, nvq.globalMean.get(d));
        }
        FloatArray rows = new FloatArray(batch * dim);
        FloatArray stats = new FloatArray(batch * subs * 3);
        ByteArray codes = new ByteArray(batch * dim);
        IntArray params = IntArray.fromElements(batch, dim, subs, nvq.learn ? 1 : 0);
        KernelContext ctx = new KernelContext();
        TaskGraph tg = new TaskGraph("nvq")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, mean, sizes, offsets)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, rows, params)
                .task("encode", NvqKernels::encode, ctx, rows, mean, sizes, offsets, params, stats, codes)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, stats, codes);
        WorkerGrid grid = new WorkerGrid1D(batch * subs * NvqKernels.THREADS);
        grid.setLocalWork(NvqKernels.THREADS, 1, 1);
        GridScheduler grids = new GridScheduler("nvq.encode", grid);

        NVQuantization.QuantizedVector[] out = new NVQuantization.QuantizedVector[n];
        var supplier = ravv.threadLocalSupplier();
        int bytesPerVector = 4 + subs * 28 + dim;
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            plan.withGridScheduler(grids);
            for (int from = 0; from < n; from += batch) {
                int size = Math.min(batch, n - from);
                int first = from;
                MemorySegment rowSegment = rows.getSegment();
                IntStream.range(0, size).parallel().forEach(i -> HostRows.copyRow(supplier.get().getVector(first + i), rowSegment, i, dim));
                params.set(NvqKernels.P_ROWS, size);
                grid.setGlobalWork((long) size * subs * NvqKernels.THREADS, 1, 1);
                plan.execute();
                MemorySegment codeSegment = codes.getSegment();
                IntStream.range(0, size).parallel().forEach(i -> {
                    // JVector's own serialized form, read back with its loader: real QuantizedVectors
                    ByteBuffer bb = ByteBuffer.allocate(bytesPerVector);
                    bb.putInt(subs);
                    for (int s = 0; s < subs; s++) {
                        int stat = 3 * (i * subs + s);
                        int length = nvq.subvectorSizesAndOffsets[s][0];
                        bb.putInt(8);
                        bb.putFloat(stats.get(stat));
                        bb.putFloat(stats.get(stat + 1));
                        bb.putFloat(stats.get(stat + 2));
                        bb.putFloat(0.0f);
                        bb.putInt(length);
                        bb.putInt(length);
                        MemorySegment.copy(codeSegment, ValueLayout.JAVA_BYTE, (long) i * dim + nvq.subvectorSizesAndOffsets[s][1],
                                bb.array(), bb.position(), length);
                        bb.position(bb.position() + length);
                    }
                    bb.flip();
                    try {
                        out[first + i] = NVQuantization.QuantizedVector.load(new ByteBufferReader(bb));
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                });
            }
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return new NVQVectors(nvq, out);
    }
}
