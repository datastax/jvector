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
package io.github.jbellis.jvector.quantization;

import com.sun.management.ThreadMXBean;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.disk.CommonHeader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Random;
import java.util.function.IntFunction;

import static org.junit.Assert.assertEquals;

/** Allocation regression and a reproducible exercise; timings include loop/volatile receiver costs. */
public class TestMemoryAccountingCost {
    private static volatile Accountable target;
    private static volatile long sink;

    @Test
    public void repeatedMemoryReadsDoNotAllocate() throws Exception {
        exercise(2048, true);
    }

    public static void main(String[] args) throws Exception {
        exercise(Integer.parseInt(args[0]), false);
    }

    private static void exercise(int iterations, boolean assertNoAllocation) throws Exception {
        var bean = (ThreadMXBean) ManagementFactory.getThreadMXBean();
        if (!bean.isThreadAllocatedMemoryEnabled()) bean.setThreadAllocatedMemoryEnabled(true);
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        var books = new VectorFloat<?>[48];
        var subspaces = new int[48][2];
        for (int j = 0; j < books.length; j++) {
            books[j] = types.createFloatVector(256);
            subspaces[j] = new int[]{1, j};
        }
        var pq = new ProductQuantization(books, 256, subspaces, null, KMeansPlusPlusClusterer.UNWEIGHTED);
        System.out.println("ACCOUNTING_LAYOUT graph=" + RamUsageEstimator.shallowSizeOfInstance(OnDiskGraphIndex.class)
                + " mutablePQ=" + RamUsageEstimator.shallowSizeOfInstance(MutablePQVectors.class)
                + " immutablePQ=" + RamUsageEstimator.shallowSizeOfInstance(ImmutablePQVectors.class)
                + " codeBuffer=" + RamUsageEstimator.shallowSizeOf(types.createByteSequence(48)));
        try {
            for (int upperNodes : new int[]{128, 1024}) {
                var path = Files.createTempFile("memory-cost-", ".index");
                try {
                    var graph = new TestUtil.RandomlyConnectedGraphIndex(List.of(
                            new CommonHeader.LayerInfo(2048, 8), new CommonHeader.LayerInfo(upperNodes, 4)), new Random(739));
                    var code = types.createByteSequence(48);
                    var vector = types.createFloatVector(48);
                    var states = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
                    try (var view = graph.getView(); var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                            .with(new InlineVectors(48)).with(new FusedPQ(8, pq)).build()) {
                        states.put(FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(vector));
                        states.put(FeatureId.FUSED_PQ, node -> new FusedPQ.State(view, ignored -> code, node));
                        writer.write(states);
                    }
                    long bytes = bean.getThreadAllocatedBytes(Thread.currentThread().getId());
                    long start = System.nanoTime();
                    try (var reader = new SimpleMappedReader.Supplier(path); var loaded = OnDiskGraphIndex.load(reader, 0)) {
                        long elapsed = System.nanoTime() - start;
                        long allocated = bean.getThreadAllocatedBytes(Thread.currentThread().getId()) - bytes;
                        System.out.println("ACCOUNTING_LOAD upperNodes=" + upperNodes + " ns=" + elapsed + " allocated=" + allocated);
                        measure("graph-" + upperNodes, loaded, iterations, assertNoAllocation, bean);
                    }
                } finally {
                    Files.delete(path);
                }
            }
            // A fused ASH graph: reporting reads its features' live totals, including the ASH model.
            var random = new Random(4127);
            var input = new VectorFloat<?>[97];
            for (int i = 0; i < input.length; i++) {
                input[i] = types.createFloatVector(64);
                for (int d = 0; d < 64; d++) input[i].set(d, random.nextFloat() - .5f);
                io.github.jbellis.jvector.vector.VectorUtil.l2normalize(input[i]);
            }
            var values = io.github.jbellis.jvector.graph.MockVectorValues.fromValues(input);
            var ash = AsymmetricHashing.initialize(values, io.github.jbellis.jvector.vector.VectorSimilarityFunction.DOT_PRODUCT,
                    AsymmetricHashing.RANDOM, AsymmetricHashing.HEADER_BITS + 33 * 2, 1, 2);
            var encoded = ash.encodeAll(values, java.util.concurrent.ForkJoinPool.commonPool());
            var ashPath = Files.createTempFile("memory-cost-ash-", ".index");
            try {
                var graph = new TestUtil.RandomlyConnectedGraphIndex(List.of(
                        new CommonHeader.LayerInfo(97, 8), new CommonHeader.LayerInfo(24, 4)), new Random(739));
                var states = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
                try (var view = graph.getView(); var writer = new OnDiskGraphIndexWriter.Builder(graph, ashPath)
                        .with(new io.github.jbellis.jvector.graph.disk.feature.FusedASH(8, ash)).with(new InlineVectors(64)).build()) {
                    states.put(FeatureId.FUSED_ASH, node -> new io.github.jbellis.jvector.graph.disk.feature.FusedASH.State(view, encoded, node));
                    states.put(FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(input[node]));
                    writer.write(states);
                }
                try (var reader = new SimpleMappedReader.Supplier(ashPath); var loaded = OnDiskGraphIndex.load(reader, 0)) {
                    measure("ash-graph", loaded, iterations, assertNoAllocation, bean);
                }
            } finally {
                Files.delete(ashPath);
            }
            for (int vectors : new int[]{1, 16385}) {
                var mutable = new MutablePQVectors(pq);
                mutable.setZero(vectors - 1);
                measure("mutablePQ-" + vectors, mutable, iterations, assertNoAllocation, bean);
                long retained = mutable.ramBytesUsed();
                for (int i = 0; i < 100; i++) mutable.setZero(0);
                assertEquals("Reusing capacity must not increase retained bytes", retained, mutable.ramBytesUsed());
            }
            var code = types.createByteSequence(384);
            measure("code-384", code, iterations, assertNoAllocation, bean);
            measure("slice-17", code.slice(1, 17), iterations, assertNoAllocation, bean);
        } finally {
            target = null;
        }
    }

    private static void measure(String name, Accountable value, int iterations, boolean check, ThreadMXBean bean) {
        target = value;
        // Warm the same bytecode/call site used for measurement. Separate loops can retain
        // different type profiles when switching from metadata sizing to a simple getter.
        sink = readMany(Math.min(200_000, Math.max(128, iterations)));
        if (!check && iterations >= 1_000_000) {
            // Long diagnostic runs allow the compilation queue to settle after changing
            // receiver types. Cold index-loading cost is reported separately above.
            long until = System.nanoTime() + 250_000_000L;
            do { sink = readMany(100_000); } while (System.nanoTime() < until);
        }
        double[] nanos = new double[5];
        long allocation = 0;
        long minAllocation = Long.MAX_VALUE;
        for (int sample = 0; sample < nanos.length; sample++) {
            long before = bean.getThreadAllocatedBytes(Thread.currentThread().getId());
            long start = System.nanoTime();
            long sum = readMany(iterations);
            long elapsed = System.nanoTime() - start;
            long allocated = bean.getThreadAllocatedBytes(Thread.currentThread().getId()) - before;
            sink = sum;
            nanos[sample] = (double) elapsed / iterations;
            allocation = Math.max(allocation, allocated);
            minAllocation = Math.min(minAllocation, allocated);
        }
        Arrays.sort(nanos);
        System.out.println("ACCOUNTING_READ name=" + name + " iterations=" + iterations + " medianNs=" + nanos[2]
                + " maxAllocatedBytes=" + allocation + " retained=" + value.ramBytesUsed());
        // A per-read allocation shows up in every sample. On JDK 11 a one-time JVM event can land in
        // one sample (a few hundred bytes, never per read), so one allocation-free sample is required.
        if (check) assertEquals(name + " reporting allocated bytes", 0, minAllocation);
    }

    private static long readMany(int iterations) {
        long sum = 0;
        for (int i = 0; i < iterations; i++) sum += target.ramBytesUsed();
        return sum;
    }
}
