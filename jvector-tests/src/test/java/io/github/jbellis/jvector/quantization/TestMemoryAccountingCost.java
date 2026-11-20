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

/** Allocation regression and a reproducible exercise; timings include loop/volatile receiver costs. Cold loading time and allocation are reported separately. */
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
        // Before JDK 21 the per-thread allocation counter can itself allocate (JDK 11 boxes the
        // thread id into two long[] per read); measure that so it is not charged to the loop.
        long overhead = 0;
        for (int i = 0; i < 3; i++) {
            long first = bean.getThreadAllocatedBytes(Thread.currentThread().getId());
            long second = bean.getThreadAllocatedBytes(Thread.currentThread().getId());
            overhead = Math.max(overhead, second - first);
        }
        double[] nanos = new double[5];
        long quietest = Long.MAX_VALUE;
        long loudest = 0;
        for (int sample = 0; sample < nanos.length; sample++) {
            long before = bean.getThreadAllocatedBytes(Thread.currentThread().getId());
            long start = System.nanoTime();
            long sum = readMany(iterations);
            long elapsed = System.nanoTime() - start;
            long allocated = Math.max(0, bean.getThreadAllocatedBytes(Thread.currentThread().getId()) - before - overhead);
            sink = sum;
            nanos[sample] = (double) elapsed / iterations;
            quietest = Math.min(quietest, allocated);
            loudest = Math.max(loudest, allocated);
        }
        Arrays.sort(nanos);
        System.out.println("ACCOUNTING_READ name=" + name + " iterations=" + iterations + " medianNs=" + nanos[2]
                + " minAllocatedBytes=" + quietest + " maxAllocatedBytes=" + loudest
                + " samplerBytes=" + overhead + " retained=" + value.ramBytesUsed());
        // A per-call allocation shows up in every sample; a one-off JIT transition (recompiling
        // the loop after its call site turns megamorphic) can land in one. Judge the steady state.
        if (check) assertEquals(name + " steady-state reporting allocated bytes", 0, quietest);
    }

    private static long readMany(int iterations) {
        long sum = 0;
        for (int i = 0; i < iterations; i++) sum += target.ramBytesUsed();
        return sum;
    }
}
