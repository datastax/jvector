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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.vector.*;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.*;
import static org.junit.Assert.*;

public class TestBufferedExactQueries {
    private static final java.util.concurrent.atomic.AtomicInteger pointReads = new java.util.concurrent.atomic.AtomicInteger();
    private static class Borrowed implements RandomAccessVectorValues {
        private final RandomAccessVectorValues source;
        private final VectorFloat<?> scratch;
        Borrowed(RandomAccessVectorValues source) {
            this.source = source;
            this.scratch = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(source.dimension());
        }
        public int size() { return source.size(); }
        public int dimension() { return source.dimension(); }
        public boolean isValueShared() { return true; }
        public RandomAccessVectorValues copy() { return new Borrowed(source); }
        public VectorFloat<?> getVector(int ordinal) {
            pointReads.incrementAndGet();
            source.getVectorInto(ordinal, scratch, 0);
            return scratch;
        }
    }

    /** Every worker must score through a copy, never through the source's shared root view. */
    @Test(timeout = 60000)
    public void concurrentFullPrecisionScorersUseIndependentViews() throws Exception {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var vectors = new ArrayList<VectorFloat<?>>();
        var random = new Random(42);
        for (int i = 0; i < 128; i++) {
            var v = vts.createFloatVector(17);
            for (int j = 0; j < 17; j++) v.set(j, random.nextFloat() * 2 - 1);
            VectorUtil.l2normalize(v);
            vectors.add(v);
        }
        var resident = new ListRandomAccessVectorValues(vectors, 17);
        var root = new Borrowed(resident) {
            @Override public VectorFloat<?> getVector(int ordinal) {
                throw new AssertionError("Shared root used by scorer");
            }
        };
        var pool = new ForkJoinPool(3);
        var workers = Executors.newFixedThreadPool(8);
        var path = Files.createTempFile("exact-scorer", ".jvector");
        try (var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(resident,
                VectorSimilarityFunction.DOT_PRODUCT), 17, 8, 32, 1.2f, 1.2f, true, true, pool, pool)) {
            OnDiskGraphIndex.write(builder.build(resident), resident, path);
            try (var index = OnDiskGraphIndex.load(ReaderSupplierFactory.open(path))) {
                for (boolean stored : new boolean[] {false, true}) {
                    for (var metric : VectorSimilarityFunction.values()) {
                        pointReads.set(0);
                        DataSet dataset = new DataSet() {
                            public boolean useDatasetPrefetching() { return stored; }
                            public String getName() { return "point-only"; }
                            public int getDimension() { return 17; }
                            public VectorSimilarityFunction getSimilarityFunction() { return metric; }
                            public List<VectorFloat<?>> getBaseVectors() { return vectors; }
                            public List<VectorFloat<?>> getQueryVectors() { return List.of(vectors.get(0)); }
                            public List<List<Integer>> getGroundTruth() { return List.of(List.of(0)); }
                            public RandomAccessVectorValues getBaseRavv() { return root; }
                        };
                        try (var system = new Grid.ConfiguredSystem(dataset, index, null, Set.of(FeatureId.INLINE_VECTORS))) {
                            var tasks = new ArrayList<Callable<Void>>();
                            for (int worker = 0; worker < 8; worker++) {
                                final int id = worker;
                                tasks.add(() -> {
                                    try (var view = index.getView()) {
                                        for (int iteration = 0; iteration < 100; iteration++) {
                                            var query = vectors.get((iteration + id) % vectors.size());
                                            var scores = system.scoreProviderFor(query, view).scoreFunction();
                                            int ordinal = (iteration * 7 + id) % vectors.size();
                                            assertEquals(metric.compare(query, vectors.get(ordinal)), scores.similarityTo(ordinal), 0f);
                                        }
                                    }
                                    return null;
                                });
                            }
                            for (var task : workers.invokeAll(tasks)) task.get();
                            assertTrue("Exact scorer bypassed original input", pointReads.get() > 0);
                        }
                    }
                }
            }
        } finally { workers.shutdown(); pool.shutdown(); Files.delete(path); }
    }
}
