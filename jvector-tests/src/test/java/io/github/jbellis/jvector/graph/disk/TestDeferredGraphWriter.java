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
package io.github.jbellis.jvector.graph.disk;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.*;
import io.github.jbellis.jvector.management.GraphIndexBuilderConfig;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import org.junit.Test;

import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.IntFunction;

import static org.junit.Assert.*;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestDeferredGraphWriter extends LuceneTestCase {
    private ListRandomAccessVectorValues values() {
        var vectors = new ArrayList<io.github.jbellis.jvector.vector.types.VectorFloat<?>>();
        var support = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(42);
        for (int n = 0; n < 259; n++) {
            var vector = support.createFloatVector(17);
            for (int d = 0; d < 17; d++) vector.set(d, random.nextFloat() - 0.5f);
            vectors.add(vector);
        }
        return new ListRandomAccessVectorValues(vectors, 17);
    }

    @Test
    public void completeAndPrewrittenFeaturesMatchSequentialV6() throws Exception {
        var directory = Files.createTempDirectory("deferred-writer");
        var values = values();
        try (var builder = new GraphIndexBuilder(values, VectorSimilarityFunction.DOT_PRODUCT,
                8, 32, 1.2f, 1.2f, true)) {
            var graph = TestUtil.buildSequentially(builder, values);
            var pq = ProductQuantization.compute(values, 3, 256, false);
            var codes = pq.encodeAll(values, ForkJoinPool.commonPool());
            var nvq = NVQuantization.compute(values, 2);
            var frozenView = graph.getView();
            try {
                for (boolean holes : new boolean[]{false, true}) {
                    OrdinalMapper mapper = new OrdinalMapper() {
                        public int maxOrdinal() { return holes ? values.size() * 2 - 1 : values.size() - 1; }
                        public int oldToNew(int n) { return (values.size() - 1 - n) * (holes ? 2 : 1); }
                        public int newToOld(int n) { return holes && n % 2 != 0 ? OMITTED : values.size() - 1 - n / (holes ? 2 : 1); }
                    };
                    for (boolean quantized : new boolean[]{false, true}) {
                        Feature vectors = quantized ? new NVQ(nvq) : new InlineVectors(values.dimension());
                        var fused = new FusedPQ(graph.maxDegree(), pq);
                        Map<FeatureId, IntFunction<Feature.State>> suppliers = new EnumMap<>(FeatureId.class);
                        suppliers.put(vectors.id(), n -> quantized ? new NVQ.State(nvq.encode(values.getVector(n)))
                                : new InlineVectors.State(values.getVector(n)));
                        suppliers.put(fused.id(), n -> new FusedPQ.State(frozenView, codes, n));
                        byte[] prefix = new byte[83];
                        Arrays.fill(prefix, (byte) 0x5a);
                        var sequential = directory.resolve("sequential");
                        Files.write(sequential, prefix);
                        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, sequential)
                                .withStartOffset(prefix.length).withVersion(6).withMapper(mapper)
                                .with(vectors).with(fused).build()) {
                            writer.write(suppliers);
                        }
                        byte[] expected = Files.readAllBytes(sequential);
                        for (int batch : new int[]{1, 7, 64, 128}) {
                            for (boolean direct : new boolean[]{false, true}) {
                                for (boolean prewritten : new boolean[]{false, true}) {
                                    var parallel = directory.resolve("parallel");
                                    Files.write(parallel, prefix);
                                    try (var writer = new OnDiskParallelGraphIndexWriter.Builder(graph, parallel)
                                            .withParallelWorkerThreads(3).withParallelBatchSize(batch)
                                            .withParallelDirectBuffers(direct).withStartOffset(prefix.length)
                                            .withVersion(6).withMapper(mapper).with(vectors).with(fused).build()) {
                                        if (prewritten) {
                                            for (int ordinal = 0; ordinal < values.size(); ordinal++) {
                                                writer.writeFeaturesInline(mapper.oldToNew(ordinal),
                                                        Map.of(vectors.id(), suppliers.get(vectors.id()).apply(ordinal)));
                                            }
                                            writer.write(Map.of(fused.id(), suppliers.get(fused.id())));
                                        } else writer.write(suppliers);
                                    }
                                    assertArrayEquals("holes=" + holes + " nvq=" + quantized + " batch=" + batch
                                            + " direct=" + direct + " prewritten=" + prewritten,
                                            expected, Files.readAllBytes(parallel));
                                }
                            }
                        }
                    }
                }
            } finally { frozenView.close(); }
        } finally { TestUtil.deleteQuietly(directory); }
    }

    @Test(timeout = 60000)
    public void failedWriteJoinsWorkersAndPreservesExternalExecutor() throws Exception {
        var directory = Files.createTempDirectory("failed-deferred-writer");
        var values = values();
        var workers = Executors.newFixedThreadPool(2);
        var caller = Executors.newSingleThreadExecutor();
        var release = new CountDownLatch(1);
        try (var builder = new GraphIndexBuilder(values, VectorSimilarityFunction.DOT_PRODUCT,
                8, 32, 1.2f, 1.2f, false)) {
            var graph = TestUtil.buildSequentially(builder, values);
            var file = directory.resolve("records");
            Files.createFile(file);
            var blocked = new CountDownLatch(1);
            var failed = new CountDownLatch(1);
            try (var out = new BufferedRandomAccessWriter(file);
                 var writer = new ParallelGraphWriter(out, graph, List.of(new InlineVectors(17)),
                         new ParallelGraphWriter.Config(2, false, 1, 7), file, workers)) {
                var writing = caller.submit(() -> {
                    writer.writeL0Records(new OrdinalMapper.IdentityMapper(values.size() - 1),
                            List.of(new InlineVectors(17)), Feature.singleStateFactory(FeatureId.INLINE_VECTORS, n -> {
                                if (n == 0) {
                                    try { if (!blocked.await(10, TimeUnit.SECONDS)) throw new AssertionError("Worker did not start"); }
                                    catch (InterruptedException e) { throw new AssertionError(e); }
                                    failed.countDown();
                                    throw new IllegalStateException("supplier failure");
                                }
                                blocked.countDown();
                                try { if (!release.await(10, TimeUnit.SECONDS)) throw new AssertionError("Writer cleanup timeout"); }
                                catch (InterruptedException e) { throw new AssertionError(e); }
                                return new InlineVectors.State(values.getVector(n));
                            }), 0);
                    return null;
                });
                assertTrue(failed.await(10, TimeUnit.SECONDS));
                assertTrue(blocked.await(10, TimeUnit.SECONDS));
                assertThrows(TimeoutException.class, () -> writing.get(100, TimeUnit.MILLISECONDS));
                release.countDown();
                var failure = assertThrows(ExecutionException.class, () -> writing.get(10, TimeUnit.SECONDS));
                assertTrue(failure.getCause() instanceof IllegalStateException);
            }
            assertEquals(42, workers.submit(() -> 42).get().intValue());
        } finally {
            release.countDown();
            caller.shutdownNow(); workers.shutdownNow();
            caller.awaitTermination(10, TimeUnit.SECONDS); workers.awaitTermination(10, TimeUnit.SECONDS);
            TestUtil.deleteQuietly(directory);
        }
    }

    @Test
    public void batchSizeValidationAndUnifiedBuilderSelection() throws Exception {
        var directory = Files.createTempDirectory("writer-batch-option");
        var values = values();
        var settings = GraphIndexBuilderConfig.getInstance();
        boolean wasParallel = settings.isParallelBuild();
        try (var graphBuilder = new GraphIndexBuilder(values, VectorSimilarityFunction.DOT_PRODUCT,
                8, 32, 1.2f, 1.2f, false)) {
            var graph = TestUtil.buildSequentially(graphBuilder, values);
            assertEquals(64, ParallelGraphWriter.Config.defaultConfig().recordsPerBatch);
            assertThrows(IllegalArgumentException.class, () -> new ParallelGraphWriter.Config(1, false, 1, 0));
            for (boolean parallel : new boolean[]{false, true}) {
                settings.setParallelBuild(parallel);
                var builder = new RandomAccessOnDiskGraphIndexWriter.Builder(graph, directory.resolve("unified"));
                assertThrows(IllegalArgumentException.class, () -> builder.withParallelBatchSize(-1));
                try (var writer = builder.withParallelBatchSize(7).withParallelWorkerThreads(2)
                        .with(new InlineVectors(17)).build()) {
                    assertEquals(parallel, writer instanceof OnDiskParallelGraphIndexWriter);
                    writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                            n -> new InlineVectors.State(values.getVector(n))));
                }
            }
            var direct = new OnDiskParallelGraphIndexWriter.Builder(graph, directory.resolve("direct"));
            assertThrows(IllegalArgumentException.class, () -> direct.withParallelBatchSize(0));
            try (var writer = direct.with(new InlineVectors(17)).build()) { }
        } finally {
            settings.setParallelBuild(wasParallel);
            TestUtil.deleteQuietly(directory);
        }
    }
}
