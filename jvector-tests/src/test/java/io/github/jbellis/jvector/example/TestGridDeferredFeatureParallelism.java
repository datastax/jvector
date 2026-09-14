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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.OnDiskGraphIndexCache;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.management.GraphIndexBuilderConfig;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Assume;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertTrue;

/// Repro for PR #747: with the default writer, Grid serializes all deferred feature work on
/// one thread.
///
/// ## The bug
///
/// Before #747, Grid encoded NVQ (and copied full-precision vectors) for each node during
/// insertion, on the parallel insertion threads, and wrote them with `writeInline`. #747 moves
/// that work into the final `writer.write(suppliers)` call for every Grid run.
///
/// Only the parallel writer calls final suppliers from several threads. It is selected only when
/// `GraphIndexBuilderConfig.isParallelBuild()` is true. The default is `false`, and BenchYAML
/// offers no way to change it, so a plain BenchYAML run (including CI) uses the sequential
/// `OnDiskGraphIndexWriter`. Its `writeL0Records` calls each supplier in turn, on the calling
/// thread. Every vector read and NVQ encode for the whole dataset then runs on one thread after
/// construction.
///
/// On Cohere-1M (preload, native, 48 workers), build+write went from 74–76 s at `4dcfe917`
/// to 100–104 s at `ec42a94b` with the default writer, about 35% slower. With the parallel writer
/// enabled, #747 is faster than before (78 s against 88–92 s). But only the PR's measurements enable it.
///
/// ## The test
///
/// Records the thread of every base-vector read made during final serialization, meaning with a
/// frame from the graph-writer package (`io.github.jbellis.jvector.graph.disk`) on the reading
/// thread's stack. That covers the sequential writer's `writeL0Records` and the parallel writer's
/// `NodeRecordTask`s. No other read in a Grid run goes through that package. The test runs Grid
/// twice on a resident dataset.
///
/// 1. **Control**, with the parallel writer (`parallelBuild = true`). Grid defers features with
///    this writer, so the final write must read base vectors, from more than one writer thread.
///    This proves the detector sees the final write. If the writers move package or the final
///    write stops reading through the dataset's vectors, the control fails instead of the check
///    below passing vacuously.
/// 2. **Requirement**, with BenchYAML's default writer selection (`parallelBuild = false`).
///    Final-write input must not be confined to one thread. Either no vectors are read during
///    the final write (features were produced during insertion), or they are read from more than
///    one thread.
///
/// At `ec42a94b` the control passes, and every read in the default run's final write comes from
/// one thread. The control needs at least two writer threads, so it's skipped on a machine with
/// one physical core.
public class TestGridDeferredFeatureParallelism {
    private static final int SIZE = 4096, DIMENSIONS = 17;

    /// Package of the graph writers: the sequential writer's `writeL0Records`, and the parallel
    /// writer's `NodeRecordTask`s, which run on worker threads whose stacks lack `writeL0Records`.
    private static final String WRITER_PACKAGE = "io.github.jbellis.jvector.graph.disk.";

    /// Resident vectors that record which threads read them during final serialization, meaning
    /// with a graph-writer frame on the reading thread's stack.
    private static final class RecordingVectors implements RandomAccessVectorValues {
        final RandomAccessVectorValues resident;
        final Set<String> writeThreads = ConcurrentHashMap.newKeySet();
        final AtomicInteger writeReads = new AtomicInteger();

        RecordingVectors(RandomAccessVectorValues resident) { this.resident = resident; }

        @Override public int size() { return resident.size(); }
        @Override public int dimension() { return resident.dimension(); }
        @Override public boolean isValueShared() { return false; }
        @Override public RandomAccessVectorValues copy() { return this; }

        @Override public VectorFloat<?> getVector(int ordinal) {
            boolean inFinalWrite = StackWalker.getInstance().walk(frames ->
                    frames.anyMatch(f -> f.getClassName().startsWith(WRITER_PACKAGE)));
            if (inFinalWrite) {
                writeReads.incrementAndGet();
                writeThreads.add(Thread.currentThread().getName());
            }
            return resident.getVector(ordinal);
        }
    }

    /// Run Grid once on `vectors` with the given writer selection, restoring the selection and
    /// removing the run's work directory and cached codebook afterwards.
    private static void runGrid(List<VectorFloat<?>> list, RecordingVectors vectors, boolean parallel)
            throws Exception {
        String name = "deferred-input-" + UUID.randomUUID();
        DataSet ds = new DataSet() {
            public int getDimension() { return DIMENSIONS; }
            public String getName() { return name; }
            public VectorSimilarityFunction getSimilarityFunction() { return VectorSimilarityFunction.DOT_PRODUCT; }
            public RandomAccessVectorValues getBaseRavv() { return vectors; }
            public List<VectorFloat<?>> getBaseVectors() { return list; }
            public List<VectorFloat<?>> getQueryVectors() { return List.of(list.get(0)); }
            public List<List<Integer>> getGroundTruth() { return List.of(List.of(0)); }
        };
        var config = GraphIndexBuilderConfig.getInstance();
        boolean previous = config.isParallelBuild();
        var work = Files.createTempDirectory("grid-deferred-input");
        var pqParameters = new CompressorParameters.PQParameters(4, 16, false, -1f);
        try {
            config.setParallelBuild(parallel);
            Grid.runOneGraph(OnDiskGraphIndexCache.initialize(work.resolve("saved")),
                    List.of(EnumSet.of(FeatureId.NVQ_VECTORS)), 8, 32, 1.2f, true, false,
                    ignored -> pqParameters, List.of(), Map.of(10, List.of(1.0)), List.of(false),
                    RunArtifacts.disabled(), ds, work);
        } finally {
            config.setParallelBuild(previous);
            TestUtil.deleteQuietly(work);
            // Grid caches codebooks by dataset name in the working directory; remove this run's.
            Files.deleteIfExists(Path.of("pq_cache").resolve(pqParameters.idStringFor(ds)));
        }
    }

    @Test(timeout = 120_000)
    public void defaultWriterDoesNotConfineDeferredFeatureInputToOneThread() throws Exception {
        Assume.assumeTrue("the control needs at least two writer threads",
                PhysicalCoreExecutor.getPhysicalCoreCount() > 1);
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(42);
        var list = new ArrayList<VectorFloat<?>>();
        for (int n = 0; n < SIZE; n++) {
            var v = vts.createFloatVector(DIMENSIONS);
            for (int d = 0; d < DIMENSIONS; d++) v.set(d, random.nextFloat() - 0.5f);
            VectorUtil.l2normalize(v);
            list.add(v);
        }
        var resident = new ListRandomAccessVectorValues(list, DIMENSIONS);

        var control = new RecordingVectors(resident);
        runGrid(list, control, true);
        assertTrue("control: the parallel writer's final write read " + control.writeReads.get()
                        + " base vectors on threads " + control.writeThreads
                        + "; the detector no longer sees the final write",
                control.writeReads.get() > 0 && control.writeThreads.size() > 1);

        var vectors = new RecordingVectors(resident);
        runGrid(list, vectors, false); // Explicit sequential selection; retain efficient insertion-time encoding.
        int reads = vectors.writeReads.get();
        assertTrue("final serialization read " + reads + " base vectors, all on one thread "
                        + vectors.writeThreads + ", with the library's default (sequential) writer",
                reads == 0 || vectors.writeThreads.size() > 1);
    }

    @Test
    public void benchmarkWriterDefaultAndOverrideSelectActualWriterTypes() throws Exception {
        var settings = GraphIndexBuilderConfig.getInstance();
        boolean previous = settings.isParallelBuild();
        String property = System.getProperty("jvector.parallel_writer");
        var path = Files.createTempFile("benchmark-writer-selection", ".idx");
        var support = VectorizationProvider.getInstance().getVectorTypeSupport();
        var values = new ListRandomAccessVectorValues(List.of(
                support.createFloatVector(new float[] {1f, 0f}),
                support.createFloatVector(new float[] {0f, 1f})), 2);
        try (var builder = new io.github.jbellis.jvector.graph.GraphIndexBuilder(values,
                VectorSimilarityFunction.DOT_PRODUCT, 4, 16, 1.2f, 1.2f, true)) {
            var graph = TestUtil.buildSequentially(builder, values);
            System.clearProperty("jvector.parallel_writer");
            Grid.configureWriterForBenchmark();
            try (var writer = new io.github.jbellis.jvector.graph.disk.RandomAccessOnDiskGraphIndexWriter.Builder(graph, path)
                    .with(new io.github.jbellis.jvector.graph.disk.feature.InlineVectors(2)).build()) {
                assertTrue(writer instanceof io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter);
            }
            System.setProperty("jvector.parallel_writer", "false");
            Grid.configureWriterForBenchmark();
            try (var writer = new io.github.jbellis.jvector.graph.disk.RandomAccessOnDiskGraphIndexWriter.Builder(graph, path)
                    .with(new io.github.jbellis.jvector.graph.disk.feature.InlineVectors(2)).build()) {
                assertTrue(writer instanceof io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter);
            }
            System.setProperty("jvector.parallel_writer", "invalid");
            org.junit.Assert.assertThrows(IllegalArgumentException.class, Grid::configureWriterForBenchmark);
            org.junit.Assert.assertFalse(settings.isParallelBuild());
        } finally {
            settings.setParallelBuild(previous);
            if (property == null) System.clearProperty("jvector.parallel_writer");
            else System.setProperty("jvector.parallel_writer", property);
            Files.deleteIfExists(path);
        }
    }
}
