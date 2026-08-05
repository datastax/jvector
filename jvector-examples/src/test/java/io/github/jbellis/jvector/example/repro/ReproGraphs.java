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

package io.github.jbellis.jvector.example.repro;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/// Shared fixtures for the memory-safety reproduction tests: tiny on-disk graphs with the
/// `INLINE_VECTORS` feature (the shape [io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor]
/// requires of its sources), raw big-endian float files for reader-level scenarios, and the
/// live-nodes / remapper boilerplate the compactor constructor needs. Graph construction mirrors
/// `TestOnDiskGraphIndexCompactor.buildSimpleSourceGraph` in jvector-tests.
public final class ReproGraphs {
    public static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();

    private ReproGraphs() {
    }

    /// Creates a fresh scratch directory under the build's `target/` tree (never `/tmp`), so the
    /// build owns cleanup and tests never need to delete anything.
    public static Path newWorkDir(String label) throws IOException {
        Path moduleTarget;
        if (Files.isDirectory(Path.of("jvector-examples", "target"))) {
            moduleTarget = Path.of("jvector-examples", "target");   // CWD = repo root (surefire config)
        } else if (Files.isDirectory(Path.of("target"))) {
            moduleTarget = Path.of("target");                       // CWD = module dir
        } else {
            moduleTarget = Path.of(System.getProperty("java.io.tmpdir"));
        }
        Path base = moduleTarget.resolve("repro-tmp");
        Files.createDirectories(base);
        return Files.createTempDirectory(base, label + "-");
    }

    /// Deterministic random vectors.
    public static List<VectorFloat<?>> randomVectors(int count, int dimension, long seed) {
        Random r = new Random(seed);
        List<VectorFloat<?>> out = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            VectorFloat<?> v = VTS.createFloatVector(dimension);
            for (int d = 0; d < dimension; d++) {
                v.set(d, r.nextFloat());
            }
            out.add(v);
        }
        return out;
    }

    /// Builds a single-layer graph with inline full-resolution vectors and writes it to
    /// `outputPath` (v5+ format, footer at the logical end of the file).
    public static Path buildInlineGraph(Path outputPath, List<VectorFloat<?>> vecs, int dimension) throws IOException {
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vecs, dimension);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN);
        var builder = new GraphIndexBuilder(bsp, dimension, 8, 32, 1.2f, 1.2f, false, true,
                ForkJoinPool.commonPool(), ForkJoinPool.commonPool());
        for (int i = 0; i < vecs.size(); i++) {
            builder.addGraphNode(i, vecs.get(i));
        }
        builder.cleanup();
        var graph = builder.getGraph();

        var writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                .withMapper(new OrdinalMapper.IdentityMapper(vecs.size() - 1))
                .with(new InlineVectors(dimension));
        var writer = writerBuilder.build();

        Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
        writeSuppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(ravv.getVector(ordinal)));
        for (int node = 0; node < vecs.size(); node++) {
            var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
            stateMap.put(FeatureId.INLINE_VECTORS, writeSuppliers.get(FeatureId.INLINE_VECTORS).apply(node));
            writer.writeInline(node, stateMap);
        }
        writer.write(writeSuppliers);
        return outputPath;
    }

    /// Builds a single-layer graph with inline full-resolution vectors AND the `FUSED_PQ`
    /// feature — the source shape that makes the compactor pick `FusedCompactionStrategy` and
    /// therefore run the pre-encode `invokeAll` fan-out. Mirrors
    /// `TestOnDiskGraphIndexCompactor.buildFusedPQ` in jvector-tests.
    public static Path buildFusedGraph(Path outputPath, List<VectorFloat<?>> vecs, int dimension) throws IOException {
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vecs, dimension);
        ProductQuantization pq = ProductQuantization.compute(ravv, 8, 256, true, UNWEIGHTED,
                ForkJoinPool.commonPool(), ForkJoinPool.commonPool());
        PQVectors pqv = (PQVectors) pq.encodeAll(ravv, ForkJoinPool.commonPool());
        var bsp = BuildScoreProvider.pqBuildScoreProvider(io.github.jbellis.jvector.vector.VectorSimilarityFunction.COSINE, pqv);
        var builder = new GraphIndexBuilder(bsp, dimension, 16, 100, 1.2f, 1.2f, false, true,
                ForkJoinPool.commonPool(), ForkJoinPool.commonPool());
        var graph = builder.getGraph();

        var writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                .withMapper(new OrdinalMapper.IdentityMapper(vecs.size() - 1))
                .with(new InlineVectors(dimension))
                .with(new FusedPQ(graph.maxDegree(), pq));
        var writer = writerBuilder.build();

        Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
        writeSuppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(ravv.getVector(ordinal)));
        for (int node = 0; node < ravv.size(); node++) {
            var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
            stateMap.put(FeatureId.INLINE_VECTORS, new InlineVectors.State(ravv.getVector(node)));
            writer.writeInline(node, stateMap);
            builder.addGraphNode(node, ravv.getVector(node));
        }
        builder.cleanup();

        writeSuppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(graph.getView(), pqv, ordinal));
        writer.write(writeSuppliers);
        return outputPath;
    }

    /// A raw file of `count * dimension` big-endian floats — the shape a reader-level scenario
    /// scans without any graph structure on top.
    public static Path writeBigEndianFloats(Path path, int count, int dimension, long seed) throws IOException {
        Random r = new Random(seed);
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path), 1 << 20))) {
            for (long i = 0; i < (long) count * dimension; i++) {
                out.writeFloat(r.nextFloat());
            }
        }
        return path;
    }

    /// All-live bitsets, one per source size.
    public static List<FixedBitSet> allLive(int... sizes) {
        List<FixedBitSet> out = new ArrayList<>(sizes.length);
        for (int size : sizes) {
            FixedBitSet bits = new FixedBitSet(size);
            bits.set(0, size);
            out.add(bits);
        }
        return out;
    }

    /// Identity remappers that stack each source's ordinals after the previous source's
    /// (source 0 keeps `[0, n0)`, source 1 gets `[n0, n0+n1)`, ...).
    public static List<OrdinalMapper> stackedRemappers(int... sizes) {
        List<OrdinalMapper> out = new ArrayList<>(sizes.length);
        int base = 0;
        for (int size : sizes) {
            Map<Integer, Integer> map = new HashMap<>();
            for (int i = 0; i < size; i++) {
                map.put(i, base + i);
            }
            out.add(new OrdinalMapper.MapMapper(map));
            base += size;
        }
        return out;
    }

    /// Copies one full-resolution vector out of a graph view as a plain float array.
    public static float[] readVector(OnDiskGraphIndex.View view, int ordinal, int dimension) {
        VectorFloat<?> buf = VTS.createFloatVector(dimension);
        view.getVectorInto(ordinal, buf, 0);
        float[] out = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            out[d] = buf.get(d);
        }
        return out;
    }

    /// Converts a [VectorFloat] to a plain float array for comparisons.
    public static float[] toArray(VectorFloat<?> v) {
        float[] out = new float[v.length()];
        for (int d = 0; d < v.length(); d++) {
            out[d] = v.get(d);
        }
        return out;
    }

    /// Loads a graph over the supplier, reflectively matching how production callers hold sources.
    public static OnDiskGraphIndex load(ReaderSupplier rs) {
        return OnDiskGraphIndex.load(rs);
    }
}
