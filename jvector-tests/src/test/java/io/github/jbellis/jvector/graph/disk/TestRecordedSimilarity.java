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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.PersistableGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedASH;
import io.github.jbellis.jvector.graph.disk.feature.InlineByteVectors;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.index.Indexes;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertEquals;

/**
 * Tests the similarity function on-disk graphs record: in the header from format version 8, and through a
 * fused ASH model before that.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestRecordedSimilarity extends RandomizedTest {
    private static final int DIMENSION = 32;
    private Path directory;

    @Before
    public void setUp() throws IOException {
        directory = Files.createTempDirectory(getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(directory);
    }

    private static ListRandomAccessVectorValues unitRavv(int n) {
        List<VectorFloat<?>> vectors = createRandomVectors(n, DIMENSION);
        vectors.forEach(VectorUtil::l2normalize);
        return new ListRandomAccessVectorValues(vectors, DIMENSION);
    }

    private static PersistableGraphIndex build(ListRandomAccessVectorValues ravv, VectorSimilarityFunction vsf) {
        return Indexes.hnswBuilder(ravv, vsf).buildAndPopulate();
    }

    /** Writes {@code graph} with inline vectors at {@code version}, or the default version if null, and loads it. */
    private OnDiskGraphIndex writeAndLoad(GraphIndex graph, ListRandomAccessVectorValues ravv, Integer version, String name)
            throws IOException {
        var path = directory.resolve(name);
        var builder = new OnDiskGraphIndexWriter.Builder(graph, path).with(new InlineVectors(DIMENSION));
        if (version != null) {
            builder.withVersion(version);
        }
        try (var writer = builder.build()) {
            writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                    node -> new InlineVectors.State(ravv.getVector(node))));
        }
        return OnDiskGraphIndex.load(new SimpleMappedReader.Supplier(path));
    }

    @Test
    public void version8IsTheDefaultAndRecordsTheGraphsFunction() throws Exception {
        var ravv = unitRavv(100);
        for (VectorSimilarityFunction vsf : VectorSimilarityFunction.values()) {
            try (var onDisk = writeAndLoad(build(ravv, vsf), ravv, null, "default-" + vsf)) {
                assertEquals(8, onDisk.version);
                assertEquals(Optional.of(vsf), onDisk.similarityFunction());
                assertEquals(Optional.empty(), onDisk.byteSimilarityFunction());
            }
            // writeTo uses the default version too
            var path = directory.resolve("writeTo-" + vsf);
            build(ravv, vsf).writeTo(path, ravv);
            try (var onDisk = OnDiskGraphIndex.load(new SimpleMappedReader.Supplier(path))) {
                assertEquals(Optional.of(vsf), onDisk.similarityFunction());
            }
        }
    }

    @Test
    public void aGraphWrittenAtVersion8ReadsBackTheSameAsAtVersion7() throws Exception {
        var ravv = unitRavv(200);
        var graph = build(ravv, VectorSimilarityFunction.COSINE);
        try (var v7 = writeAndLoad(graph, ravv, 7, "v7"); var v8 = writeAndLoad(graph, ravv, 8, "v8")) {
            assertEquals(7, v7.version);
            assertEquals(8, v8.version);
            // a requested older version is honored, and records nothing
            assertEquals(Optional.empty(), v7.similarityFunction());
            TestUtil.assertGraphEquals(v7, v8);
            try (var view7 = v7.getView(); var view8 = v8.getView()) {
                for (int node = 0; node < ravv.size(); node++) {
                    assertEquals(view7.getVector(node), view8.getVector(node));
                }
            }
        }
    }

    @Test
    public void aGraphWhoseProviderNamesNoFunctionRecordsNone() throws Exception {
        var ravv = unitRavv(100);
        var inner = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.DOT_PRODUCT);
        var custom = new BuildScoreProvider() {
            @Override public boolean isExact() { return true; }
            @Override public VectorFloat<?> approximateCentroid() { return inner.approximateCentroid(); }
            @Override public io.github.jbellis.jvector.graph.similarity.SearchScoreProvider searchProviderFor(VectorFloat<?> v) { return inner.searchProviderFor(v); }
            @Override public io.github.jbellis.jvector.graph.similarity.SearchScoreProvider searchProviderFor(int node) { return inner.searchProviderFor(node); }
            @Override public io.github.jbellis.jvector.graph.similarity.SearchScoreProvider diversityProviderFor(int node) { return inner.diversityProviderFor(node); }
        };
        try (var builder = new GraphIndexBuilder(custom, DIMENSION, 16, 50, 1.2f, 1.2f, true);
             var onDisk = writeAndLoad(builder.build(ravv), ravv, null, "custom")) {
            assertEquals(8, onDisk.version);
            assertEquals(Optional.empty(), onDisk.similarityFunction());
            assertEquals(Optional.empty(), onDisk.byteSimilarityFunction());
        }
    }

    @Test
    public void int8GraphsRecordTheirByteFunction() throws Exception {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(7);
        List<ByteSequence<?>> vectors = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            byte[] raw = new byte[DIMENSION];
            random.nextBytes(raw);
            raw[0] |= 1; // never the zero vector, which COSINE rejects
            vectors.add(vts.createByteSequence(raw));
        }
        var ravv = new ListRandomAccessByteVectorValues(vectors, DIMENSION);
        for (ByteVectorSimilarityFunction bvsf : ByteVectorSimilarityFunction.values()) {
            var bsp = BuildScoreProvider.byteVectorScoreProvider(ravv, bvsf);
            try (var builder = new GraphIndexBuilder(bsp, DIMENSION, 16, 50, 1.2f, 1.2f, true)) {
                var graph = builder.build(ravv);
                assertEquals(Optional.of(bvsf), graph.byteSimilarityFunction());
                assertEquals(Optional.empty(), graph.similarityFunction());

                var path = directory.resolve("int8-" + bvsf);
                try (var writer = new OnDiskGraphIndexWriter.Builder(graph, path).with(new InlineByteVectors(DIMENSION)).build()) {
                    writer.write(Feature.singleStateFactory(FeatureId.INLINE_BYTE_VECTORS,
                            node -> new InlineByteVectors.State(vectors.get(node))));
                }
                try (var onDisk = OnDiskGraphIndex.load(new SimpleMappedReader.Supplier(path))) {
                    assertEquals(Optional.of(bvsf), onDisk.byteSimilarityFunction());
                    assertEquals(Optional.empty(), onDisk.similarityFunction());
                }
            }
        }
    }

    /** Writes {@code graph} with inline vectors and fused ASH codes from a model trained for {@code modelFunction}. */
    private OnDiskGraphIndex writeFusedAsh(GraphIndex graph, ListRandomAccessVectorValues ravv,
                                           VectorSimilarityFunction modelFunction, int version, String name) throws IOException {
        var ash = AsymmetricHashing.initialize(ravv, modelFunction, AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + DIMENSION * 2, 1, 2);
        var encoded = (ASHVectors) ash.encodeAll(ravv, ForkJoinPool.commonPool());
        var path = directory.resolve(name);
        var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
        try (var view = graph.getView();
             var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                     .withVersion(version)
                     .with(new FusedASH(graph.maxDegree(), ash))
                     .with(new InlineVectors(DIMENSION))
                     .build()) {
            suppliers.put(FeatureId.FUSED_ASH, n -> new FusedASH.State(view, encoded, n));
            suppliers.put(FeatureId.INLINE_VECTORS, n -> new InlineVectors.State(ravv.getVector(n)));
            writer.write(suppliers);
        }
        return OnDiskGraphIndex.load(new SimpleMappedReader.Supplier(path));
    }

    @Test
    public void fusedAshGraphsReportTheModelsFunctionAtEveryVersion() throws Exception {
        var ravv = unitRavv(100);
        var graph = build(ravv, VectorSimilarityFunction.EUCLIDEAN);
        for (int version : new int[] {7, 8}) {
            try (var onDisk = writeFusedAsh(graph, ravv, VectorSimilarityFunction.EUCLIDEAN, version, "ash-agree-" + version)) {
                assertEquals(Optional.of(VectorSimilarityFunction.EUCLIDEAN), onDisk.similarityFunction());
            }
        }
    }

    @Test
    public void aFusedAshModelForAnotherFunctionIsRecordedNotRejected() throws Exception {
        // A EUCLIDEAN graph over unit vectors fused with a DOT_PRODUCT model is a legitimate pairing that wrote
        // before version 8; it still writes, and records the function the file can be searched with.
        var ravv = unitRavv(100);
        var graph = build(ravv, VectorSimilarityFunction.EUCLIDEAN);
        try (var onDisk = writeFusedAsh(graph, ravv, VectorSimilarityFunction.DOT_PRODUCT, 8, "ash-disagree")) {
            assertEquals(Optional.of(VectorSimilarityFunction.DOT_PRODUCT), onDisk.similarityFunction());
        }
    }

    /** Compacts two graphs built with {@code built} using a compactor for {@code compactWith}, at {@code version} or the default. */
    private OnDiskGraphIndex compact(VectorSimilarityFunction built, VectorSimilarityFunction compactWith, Integer version,
                                     String name) throws IOException {
        var sources = new ArrayList<OnDiskGraphIndex>();
        var live = new ArrayList<FixedBitSet>();
        for (int s = 0; s < 2; s++) {
            var ravv = unitRavv(100);
            sources.add(writeAndLoad(build(ravv, built), ravv, null, name + "-source" + s));
            var all = new FixedBitSet(ravv.size());
            all.set(0, ravv.size());
            live.add(all);
        }
        try {
            var compactor = new OnDiskGraphIndexCompactor(sources, live, compactWith, ForkJoinPool.commonPool());
            if (version != null) {
                compactor.withVersion(version);
            }
            var path = directory.resolve(name);
            compactor.compact(path);
            return OnDiskGraphIndex.load(new SimpleMappedReader.Supplier(path));
        } finally {
            for (var source : sources) {
                source.close();
            }
        }
    }

    @Test
    public void compactionRecordsTheCompactorsFunction() throws Exception {
        for (VectorSimilarityFunction vsf : VectorSimilarityFunction.values()) {
            try (var compacted = compact(vsf, vsf, null, "compact-" + vsf)) {
                assertEquals(8, compacted.version);
                assertEquals(Optional.of(vsf), compacted.similarityFunction());
            }
        }
        try (var compacted = compact(VectorSimilarityFunction.COSINE, VectorSimilarityFunction.COSINE, 7, "compact-v7")) {
            assertEquals(Optional.empty(), compacted.similarityFunction());
        }
    }

    @Test
    public void aCompactorForAnotherFunctionIsRecordedNotRejected() throws Exception {
        // as before V8, compacting with another function works; it warns, and the output records the function used
        try (var compacted = compact(VectorSimilarityFunction.COSINE, VectorSimilarityFunction.DOT_PRODUCT, null, "compact-disagree")) {
            assertEquals(Optional.of(VectorSimilarityFunction.DOT_PRODUCT), compacted.similarityFunction());
        }
    }

    @Test
    public void theHeaderGrowsByTheRecordedFunction() {
        var v7 = GraphIndexFormatFactory.forVersion(7);
        var v8 = GraphIndexFormatFactory.forVersion(8);
        assertEquals(v7.commonHeaderSize() + 2 * Integer.BYTES, v8.commonHeaderSize());
        assertEquals(v7.getSupportedFeatures(), v8.getSupportedFeatures());
        assertEquals(8, OnDiskGraphIndex.CURRENT_VERSION);
    }
}
