/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.ByteBufferIndexWriter;
import io.github.jbellis.jvector.disk.ByteBufferReader;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedASH;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2ObjectHashMap;
import org.junit.Test;
import java.util.Random;
import java.util.EnumMap;
import java.util.function.IntFunction;
import java.nio.file.Files;
import java.util.concurrent.ForkJoinPool;
import static org.junit.Assert.*;

/** Exercises the production factories and decoder, not only their layout helpers. */
public class TestASHScoringDispatch {
    private static final String BLOCK = "jvector.ash.blockKernel";
    private static final String SINGLE = "jvector.ash.singleKernel";

    @Test
    public void standaloneAndFusedUseSelectedKernels() throws Exception {
        String oldBlock = System.getProperty(BLOCK), oldSingle = System.getProperty(SINGLE);
        try {
            var provider = VectorizationProvider.getInstance();
            boolean simd = provider.getVectorUtilSupport().supportsAshLutScoring();
            VectorFloat<?>[] input = new VectorFloat<?>[97];
            Random random = new Random(6234);
            for (int n = 0; n < input.length; n++) {
                input[n] = provider.getVectorTypeSupport().createFloatVector(64);
                for (int i = 0; i < 64; i++) input[n].set(i, random.nextFloat() - 0.5f);
                VectorUtil.l2normalize(input[n]);
            }
            var values = MockVectorValues.fromValues(input);
            for (int bits : new int[]{1, 2, 4}) {
                var ash = AsymmetricHashing.initialize(values, AsymmetricHashing.RANDOM,
                        AsymmetricHashing.HEADER_BITS + 33 * bits, 2, bits);
                var encoded = ash.encodeAll(values, ForkJoinPool.commonPool());
                // Reload canonical metadata/body bytes before exercising the new factories.
                try (var writer = ByteBufferIndexWriter.create(1024 * 1024, false)) {
                    encoded.write(writer, OnDiskGraphIndex.CURRENT_VERSION);
                    encoded = ASHVectors.load(new ByteBufferReader(writer.getWrittenData()));
                }
                checkDiskGraph(encoded, input);
                for (int q : new int[]{3, 68}) {
                    System.setProperty(SINGLE, "scalar");
                    var reference = new ASHScorer(ash).scoreFunctionFor(input[q], VectorSimilarityFunction.DOT_PRODUCT);
                    float[] expected = new float[input.length];
                    for (int i = 0; i < expected.length; i++) {
                        expected[i] = Math.max(0f, (1f + reference.similarityTo(encoded.get(i))) / 2f);
                    }
                    for (String mode : simd ? new String[]{"scalar", "auto", "simd"} : new String[]{"scalar", "auto"}) {
                        System.setProperty(BLOCK, mode);
                        System.setProperty(SINGLE, mode);
                        var single = encoded.scoreFunctionFor(input[q], VectorSimilarityFunction.DOT_PRODUCT);
                        for (int i = 0; i < expected.length; i++) assertEquals(expected[i], single.similarityTo(i), 0.00005f);
                        for (int size : new int[]{8, 16, 32}) {
                            var block = encoded.blockScorerFor(input[q], VectorSimilarityFunction.DOT_PRODUCT, size);
                            if (bits != 1) assertTrue(block.toString(), block.toString().contains(
                                    mode.equals("scalar") || !simd ? "scalar ASH LUT" : "SIMD ASH LUT"));
                            for (int start : new int[]{0, 1, 7, 15, 31, 65, 96, 97}) {
                                int count = Math.min(32, input.length - start);
                                float[] actual = new float[count];
                                block.scoreRange(start, count, actual);
                                for (int i = 0; i < count; i++) assertEquals(expected[start + i], actual[i], 0.00005f);
                            }
                            checkFused(ash, encoded, input[q], expected, size);
                        }
                        // The two-argument overload deliberately remains an independent scalar oracle.
                        float[] referenceBlock = new float[32];
                        encoded.blockScorerFor(input[q], VectorSimilarityFunction.DOT_PRODUCT).scoreRange(0, 32, referenceBlock);
                        for (int i = 0; i < 32; i++) assertEquals(expected[i], referenceBlock[i], 0.00005f);
                    }
                }
            }
        } finally {
            restore(BLOCK, oldBlock);
            restore(SINGLE, oldSingle);
        }
    }

    private static void checkDiskGraph(ASHVectors encoded, VectorFloat<?>[] input) throws Exception {
        var graph = new TestUtil.RandomlyConnectedGraphIndex(input.length, 37, new Random(9182));
        var path = Files.createTempFile("ash-simd-roundtrip-", ".index");
        try {
            var builder = new OnDiskGraphIndexWriter.Builder(graph, path)
                    .with(new FusedASH(37, encoded.getCompressor(), 32))
                    .with(new InlineVectors(input[0].length()));
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            try (var graphView = graph.getView(); var writer = builder.build()) {
                suppliers.put(FeatureId.FUSED_ASH, ordinal -> new FusedASH.State(graphView, encoded, ordinal));
                suppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(input[ordinal]));
                writer.write(suppliers);
            }
            try (var reader = new SimpleMappedReader.Supplier(path);
                 var loaded = OnDiskGraphIndex.load(reader, 0);
                 var view = loaded.getView()) {
                var reference = encoded.scoreFunctionFor(input[3], VectorSimilarityFunction.DOT_PRODUCT);
                var fused = view.approximateScoreFunctionFor(input[3], VectorSimilarityFunction.DOT_PRODUCT);
                assertTrue(fused.toString(), fused instanceof FusedASHDecoder);
                for (int origin : new int[]{0, 7, 63, 96}) {
                    fused.enableSimilarityToNeighbors(origin);
                    var neighbors = view.getNeighborsIterator(0, origin);
                    for (int lane = 0; neighbors.hasNext(); lane++) {
                        assertEquals(reference.similarityTo(neighbors.nextInt()),
                                fused.similarityToNeighbor(origin, lane), 0.002f);
                    }
                }
            }
        } finally { Files.deleteIfExists(path); }
    }

    private static void checkFused(AsymmetricHashing ash, ASHVectors vectors,
                                   VectorFloat<?> query, float[] expected, int size) {
        int degree = 37;
        byte[] bytes = new byte[FusedASHLayout.featureSize(degree, ash.quantizedDim, ash.bitsPerDimension, size)];
        for (int i = 0; i < degree; i++) {
            int offset = FusedASHLayout.blockOffset(i / size, ash.quantizedDim, ash.bitsPerDimension, size);
            var v = vectors.get(i);
            FusedASHLayout.packQuantizedVector(bytes, offset, i % size, v, ash.quantizedDim, ash.bitsPerDimension, size);
            if (ash.bitsPerDimension == 1) {
                float landmarkDot = 0;
                for (int d = 0; d < ash.quantizedDim; d++) landmarkDot += ash.landmarkProj[v.landmark & 255][d]
                        * (((v.binaryVector[d >>> 6] >>> (d & 63)) & 1) == 0 ? -1 : 1);
                FusedASHLayout.writeLaneHeader(bytes, offset, i % size, ash.quantizedDim, 1, size,
                        v.scale, v.offset - v.scale * landmarkDot, v.landmark);
            }
        }
        int[] reads = {0};
        var source = new FusedASHDecoder.PackedNeighborhoods() {
            public void readInto(int origin, byte[] dest) {
                reads[0]++;
                if (origin == 2) throw new IllegalStateException("Injected read failure");
                System.arraycopy(bytes, 0, dest, 0, bytes.length);
            }
            public int maxDegree() { return degree; }
            public int featureSize() { return bytes.length; }
        };
        var cache = new Int2ObjectHashMap<FusedFeature.InlineSource>();
        cache.put(0, () -> 0L);
        var decoder = FusedASHDecoder.newDecoder(source, ash, cache, ignored -> vectors.get(0), query,
                new byte[bytes.length], new float[degree], size, VectorSimilarityFunction.DOT_PRODUCT);
        decoder.enableSimilarityToNeighbors(1);
        for (int i = 0; i < degree; i++) assertEquals("fused bits=" + ash.bitsPerDimension + " lane=" + i,
                expected[i], decoder.similarityToNeighbor(1, i), 0.002f);
        assertEquals(expected[0], decoder.similarityTo(0), 0.00005f);
        decoder.enableSimilarityToNeighbors(1);
        assertEquals(1, reads[0]);
        assertThrows(IllegalStateException.class, () -> decoder.enableSimilarityToNeighbors(2));
        assertThrows(IllegalArgumentException.class, () -> decoder.similarityToNeighbor(1, 0));
        decoder.enableSimilarityToNeighbors(1);
        assertEquals(3, reads[0]);
    }

    @Test
    public void forcedSimdCannotSilentlyFallBack() {
        String old = System.getProperty(BLOCK);
        try {
            System.setProperty(BLOCK, "simd");
            if (VectorizationProvider.getInstance().getVectorUtilSupport().supportsAshLutScoring()) {
                assertTrue(new ASHLutKernel().toString().contains("SIMD"));
            } else {
                assertThrows(IllegalStateException.class, ASHLutKernel::new);
            }
            System.setProperty(BLOCK, "misspelled");
            assertThrows(IllegalArgumentException.class, ASHLutKernel::new);
        } finally { restore(BLOCK, old); }
    }

    private static void restore(String property, String value) {
        if (value == null) System.clearProperty(property); else System.setProperty(property, value);
    }
}
