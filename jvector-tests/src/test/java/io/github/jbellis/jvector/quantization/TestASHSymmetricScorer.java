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

import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.util.Random;
import static org.junit.Assert.*;

public class TestASHSymmetricScorer {
    // Deliberately independent unpacking oracle, including padding with arbitrary bits.
    private static float component(AsymmetricHashing.QuantizedVector v, int j, int bits) {
        if (bits == 1) return ((v.binaryVector[j / 64] >>> (j % 64)) & 1) == 0 ? -1f : 1f;
        if (bits == 2 || bits == 4) {
            int field = (v.extraBits[j * bits / 8] >>> (j * bits % 8)) & ((1 << bits) - 1);
            float magnitude = (field & ((1 << (bits - 1)) - 1)) + 0.5f;
            return (field & (1 << (bits - 1))) == 0 ? -magnitude : magnitude;
        }
        int code = 0;
        for (int k = 0; k < bits - 1; k++) {
            int pos = j * (bits - 1) + k;
            code |= ((v.extraBits[pos / 8] >>> (pos % 8)) & 1) << k;
        }
        code |= ((v.binaryVector[j / 64] >>> (j % 64)) & 1) << (bits - 1);
        return code - ((1 << (bits - 1)) - 0.5f);
    }

    @Test public void packedDotMatchesIndependentDecoderAndSimdIncludingTails() {
        Random random = new Random(981234);
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        for (int bits = 1; bits <= 9; bits++) {
            for (int d = 1; d <= 749; d++) {
                var a = AsymmetricHashing.QuantizedVector.createEmpty(d, bits);
                var b = AsymmetricHashing.QuantizedVector.createEmpty(d, bits);
                for (int w = 0; w < a.binaryVector.length; w++) {
                    a.binaryVector[w] = random.nextLong(); b.binaryVector[w] = random.nextLong();
                }
                random.nextBytes(a.extraBits); random.nextBytes(b.extraBits);
                double expected = 0;
                float[] unpacked = new float[d];
                for (int j = 0; j < d; j++) {
                    unpacked[j] = component(a, j, bits);
                    expected += unpacked[j] * component(b, j, bits);
                }
                float actual = ASHSymmetricScorer.codeDot(a, b, d, bits);
                assertEquals("bits=" + bits + " d=" + d, (float) expected, actual, 0f);
                assertEquals(actual, ASHSymmetricScorer.codeDot(b, a, d, bits), 0f);
                if ((bits == 2 || bits == 4) && backend.supportsAshProjectionScoring()) {
                    assertEquals(actual, backend.ashProjectionDot(unpacked, b.extraBits, d, bits), 0f);
                }
            }
        }
    }

    private static VectorFloat<?>[] inputs(int n, int d) {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        Random r = new Random(773);
        VectorFloat<?>[] input = new VectorFloat<?>[n];
        for (int i = 0; i < n; i++) {
            input[i] = vts.createFloatVector(d);
            for (int j = 0; j < d; j++) input[i].set(j, r.nextFloat() - 0.15f);
            VectorUtil.l2normalize(input[i]);
        }
        return input;
    }

    @Test public void appendixBMatchesDecodedReconstructionAndHeaderRoundTrip() throws Exception {
        var input = inputs(25, 40);
        var values = MockVectorValues.fromValues(input);
        for (int bits = 1; bits <= 9; bits++) {
            var ash = AsymmetricHashing.initialize(values, AsymmetricHashing.RANDOM,
                    AsymmetricHashing.HEADER_BITS + 31 * bits, 1, bits);
            var encoded = ash.encodeAll(values, java.util.concurrent.ForkJoinPool.commonPool());
            var scorer = new ASHSymmetricScorer(encoded);
            var bsp = BuildScoreProvider.ashBuildScoreProvider(VectorSimilarityFunction.DOT_PRODUCT, encoded);
            for (int i = 0; i < input.length; i++) {
                for (int j = 0; j < input.length; j++) {
                    var a = encoded.get(i); var b = encoded.get(j);
                    double dot = 0, ma = 0, mb = 0;
                    var decodedX = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(40);
                    // Explicit W^T reconstruction exercises the orthonormal simplification.
                    for (int k = 0; k < 40; k++) {
                        double x = 0, y = 0;
                        for (int l = 0; l < 31; l++) {
                            x += ash.stiefelTransform.AFloat[l][k] * component(a, l, bits);
                            y += ash.stiefelTransform.AFloat[l][k] * component(b, l, bits);
                        }
                        decodedX.set(k, ash.landmarks[0].get(k) + a.scale * (float) x);
                        dot += x * y;
                        ma += x * ash.landmarks[0].get(k);
                        mb += y * ash.landmarks[0].get(k);
                    }
                    double oa = a.offset, ob = b.offset;
                    if (bits == 2 || bits == 4) { oa += a.scale * ma; ob += b.scale * mb; }
                    double expected = a.scale * b.scale * dot + oa + ob
                            + VectorUtil.dotProduct(ash.landmarks[0], ash.landmarks[0]);
                    assertEquals(expected, scorer.dotProduct(i,j), 2e-5);
                    // ASH preserves the original landmark dot via the header, rather than
                    // substituting the reconstructed source's landmark dot.
                    double sourceCorrection = a.offset;
                    if (bits != 2 && bits != 4) sourceCorrection -= a.scale * ma;
                    float decodedAsymmetric = new ASHScorer(ash)
                            .scoreFunctionFor(decodedX, VectorSimilarityFunction.DOT_PRODUCT).similarityTo(b);
                    assertEquals(decodedAsymmetric + sourceCorrection, scorer.dotProduct(i,j), 2e-5);
                    assertEquals(scorer.dotProduct(i,j), scorer.dotProduct(j,i), 0f);
                    assertEquals(ASHScorer.toSimilarity(scorer.dotProduct(i,j)),
                            bsp.searchProviderFor(i).scoreFunction().similarityTo(j), 0f);
                    assertEquals(bsp.searchProviderFor(i).scoreFunction().similarityTo(j),
                            bsp.diversityScoreFunctionFor(i).similarityTo(j), 0f);
                }
                assertEquals("Self score should reconstruct the input norm, up to FP16 headers",
                        VectorUtil.dotProduct(input[i],input[i]), scorer.dotProduct(i,i), 0.002);
            }
            try (var out = io.github.jbellis.jvector.disk.ByteBufferIndexWriter.create(1024*1024,false)) {
                encoded.write(out, io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex.CURRENT_VERSION);
                var reloaded = ASHVectors.load(new io.github.jbellis.jvector.disk.ByteBufferReader(out.getWrittenData()));
                assertEquals(scorer.dotProduct(2,7), new ASHSymmetricScorer(reloaded).dotProduct(2,7), 0f);
            }
        }
    }

    @Test public void zeroResidualAtLandmarkStaysFinite() throws Exception {
        var input = inputs(20,40);
        for (int i = 1; i < input.length; i++) input[i] = input[0].copy();
        var values = MockVectorValues.fromValues(input);
        for (int bits : new int[]{1,2,4,9}) {
            var ash = AsymmetricHashing.initialize(values, AsymmetricHashing.RANDOM,
                    AsymmetricHashing.HEADER_BITS + 31*bits,1,bits);
            var encoded = ash.encodeAll(values, java.util.concurrent.ForkJoinPool.commonPool());
            float score = new ASHSymmetricScorer(encoded).dotProduct(0,1);
            assertEquals(VectorUtil.dotProduct(input[0],input[1]), score, 0.0001f);
        }
    }

    @Test public void rejectsMultipleLandmarks() throws Exception {
        var values = MockVectorValues.fromValues(inputs(25,40));
        var ash = AsymmetricHashing.initialize(values,AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS+31*2,2,2);
        assertThrows(IllegalArgumentException.class, () -> new ASHSymmetricScorer(ash.encodeAll(values, java.util.concurrent.ForkJoinPool.commonPool())));
    }

    @Test public void constructionWorksWithPackedCodesAndNegativeSimilarities() throws Exception {
        var input = inputs(80,40);
        for (int d = 0; d < 40; d++) input[1].set(d, -input[0].get(d));
        var values = MockVectorValues.fromValues(input);
        for (int bits : new int[]{1,2,4}) {
            var ash = AsymmetricHashing.initialize(values,AsymmetricHashing.RANDOM,
                    AsymmetricHashing.HEADER_BITS+31*bits,1,bits);
            var encoded = ash.encodeAll(values, java.util.concurrent.ForkJoinPool.commonPool());
            var bsp = BuildScoreProvider.ashBuildScoreProvider(VectorSimilarityFunction.DOT_PRODUCT,encoded);
            assertTrue(new ASHSymmetricScorer(encoded).dotProduct(0,1) < 0f);
            assertFalse(bsp.searchProviderFor(0).adaptiveScoresAreRawDotProducts());
            try (var builder = new GraphIndexBuilder(bsp,40,16,40,1.2f,1.2f,true,true)) {
                for (int i=0;i<input.length;i++) builder.addGraphNode(i,input[i]);
                builder.cleanup();
                try (var searcher = new GraphSearcher(builder.getGraph())) {
                    searcher.usePruning(false);
                    assertEquals(10,searcher.search(bsp.searchProviderFor(input[3]),10,10,0f,0f,Bits.ALL).getNodes().length);
                    assertEquals(10,searcher.search(bsp.searchProviderFor(3),10,10,0f,0f,Bits.ALL).getNodes().length);
                }
            }
        }
    }
}
