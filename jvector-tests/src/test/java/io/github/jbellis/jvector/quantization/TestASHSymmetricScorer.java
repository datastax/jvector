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
                if ((bits == 1 || bits == 2 || bits == 4) && backend.supportsAshSymmetricScoring()) {
                    assertEquals("symmetric SIMD bits=" + bits + " d=" + d, actual,
                            backend.ashSymmetricDot(a.binaryVector, a.extraBits, b.binaryVector, b.extraBits, d, bits), 0f);
                }
                if ((bits == 2 || bits == 4) && backend.supportsAshProjectionScoring()) {
                    assertEquals(actual, backend.ashProjectionDot(unpacked, b.extraBits, d, bits), 0f);
                }
            }
        }
    }

    @Test public void integerBlockLutPreservesTailsAndAvoidsShortOverflow() {
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        Random random = new Random(77531);
        for (int groups : new int[]{1,2,31,63,64,65,127,257,4097}) {
            int stride = 37;
            byte[] codes = new byte[((groups+1)/2)*stride];
            random.nextBytes(codes);
            short[] lut = new short[groups*32];
            for (int g = 0; g < groups; g++) for (int code = 0; code < 16; code++) {
                short value = groups == 4097 ? (short)225 : (short)(random.nextInt(451)-225);
                for (int copy = 0; copy < 2; copy++) lut[g*32+copy*16+code] = value;
            }
            for (int count : new int[]{1,7,16,17,32,34}) {
                float[] out = new float[count+2];
                java.util.Arrays.fill(out, Float.NaN);
                backend.ashSymmetricLutScore(codes, groups, stride, 3, count, lut, out, 1);
                assertTrue(Float.isNaN(out[0])); assertTrue(Float.isNaN(out[count+1]));
                for (int i = 0; i < count; i++) {
                    long expected = 0;
                    for (int g = 0; g < groups; g++) expected += lut[g*32+((codes[(g/2)*stride+3+i] >>> (4*(g%2))) & 15)];
                    assertEquals("groups="+groups+" count="+count, expected*.25f,out[i+1],0f);
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
                    assertEquals(scorer.scoreFunctionFor(i).similarityTo(j),
                            scorer.scoreFunctionFor(encoded.get(i)).similarityTo(j), 0f);
                    assertEquals(ASHScorer.toSimilarity(scorer.dotProduct(i,j)),
                            bsp.searchProviderFor(i).scoreFunction().similarityTo(j), 0f);
                    assertEquals(bsp.searchProviderFor(i).scoreFunction().similarityTo(j),
                            bsp.diversityScoreFunctionFor(i).similarityTo(j), 0f);
                }
                for (var kernel : ASHSymmetricScorer.Kernel.values()) {
                    if (kernel == ASHSymmetricScorer.Kernel.SIMD &&
                            (!(bits == 1 || bits == 2 || bits == 4) ||
                             !VectorizationProvider.getInstance().getVectorUtilSupport().supportsAshSymmetricScoring())) continue;
                    var block = scorer.blockScorerFor(encoded.get(i), 8, kernel);
                    float[] scores = new float[19];
                    block.scoreRange(3, scores.length, scores);
                    for (int j = 0; j < scores.length; j++) assertEquals("C1 block bits="+bits+" kernel="+kernel,
                            ASHScorer.toSimilarity(scorer.dotProduct(i,j+3)), scores[j], 5e-5f);
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

    @Test public void graphProviderStillRejectsMultipleLandmarks() throws Exception {
        var values = MockVectorValues.fromValues(inputs(25,40));
        var ash = AsymmetricHashing.initialize(values,AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS+31*2,2,2);
        var encoded = ash.encodeAll(values, java.util.concurrent.ForkJoinPool.commonPool());
        new ASHSymmetricScorer(encoded);
        assertThrows(IllegalArgumentException.class, () -> BuildScoreProvider.ashBuildScoreProvider(VectorSimilarityFunction.DOT_PRODUCT, encoded));
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
    @Test public void multipleLandmarksMatchCalibratedReconstructionAndExistingKernels() throws Exception {
        var input = inputs(300,40);
        var values = MockVectorValues.fromValues(input);
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        for (int centers : new int[]{2,64,256}) {
            for (int bits : new int[]{1,2,4}) {
                var ash = AsymmetricHashing.initialize(values, AsymmetricHashing.RANDOM,
                        AsymmetricHashing.HEADER_BITS + 31*bits, centers, bits);
                var encoded = ash.encodeAll(values, java.util.concurrent.ForkJoinPool.commonPool());
                var scalar = new ASHSymmetricScorer(encoded, ASHSymmetricScorer.Kernel.SCALAR);
                var vector = backend.supportsAshSymmetricScoring()
                        ? new ASHSymmetricScorer(encoded, ASHSymmetricScorer.Kernel.SIMD) : scalar;
                for (int i = 0; i < input.length; i++) {
                    int j = (i*31+7) % input.length;
                    var a = encoded.get(i); var b = encoded.get(j);
                    int ca = a.landmark & 255, cb = b.landmark & 255;
                    double expected = 0, pa = 0, pb = 0;
                    for (int k = 0; k < 40; k++) {
                        double x = 0, y = 0;
                        for (int l = 0; l < 31; l++) {
                            x += ash.stiefelTransform.AFloat[l][k] * component(a,l,bits);
                            y += ash.stiefelTransform.AFloat[l][k] * component(b,l,bits);
                        }
                        pa += x * ash.landmarks[ca].get(k);
                        pb += y * ash.landmarks[cb].get(k);
                        expected += (ash.landmarks[ca].get(k) + a.scale*x)
                                * (ash.landmarks[cb].get(k) + b.scale*y);
                    }
                    double correctionA = a.offset, correctionB = b.offset;
                    if (bits != 2 && bits != 4) {
                        correctionA -= a.scale*pa; correctionB -= b.scale*pb;
                    }
                    expected += correctionA + correctionB;
                    assertEquals(expected, scalar.dotProduct(i,j), 2e-5);
                    assertEquals(scalar.dotProduct(i,j), scalar.dotProduct(j,i), 0f);
                    float similarity = ASHScorer.toSimilarity((float) expected);
                    assertEquals(similarity, scalar.scoreFunctionFor(a).similarityTo(j), 2e-5f);
                    assertEquals(similarity, vector.scoreFunctionFor(a).similarityTo(j), 2e-5f);
                }
                if (centers == 256) {
                    try (var out = io.github.jbellis.jvector.disk.ByteBufferIndexWriter.create(1024*1024,false)) {
                        encoded.write(out, io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex.CURRENT_VERSION);
                        var loaded = ASHVectors.load(new io.github.jbellis.jvector.disk.ByteBufferReader(out.getWrittenData()));
                        assertEquals(256, loaded.getCompressor().landmarkCount);
                        for (int i = 0; i < encoded.count(); i++) {
                            assertEquals(encoded.get(i).landmark & 255, loaded.get(i).landmark & 255);
                        }
                        assertEquals(scalar.dotProduct(7,11), new ASHSymmetricScorer(loaded,ASHSymmetricScorer.Kernel.SCALAR).dotProduct(7,11), 2e-6f);
                        assertEquals(encoded.scoreFunctionFor(input[3],VectorSimilarityFunction.DOT_PRODUCT).similarityTo(7),
                                loaded.scoreFunctionFor(input[3],VectorSimilarityFunction.DOT_PRODUCT).similarityTo(7), 2e-6f);
                    }
                }
                for (int blockSize : new int[]{8,16,32}) {
                    var query = encoded.get(7);
                    var block = vector.blockScorerFor(query, blockSize, backend.supportsAshSymmetricScoring()
                            ? ASHSymmetricScorer.Kernel.SIMD : ASHSymmetricScorer.Kernel.SCALAR);
                    float[] scores = new float[41];
                    block.scoreRange(3,41,scores);
                    for (int k = 0; k < 41; k++) assertEquals(scalar.scoreFunctionFor(query).similarityTo(k+3), scores[k], 0.005f);
                }
            }
        }
    }

}
