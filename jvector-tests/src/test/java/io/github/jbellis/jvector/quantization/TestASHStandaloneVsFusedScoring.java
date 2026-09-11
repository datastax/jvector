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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Compares standalone ASH scores with scores produced from the fused byte
 * layout for the same queries and quantized vectors.
 *
 * This does not involve graph construction, graph search, or disk I/O.
 */
public class TestASHStandaloneVsFusedScoring {
    private static final int ORIGINAL_DIMENSION = 128;
    private static final int PROJECTED_DIMENSION = 64;
    private static final int VECTOR_COUNT = 1_024;
    private static final int QUERY_COUNT = 32;
    private static final int LANDMARK_COUNT = 1;
    private static final int MAX_DEGREE = 64;

    /*
     * The fused layout stores scale and offset as float16. Allow for the
     * resulting rounding error when comparing against standalone scoring.
     */
    private static final float ABSOLUTE_TOLERANCE = 2.0e-3f;
    private static final float RELATIVE_TOLERANCE = 2.0e-3f;

    @Test
    public void testOneBitStandaloneMatchesFused() throws Exception {
        compareStandaloneAndFused(1);
    }

    @Test
    public void testTwoBitStandaloneMatchesFused() throws Exception {
        compareStandaloneAndFused(2);
    }

    @Test
    public void testFourBitStandaloneMatchesFused() throws Exception {
        compareStandaloneAndFused(4);
    }

    private static void compareStandaloneAndFused(int bitsPerDimension)
            throws Exception {
        VectorFloat<?>[] vectors = createNormalizedVectors(
                VECTOR_COUNT,
                ORIGINAL_DIMENSION,
                0x5eedL);

        var ravv = MockVectorValues.fromValues(vectors);

        int encodedBits =
                AsymmetricHashing.HEADER_BITS
                        + PROJECTED_DIMENSION * bitsPerDimension;

        AsymmetricHashing ash = AsymmetricHashing.initialize(
                ravv,
                AsymmetricHashing.ITQ,
                encodedBits,
                LANDMARK_COUNT,
                bitsPerDimension);

        ASHVectors encoded =
                ash.encodeAll(ravv, ForkJoinPool.commonPool());

        int blockSize =
                FusedASHLayout.chooseBlockSize(MAX_DEGREE);

        byte[] fusedBlock = new byte[
                FusedASHLayout.blockBytes(
                        ash.quantizedDim,
                        bitsPerDimension,
                        blockSize)];

        /*
         * Fill one complete fused block. Each encoded vector occupies the
         * lane with the same ordinal.
         */
        for (int lane = 0; lane < blockSize; lane++) {
            packForFused(
                    fusedBlock,
                    0,
                    lane,
                    encoded.get(lane),
                    ash,
                    blockSize);
        }

        int groups = FusedASHLayout.codeGroups(
                ash.quantizedDim,
                bitsPerDimension);

        float[] queryLut = new float[groups * 16];
        float[] dotQMuByLandmark =
                new float[ash.landmarkCount];

        double maximumDelta = 0.0;
        int maximumDeltaQuery = -1;
        int maximumDeltaLane = -1;
        float maximumDeltaStandalone = Float.NaN;
        float maximumDeltaFused = Float.NaN;

        for (int queryOrdinal = 0;
             queryOrdinal < QUERY_COUNT;
             queryOrdinal++) {
            VectorFloat<?> query = vectors[queryOrdinal];

            ScoreFunction.ApproximateScoreFunction standaloneScorer =
                    encoded.scoreFunctionFor(
                            query,
                            VectorSimilarityFunction.DOT_PRODUCT);

            precomputeFusedQuery(
                    query,
                    ash,
                    queryLut,
                    dotQMuByLandmark);

            for (int lane = 0; lane < blockSize; lane++) {
                float standaloneScore =
                        standaloneScorer.similarityTo(lane);

                float fusedScore =
                        ASHScorer.toSimilarity(FusedASHLayout.scoreLane(
                                fusedBlock,
                                0,
                                lane,
                                ash.quantizedDim,
                                bitsPerDimension,
                                blockSize,
                                queryLut,
                                dotQMuByLandmark));

                assertTrue(
                        failurePrefix(
                                bitsPerDimension,
                                queryOrdinal,
                                lane,
                                standaloneScore,
                                fusedScore)
                                + ": standalone score is not finite",
                        Float.isFinite(standaloneScore));

                assertTrue(
                        failurePrefix(
                                bitsPerDimension,
                                queryOrdinal,
                                lane,
                                standaloneScore,
                                fusedScore)
                                + ": fused score is not finite",
                        Float.isFinite(fusedScore));

                double delta =
                        Math.abs(standaloneScore - fusedScore);

                if (delta > maximumDelta) {
                    maximumDelta = delta;
                    maximumDeltaQuery = queryOrdinal;
                    maximumDeltaLane = lane;
                    maximumDeltaStandalone = standaloneScore;
                    maximumDeltaFused = fusedScore;
                }

                float tolerance =
                        ABSOLUTE_TOLERANCE
                                + RELATIVE_TOLERANCE
                                * Math.abs(standaloneScore);

                assertEquals(
                        failurePrefix(
                                bitsPerDimension,
                                queryOrdinal,
                                lane,
                                standaloneScore,
                                fusedScore)
                                + ", tolerance=" + tolerance,
                        standaloneScore,
                        fusedScore,
                        tolerance);
            }
        }

        System.out.printf(
                "ASH standalone/fused parity"
                        + ": bitsPerDimension=%d"
                        + ", blockSize=%d"
                        + ", maxDelta=%.9f"
                        + ", query=%d"
                        + ", lane=%d"
                        + ", standalone=%.9f"
                        + ", fused=%.9f%n",
                bitsPerDimension,
                blockSize,
                maximumDelta,
                maximumDeltaQuery,
                maximumDeltaLane,
                maximumDeltaStandalone,
                maximumDeltaFused);
    }

    /**
     * Mirrors FusedASH.packForFused().
     *
     * Multibit vectors already contain the projection-mode adjusted offset.
     * One-bit vectors require the same adjustment performed by
     * FusedASH.packOneBitForFused().
     */
    private static void packForFused(
            byte[] destination,
            int blockOffset,
            int lane,
            AsymmetricHashing.QuantizedVector vector,
            AsymmetricHashing ash,
            int blockSize) {
        if (ash.bitsPerDimension == 1) {
            packOneBitForFused(
                    destination,
                    blockOffset,
                    lane,
                    vector,
                    ash,
                    blockSize);
        } else {
            FusedASHLayout.packQuantizedVector(
                    destination,
                    blockOffset,
                    lane,
                    vector,
                    ash.quantizedDim,
                    ash.bitsPerDimension,
                    blockSize);
        }
    }

    /**
     * Mirrors FusedASH.packOneBitForFused().
     */
    private static void packOneBitForFused(
            byte[] destination,
            int blockOffset,
            int lane,
            AsymmetricHashing.QuantizedVector vector,
            AsymmetricHashing ash,
            int blockSize) {
        int groups = FusedASHLayout.codeGroups(
                ash.quantizedDim,
                ash.bitsPerDimension);

        for (int group = 0; group < groups; group++) {
            int nibble = FusedASHLayout.signNibbleFromWords(
                    vector.binaryVector,
                    group,
                    ash.quantizedDim);

            FusedASHLayout.setPackedNibble(
                    destination,
                    blockOffset,
                    lane,
                    group,
                    blockSize,
                    nibble);
        }

        int landmark = vector.landmark & 0xFF;

        assertTrue(
                "Invalid landmark " + landmark
                        + " for landmarkCount=" + ash.landmarkCount,
                landmark < ash.landmarkCount);

        float landmarkDot = dotOneBitProjectionCode(
                ash.landmarkProj[landmark],
                vector.binaryVector,
                ash.quantizedDim);

        float adjustedOffset =
                vector.offset - vector.scale * landmarkDot;

        FusedASHLayout.writeLaneHeader(
                destination,
                blockOffset,
                lane,
                ash.quantizedDim,
                ash.bitsPerDimension,
                blockSize,
                vector.scale,
                adjustedOffset,
                vector.landmark);
    }

    private static float dotOneBitProjectionCode(
            float[] projected,
            long[] signWords,
            int quantizedDimension) {
        float sum = 0.0f;

        for (int dimension = 0;
             dimension < quantizedDimension;
             dimension++) {
            boolean positive =
                    ((signWords[dimension >>> 6]
                            >>> (dimension & 63)) & 1L) != 0L;

            sum += positive
                    ? projected[dimension]
                    : -projected[dimension];
        }

        return sum;
    }

    /**
     * Mirrors FusedASHDecoder.precomputeQuery().
     */
    private static void precomputeFusedQuery(
            VectorFloat<?> query,
            AsymmetricHashing ash,
            float[] queryLut,
            float[] dotQMuByLandmark) {
        int originalDimension = query.length();
        int quantizedDimension = ash.quantizedDim;

        float[] queryArray =
                new float[originalDimension];

        for (int dimension = 0;
             dimension < originalDimension;
             dimension++) {
            queryArray[dimension] = query.get(dimension);
        }

        float[][] projectionMatrix =
                ash.stiefelTransform.AFloat;

        var vectorUtil = VectorizationProvider
                .getInstance()
                .getVectorUtilSupport();

        float[] projectedQuery =
                new float[quantizedDimension];

        for (int projectedDimension = 0;
             projectedDimension < quantizedDimension;
             projectedDimension++) {
            projectedQuery[projectedDimension] =
                    vectorUtil.ashDotRow(
                            projectionMatrix[projectedDimension],
                            queryArray);
        }

        FusedASHLayout.buildQueryLut(
                projectedQuery,
                quantizedDimension,
                ash.bitsPerDimension,
                queryLut);

        for (int landmark = 0;
             landmark < ash.landmarkCount;
             landmark++) {
            dotQMuByLandmark[landmark] =
                    VectorUtil.dotProduct(
                            query,
                            ash.landmarks[landmark]);
        }
    }

    private static VectorFloat<?>[] createNormalizedVectors(
            int count,
            int dimension,
            long seed) {
        Random random = new Random(seed);
        VectorFloat<?>[] vectors =
                new VectorFloat<?>[count];

        for (int i = 0; i < count; i++) {
            VectorFloat<?> vector =
                    TestUtil.randomVector(random, dimension);

            for (int d = 0; d < dimension; d++) {
                vector.set(
                        d,
                        (float) random.nextGaussian());
            }

            VectorUtil.l2normalize(vector);
            vectors[i] = vector;
        }

        return vectors;
    }

    private static String failurePrefix(
            int bitsPerDimension,
            int queryOrdinal,
            int lane,
            float standaloneScore,
            float fusedScore) {
        return "Standalone/fused score mismatch"
                + ": bitsPerDimension=" + bitsPerDimension
                + ", query=" + queryOrdinal
                + ", lane=" + lane
                + ", standalone=" + standaloneScore
                + ", fused=" + fusedScore;
    }
}
