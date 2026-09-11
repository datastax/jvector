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
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Exercises standalone ASH scoring without a graph, an on-disk index,
 * FusedASH, or YAML configuration.
 *
 * The projected dimension is held constant while bitsPerDimension changes.
 * This isolates the multibit ASH encoder and ASHScorer paths.
 */
public class TestASHStandaloneScoring {
    private static final int ORIGINAL_DIMENSION = 128;
    private static final int PROJECTED_DIMENSION = 64;
    private static final int VECTOR_COUNT = 1_024;
    private static final int QUERY_COUNT = 64;
    private static final int LANDMARK_COUNT = 1;

    @Test
    public void testOneBitStandaloneScoring() throws Exception {
        runStandaloneScoringTest(1);
    }

    @Test
    public void testTwoBitStandaloneScoring() throws Exception {
        runStandaloneScoringTest(2);
    }

    @Test
    public void testFourBitStandaloneScoring() throws Exception {
        runStandaloneScoringTest(4);
    }

    @Test
    public void testAllSupportedBitWidthsTogether() throws Exception {
        var vectors = createNormalizedVectors(
                VECTOR_COUNT,
                ORIGINAL_DIMENSION,
                0x5eedL);

        double oneBitCorrelation = scoreCorrelation(vectors, 1);
        double twoBitCorrelation = scoreCorrelation(vectors, 2);
        double fourBitCorrelation = scoreCorrelation(vectors, 4);

        System.out.printf(
                "Standalone ASH correlation: 1-bit=%.6f, 2-bit=%.6f, 4-bit=%.6f%n",
                oneBitCorrelation,
                twoBitCorrelation,
                fourBitCorrelation);

        assertTrue(
                "1-bit correlation is too low: " + oneBitCorrelation,
                oneBitCorrelation > 0.20);

        assertTrue(
                "2-bit correlation is too low: " + twoBitCorrelation,
                twoBitCorrelation > 0.20);

        assertTrue(
                "4-bit correlation is too low: " + fourBitCorrelation,
                fourBitCorrelation > 0.20);
        /*
         * More stored bits should not catastrophically degrade scoring.
         * The tolerance allows small variations from training and quantization.
         */
        assertTrue(
                "2-bit scoring degraded unexpectedly"
                        + ": oneBit=" + oneBitCorrelation
                        + ", twoBit=" + twoBitCorrelation,
                twoBitCorrelation >= oneBitCorrelation - 0.10);

        assertTrue(
                "4-bit scoring degraded unexpectedly"
                        + ": oneBit=" + oneBitCorrelation
                        + ", fourBit=" + fourBitCorrelation,
                fourBitCorrelation >= oneBitCorrelation - 0.10);
    }

    private static void runStandaloneScoringTest(int bitsPerDimension)
            throws Exception {
        var vectors = createNormalizedVectors(
                VECTOR_COUNT,
                ORIGINAL_DIMENSION,
                0x5eedL);

        double correlation = scoreCorrelation(vectors, bitsPerDimension);

        System.out.printf(
                "Standalone ASH bitsPerDimension=%d, correlation=%.6f%n",
                bitsPerDimension,
                correlation);

        assertTrue(
                "Expected positive exact/approximate correlation"
                        + " for bitsPerDimension=" + bitsPerDimension
                        + ", but got " + correlation,
                correlation > 0.2);
    }

    private static double scoreCorrelation(
            VectorFloat<?>[] vectors,
            int bitsPerDimension) throws Exception {
        var ravv = MockVectorValues.fromValues(vectors);

        /*
         * encodedBits includes the ASH header. Keeping PROJECTED_DIMENSION
         * constant reproduces the important distinction between:
         *
         *   1 bit: HEADER_BITS + 32
         *   2 bit: HEADER_BITS + 64
         *   4 bit: HEADER_BITS + 128
         */
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

        int sampleCount = QUERY_COUNT * VECTOR_COUNT;
        double[] exactScores = new double[sampleCount];
        double[] approximateScores = new double[sampleCount];

        int sample = 0;

        for (int queryOrdinal = 0;
             queryOrdinal < QUERY_COUNT;
             queryOrdinal++) {
            VectorFloat<?> query = vectors[queryOrdinal];

            int querySampleStart = sample;

            /*
             * Calculate ground truth before constructing the ASH scorer.
             * This also protects the test against query preprocessing that
             * uses or modifies mutable vector storage.
             */
            for (int vectorOrdinal = 0;
                 vectorOrdinal < VECTOR_COUNT;
                 vectorOrdinal++) {
                exactScores[sample++] = VectorUtil.dotProduct(
                        query,
                        vectors[vectorOrdinal]);
            }

            ScoreFunction.ApproximateScoreFunction scorer =
                    encoded.scoreFunctionFor(
                            query,
                            VectorSimilarityFunction.DOT_PRODUCT);

            for (int vectorOrdinal = 0;
                 vectorOrdinal < VECTOR_COUNT;
                 vectorOrdinal++) {
                double approximate =
                        scorer.similarityTo(vectorOrdinal);

                assertFinite(
                        approximate,
                        bitsPerDimension,
                        queryOrdinal,
                        vectorOrdinal,
                        encoded.get(vectorOrdinal));

                approximateScores[querySampleStart + vectorOrdinal] =
                        approximate;
            }
        }

        assertEquals(sampleCount, sample);

        double correlation =
                pearsonCorrelation(exactScores, approximateScores);

        assertTrue(
                "Correlation is not finite for bitsPerDimension="
                        + bitsPerDimension
                        + ": " + correlation,
                Double.isFinite(correlation));

        return correlation;
    }

    private static void assertFinite(
            double score,
            int bitsPerDimension,
            int queryOrdinal,
            int vectorOrdinal,
            AsymmetricHashing.QuantizedVector encoded) {
        if (!Double.isFinite(score)) {
            fail(
                    "Non-finite standalone ASH score"
                            + ": bitsPerDimension=" + bitsPerDimension
                            + ", query=" + queryOrdinal
                            + ", vector=" + vectorOrdinal
                            + ", score=" + score
                            + ", scale=" + encoded.scale
                            + ", offset=" + encoded.offset
                            + ", landmark="
                            + (encoded.landmark & 0xFF));
        }

        if (!Float.isFinite(encoded.scale)) {
            fail(
                    "Non-finite ASH scale"
                            + ": bitsPerDimension=" + bitsPerDimension
                            + ", vector=" + vectorOrdinal
                            + ", scale=" + encoded.scale);
        }

        if (!Float.isFinite(encoded.offset)) {
            fail(
                    "Non-finite ASH offset"
                            + ": bitsPerDimension=" + bitsPerDimension
                            + ", vector=" + vectorOrdinal
                            + ", offset=" + encoded.offset);
        }

        int landmark = encoded.landmark & 0xFF;
        assertTrue(
                "Invalid landmark"
                        + ": bitsPerDimension=" + bitsPerDimension
                        + ", vector=" + vectorOrdinal
                        + ", landmark=" + landmark,
                landmark >= 0 && landmark < LANDMARK_COUNT);
    }

    private static VectorFloat<?>[] createNormalizedVectors(
            int count,
            int dimension,
            long seed) {
        Random random = new Random(seed);
        VectorFloat<?>[] vectors = new VectorFloat<?>[count];

        for (int i = 0; i < count; i++) {
            VectorFloat<?> vector =
                    TestUtil.randomVector(random, dimension);

            // Replace TestUtil's values with zero-mean Gaussian components.
            for (int d = 0; d < dimension; d++) {
                vector.set(d, (float) random.nextGaussian());
            }

            VectorUtil.l2normalize(vector);
            vectors[i] = vector;
        }

        return vectors;
    }

    private static double pearsonCorrelation(
            double[] left,
            double[] right) {
        assertEquals(left.length, right.length);

        double leftMean = 0.0;
        double rightMean = 0.0;

        for (int i = 0; i < left.length; i++) {
            leftMean += left[i];
            rightMean += right[i];
        }

        leftMean /= left.length;
        rightMean /= right.length;

        double covariance = 0.0;
        double leftVariance = 0.0;
        double rightVariance = 0.0;

        for (int i = 0; i < left.length; i++) {
            double leftDelta = left[i] - leftMean;
            double rightDelta = right[i] - rightMean;

            covariance += leftDelta * rightDelta;
            leftVariance += leftDelta * leftDelta;
            rightVariance += rightDelta * rightDelta;
        }

        assertTrue(
                "Exact scores have zero variance",
                leftVariance > 0.0);

        assertTrue(
                "Approximate scores have zero variance",
                rightVariance > 0.0);

        return covariance
                / Math.sqrt(leftVariance * rightVariance);
    }
}
