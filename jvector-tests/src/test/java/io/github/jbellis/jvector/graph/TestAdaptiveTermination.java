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

package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.Random;

import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.COSINE;
import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.DOT_PRODUCT;
import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The stopping rule on its own: scores computed by the similarity functions from real vectors must lead to the
 * decision the rule makes on the distances between those vectors.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestAdaptiveTermination extends RandomizedTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Test
    public void rejectsInvalidGamma() {
        for (float gamma : new float[] {-0.01f, -1f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
            assertThrows(String.valueOf(gamma), IllegalArgumentException.class, () -> AdaptiveTermination.of(COSINE, gamma));
        }
        assertThrows(NullPointerException.class, () -> AdaptiveTermination.of(null, 0.1f));
        var rule = AdaptiveTermination.of(EUCLIDEAN, 0f);
        assertEquals(0f, rule.getGamma(), 0f);
        assertEquals(EUCLIDEAN, rule.getSimilarityFunction());
    }

    /** Points on a line through the query, so the distances are known exactly. */
    @Test
    public void euclideanStopsAtTheGammaBoundary() {
        var q = vector(0, 0, 0);
        var kth = vector(1, 0, 0);      // distance 1
        var candidate = vector(0, 1.5f, 0); // distance 1.5
        float kthScore = EUCLIDEAN.compare(q, kth);
        float candidateScore = EUCLIDEAN.compare(q, candidate);
        // the candidate is 1.5x as far as the k-th result
        assertTrue(AdaptiveTermination.of(EUCLIDEAN, 0f).shouldStop(candidateScore, kthScore));
        assertTrue(AdaptiveTermination.of(EUCLIDEAN, 0.4f).shouldStop(candidateScore, kthScore));
        assertTrue(AdaptiveTermination.of(EUCLIDEAN, 0.49f).shouldStop(candidateScore, kthScore));
        assertFalse(AdaptiveTermination.of(EUCLIDEAN, 0.51f).shouldStop(candidateScore, kthScore));
        assertFalse(AdaptiveTermination.of(EUCLIDEAN, 3f).shouldStop(candidateScore, kthScore));
        // a candidate closer than the k-th result never stops the search
        assertFalse(AdaptiveTermination.of(EUCLIDEAN, 0f).shouldStop(kthScore, candidateScore));
    }

    /** COSINE distance is the chord between the normalized vectors, whatever the lengths of the inputs. */
    @Test
    public void cosineUsesTheDistanceBetweenNormalizedVectors() {
        var q = vector(5, 0);
        // 60 degrees: chord 1; 90 degrees: chord sqrt(2) = 1.414...
        var kth = vector(0.5f * 3, (float) Math.sqrt(3) / 2 * 3);
        var candidate = vector(0, 0.25f);
        float kthScore = COSINE.compare(q, kth);
        float candidateScore = COSINE.compare(q, candidate);
        assertTrue(AdaptiveTermination.of(COSINE, 0.41f).shouldStop(candidateScore, kthScore));
        assertFalse(AdaptiveTermination.of(COSINE, 0.42f).shouldStop(candidateScore, kthScore));
    }

    @Test
    public void tiesStopOnlyWithZeroGamma() {
        for (var vsf : VectorSimilarityFunction.values()) {
            assertTrue(vsf.toString(), AdaptiveTermination.of(vsf, 0f).shouldStop(0.7f, 0.7f));
            assertFalse(vsf.toString(), AdaptiveTermination.of(vsf, 0.01f).shouldStop(0.7f, 0.7f));
        }
    }

    /** Nothing can beat a result at distance 0, so the search can stop whatever the gamma. */
    @Test
    public void exactMatchAsKthResultAlwaysStops() {
        for (var vsf : VectorSimilarityFunction.values()) {
            var rule = AdaptiveTermination.of(vsf, 1000f);
            assertTrue(vsf.toString(), rule.shouldStop(0.999f, 1.0f));
            assertTrue(vsf.toString(), rule.shouldStop(1.0f, 1.0f));
        }
    }

    /**
     * For random vectors and gammas, the rule's decision on scores matches the paper's rule on true distances
     * ((1 + gamma) * d(q, kth) <= d(q, candidate)), except where the two sides are too close for float scores to tell
     * apart.
     */
    @Test
    public void matchesTheRuleOnTrueDistances() {
        Random random = getRandom();
        int checked = 0, stops = 0;
        for (int trial = 0; trial < 20_000; trial++) {
            int dim = 2 + random.nextInt(30);
            var vsf = random.nextBoolean() ? EUCLIDEAN : COSINE;
            var q = randomVector(random, dim);
            var a = randomVector(random, dim);
            var b = randomVector(random, dim);
            float gamma = random.nextInt(5) == 0 ? 0f : random.nextFloat() * 2;
            double da = trueDistance(vsf, q, a), db = trueDistance(vsf, q, b);
            double lhs = (1 + (double) gamma) * da;
            if (Math.abs(lhs - db) <= 1e-3 * Math.max(lhs, db)) {
                continue; // too close to call with float scores
            }
            boolean expected = lhs <= db;
            boolean actual = AdaptiveTermination.of(vsf, gamma).shouldStop(vsf.compare(q, b), vsf.compare(q, a));
            assertEquals(vsf + " gamma=" + gamma + " d(kth)=" + da + " d(candidate)=" + db, expected, actual);
            checked++;
            if (actual) stops++;
        }
        // both outcomes were exercised
        assertTrue(checked > 15_000);
        assertTrue(stops > checked / 10 && stops < checked * 9 / 10);
    }

    /** For unit vectors DOT_PRODUCT and COSINE scores are the same, and so are the decisions. */
    @Test
    public void dotProductOnUnitVectorsDecidesLikeCosine() {
        Random random = getRandom();
        for (int trial = 0; trial < 2_000; trial++) {
            int dim = 2 + random.nextInt(30);
            var q = unit(randomVector(random, dim));
            var a = unit(randomVector(random, dim));
            var b = unit(randomVector(random, dim));
            float gamma = random.nextFloat();
            assertEquals(AdaptiveTermination.of(COSINE, gamma).shouldStop(COSINE.compare(q, b), COSINE.compare(q, a)),
                         AdaptiveTermination.of(DOT_PRODUCT, gamma).shouldStop(DOT_PRODUCT.compare(q, b), DOT_PRODUCT.compare(q, a)));
        }
    }

    @Test
    public void largerGammaNeverStopsWhereASmallerOneDoesNot() {
        Random random = getRandom();
        for (int trial = 0; trial < 20_000; trial++) {
            var vsf = VectorSimilarityFunction.values()[random.nextInt(3)];
            float s1 = score(random, vsf), s2 = score(random, vsf);
            float g1 = random.nextInt(5) == 0 ? 0f : random.nextFloat() * 2, g2 = g1 + random.nextFloat();
            if (AdaptiveTermination.of(vsf, g2).shouldStop(s1, s2)) {
                assertTrue(vsf + " " + s1 + " " + s2 + " " + g1 + " " + g2, AdaptiveTermination.of(vsf, g1).shouldStop(s1, s2));
            }
        }
    }

    @Test
    public void aCandidateBetterThanTheKthResultNeverStops() {
        Random random = getRandom();
        for (int trial = 0; trial < 20_000; trial++) {
            var vsf = VectorSimilarityFunction.values()[random.nextInt(3)];
            float s1 = score(random, vsf), s2 = score(random, vsf);
            float gamma = random.nextInt(5) == 0 ? 0f : random.nextFloat() * 2;
            if (s1 > s2) {
                assertFalse(vsf + " " + s1 + " " + s2, AdaptiveTermination.of(vsf, gamma).shouldStop(s1, s2));
            }
        }
    }

    /**
     * DOT_PRODUCT of vectors that are not unit length scores (1 + dot) / 2, which can be negative or above 1, where
     * 1 - score is not a distance. A k-th result scoring 1 or more stops the search at the first candidate that is
     * not better, for any gamma.
     */
    @Test
    public void unnormalizedDotProductKthAboveAnExactMatch() {
        for (float gamma : new float[] {0f, 0.1f, 5f}) {
            var rule = AdaptiveTermination.of(DOT_PRODUCT, gamma);
            assertTrue(rule.shouldStop(0.9f, 3f));
            assertTrue(rule.shouldStop(-2f, 1f));
            assertFalse(rule.shouldStop(3.5f, 3f));
        }
        // and below 1 it is the COSINE rule
        assertTrue(AdaptiveTermination.of(DOT_PRODUCT, 0.4f).shouldStop(0.5f, 0.75f)); // 1 - s: 0.5 vs 0.25 * 1.96
        assertFalse(AdaptiveTermination.of(DOT_PRODUCT, 0.5f).shouldStop(0.5f, 0.75f)); // 0.5 vs 0.25 * 2.25
    }

    /** scores as the similarity function can produce them; DOT_PRODUCT ranges over vectors that are not unit length */
    private static float score(Random random, VectorSimilarityFunction vsf) {
        if (vsf == DOT_PRODUCT && random.nextBoolean()) {
            return (float) (random.nextGaussian() * 3);
        }
        return random.nextInt(50) == 0 ? 1f : random.nextFloat();
    }

    private static double trueDistance(VectorSimilarityFunction vsf, VectorFloat<?> x, VectorFloat<?> y) {
        if (vsf == COSINE) {
            x = unit(x);
            y = unit(y);
        }
        double sum = 0;
        for (int i = 0; i < x.length(); i++) {
            double d = (double) x.get(i) - y.get(i);
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    private static VectorFloat<?> unit(VectorFloat<?> v) {
        double norm = 0;
        for (int i = 0; i < v.length(); i++) norm += (double) v.get(i) * v.get(i);
        norm = Math.sqrt(norm);
        var out = vts.createFloatVector(v.length());
        for (int i = 0; i < v.length(); i++) out.set(i, (float) (v.get(i) / norm));
        return out;
    }

    private static VectorFloat<?> randomVector(Random random, int dim) {
        var v = vts.createFloatVector(dim);
        float scale = random.nextBoolean() ? 1f : 0.1f + 3 * random.nextFloat();
        for (int i = 0; i < dim; i++) v.set(i, (float) random.nextGaussian() * scale);
        return v;
    }

    private static VectorFloat<?> vector(float... values) {
        return vts.createFloatVector(values);
    }
}
