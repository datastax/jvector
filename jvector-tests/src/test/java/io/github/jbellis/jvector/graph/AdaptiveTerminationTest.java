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

import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static io.github.jbellis.jvector.graph.AdaptiveTermination.Decision.CONTINUE;
import static io.github.jbellis.jvector.graph.AdaptiveTermination.Decision.FALL_BACK_TO_STANDARD;
import static io.github.jbellis.jvector.graph.AdaptiveTermination.Decision.TERMINATE;
import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.COSINE;
import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.DOT_PRODUCT;
import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;
import static org.junit.jupiter.api.Assertions.*;

class AdaptiveTerminationTest {
    /** The gamma whose squared-distance factor (1 + gamma)^2 is {@code factor}. */
    private static float gammaFor(double factor) {
        return (float) (Math.sqrt(factor) - 1);
    }

    private static NodeQueue results(float... scores) {
        NodeQueue q = new NodeQueue(new BoundedLongHeap(scores.length), NodeQueue.Order.MIN_HEAP);
        for (int i = 0; i < scores.length; i++) q.push(i, scores[i]);
        return q;
    }

    private static AdaptiveTermination rule(int k, double factor, VectorSimilarityFunction similarity, float bound) {
        var t = new AdaptiveTermination();
        assertTrue(t.reset(k, gammaFor(factor), similarity, bound));
        return t;
    }

    @Test void requiresKResults() {
        var t = rule(2, 1.1, COSINE, Float.NaN);
        assertEquals(CONTINUE, t.shouldTerminate(0.1f, results(0.9f)));
    }

    @Test void cosineReadsOneMinusScore() {
        var t = rule(1, 1.1, COSINE, Float.NaN);
        NodeQueue q = results(0.9f);
        assertEquals(CONTINUE, t.shouldTerminate(0.89f, q));
        assertEquals(TERMINATE, t.shouldTerminate(0.88f, q));
    }

    @Test void euclideanReadsSquaredDistance() {
        // Score 0.5 is d^2 = 1; with factor 1.1 the cutoff is d^2 = 1.1, score 1 / 2.1 = 0.476.
        var t = rule(1, 1.1, EUCLIDEAN, Float.NaN);
        NodeQueue q = results(0.5f);
        assertEquals(CONTINUE, t.shouldTerminate(0.48f, q));
        assertEquals(TERMINATE, t.shouldTerminate(0.47f, q));
    }

    @Test void dotProductReadsDistanceFromTheBound() {
        // Bound 9: top = 5. The k-th dot product 7 scores 4, distance 1; factor 1.1 puts the cutoff at score 3.9.
        var t = rule(1, 1.1, DOT_PRODUCT, 9f);
        NodeQueue q = results(4f);
        assertEquals(CONTINUE, t.shouldTerminate(3.95f, q));
        assertEquals(TERMINATE, t.shouldTerminate(3.85f, q));
    }

    @Test void dotProductDecisionsDoNotDependOnScale() {
        var random = new Random(7);
        for (int i = 0; i < 10_000; i++) {
            float bound = 1 + 9 * random.nextFloat();
            float kthDot = bound * (2 * random.nextFloat() - 1);
            float candidateDot = kthDot - bound * random.nextFloat();
            double margin = Math.abs((bound - candidateDot) - 1.02 * (bound - kthDot)) / bound;
            if (margin < 1e-4) continue; // too close to the cutoff for float scores at every scale
            AdaptiveTermination.Decision reference = null;
            for (float scale : new float[] {0.01f, 1f, 100f}) {
                var t = rule(1, 1.02, DOT_PRODUCT, bound * scale);
                var decision = t.shouldTerminate((1 + candidateDot * scale) / 2, results((1 + kthDot * scale) / 2));
                if (reference == null) reference = decision;
                assertEquals(reference, decision, "scale " + scale + " bound " + bound + " kth " + kthDot + " candidate " + candidateDot);
            }
        }
    }

    @Test void matchesThePreviousRuleOnUnitLengthScores() {
        // The rule this replaced read 1 - score as the distance and compared it against (1 + 0.01) times the k-th's.
        var random = new Random(11);
        for (var similarity : new VectorSimilarityFunction[] {COSINE, DOT_PRODUCT}) {
            var t = new AdaptiveTermination();
            assertTrue(t.reset(1, AdaptiveTermination.DEFAULT_GAMMA, similarity, 1f));
            for (int i = 0; i < 100_000; i++) {
                float kth = random.nextFloat(), candidate = kth * random.nextFloat();
                double previous = (1 - candidate) - (1.01 * (1 - kth) + 1e-6);
                if (Math.abs(previous) < 1e-5) continue;
                assertEquals(previous > 0 ? TERMINATE : CONTINUE, t.shouldTerminate(candidate, results(kth)),
                        similarity + " kth " + kth + " candidate " + candidate);
            }
        }
    }

    @Test void scoresAboveTheBoundRaiseIt() {
        var t = rule(1, 1.1, DOT_PRODUCT, 1f);
        NodeQueue q = results(0.9f);
        assertEquals(TERMINATE, t.shouldTerminate(0.88f, q));
        // An approximate score of 1.2 moves the zero point up, which loosens the margin: 0.32 <= 1.1 * 0.3.
        t.observe(1.2f);
        t.observe(1.1f);
        assertEquals(CONTINUE, t.shouldTerminate(0.88f, q));
        assertEquals(TERMINATE, t.shouldTerminate(0.85f, q));
    }

    @Test void theCapStopsCandidatesTheMarginWouldContinue() {
        // A wide margin continues 0.5 against a k-th best of 0.9, until the cap rises above it.
        var t = rule(1, 4.0, COSINE, Float.NaN);
        NodeQueue q = results(0.9f);
        assertEquals(CONTINUE, t.shouldTerminate(0.7f, q));
        t.raiseCap(0.8f);
        assertEquals(TERMINATE, t.shouldTerminate(0.7f, q));
        assertEquals(CONTINUE, t.shouldTerminate(0.85f, q));
        assertTrue(t.lowestContinuableScore(0.9f) >= 0.8f);
        // The cap only rises.
        t.raiseCap(0.5f);
        assertEquals(TERMINATE, t.shouldTerminate(0.7f, q));
    }

    @Test void theBarNeverFallsDuringASearch() {
        var t = rule(1, 1.1, DOT_PRODUCT, 1f);
        float before = t.lowestContinuableScore(0.9f);
        // An approximate score above the bound raises the zero point, which would lower a recomputed bar.
        t.observe(1.5f);
        assertTrue(t.lowestContinuableScore(0.9f) >= before);
        assertEquals(TERMINATE, t.shouldTerminate(Math.nextDown(before), results(0.9f)));
    }

    @Test void dotProductWithoutABoundUsesTheStandardRule() {
        var t = new AdaptiveTermination();
        assertFalse(t.reset(1, 0.01f, DOT_PRODUCT, Float.NaN));
        assertEquals(FALL_BACK_TO_STANDARD, t.shouldTerminate(0.1f, results(0.9f)));
        assertEquals(Float.NEGATIVE_INFINITY, t.lowestContinuableScore(0.9f));
        assertFalse(t.reset(1, 0.01f, DOT_PRODUCT, Float.POSITIVE_INFINITY));
        assertFalse(t.reset(1, 0.01f, DOT_PRODUCT, -1f));
    }

    @Test void nonFiniteScoresFallBackUntilReset() {
        var t = rule(1, 1.1, COSINE, Float.NaN);
        assertEquals(FALL_BACK_TO_STANDARD, t.shouldTerminate(Float.NaN, results(0.9f)));
        assertEquals(FALL_BACK_TO_STANDARD, t.shouldTerminate(0.1f, results(0.9f)));
        assertTrue(t.reset(1, gammaFor(1.1), COSINE, Float.NaN));
        assertEquals(TERMINATE, t.shouldTerminate(0.88f, results(0.9f)));
        assertEquals(CONTINUE, t.shouldTerminate(Float.POSITIVE_INFINITY, results(0.9f)));
        assertEquals(TERMINATE, t.shouldTerminate(Float.NEGATIVE_INFINITY, results(0.9f)));
    }

    @Test void theExpansionBarNeverExcludesAContinuedCandidate() {
        var random = new Random(5);
        for (var similarity : VectorSimilarityFunction.values()) {
            for (int i = 0; i < 20_000; i++) {
                float bound = 10 * random.nextFloat();
                var t = new AdaptiveTermination();
                assertTrue(t.reset(1, 0.05f * random.nextFloat(), similarity, bound));
                float kth, candidate;
                if (similarity == EUCLIDEAN) {
                    kth = random.nextFloat();
                    candidate = kth * random.nextFloat();
                } else {
                    float top = (1 + bound) / 2;
                    kth = top - top * random.nextFloat();
                    candidate = kth - top * random.nextFloat();
                    if (random.nextInt(4) == 0) t.observe(top + random.nextFloat());
                }
                float bar = t.lowestContinuableScore(kth);
                if (t.shouldTerminate(candidate, results(kth)) == CONTINUE) {
                    assertTrue(candidate >= bar, similarity + " kth " + kth + " candidate " + candidate + " bar " + bar);
                }
            }
        }
    }
}
