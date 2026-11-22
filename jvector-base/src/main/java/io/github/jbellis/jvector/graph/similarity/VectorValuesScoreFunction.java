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

package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Arrays;

/**
 * Exact similarity of a fixed vector to the vectors of a {@link RandomAccessVectorValues}.
 * <p>
 * A batch of nodes is scored with one bulk {@link VectorSimilarityFunction#compare(VectorFloat, VectorFloat[], int, float[])}
 * call, which lets the vector provider overlap the memory loads of the targets. That needs every target vector at once,
 * so it is only done when the values are not shared; a shared source is scored one node at a time.
 * <p>
 * Threadsafe: the scratch space of the bulk path is per thread.
 */
final class VectorValuesScoreFunction implements ScoreFunction.ExactScoreFunction {
    /** how many targets the provider scores at once */
    private static final int BATCH = 4;
    /** the target vectors of the bulk call, per thread */
    private static final ThreadLocal<VectorFloat<?>[]> TARGETS = ThreadLocal.withInitial(() -> new VectorFloat<?>[BATCH]);
    /** the scores of the bulk call, per thread */
    private static final ThreadLocal<float[]> SCORES = ThreadLocal.withInitial(() -> new float[BATCH]);

    private final VectorFloat<?> vector;
    private final VectorSimilarityFunction similarityFunction;
    private final RandomAccessVectorValues values;
    /** maps node ids to ordinals of {@code values}, or null if they are the same */
    private final int[] ordinals;

    VectorValuesScoreFunction(VectorFloat<?> vector, VectorSimilarityFunction similarityFunction, RandomAccessVectorValues values, int[] ordinals) {
        this.vector = vector;
        this.similarityFunction = similarityFunction;
        this.values = values;
        this.ordinals = ordinals;
    }

    private int ordinal(int node) {
        return ordinals == null ? node : ordinals[node];
    }

    @Override
    public float similarityTo(int node2) {
        return similarityFunction.compare(vector, values.getVector(ordinal(node2)));
    }

    @Override
    public boolean supportsBatchScoring() {
        // a shared source hands out the same vector again on the next call, so its vectors cannot be held
        return !values.isValueShared();
    }

    @Override
    public void similarityTo(int[] nodes, int count, float[] results) {
        if (count < 2 || values.isValueShared()) {
            ExactScoreFunction.super.similarityTo(nodes, count, results);
            return;
        }
        // Score four targets at a time: the provider keeps four load streams in flight, and only four target
        // vectors are held at once, so a source that materializes a vector per call does not have to keep them
        // all alive.
        var targets = TARGETS.get();
        var scores = SCORES.get();
        for (int i = 0; i < count; i += BATCH) {
            int n = Math.min(BATCH, count - i);
            for (int j = 0; j < n; j++) {
                targets[j] = values.getVector(ordinal(nodes[i + j]));
            }
            similarityFunction.compare(vector, targets, n, scores);
            System.arraycopy(scores, 0, results, i, n);
            // don't keep vectors the source may have materialized for this call
            Arrays.fill(targets, 0, n, null);
        }
    }
}
