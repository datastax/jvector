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
 * Not threadsafe: the bulk path keeps scratch space.
 */
final class VectorValuesScoreFunction implements ScoreFunction.ExactScoreFunction {
    private static final VectorFloat<?>[] NO_VECTORS = new VectorFloat<?>[0];

    private final VectorFloat<?> vector;
    private final VectorSimilarityFunction similarityFunction;
    private final RandomAccessVectorValues values;
    /** maps node ids to ordinals of {@code values}, or null if they are the same */
    private final int[] ordinals;
    private VectorFloat<?>[] targets = NO_VECTORS;

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
    public void similarityTo(int[] nodes, int count, float[] results) {
        if (count < 2 || values.isValueShared()) {
            ExactScoreFunction.super.similarityTo(nodes, count, results);
            return;
        }
        if (targets.length < count) {
            targets = new VectorFloat<?>[Math.max(count, 2 * targets.length)];
        }
        for (int i = 0; i < count; i++) {
            targets[i] = values.getVector(ordinal(nodes[i]));
        }
        similarityFunction.compare(vector, targets, count, results);
        // don't keep vectors the source may have materialized for this call
        Arrays.fill(targets, 0, count, null);
    }
}
