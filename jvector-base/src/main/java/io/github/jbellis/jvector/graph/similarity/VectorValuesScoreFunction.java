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


/**
 * Exact similarity of a fixed vector to the vectors of a {@link RandomAccessVectorValues}. A batch of nodes is scored by
 * the source's {@link RandomAccessVectorValues#similarityTo(VectorFloat, VectorSimilarityFunction, int[], int, float[])},
 * so each source decides how to read and score its own vectors.
 * <p>
 * Threadsafe: {@link Mapped}'s only scratch, the mapped ordinals, is per thread.
 */
class VectorValuesScoreFunction implements ScoreFunction.ExactScoreFunction {
    final VectorFloat<?> vector;
    final VectorSimilarityFunction similarityFunction;
    final RandomAccessVectorValues values;

    VectorValuesScoreFunction(VectorFloat<?> vector, VectorSimilarityFunction similarityFunction, RandomAccessVectorValues values) {
        this.vector = vector;
        this.similarityFunction = similarityFunction;
        this.values = values;
    }

    @Override
    public float similarityTo(int node2) {
        return similarityFunction.compare(vector, values.getVector(node2));
    }

    @Override
    public boolean supportsBatchScoring() {
        return true;
    }

    @Override
    public void similarityTo(int[] nodes, int count, float[] results) {
        values.similarityTo(vector, similarityFunction, nodes, count, results);
    }

    /** Scores nodes whose ordinals in {@code values} differ from their ids. */
    static final class Mapped extends VectorValuesScoreFunction {
        /** node ids mapped to ordinals of {@code values}, per thread */
        private static final ThreadLocal<int[]> MAPPED = ThreadLocal.withInitial(() -> new int[64]);

        private final int[] ordinals;

        Mapped(VectorFloat<?> vector, VectorSimilarityFunction similarityFunction, RandomAccessVectorValues values, int[] ordinals) {
            super(vector, similarityFunction, values);
            this.ordinals = ordinals;
        }

        @Override
        public float similarityTo(int node2) {
            return similarityFunction.compare(vector, values.getVector(ordinals[node2]));
        }

        @Override
        public void similarityTo(int[] nodes, int count, float[] results) {
            int[] mapped = MAPPED.get();
            if (mapped.length < count) {
                mapped = new int[count];
                MAPPED.set(mapped);
            }
            for (int i = 0; i < count; i++) {
                mapped[i] = ordinals[nodes[i]];
            }
            values.similarityTo(vector, similarityFunction, mapped, count, results);
        }
    }
}
