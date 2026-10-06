/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.function.Supplier;

/**
 * Provides random access to float32 vectors by dense ordinal. This interface is used by graph-based
 * implementations of KNN search.
 * <p>
 * For int8 vectors see {@link RandomAccessByteVectorValues}.
 * Both extend the common super-interface {@link VectorValues}.
 */
public interface RandomAccessVectorValues extends VectorValues<VectorFloat<?>> {

    /**
     * Return the number of vector values.
     * <p>
     * All copies of a given RAVV should have the same size.  Typically this is achieved by either
     * (1) implementing a threadsafe, un-shared RAVV, where `copy` returns `this`, or
     * (2) implementing a fixed-size RAVV.
     */
    @Override
    int size();

    @Deprecated
    default VectorFloat<?> vectorValue(int targetOrd) {
        return getVector(targetOrd);
    }

    /**
     * Retrieve the vector associated with a given node, and store it in the destination vector at the given offset.
     * @param node the node to retrieve
     * @param destinationVector the vector to store the result in
     * @param offset the offset in the destination vector to store the result
     */
    default void getVectorInto(int node, VectorFloat<?> destinationVector, int offset) {
        destinationVector.copyFrom(getVector(node), 0, offset, dimension());
    }

    /**
     * Creates a new copy of this {@link RandomAccessVectorValues}. This is helpful when you need to
     * access different values at once, to avoid overwriting the underlying float vector returned by
     * a shared {@link RandomAccessVectorValues#getVector}.
     * <p>
     * Un-shared implementations may simply return `this`.
     */
    @Override
    RandomAccessVectorValues copy();

    /**
     * Convenience method to create an ExactScoreFunction for reranking.  The resulting function is NOT thread-safe.
     */
    default ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
        return new ScoreFunction.ExactScoreFunction() {
            private final VectorFloat<?> scratch = vts.createFloatVector(dimension());

            @Override
            public float similarityTo(int node2) {
                getVectorInto(node2, scratch, 0);
                return vsf.compare(queryVector, scratch);
            }
        };
    }
}
