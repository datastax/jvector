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

package io.github.jbellis.jvector.graph.disk.feature;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.disk.CommonHeader;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;

/**
 * Implements the storage of full-resolution vectors inline into an OnDiskGraphIndex. These can be used for exact scoring.
 */
public class InlineVectors implements Feature {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final int dimension;

    /**
     * Constructs an InlineVectors feature with the specified dimension.
     *
     * @param dimension the vector dimension
     */
    public InlineVectors(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public FeatureId id() {
        return FeatureId.INLINE_VECTORS;
    }

    @Override
    public int headerSize() {
        return 0;
    }

    /**
     * Returns the size in bytes of each inline vector.
     *
     * @return the feature size in bytes
     */
    public int featureSize() {
        return dimension * Float.BYTES;
    }

    /**
     * Returns the dimension of the stored vectors.
     *
     * @return the vector dimension
     */
    public int dimension() {
        return dimension;
    }

    /**
     * Loads an InlineVectors feature from the reader.
     *
     * @param header the common header containing dimension information
     * @param reader the reader (not used, dimension comes from header)
     * @return a new InlineVectors instance
     */
    static InlineVectors load(CommonHeader header, RandomAccessReader reader) {
        return new InlineVectors(header.dimension);
    }

    @Override
    public void writeHeader(DataOutput out) {
        // common header contains dimension, which is sufficient
    }

    @Override
    public void writeInline(DataOutput out, Feature.State state) throws IOException {
        vectorTypeSupport.writeFloatVector(out, ((InlineVectors.State) state).vector);
    }

    /**
     * State holder for an inline vector being written.
     */
    public static class State implements Feature.State {
        /** The vector to be written inline. */
        public final VectorFloat<?> vector;

        /**
         * Constructs a State with the given vector.
         *
         * @param vector the vector to store
         */
        public State(VectorFloat<?> vector) {
            this.vector = vector;
        }
    }
}
