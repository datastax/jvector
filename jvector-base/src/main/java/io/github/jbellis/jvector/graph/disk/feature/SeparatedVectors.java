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
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * A separated feature implementation for full-resolution float vectors.
 * Stores vector data separately from the graph structure, with only offset information in the header.
 */
public class SeparatedVectors implements SeparatedFeature {
    /** Vectorization support for reading and writing vectors efficiently. */
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    /** The dimensionality of the vectors stored by this feature. */
    private final int dimension;
    /** The file offset where the separated vector data begins. */
    private long offset;

    /**
     * Constructs a SeparatedVectors feature with the specified dimension and offset.
     *
     * @param dimension the dimensionality of the vectors
     * @param offset the file offset where the vector data begins
     */
    public SeparatedVectors(int dimension, long offset) {
        this.dimension = dimension;
        this.offset = offset;
    }

    @Override
    public void setOffset(long offset) {
        this.offset = offset;
    }

    @Override
    public long getOffset() {
        return offset;
    }

    @Override
    public FeatureId id() {
        return FeatureId.SEPARATED_VECTORS;
    }

    @Override
    public int headerSize() {
        return Long.BYTES;
    }

    @Override
    public int featureSize() {
        return dimension * Float.BYTES;
    }

    @Override
    public void writeHeader(DataOutput out) throws IOException {
        out.writeLong(offset);
    }

    @Override
    public void writeSeparately(DataOutput out, State state_) throws IOException {
        var state = (InlineVectors.State) state_;
        if (state.vector != null) {
            vectorTypeSupport.writeFloatVector(out, state.vector);
        } else {
            // Write zeros for missing vector
            for (int j = 0; j < dimension; j++) {
                out.writeFloat(0.0f);
            }
        }
    }

    // Using InlineVectors.State

    /**
     * Loads a SeparatedVectors feature from the specified reader.
     * The dimension is taken from the common header.
     *
     * @param header the common header containing dimension information
     * @param reader the reader to load from
     * @return the loaded SeparatedVectors instance
     * @throws UncheckedIOException if an I/O error occurs during loading
     */
    static SeparatedVectors load(CommonHeader header, RandomAccessReader reader) {
        try {
            long offset = reader.readLong();
            return new SeparatedVectors(header.dimension, offset);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Returns the dimensionality of the vectors stored by this feature.
     *
     * @return the vector dimension
     */
    public int dimension() {
        return dimension;
    }
}
