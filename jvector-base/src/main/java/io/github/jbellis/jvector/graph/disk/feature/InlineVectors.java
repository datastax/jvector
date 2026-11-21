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

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.disk.CommonHeader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Implements the storage of full-resolution vectors inline into an OnDiskGraphIndex. These can be used for exact scoring.
 */
public class InlineVectors extends AbstractFeature {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final int dimension;
    // The largest length of the vectors this feature has written, for the header.
    private final MaxVectorNorm written = new MaxVectorNorm();
    // The largest length stored in a loaded index's header; NaN when unknown.
    private final float stored;

    public InlineVectors(int dimension) {
        this(dimension, Float.NaN);
    }

    private InlineVectors(int dimension, float stored) {
        this.dimension = dimension;
        this.stored = stored;
    }

    /**
     * The largest length of the stored vectors: from the header of a loaded index, or of the vectors written so far.
     * NaN when unknown, as for an index written before format version 7 or by a writer that wrote its header first.
     */
    public float maxVectorNorm() {
        return Float.isNaN(stored) ? written.get() : stored;
    }

    @Override
    public FeatureId id() {
        return FeatureId.INLINE_VECTORS;
    }

    @Override
    public int headerSize() {
        return headerSize(OnDiskGraphIndex.CURRENT_VERSION);
    }

    @Override
    public int headerSize(int version) {
        return MaxVectorNorm.headerSize(version);
    }

    public int featureSize() {
        return dimension * Float.BYTES;
    }

    public int dimension() {
        return dimension;
    }

    static InlineVectors load(CommonHeader header, RandomAccessReader reader) {
        try {
            return new InlineVectors(header.dimension, MaxVectorNorm.read(reader, header.version));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Deprecated
    @Override
    public void writeHeader(IndexWriter out) throws IOException {
        writeHeader(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    @Override
    public void writeHeader(IndexWriter out, int version) throws IOException {
        // The common header holds the dimension; from version 7 the header also stores the largest vector length.
        written.write(out, version);
    }

    /**
     * Counts {@code vector} in the largest length this feature's header stores, for writers that serialize inline
     * vectors themselves instead of through {@link #writeInline}.
     */
    public void recordWritten(VectorFloat<?> vector) {
        written.record(vector);
    }

    @Override
    public void writeInline(IndexWriter out, Feature.State state) throws IOException {
        var vector = ((InlineVectors.State) state).vector;
        written.record(vector);
        vectorTypeSupport.writeFloatVector(out, vector);
    }

    public static class State implements Feature.State {
        public final VectorFloat<?> vector;

        public State(VectorFloat<?> vector) {
            this.vector = vector;
        }
    }
}
