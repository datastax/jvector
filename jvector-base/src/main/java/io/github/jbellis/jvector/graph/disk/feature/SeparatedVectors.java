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
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.io.UncheckedIOException;

public class SeparatedVectors extends AbstractSeparatedFeature {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final int dimension;
    private long offset;
    // The largest length of the vectors this feature has written, for the header.
    private final MaxVectorNorm written = new MaxVectorNorm();
    // The largest length stored in a loaded index's header; NaN when unknown.
    private final float stored;

    public SeparatedVectors(int dimension, long offset) {
        this(dimension, offset, Float.NaN);
    }

    private SeparatedVectors(int dimension, long offset, float stored) {
        this.dimension = dimension;
        this.offset = offset;
        this.stored = stored;
    }

    /**
     * The largest length of the stored vectors: from the header of a loaded index, or of the vectors written so far.
     * NaN when unknown, as for an index written before format version 7.
     */
    public float maxVectorNorm() {
        return Float.isNaN(stored) ? written.get() : stored;
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
        return headerSize(OnDiskGraphIndex.CURRENT_VERSION);
    }

    @Override
    public int headerSize(int version) {
        return Long.BYTES + MaxVectorNorm.headerSize(version);
    }

    @Override
    public int featureSize() {
        return dimension * Float.BYTES;
    }

    @Deprecated
    @Override
    public void writeHeader(IndexWriter out) throws IOException {
        writeHeader(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    @Override
    public void writeHeader(IndexWriter out, int version) throws IOException {
        out.writeLong(offset);
        written.write(out, version);
    }

    @Override
    public void writeSeparately(IndexWriter out, State state_) throws IOException {
        var state = (InlineVectors.State) state_;
        if (state.vector != null) {
            written.record(state.vector);
            vectorTypeSupport.writeFloatVector(out, state.vector);
        } else {
            // Write zeros for missing vector
            for (int j = 0; j < dimension; j++) {
                out.writeFloat(0.0f);
            }
        }
    }

    // Using InlineVectors.State

    static SeparatedVectors load(CommonHeader header, RandomAccessReader reader) {
        try {
            long offset = reader.readLong();
            return new SeparatedVectors(header.dimension, offset, MaxVectorNorm.read(reader, header.version));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public int dimension() {
        return dimension;
    }
}
