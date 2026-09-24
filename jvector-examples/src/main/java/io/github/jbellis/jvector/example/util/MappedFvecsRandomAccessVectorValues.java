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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

/// A memory-mapped, read-only {@link RandomAccessVectorValues} over a standard `.fvecs` file.
///
/// Each fvecs record is a little-endian `int` dimension followed by that many little-endian
/// `float`s. All records must have the same dimension, which is taken from the first record and
/// verified on every read. The file is mapped in slabs of at most {@link #DEFAULT_MAX_SLAB_BYTES}
/// so files larger than 2 GB work; a record never straddles two slabs.
///
/// Reads are served by copying the record out of the mapping. Consequently {@link #getVector(int)}
/// returns a per-instance scratch vector and {@link #isValueShared()} is `true`; use
/// {@link #copy()} (or {@link #threadLocalSupplier()}) for concurrent readers, which share the
/// mapping and cost only a scratch vector each. Bulk in-memory caching should go through
/// {@link #range(int, int)} plus {@link #getVectorInto(int, VectorFloat, int)}.
///
/// The mapping is released when the instance (and all copies) become unreachable; there is
/// deliberately no explicit unmap, since unmapping under a concurrent read crashes the JVM.
public final class MappedFvecsRandomAccessVectorValues implements RandomAccessVectorValues {
    /// Upper bound on the bytes mapped per slab: 1 GiB, comfortably under the `int` limit of a single mapping.
    public static final long DEFAULT_MAX_SLAB_BYTES = 1L << 30;

    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final Path path;
    private final int dimension;
    private final int size;
    private final int stride;
    private final int recordsPerSlab;
    private final MappedByteBuffer[] slabs;
    private final VectorFloat<?> scratch;

    /// Maps the fvecs file at `path` using the default slab size.
    ///
    /// @param path the `.fvecs` file
    /// @throws IOException if the file cannot be read, is empty, or is not a well-formed fvecs file
    public MappedFvecsRandomAccessVectorValues(Path path) throws IOException {
        this(path, DEFAULT_MAX_SLAB_BYTES);
    }

    /// Maps the fvecs file at `path`, limiting each slab to `maxSlabBytes` (rounded down to a whole
    /// number of records). Exposed for tests that need to exercise slab boundaries on small files.
    ///
    /// @param path         the `.fvecs` file
    /// @param maxSlabBytes the maximum bytes per mapped slab; must hold at least one record
    /// @throws IOException if the file cannot be read, is empty, or is not a well-formed fvecs file
    public MappedFvecsRandomAccessVectorValues(Path path, long maxSlabBytes) throws IOException {
        this.path = path;
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long length = channel.size();
            if (length < Integer.BYTES) {
                throw new IOException("fvecs file is empty or truncated: " + path);
            }
            ByteBuffer header = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(header, 0);
            int dim = header.getInt(0);
            if (dim <= 0) {
                throw new IOException("Corrupt fvecs file: negative or zero dimension " + dim + " (possible file corruption or wrong format): " + path);
            }
            if (dim > 100_000) {
                throw new IOException("Unreasonable dimension " + dim + " in fvecs file (possible file corruption or wrong format): " + path);
            }
            this.dimension = dim;
            this.stride = Integer.BYTES + dim * Float.BYTES;
            if (length % stride != 0) {
                throw new IOException("fvecs file length " + length + " is not a multiple of the " + stride
                        + "-byte record size for dimension " + dim + " (truncated or mixed dimensions): " + path);
            }
            long count = length / stride;
            if (count > Integer.MAX_VALUE) {
                throw new IOException("fvecs file holds " + count + " vectors, more than can be addressed by ordinal: " + path);
            }
            this.size = (int) count;
            if (maxSlabBytes < stride) {
                throw new IllegalArgumentException("maxSlabBytes " + maxSlabBytes + " is smaller than one record of " + stride + " bytes");
            }
            this.recordsPerSlab = (int) Math.min(count, maxSlabBytes / stride);
            int slabCount = (int) ((count + recordsPerSlab - 1) / recordsPerSlab);
            this.slabs = new MappedByteBuffer[slabCount];
            long slabBytes = (long) recordsPerSlab * stride;
            for (int s = 0; s < slabCount; s++) {
                long offset = s * slabBytes;
                long slabLength = Math.min(slabBytes, length - offset);
                MappedByteBuffer slab = channel.map(FileChannel.MapMode.READ_ONLY, offset, slabLength);
                slab.order(ByteOrder.LITTLE_ENDIAN);
                slabs[s] = slab;
            }
        }
        this.scratch = vts.createFloatVector(dimension);
    }

    private MappedFvecsRandomAccessVectorValues(MappedFvecsRandomAccessVectorValues other) {
        this.path = other.path;
        this.dimension = other.dimension;
        this.size = other.size;
        this.stride = other.stride;
        this.recordsPerSlab = other.recordsPerSlab;
        this.slabs = other.slabs;
        this.scratch = vts.createFloatVector(dimension);
    }

    /// @return the mapped file
    public Path getPath() {
        return path;
    }

    /// @return the number of mapped slabs backing this file
    public int slabCount() {
        return slabs.length;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public VectorFloat<?> getVector(int nodeId) {
        read(nodeId, scratch, 0);
        return scratch;
    }

    @Override
    public void getVectorInto(int node, VectorFloat<?> destinationVector, int offset) {
        read(node, destinationVector, offset);
    }

    @Override
    public boolean isValueShared() {
        return true;
    }

    /// Returns a reader that shares the file mapping but owns its own scratch vector.
    @Override
    public RandomAccessVectorValues copy() {
        return new MappedFvecsRandomAccessVectorValues(this);
    }

    private void read(int node, VectorFloat<?> dest, int destOffset) {
        int idx = Objects.checkIndex(node, size);
        MappedByteBuffer slab = slabs[idx / recordsPerSlab];
        int pos = (idx % recordsPerSlab) * stride;
        int recordDim = slab.getInt(pos);
        if (recordDim != dimension) {
            throw new IllegalStateException("Corrupt fvecs record " + idx + " in " + path + ": dimension " + recordDim + " != " + dimension);
        }
        int dataPos = pos + Integer.BYTES;
        Object backing = dest.get();
        if (backing instanceof float[]) {
            ByteBuffer view = slab.duplicate().order(ByteOrder.LITTLE_ENDIAN);
            view.position(dataPos);
            view.asFloatBuffer().get((float[]) backing, dest.offset(destOffset), dimension);
        } else {
            for (int i = 0; i < dimension; i++) {
                dest.set(destOffset + i, slab.getFloat(dataPos + i * Float.BYTES));
            }
        }
    }
}
