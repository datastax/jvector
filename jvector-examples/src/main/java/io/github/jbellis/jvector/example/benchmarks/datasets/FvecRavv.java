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

package io.github.jbellis.jvector.example.benchmarks.datasets;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.disk.ByteBufferReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * A {@link RandomAccessVectorValues} over a memory-mapped Fvec file.
 */
class FvecRavv implements RandomAccessVectorValues {

    private static final ByteOrder byteOrder = ByteOrder.LITTLE_ENDIAN;
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static int readInt(FileChannel ch, long position) throws IOException {
        ByteBuffer bf = ByteBuffer.allocate(Integer.BYTES).order(byteOrder);
        ch.read(bf, position);
        bf.flip();
        return bf.asIntBuffer().get();
    }

    private final List<MappedByteBuffer> mbbs;
    private final int dim;
    private final int nvecs;
    private final int vecsPerGroup;

    private final VectorFloat<?> bufVec;

    private FvecRavv(List<MappedByteBuffer> mbbs, int dim, int nvecs, int vecsPerGroup) {
        this.mbbs = mbbs;
        this.dim = dim;
        this.nvecs = nvecs;
        this.vecsPerGroup = vecsPerGroup;
        this.bufVec = vts.createFloatVector(dim);
    }

    static FvecRavv of(Path path) throws IOException {
        return FvecRavv.of(path, Integer.MAX_VALUE);
    }

    /** Use {@link #of(Path)} instead */
    @VisibleForTesting
    static FvecRavv of(Path path, int maxBytesPerGroup) throws IOException {
        // This static factory method avoids the ReaderSupplierFactory.open() indirection
        // used elsewhere in JVector.
        // ReaderSuppliers and RandomAccessReaders are AutoCloseable, which would require this RAVV
        // to be AutoCloseable, which complicates upstream integration.
        // This approach sidesteps the problem simce MappedByteBuffers are valid until garbage-collected,
        // even after the original FileChannel is closed.
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            long size = ch.size();
            int dim = readInt(ch, 0);
            if (dim <= 0) {
                throw new RuntimeException("Fvec dimension is negative");
            }

            int vecBytes = Integer.BYTES + dim * Float.BYTES;
            int vecsPerGroup = maxBytesPerGroup / vecBytes;
            int groupSizeBytes = vecsPerGroup * vecBytes;

            int nvecs = Math.toIntExact(size / vecBytes);
            if (nvecs * (long) vecBytes != size) {
                throw new RuntimeException("File size is not divisible by row size");
            }

            var mbbs = new ArrayList<MappedByteBuffer>();
            for (long i = 0; i < size; i += groupSizeBytes) {
                int mapSize = Math.toIntExact(Math.min(size - i, groupSizeBytes));
                var mbb = ch.map(MapMode.READ_ONLY, i, mapSize);
                mbbs.add(mbb);
            }

            return new FvecRavv(Collections.unmodifiableList(mbbs), dim, nvecs, vecsPerGroup);
        }
    }

	@Override
	public int size() {
        return nvecs;
	}

	@Override
	public int dimension() {
        return dim;
	}

    @Override
    public void getVectorInto(int node, VectorFloat<?> destinationVector, int offset) {
        if (node >= nvecs) {
            throw new IndexOutOfBoundsException(node);
        }
        int groupId = node / vecsPerGroup;
        int inGroupId = node % vecsPerGroup;
        int inGroupByteOffset = (inGroupId + 1) * Integer.BYTES + inGroupId * dim * Float.BYTES;

        var slice = mbbs.get(groupId)
            .slice()
            .position(inGroupByteOffset)
            .limit(inGroupByteOffset + dim * Float.BYTES)
            .order(byteOrder);

        // close is expected to be a no-op for ByteBufferReader and therefore cheap
        try (var reader = new ByteBufferReader(slice)) {
            vts.readFloatVector(reader, dim, destinationVector, offset);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

	@Override
	public VectorFloat<?> getVector(int nodeId) {
        this.getVectorInto(nodeId, bufVec, 0);
        return bufVec;
	}

	@Override
	public boolean isValueShared() {
        return true;
	}

	@Override
	public RandomAccessVectorValues copy() {
        return new FvecRavv(mbbs, dim, nvecs, vecsPerGroup);
	}

    @VisibleForTesting
    public int getNumGroups() {
        return mbbs.size();
    }
}
