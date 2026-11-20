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

package io.github.jbellis.jvector.gpu;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Vectors in a memory-mapped file of raw row-major little-endian float32 ({@code n x dim}, no header). The
 * {@link TornadoGraphBuildAccelerator} reads the mapping directly, so a dataset needs no second copy in memory: cuVS
 * builds its batches from it and the pruner packs it to FP16. {@link #getVector} copies a row, as JVector expects.
 */
public final class MappedVectors implements RandomAccessVectorValues, AutoCloseable {

    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final Arena arena;
    private final MemorySegment segment;
    private final int rows;
    private final int dim;

    /**
     * @param file a file of raw float32 rows
     * @param dim  the dimension
     * @param rows the first {@code rows} rows of the file, or -1 for all
     */
    public MappedVectors(Path file, int dim, int rows) {
        this.dim = dim;
        this.arena = Arena.ofShared();
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            long available = channel.size() / ((long) dim * Float.BYTES);
            this.rows = (int) (rows < 0 ? available : Math.min(rows, available));
            this.segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, (long) this.rows * dim * Float.BYTES, arena);
        } catch (IOException e) {
            arena.close();
            throw new UncheckedIOException(e);
        }
    }

    /** The mapped rows: {@code size() x dimension()} float32, row-major. */
    MemorySegment segment() {
        return segment;
    }

    @Override
    public int size() {
        return rows;
    }

    @Override
    public int dimension() {
        return dim;
    }

    @Override
    public VectorFloat<?> getVector(int node) {
        float[] row = new float[dim];
        MemorySegment.copy(segment, ValueLayout.JAVA_FLOAT_UNALIGNED, (long) node * dim * Float.BYTES, row, 0, dim);
        return VTS.createFloatVector(row);
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public RandomAccessVectorValues copy() {
        return this;
    }

    @Override
    public void close() {
        arena.close();
    }
}
