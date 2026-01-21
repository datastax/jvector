package io.github.jbellis.jvector.disk;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Provides a RAVV backed by a fvec file.
 * Especially useful for handling larger-than-memory datasets.
 */
@Experimental
public class FvecSegmentRavv implements RandomAccessVectorValues, AutoCloseable {

    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int elemSize = 4;  // size of int and float

    private final long numVecs;
    private final int dim;
    private final FileChannel channel;
    private final Arena arena;
    private final MemorySegment segment;

    private final int rowSize;

    public FvecSegmentRavv(Path path) throws IOException {
        channel = FileChannel.open(path, StandardOpenOption.READ);
        arena = Arena.ofShared();
        segment = channel.map(MapMode.READ_ONLY, 0, channel.size(), arena);

        dim = (int) segment.get(ValueLayout.JAVA_INT.withOrder(ByteOrder.LITTLE_ENDIAN), 0);
        rowSize = (dim + 1) * elemSize;
        assert segment.byteSize() % rowSize == 0;
        numVecs = segment.byteSize() / rowSize;
    }

    /**
     * @throws ArithmeticException if the vector count is too large for an int
     */
    @Override
    public int size() {
        return Math.toIntExact(numVecs);
    }

    /**
     * @return the number of vectors in this RAVV
     */
    public long sizeLong() {
        return numVecs;
    }

    @Override
    public int dimension() {
        return dim;
    }

    @Override
    public VectorFloat<?> getVector(int nodeId) {
        var arr = new float[dim];
        MemorySegment.copy(
            segment, ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN), rowSize * (long) nodeId + elemSize,
            arr, 0,
            dim);
        return vts.createFloatVector(arr);
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
	public void close() throws IOException {
        arena.close();
        channel.close();
	}
}
