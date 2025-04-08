package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.function.Supplier;

/**
 * Lazy loader for fvecs files.
 * Each record is stored as:
 *   [int dimension][float vector[dimension]]
 * This class reads a single vector record on demand.
 */
public class LazyFvecsRandomAccessVectorValues implements RandomAccessVectorValues {

    private final Path filePath;
    private final int dimension;
    private final int size;         // number of vectors in the file
    private final int recordSize;   // total bytes per record (4 + dimension * 4)
    private final FileChannel channel;
    private final VectorTypeSupport vectorTypeSupport;

    public LazyFvecsRandomAccessVectorValues(Path filePath, int dimension, int size, VectorTypeSupport vectorTypeSupport) throws IOException {
        this.filePath = filePath;
        this.dimension = dimension;
        this.size = size;
        this.vectorTypeSupport = vectorTypeSupport;
        this.recordSize = 4 + dimension * Float.BYTES; // header (4 bytes) + data
        this.channel = FileChannel.open(filePath, StandardOpenOption.READ);
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    /**
     * Reads the vector at the specified index.
     * It reads the 4-byte header (to verify the dimension) and then the vector data.
     */
    @Override
    public VectorFloat<?> getVector(int index) {
        try {
            long recordOffset = (long) index * recordSize;
            // Read the header (4 bytes)
            ByteBuffer headerBuffer = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            int bytesHeader = channel.read(headerBuffer, recordOffset);
            if (bytesHeader != 4) {
                throw new IOException("Unable to read header for vector at index " + index);
            }
            headerBuffer.flip();
            int vectorDim = headerBuffer.getInt();
            if (vectorDim != dimension) {
                throw new IOException("Dimension mismatch at index " + index + ": expected " + dimension + ", got " + vectorDim);
            }
            // Read vector data
            ByteBuffer dataBuffer = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            int bytesData = channel.read(dataBuffer, recordOffset + 4);
            if (bytesData != dimension * Float.BYTES) {
                throw new IOException("Could not read full vector data at index " + index);
            }
            dataBuffer.flip();
            float[] vectorData = new float[dimension];
            dataBuffer.asFloatBuffer().get(vectorData);
            return vectorTypeSupport.createFloatVector(vectorData);
        } catch (IOException e) {
            throw new RuntimeException("Error reading vector at index " + index, e);
        }
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
    public Supplier<RandomAccessVectorValues> threadLocalSupplier() {
        return () -> this;
    }

    public void close() {
        try {
            channel.close();
        } catch (IOException e) {
            // Optionally log the exception.
        }
    }
}

