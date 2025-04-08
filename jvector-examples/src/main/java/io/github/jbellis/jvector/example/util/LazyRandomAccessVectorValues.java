package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.function.Supplier;

/**
 * A lazy-loading implementation of RandomAccessVectorValues that reads vectors
 * from a binary file on demand. Assumes that vectors are stored back-to-back as floats.
 */
public class LazyRandomAccessVectorValues implements RandomAccessVectorValues {

    private final Path filePath;
    private final int dimension;
    private final int size;
    private final FileChannel channel;
    private final VectorTypeSupport vectorTypeSupport; // Our factory support

    public LazyRandomAccessVectorValues(Path filePath, int dimension, int size, VectorTypeSupport vectorTypeSupport) throws IOException {
        this.filePath = filePath;
        this.dimension = dimension;
        this.size = size;
        this.vectorTypeSupport = vectorTypeSupport;
        // Open the file channel for reading.
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
     * Reads the vector at the specified index from disk.
     * Assumes that each vector is stored as contiguous floats (without an extra header per vector).
     */
    @Override
    public VectorFloat<?> getVector(int index) {
        try {
            int vectorByteSize = dimension * Float.BYTES;
            long offset = (long) index * vectorByteSize;
            ByteBuffer buffer = ByteBuffer.allocate(vectorByteSize);
            int bytesRead = channel.read(buffer, offset);
            if (bytesRead != vectorByteSize) {
                throw new IOException("Could not read complete vector at index " + index);
            }
            buffer.flip();
            float[] vectorData = new float[dimension];
            buffer.asFloatBuffer().get(vectorData);
            // Delegate vector creation to vectorTypeSupport
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

    /**
     * Closes the underlying file channel to free system resources.
     */
    public void close() {
        try {
            channel.close();
        } catch (IOException e) {
            // Optionally log or handle this exception.
        }
    }
}

