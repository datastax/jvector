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

import com.indeed.util.mmap.MMapBuffer;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.Closeable;
import java.io.File;
import java.io.IOError;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

/**
 * Memory-mapped implementation of RandomAccessVectorValues that provides efficient
 * random access to vectors stored in a file. Uses memory mapping for fast I/O.
 */
public class MMapRandomAccessVectorValues implements RandomAccessVectorValues, Closeable {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    /** The dimension of each vector. */
    final int dimension;
    /** The number of vectors in the file. */
    final int rows;
    /** The file containing the vectors. */
    final File file;
    /** Reusable buffer for reading vector values. */
    final float[] valueBuffer;

    /** The memory-mapped file reader. */
    final MMapBuffer fileReader;

    /**
     * Constructs a MMapRandomAccessVectorValues for the specified file.
     *
     * @param f the file containing vectors, which must exist and be readable
     * @param dimension the dimension of each vector in the file
     * @throws IOError if an I/O error occurs during initialization
     */
    public MMapRandomAccessVectorValues(File f, int dimension) {
        assert f != null && f.exists() && f.canRead();
        assert f.length() % ((long) dimension * Float.BYTES) == 0;

        try {
            this.file = f;
            this.fileReader = new MMapBuffer(f, FileChannel.MapMode.READ_ONLY, ByteOrder.LITTLE_ENDIAN);
            this.dimension = dimension;
            this.rows = ((int) f.length()) / dimension;
            this.valueBuffer = new float[dimension];
        } catch (IOException e) {
            throw new IOError(e);
        }
    }

    @Override
    public int size() {
        return (int) (file.length() / ((long) dimension * Float.BYTES));
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public VectorFloat<?> getVector(int targetOrd) {
        long offset = (long) targetOrd * dimension * Float.BYTES;
        int i = 0;
        for (long o = offset; o < offset + ((long) dimension * Float.BYTES); o += Float.BYTES, i++)
            valueBuffer[i] = fileReader.memory().getFloat(o);

        return vectorTypeSupport.createFloatVector(valueBuffer);
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public RandomAccessVectorValues copy() {
        return new MMapRandomAccessVectorValues(file, dimension);
    }

    @Override
    public void close() {
        try {
            this.fileReader.close();
        } catch (IOException e) {
            throw new IOError(e);
        }
    }
}
