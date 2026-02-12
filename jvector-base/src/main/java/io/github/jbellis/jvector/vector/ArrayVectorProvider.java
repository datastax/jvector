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

package io.github.jbellis.jvector.vector;


import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/**
 * VectorTypeSupport implementation that uses on-heap arrays for ByteSequence/VectorFloat.
 */
final class ArrayVectorProvider implements VectorTypeSupport
{

    @Override
    public VectorFloat<?> createFloatVector(Object data)
    {
        return new ArrayVectorFloat((float[]) data);
    }

    @Override
    public VectorFloat<?> createFloatVector(int length)
    {
        return new ArrayVectorFloat(length);
    }

    @Override
    public VectorFloat<?> readFloatVector(RandomAccessReader r, int size) throws IOException
    {
        float[] vector = new float[size];
        r.readFully(vector);
        return new ArrayVectorFloat(vector);
    }

    @Override
    public void readFloatVector(RandomAccessReader r, int size, VectorFloat<?> vector, int offset) throws IOException {
        r.read(((ArrayVectorFloat) vector).get(), offset, size);
    }

    @Override
    public void writeFloatVector(IndexWriter out, VectorFloat<?> vector) throws IOException
    {
        ArrayVectorFloat v = (ArrayVectorFloat)vector;
        out.writeFloats(v.get(), 0, v.length());
    }

    @Override
    public ByteSequence<?> createByteSequence(Object data)
    {
        return new ArrayByteSequence((byte[]) data);
    }

    @Override
    public ByteSequence<?> createByteSequence(int length)
    {
        return new ArrayByteSequence(length);
    }

    @Override
    public ByteSequence<?> readByteSequence(RandomAccessReader r, int size) throws IOException
    {
        byte[] vector = new byte[size];
        r.readFully(vector);
        return new ArrayByteSequence(vector);
    }

    @Override
    public void readByteSequence(RandomAccessReader r, ByteSequence<?> sequence) throws IOException {
        ArrayByteSequence v = (ArrayByteSequence) sequence;
        r.readFully(v.get());
    }

    @Override
    public void writeByteSequence(IndexWriter out, ByteSequence<?> sequence) throws IOException
    {
        ArrayByteSequence v = (ArrayByteSequence) sequence;
        out.write(v.get());
    }
}
