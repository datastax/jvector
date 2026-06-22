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
import java.lang.foreign.MemorySegment;
import java.nio.Buffer;

/**
 * VectorTypeSupport using off-heap (native) MemorySegments for improved performance
 * and reduced GC pressure.
 */
public class MemorySegmentVectorProvider implements VectorTypeSupport
{
    @Override
    public VectorFloat<?> createFloatVector(Object data)
    {
        if (data instanceof Buffer)
            return new MemorySegmentVectorFloat((Buffer) data);

        return new MemorySegmentVectorFloat((float[]) data);
    }

    @Override
    public VectorFloat<?> createFloatVector(int length)
    {
        return new MemorySegmentVectorFloat(length);
    }

    @Override
    public VectorFloat<?> readFloatVector(RandomAccessReader r, int size) throws IOException
    {
        float[] data = new float[size];
        r.readFully(data);
        return new MemorySegmentVectorFloat(data);
    }

    @Override
    public void readFloatVector(RandomAccessReader r, int count, VectorFloat<?> vector, int offset) throws IOException {
        // Read into temporary array then copy to native segment
        float[] temp = new float[count];
        r.read(temp, 0, count);
        
        // Copy from temp array to native segment
        MemorySegment segment = ((MemorySegmentVectorFloat) vector).get();
        MemorySegment.copy(MemorySegment.ofArray(temp), 0,
                          segment, (long) offset * Float.BYTES,
                          (long) count * Float.BYTES);
    }

    @Override
    public void writeFloatVector(IndexWriter out, VectorFloat<?> vector) throws IOException
    {
        // Copy from native segment to temporary array then write
        int length = vector.length();
        float[] temp = new float[length];
        MemorySegment segment = ((MemorySegmentVectorFloat) vector).get();
        MemorySegment.copy(segment, 0, MemorySegment.ofArray(temp), 0, (long) length * Float.BYTES);
        out.writeFloats(temp, 0, length);
    }

    @Override
    public ByteSequence<?> createByteSequence(Object data)
    {
        if (data instanceof Buffer)
            return new MemorySegmentByteSequence((Buffer) data);

        return new MemorySegmentByteSequence((byte[]) data);
    }

    @Override
    public ByteSequence<?> createByteSequence(int length)
    {
        return new MemorySegmentByteSequence(length);
    }

    @Override
    public ByteSequence<?> readByteSequence(RandomAccessReader r, int size) throws IOException
    {
        var vector = new MemorySegmentByteSequence(size);
        r.readFully(vector.get().asByteBuffer());
        return vector;
    }

    @Override
    public void readByteSequence(RandomAccessReader r, ByteSequence<?> sequence) throws IOException {
        r.readFully(((MemorySegmentByteSequence) sequence).get().asByteBuffer());
    }


    @Override
    public void writeByteSequence(IndexWriter out, ByteSequence<?> sequence) throws IOException
    {
        // Copy from native segment to temporary array then write
        int length = sequence.length();
        byte[] temp = new byte[length];
        MemorySegment segment = ((MemorySegmentByteSequence) sequence).get();
        MemorySegment.copy(segment, 0, MemorySegment.ofArray(temp), 0, length);
        out.write(temp, 0, length);
    }
}
