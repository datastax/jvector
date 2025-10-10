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

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.nio.Buffer;

/**
 * Implementation of {@link VectorTypeSupport} that uses off-heap memory segments for vector storage.
 * <p>
 * This provider leverages Java's Foreign Function &amp; Memory API (introduced in Java 19 and
 * finalized in Java 22) to store vector data in native memory segments rather than on-heap arrays.
 * This approach provides several advantages for high-performance vector operations:
 * <ul>
 * <li><strong>Reduced GC pressure:</strong> Vector data is stored off-heap, minimizing
 *     garbage collection overhead for large vector datasets</li>
 * <li><strong>Direct memory access:</strong> Enables efficient interoperability with native
 *     SIMD implementations through {@link java.lang.foreign.MemorySegment} pointers</li>
 * <li><strong>Memory-mapped I/O:</strong> Supports zero-copy access to memory-mapped vector
 *     files via {@link io.github.jbellis.jvector.disk.MemorySegmentReader}</li>
 * <li><strong>Deterministic cleanup:</strong> Memory segments are explicitly managed through
 *     the arena API, providing predictable memory lifecycle</li>
 * </ul>
 * This provider is typically used in conjunction with {@link NativeVectorizationProvider} to
 * enable native SIMD acceleration for vector similarity computations. It can handle both
 * float vectors ({@link VectorFloat}) and byte sequences ({@link ByteSequence}) backed by
 * native memory.
 * <p>
 * <strong>Thread safety:</strong> This provider is thread-safe for creating new vectors.
 * However, individual vector instances are not thread-safe unless they are backed by
 * memory segments from a shared arena.
 *
 * @see MemorySegmentVectorFloat
 * @see MemorySegmentByteSequence
 * @see VectorTypeSupport
 */
public class MemorySegmentVectorProvider implements VectorTypeSupport
{
    /**
     * Constructs a new MemorySegmentVectorProvider with default settings.
     * <p>
     * This provider creates vectors backed by off-heap memory segments managed by
     * the Foreign Function &amp; Memory API. The provider itself is stateless and
     * lightweight - memory management is handled at the individual vector level
     * through their associated arenas.
     */
    public MemorySegmentVectorProvider() {
    }
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
        float[] dest = (float[]) ((MemorySegmentVectorFloat) vector).get().heapBase().get();
        r.read(dest, offset, count);
    }

    @Override
    public void writeFloatVector(DataOutput out, VectorFloat<?> vector) throws IOException
    {
        for (int i = 0; i < vector.length(); i++)
            out.writeFloat(vector.get(i));
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
    public void writeByteSequence(DataOutput out, ByteSequence<?> sequence) throws IOException
    {
        for (int i = 0; i < sequence.length(); i++)
            out.writeByte(sequence.get(i));
    }
}
