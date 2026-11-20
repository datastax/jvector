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
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.types.FloatArray;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.Buffer;
import java.util.Objects;

/**
 * VectorFloat backed by a float array with a lazily created on-heap MemorySegment view.
 */
final public class MemorySegmentVectorFloat implements VectorFloat<MemorySegment>, FloatArray
{
    private final float[] data;
    // Plain (non-volatile) cache: racing threads may each create a view, but heap segments have
    // only final fields, so any published reference is fully constructed and addresses the same
    // array. Avoids a store fence on every ephemeral vector's first native call.
    private MemorySegment segment;

    MemorySegmentVectorFloat(int length) {
        this(new float[length]);
    }

    MemorySegmentVectorFloat(Buffer buffer) {
        this(buffer.remaining());
        get().copyFrom(MemorySegment.ofBuffer(buffer));
    }

    MemorySegmentVectorFloat(float[] data) {
        this.data = java.util.Objects.requireNonNull(data);
    }

    @Override
    public long ramBytesUsed()
    {
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        return OH_BYTES + 2L * REF_BYTES + (long) length() * Float.BYTES;
    }

    @Override
    public MemorySegment get()
    {
        MemorySegment view = segment;
        if (view == null) {
            segment = view = MemorySegment.ofArray(data);
        }
        return view;
    }

    @Override
    public float get(int n)
    {
        return data[n];
    }

    @Override
    public void set(int n, float value)
    {
        data[n] = value;
    }

    @Override
    public void zero() {
        java.util.Arrays.fill(data, 0.0f);
    }

    @Override
    public int length() {
        return data.length;
    }

    @Override
    public int offset(int i)
    {
        return i * Float.BYTES;
    }

    @Override
    public void writeTo(IndexWriter writer) throws IOException {
        writer.writeFloats(data, 0, data.length);
    }

    @Override
    public VectorFloat<MemorySegment> copy()
    {
        MemorySegmentVectorFloat copy = new MemorySegmentVectorFloat(length());
        copy.copyFrom(this, 0, 0, length());
        return copy;
    }

    @Override
    public void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int length)
    {
        if (src instanceof FloatArray csrc) {
            System.arraycopy(csrc.array(), srcOffset, data, destOffset, length);
        } else {
            Objects.checkFromIndexSize(srcOffset, length, src.length());
            Objects.checkFromIndexSize(destOffset, length, data.length);
            for (int i = 0; i < length; i++) {
                data[destOffset + i] = src.get(srcOffset + i);
            }
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < Math.min(length(), 25); i++) {
            sb.append(get(i));
            if (i < length() - 1) {
                sb.append(", ");
            }
        }
        if (length() > 25) {
            sb.append("...");
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MemorySegmentVectorFloat that = (MemorySegmentVectorFloat) o;
        return java.util.Arrays.equals(data, that.data);
    }

    @Override
    public int hashCode() {
        return this.getHashCode();
    }

    @Override
    public float[] array() {
        return data;
    }
}
