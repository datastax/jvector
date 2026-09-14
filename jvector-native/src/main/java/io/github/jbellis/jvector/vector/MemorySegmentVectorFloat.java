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
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.Buffer;
import java.util.Objects;

/**
 * VectorFloat backed by a float array with a lazily created on-heap MemorySegment view.
 */
final public class MemorySegmentVectorFloat implements VectorFloat<MemorySegment>, FloatArray
{
    private static final VarHandle STORAGE;
    static {
        try {
            STORAGE = MethodHandles.lookup().findVarHandle(MemorySegmentVectorFloat.class, "segment", Object.class);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    // Reuse the original single reference slot: an array until native access,
    // then its MemorySegment view. The view keeps the same array alive.
    private volatile Object segment;

    MemorySegmentVectorFloat(int length) {
        this(new float[length]);
    }

    MemorySegmentVectorFloat(Buffer buffer) {
        this(buffer.remaining());
        get().copyFrom(MemorySegment.ofBuffer(buffer));
    }

    MemorySegmentVectorFloat(float[] data) {
        this.segment = java.util.Objects.requireNonNull(data);
    }

    @Override
    public long ramBytesUsed()
    {
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        return OH_BYTES + REF_BYTES + (long) length() * Float.BYTES;
    }

    @Override
    public MemorySegment get()
    {
        Object current = segment;
        if (current instanceof MemorySegment view) {
            return view;
        }
        synchronized (this) {
            current = segment;
            if (current instanceof MemorySegment view) {
                return view;
            }
            MemorySegment view = MemorySegment.ofArray((float[]) current);
            segment = view;
            return view;
        }
    }

    // Plain reads may keep the original array after native publication: both states
    // address the same elements. Heap-segment bounds/base/scope metadata is final.
    // Retain volatile publication in get() and its single shared-view identity.
    @Override
    public float get(int n)
    {
        Object current = STORAGE.get(this);
        if (current instanceof float[] data) {
            return data[n];
        }
        MemorySegment view = (MemorySegment) current;
        if (n < 0 || n >= view.byteSize() / Float.BYTES) {
            throw new ArrayIndexOutOfBoundsException(n);
        }
        // heapBase() returns an Optional that can escape in scalar callers.
        return view.getAtIndex(ValueLayout.JAVA_FLOAT, n);
    }

    @Override
    public void set(int n, float value)
    {
        Object current = STORAGE.get(this);
        if (current instanceof float[] data) {
            data[n] = value;
        } else {
            MemorySegment view = (MemorySegment) current;
            if (n < 0 || n >= view.byteSize() / Float.BYTES) {
                throw new ArrayIndexOutOfBoundsException(n);
            }
            view.setAtIndex(ValueLayout.JAVA_FLOAT, n, value);
        }
    }

    @Override
    public void zero() {
        Object current = STORAGE.get(this);
        if (current instanceof float[] data) {
            java.util.Arrays.fill(data, 0.0f);
        } else {
            ((MemorySegment) current).fill((byte) 0);
        }
    }

    @Override
    public int length() {
        Object current = STORAGE.get(this);
        return current instanceof float[] data
                ? data.length : (int) (((MemorySegment) current).byteSize() / Float.BYTES);
    }

    @Override
    public int offset(int i)
    {
        return i * Float.BYTES;
    }

    @Override
    public void writeTo(IndexWriter writer) throws IOException {
        for (int i = 0; i < length(); i++) {
            writer.writeFloat(get(i));
        }
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
            System.arraycopy(csrc.array(), srcOffset, array(), destOffset, length);
        } else {
            Objects.checkFromIndexSize(srcOffset, length, src.length());
            Objects.checkFromIndexSize(destOffset, length, length());
            for (int i = 0; i < length; i++) {
                set(destOffset + i, src.get(srcOffset + i));
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
        return get().mismatch(that.get()) == -1;
    }

    @Override
    public int hashCode() {
        return this.getHashCode();
    }

    @Override
    public float[] array() {
        Object current = segment;
        return current instanceof float[] data ? data : (float[]) ((MemorySegment) current).heapBase().get();
    }
}
