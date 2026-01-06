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

import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.lang.foreign.MemorySegment;
import java.nio.Buffer;

/**
 * VectorFloat implementation backed by an on-heap MemorySegment.
 */
final public class MemorySegmentVectorFloat implements VectorFloat<MemorySegment>
{
    private final MemorySegment segment;

    MemorySegmentVectorFloat(int length) {
        segment = MemorySegment.ofArray(new float[length]);
    }

    MemorySegmentVectorFloat(Buffer buffer) {
        this(buffer.remaining());
        segment.copyFrom(MemorySegment.ofBuffer(buffer));
    }

    MemorySegmentVectorFloat(float[] data) {
        this.segment = MemorySegment.ofArray(data);
    }

    @Override
    public long ramBytesUsed()
    {
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        return OH_BYTES + REF_BYTES + segment.byteSize();
    }

    @Override
    public MemorySegment get()
    {
        return segment;
    }

    @Override
    public float get(int n)
    {
        // this is (unfortunately) meaningfully better performing than getting at an offset in the memory segment
        return ((float[])segment.heapBase().get())[n];
    }

    @Override
    public void set(int n, float value)
    {
        // this is (unfortunately) meaningfully better performing than setting at an offset in the memory segment
        ((float[])segment.heapBase().get())[n] = value;
    }

    @Override
    public void zero() {
        segment.fill((byte) 0);
    }

    @Override
    public int length() {
        return (int) (segment.byteSize() / Float.BYTES);
    }

    @Override
    public int offset(int i)
    {
        return i * Float.BYTES;
    }

    @Override
    public boolean isExact() {
        return true;
    }

    @Override
    public VectorFloat<MemorySegment> copy()
    {
        MemorySegmentVectorFloat copy = new MemorySegmentVectorFloat(length());
        copy.copyFrom(this, length());
        return copy;
    }

    @Override
    public VectorFloat<?> decode() {
        return this;
    }

    private void copyFrom(VectorFloat<?> src, int length)
    {
        MemorySegmentVectorFloat csrc = (MemorySegmentVectorFloat) src;
        segment.asSlice(0, (long) length * Float.BYTES)
                .copyFrom(csrc.segment.asSlice(0, (long) length * Float.BYTES));
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
        return segment.mismatch(that.segment) == -1;
    }

    @Override
    public int hashCode() {
        return this.getHashCode();
    }
}
