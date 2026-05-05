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
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.util.Arrays;

/**
 * A {@link VectorFloat} implementation that represents a slice of another {@link VectorFloat}.
 */
public class ArraySliceVectorFloat implements VectorFloat<float[]> {
    private final VectorFloat<float[]> data;
    private final int offset;
    private final int length;

    // TODO perhaps this class should accept a float[] instead of a VectorFloat<float[]>?
    public ArraySliceVectorFloat(VectorFloat<float[]> data, int offset, int length) {
        if (offset < 0 || length < 0 || offset + length > data.length()) {
            throw new IllegalArgumentException("Invalid offset or length");
        }
        this.data = data;
        this.offset = offset;
        this.length = length;
    }

    @Override
    public float[] get() {
        return data.get();
    }

    @Override
    public int length() {
        return length;
    }

    @Override
    public int offset(int i) {
        return offset + i;
    }

    @Override
    public void writeTo(IndexWriter writer) throws IOException {
        writer.writeFloats(data.get(), offset, length);
    }

    @Override
    public VectorFloat<float[]> copy() {
        float[] newData = Arrays.copyOfRange(data.get(), offset, offset + length);
        return new ArrayVectorFloat(newData);
    }

    @Override
    public VectorFloat<float[]> slice(int sliceOffset, int sliceLength) {
        if (sliceOffset < 0 || sliceLength < 0 || sliceOffset + sliceLength > length) {
            throw new IllegalArgumentException("Invalid slice parameters");
        }
        if (sliceOffset == 0 && sliceLength == length) {
            return this;
        }
        return new ArraySliceVectorFloat(data, offset + sliceOffset, sliceLength);
    }

    @Override
    public void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int copyLength) {
        if (src instanceof ArraySliceVectorFloat) {
            ArraySliceVectorFloat srcSlice = (ArraySliceVectorFloat) src;
            data.copyFrom(srcSlice.data, srcSlice.offset + srcOffset, offset + destOffset, copyLength);
        } else {
            for (int i = 0; i < copyLength; i++) {
                data.set(offset + destOffset + i, src.get(srcOffset + i));
            }
        }
    }

    @Override
    public float get(int i) {
        return data.get(offset + i);
    }

    @Override
    public void set(int i, float value) {
        data.set(offset + i, value);
    }

    @Override
    public void zero() {
        for (int i = 0; i < length; i++) {
            data.set(offset + i, 0);
        }
    }

    @Override
    public long ramBytesUsed() {
        return RamUsageEstimator.NUM_BYTES_OBJECT_HEADER +
               data.ramBytesUsed() +
               (2 * Integer.BYTES);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < Math.min(length, 25); i++) {
            sb.append(get(i));
            if (i < length - 1) {
                sb.append(", ");
            }
        }
        if (length > 25) {
            sb.append("...");
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (!(o instanceof VectorFloat)) return false;
        VectorFloat<?> that = (VectorFloat<?>) o;
        if (length() != that.length()) return false;
        for (int i = 0; i < length(); i++) {
            if (Float.compare(get(i), that.get(i)) != 0) return false;
        }
        return true;
    }

    @Override
    public int hashCode()
    {
        return this.getHashCode();
    }
}
