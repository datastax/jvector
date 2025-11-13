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
import io.github.jbellis.jvector.vector.types.ByteSequence;
import java.util.Arrays;

/**
 * A {@link ByteSequence} implementation that represents a slice of another {@link ByteSequence}.
 */
public class ArraySliceByteSequence implements ByteSequence<byte[]> {
    private final ByteSequence<byte[]> data;
    private final int offset;
    private final int length;

    public ArraySliceByteSequence(ByteSequence<byte[]> data, int offset, int length) {
        if (offset < 0 || length < 0 || offset + length > data.length()) {
            throw new IllegalArgumentException("Invalid offset or length");
        }
        this.data = data;
        this.offset = offset;
        this.length = length;
    }

    @Override
    public byte[] get() {
        return data.get();
    }

    @Override
    public int offset() {
        return offset;
    }

    @Override
    public byte get(int n) {
        return data.get(offset + n);
    }

    @Override
    public void set(int n, byte value) {
        data.set(offset + n, value);
    }

    @Override
    public void zero() {
        for (int i = 0; i < length; i++) {
            data.set(offset + i, (byte) 0);
        }
    }

    @Override
    public int length() {
        return length;
    }

    @Override
    public ByteSequence<byte[]> copy() {
        byte[] newData = Arrays.copyOfRange(data.get(), offset, offset + length);
        return new ArrayByteSequence(newData);
    }

    @Override
    public ByteSequence<byte[]> slice(int sliceOffset, int sliceLength) {
        if (sliceOffset < 0 || sliceLength < 0 || sliceOffset + sliceLength > length) {
            throw new IllegalArgumentException("Invalid slice parameters");
        }
        if (sliceOffset == 0 && sliceLength == length) {
            return this;
        }
        return new ArraySliceByteSequence(data, offset + sliceOffset, sliceLength);
    }

    @Override
    public long ramBytesUsed() {
        // Only count the overhead of this slice object, not the underlying array
        // since that's shared and counted elsewhere
        return RamUsageEstimator.NUM_BYTES_OBJECT_HEADER +
                data.ramBytesUsed() +
                (2 * Integer.BYTES); // offset, length
    }

    @Override
    public void copyFrom(ByteSequence<?> src, int srcOffset, int destOffset, int copyLength) {
        if (src instanceof ArraySliceByteSequence) {
            ArraySliceByteSequence srcSlice = (ArraySliceByteSequence) src;
            data.copyFrom(srcSlice.data, srcSlice.offset + srcOffset, offset + destOffset, copyLength);
        } else {
            for (int i = 0; i < copyLength; i++) {
                data.set(offset + destOffset + i, src.get(srcOffset + i));
            }
        }
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
    public boolean equals(Object o) {
        return this.equalTo(o);
    }

    @Override
    public int hashCode() {
        return this.getHashCode();
    }
}
