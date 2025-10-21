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
 * ByteSequence implementation backed by an on-heap byte array.
 */
final public class ArrayByteSequence implements ByteSequence<byte[]>
{
    private final byte[] data;

    ArrayByteSequence(int length) {
        this.data = new byte[length];
    }

    ArrayByteSequence(byte[] data) {
        this.data = data;
    }

    @Override
    public byte[] get() {
        return data;
    }

    @Override
    public int offset() {
        return 0;
    }

    @Override
    public byte get(int n) {
        return data[n];
    }

    @Override
    public void set(int n, byte value) {
        data[n] = value;
    }

    @Override
    public void zero() {
        Arrays.fill(data, (byte) 0);
    }

    @Override
    public int length() {
        return data.length;
    }

    @Override
    public ArrayByteSequence copy() {
        return new ArrayByteSequence(Arrays.copyOf(data, data.length));
    }

    @Override
    public ByteSequence<byte[]> slice(int offset, int length) {
        if (offset == 0 && length == data.length) {
            return this;
        }
        return new ArraySliceByteSequence(this, offset, length);
    }

    @Override
    public long ramBytesUsed() {
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        return OH_BYTES + RamUsageEstimator.sizeOf(data);
    }

    @Override
    public void copyFrom(ByteSequence<?> src, int srcOffset, int destOffset, int length) {
        ArrayByteSequence csrc = (ArrayByteSequence) src;
        System.arraycopy(csrc.data, srcOffset, data, destOffset, length);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < Math.min(data.length, 25); i++) {
            sb.append(data[i]);
            if (i < data.length - 1) {
                sb.append(", ");
            }
        }
        if (data.length > 25) {
            sb.append("...");
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public boolean equals(Object o)
    {
        return this.equalTo(o);
    }

    @Override
    public int hashCode()
    {
        return this.getHashCode();
    }
}
