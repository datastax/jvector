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
 * VectorFloat implementation backed by an on-heap float array.
 */
final public class ArrayVectorFloat implements VectorFloat<float[]>
{
    private final float[] data;

    ArrayVectorFloat(int length)
    {
        this.data = new float[length];
    }

    ArrayVectorFloat(float[] data)
    {
        this.data = data;
    }

    @Override
    public float[] get()
    {
        return data;
    }

    @Override
    public float get(int n) {
        return data[n];
    }

    @Override
    public void set(int n, float value) {
        data[n] = value;
    }

    @Override
    public void zero() {
        Arrays.fill(data, 0);
    }

    @Override
    public int length()
    {
        return data.length;
    }

    @Override
    public void writeTo(IndexWriter writer) throws IOException {
        writer.writeFloats(this.get(), 0, data.length);
    }

    @Override
    public VectorFloat<float[]> copy()
    {
        return new ArrayVectorFloat(Arrays.copyOf(data, data.length));
    }

    @Override
    public void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int length)
    {
        ArrayVectorFloat csrc = (ArrayVectorFloat) src;
        System.arraycopy(csrc.data, srcOffset, data, destOffset, length);
    }

    @Override
    public long ramBytesUsed()
    {
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        return OH_BYTES + RamUsageEstimator.sizeOf(data);
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
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ArrayVectorFloat that = (ArrayVectorFloat) o;
        return Arrays.equals(data, that.data);
    }

    @Override
    public int hashCode()
    {
        return this.getHashCode();
    }
}

