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

package io.github.jbellis.jvector.disk;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * RandomAccessReader that reads from a ByteBuffer
 */
public class ByteBufferReader implements RandomAccessReader {
    protected final ByteBuffer bb;

    public ByteBufferReader(ByteBuffer sourceBB) {
        bb = sourceBB;
    }

    @Override
    public void seek(long offset) {
        bb.position(Math.toIntExact(offset));
    }

    @Override
    public long getPosition() {
        return bb.position();
    }

    @Override
    public void readFully(float[] buffer) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = bb.getFloat();
        }
    }

    @Override
    public void readFully(byte[] b) {
        bb.get(b);
    }

    @Override
    public void readFully(ByteBuffer buffer) {
        // slice mbb from current position to buffer.remaining()
        var slice = bb.slice();
        var remaining = buffer.remaining();
        slice.limit(remaining);
        buffer.put(slice);
        bb.position(bb.position() + remaining);
    }

    @Override
    public void readFully(long[] vector) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = bb.getLong();
        }
    }

    @Override
    public int readInt() {
        return bb.getInt();
    }

    @Override
    public long readLong() {
        return bb.getLong();
    }

    @Override
    public float readFloat() {
        return bb.getFloat();
    }

    @Override
    public void read(int[] ints, int offset, int count) {
        for (int i = 0; i < count; i++) {
            ints[offset + i] = bb.getInt();
        }
    }

    @Override
    public void read(float[] floats, int offset, int count) {
        for (int i = 0; i < count; i++) {
            floats[offset + i] = bb.getFloat();
        }
    }

    @Override
    public void close() {
    }
}
