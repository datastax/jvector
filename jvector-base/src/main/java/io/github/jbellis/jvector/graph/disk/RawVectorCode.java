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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.FloatArray;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ForkJoinPool;

/**
 * The scratch "code" of a full-precision merge: the vector itself, as little-endian floats. With it the
 * pre-encode pass turns the scratch region into a vector store in merged-ordinal (cell) order, which the
 * cell join scans and reads exactly.
 */
final class RawVectorCode implements VectorCompressor<ByteSequence<?>> {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final int dimension;

    RawVectorCode(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public void encodeTo(VectorFloat<?> v, ByteSequence<?> dest) {
        Object raw = dest.get();
        if (raw instanceof byte[] && v instanceof FloatArray && dest.offset() == 0) {
            ByteBuffer.wrap((byte[]) raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(((FloatArray) v).array(), 0, dimension);
            return;
        }
        for (int i = 0; i < dimension; i++) {
            int bits = Float.floatToRawIntBits(v.get(i));
            dest.set(4 * i, (byte) bits);
            dest.set(4 * i + 1, (byte) (bits >>> 8));
            dest.set(4 * i + 2, (byte) (bits >>> 16));
            dest.set(4 * i + 3, (byte) (bits >>> 24));
        }
    }

    /** Reads the vector stored by {@link #encodeTo} back into {@code dst}. */
    static void decodeInto(byte[] code, VectorFloat<?> dst, int dimension) {
        if (dst instanceof FloatArray) {
            ByteBuffer.wrap(code).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(((FloatArray) dst).array(), 0, dimension);
            return;
        }
        ByteBuffer b = ByteBuffer.wrap(code).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < dimension; i++) {
            dst.set(i, b.getFloat(4 * i));
        }
    }

    @Override
    public int compressedVectorSize() {
        return 4 * dimension;
    }

    @Override
    public ByteSequence<?> encode(VectorFloat<?> v) {
        ByteSequence<?> out = vts.createByteSequence(compressedVectorSize());
        encodeTo(v, out);
        return out;
    }

    @Override
    public CompressedVectors encodeAll(RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        throw new UnsupportedOperationException("scratch-only");
    }

    @Override
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        throw new UnsupportedOperationException("scratch-only");
    }

    @Override
    public int compressorSize() {
        return 0;
    }

    @Override
    public void write(IndexWriter out, int version) {
        throw new UnsupportedOperationException("scratch-only");
    }

    @Override
    public double reconstructionError(VectorFloat<?> vector) {
        return 0;
    }
}
