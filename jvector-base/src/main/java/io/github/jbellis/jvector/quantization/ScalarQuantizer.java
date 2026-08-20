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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.ListRandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Per-dimension scalar quantizer: maps float32 vectors to signed int8 using
 * per-dimension min/max derived from a base vector set.
 *
 * <p>Implements {@link VectorCompressor} so the fitted parameters can be serialized into
 * the index header and reloaded when the index is reopened from disk.
 *
 * <p>Usage:
 * <pre>
 *   ScalarQuantizer sq = ScalarQuantizer.fit(baseRavv);
 *   SQVectors sqv       = (SQVectors) sq.encodeAll(baseRavv);
 *   ByteSequence&lt;?&gt; qb  = sq.encode(queryVector);
 *   VectorFloat&lt;?&gt;  rec = sq.dequantize(byteVec);
 * </pre>
 */
public class ScalarQuantizer implements VectorCompressor<ByteSequence<?>> {
    private static final VectorTypeSupport VTS =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    private final float[] dimMin;
    private final float[] dimMax;

    public ScalarQuantizer(float[] dimMin, float[] dimMax) {
        this.dimMin = dimMin;
        this.dimMax = dimMax;
    }

    // ── factory ──────────────────────────────────────────────────────────────

    /**
     * Scans all base vectors and computes per-dimension min and max.
     */
    public static ScalarQuantizer fit(RandomAccessVectorValues ravv) {
        int dim = ravv.dimension();
        float[] dimMin = new float[dim];
        float[] dimMax = new float[dim];
        for (int d = 0; d < dim; d++) {
            dimMin[d] = Float.MAX_VALUE;
            dimMax[d] = -Float.MAX_VALUE;
        }
        for (int i = 0; i < ravv.size(); i++) {
            VectorFloat<?> v = ravv.getVector(i);
            for (int d = 0; d < dim; d++) {
                float x = v.get(d);
                if (x < dimMin[d]) dimMin[d] = x;
                if (x > dimMax[d]) dimMax[d] = x;
            }
        }
        return new ScalarQuantizer(dimMin, dimMax);
    }

    // ── VectorCompressor ─────────────────────────────────────────────────────

    /**
     * Encodes a single float32 vector to a signed int8 {@link ByteSequence}.
     * Each component is mapped linearly from {@code [dimMin[d], dimMax[d]]} to {@code [-128, 127]}.
     */
    @Override
    public ByteSequence<?> encode(VectorFloat<?> v) {
        int dim = dimMin.length;
        ByteSequence<?> out = VTS.createByteSequence(dim);
        encodeTo(v, out);
        return out;
    }

    @Override
    public void encodeTo(VectorFloat<?> v, ByteSequence<?> dest) {
        int dim = dimMin.length;
        for (int d = 0; d < dim; d++) {
            float range = dimMax[d] - dimMin[d];
            float scaled = range == 0f ? 0f : (v.get(d) - dimMin[d]) / range * 255f - 128f;
            int rounded = Math.round(scaled);
            dest.set(d, (byte) Math.max(-128, Math.min(127, rounded)));
        }
    }

    /** Encodes all vectors and returns an {@link SQVectors} instance. */
    @Override
    public CompressedVectors encodeAll(RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        var ravvCopy = ravv.threadLocalSupplier();
        ByteSequence<?>[] encoded = simdExecutor.submit(() ->
                IntStream.range(0, ravv.size())
                         .parallel()
                         .mapToObj(i -> {
                             VectorFloat<?> v = ravvCopy.get().getVector(i);
                             return v == null ? VTS.createByteSequence(dimMin.length) : encode(v);
                         })
                         .toArray(ByteSequence[]::new)
        ).join();
        return new SQVectors(this, encoded);
    }

    @Override
    @Deprecated
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        ByteSequence<?>[] seqs = Arrays.copyOf(compressedVectors, compressedVectors.length, ByteSequence[].class);
        return new SQVectors(this, seqs);
    }

    /** Serialized size of the compressor parameters (dimMin + dimMax arrays). */
    @Override
    public int compressorSize() {
        // dimension count + two float arrays of that length
        return Integer.BYTES + 2 * Float.BYTES * dimMin.length;
    }

    /** Each encoded vector is {@code dim} bytes. */
    @Override
    public int compressedVectorSize() {
        return dimMin.length;
    }

    @Override
    public void write(IndexWriter out, int version) throws IOException {
        out.writeInt(dimMin.length);
        for (float v : dimMin) out.writeFloat(v);
        for (float v : dimMax) out.writeFloat(v);
    }

    /** Deserializes a {@code ScalarQuantizer} written by {@link #write}. */
    public static ScalarQuantizer load(RandomAccessReader in) throws IOException {
        int dim = in.readInt();
        float[] dimMin = new float[dim];
        float[] dimMax = new float[dim];
        for (int d = 0; d < dim; d++) dimMin[d] = in.readFloat();
        for (int d = 0; d < dim; d++) dimMax[d] = in.readFloat();
        return new ScalarQuantizer(dimMin, dimMax);
    }

    @Override
    public double reconstructionError(VectorFloat<?> vector) {
        int dim = dimMin.length;
        ByteSequence<?> encoded = encode(vector);
        VectorFloat<?> reconstructed = dequantize(encoded);
        double sum = 0;
        for (int d = 0; d < dim; d++) {
            double diff = vector.get(d) - reconstructed.get(d);
            sum += diff * diff;
        }
        return sum / dim;
    }

    // ── SQ-specific helpers ──────────────────────────────────────────────────

    /**
     * Convenience batch-quantize returning a {@link ListRandomAccessByteVectorValues}.
     * Useful for constructing the in-memory byte RAVV passed to {@link io.github.jbellis.jvector.graph.GraphIndexBuilder}.
     */
    public ListRandomAccessByteVectorValues quantizeAll(RandomAccessVectorValues ravv) {
        List<ByteSequence<?>> result = new ArrayList<>(ravv.size());
        for (int i = 0; i < ravv.size(); i++) {
            result.add(encode(ravv.getVector(i)));
        }
        return new ListRandomAccessByteVectorValues(result, ravv.dimension());
    }

    /**
     * Reconstructs a float32 vector from a signed int8 {@link ByteSequence}.
     * Inverse of {@link #encode}: byte -128 → dimMin, byte 127 → dimMax.
     */
    public VectorFloat<?> dequantize(ByteSequence<?> b) {
        int dim = dimMin.length;
        VectorFloat<?> out = VTS.createFloatVector(dim);
        for (int d = 0; d < dim; d++) {
            float range = dimMax[d] - dimMin[d];
            out.set(d, ((b.get(d) + 128) / 255f) * range + dimMin[d]);
        }
        return out;
    }

    public float[] getDimMin() { return dimMin; }
    public float[] getDimMax() { return dimMax; }

    @Override
    public String toString() {
        return "SQ(per_dim)";
    }
}
