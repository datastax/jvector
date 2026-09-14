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
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;
import java.io.*;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;
import static org.junit.jupiter.api.Assertions.*;

/** Compare direct subspace shifting with the previous complete encode path. */
class NvqEncodingTest {
    private static class Writer extends DataOutputStream implements IndexWriter {
        final ByteArrayOutputStream bytes;
        Writer() { this(new ByteArrayOutputStream()); }
        Writer(ByteArrayOutputStream bytes) { super(bytes); this.bytes = bytes; }
        public long position() { return bytes.size(); }
    }

    private static byte[] bytes(NVQuantization.QuantizedVector vector) throws IOException {
        var out = new Writer(); vector.write(out); return out.bytes.toByteArray();
    }

    private static NVQuantization.QuantizedVector reference(NVQuantization q, VectorFloat<?> vector) {
        var dest = NVQuantization.QuantizedVector.createEmpty(q.subvectorSizesAndOffsets, q.bitsPerDimension);
        var shifted = VectorUtil.nvqSub(vector, q.globalMean);
        NVQuantization.QuantizedVector.quantizeTo(q.getSubVectors(shifted), q.bitsPerDimension, q.learn, dest);
        return dest;
    }

    @Test void preservesEncodingForOffsetsTailsAndReusedDestinations() throws IOException {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        Random random = new Random(713);
        for (int dim : new int[]{1, 3, 8, 17, 33, 256, 768, 1536, 3072}) {
            for (int subspaces : new int[]{1, Math.min(2, dim), Math.min(7, dim)}) {
                float[] mean = new float[dim];
                for (int i = 0; i < dim; i++) mean[i] = (float)random.nextGaussian();
                var q = NVQuantization.create(types.createFloatVector(mean.clone()), subspaces);
                var dest = NVQuantization.QuantizedVector.createEmpty(q.subvectorSizesAndOffsets, q.bitsPerDimension);
                for (boolean learn : new boolean[]{false, true}) {
                    q.learn = learn;
                    for (int trial = 0; trial < 4; trial++) {
                        float[] data = new float[dim];
                        for (int i = 0; i < dim; i++) data[i] = trial == 0 ? mean[i] :
                            trial == 1 ? (i % 2 == 0 ? -0.0f : 0.0f) : (float)random.nextGaussian();
                        var vector = types.createFloatVector(data.clone());
                        byte[] expected = bytes(reference(q, vector));
                        assertArrayEquals(expected, bytes(q.encode(vector)), "dim=" + dim + " subspaces=" + subspaces);
                        q.encodeTo(vector, dest);
                        assertArrayEquals(expected, bytes(dest), "Reused destination");
                        for (int i = 0; i < dim; i++) {
                            assertEquals(Float.floatToRawIntBits(data[i]), Float.floatToRawIntBits(vector.get(i)));
                            assertEquals(Float.floatToRawIntBits(mean[i]), Float.floatToRawIntBits(q.globalMean.get(i)));
                        }
                    }
                }
            }
        }
    }

    @Test void rejectsMismatchedDimensionsWithoutModifyingDestination() throws IOException {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        var q = NVQuantization.create(types.createFloatVector(new float[]{1, 2, 3, 4, 5, 6, 7, 8}), 2);
        q.learn = false;
        var dest = q.encode(types.createFloatVector(new float[]{8, 7, 6, 5, 4, 3, 2, 1}));
        byte[] original = bytes(dest);
        for (int length : new int[]{0, 7, 9}) {
            var input = types.createFloatVector(new float[length]);
            assertThrows(IllegalArgumentException.class, () -> q.encodeTo(input, dest), "length=" + length);
            assertArrayEquals(original, bytes(dest), "Destination changed for length=" + length);
        }
    }

    @Test void allowsConcurrentCallsWithIndependentDestinations() throws Exception {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        float[] mean = new float[1536], data = new float[1536];
        Random random = new Random(48);
        for (int i = 0; i < data.length; i++) { mean[i] = random.nextFloat(); data[i] = random.nextFloat(); }
        var q = NVQuantization.create(types.createFloatVector(mean), 2);
        var vector = types.createFloatVector(data);
        byte[] expected = bytes(reference(q, vector));
        try (var pool = new ForkJoinPool(48)) {
            pool.submit(() -> IntStream.range(0, 480).parallel().forEach(i -> {
                try { assertArrayEquals(expected, bytes(q.encode(vector))); }
                catch (IOException e) { throw new UncheckedIOException(e); }
            })).join();
        }
    }
}
