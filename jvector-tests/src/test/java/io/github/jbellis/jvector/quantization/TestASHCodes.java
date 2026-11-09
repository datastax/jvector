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

import io.github.jbellis.jvector.disk.ByteBufferIndexWriter;
import io.github.jbellis.jvector.disk.ByteBufferReader;
import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Assume;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.*;

/**
 * The compact code store must hold exactly the codes it is given, serialize them in the
 * standalone vector format, and score them identically to standalone vectors, including
 * across page boundaries.
 */
public class TestASHCodes {
    private static final int[] BITS = {1, 2, 3, 4, 9};
    // Small pages so that a few hundred vectors span many of them.
    private static final int SMALL_PAGE_BYTES = 256;

    private static AsymmetricHashing.QuantizedVector randomVector(int dims, int bits, Random random) {
        var v = AsymmetricHashing.QuantizedVector.createEmpty(dims, bits);
        for (int w = 0; w < v.binaryVector.length; w++) v.binaryVector[w] = random.nextLong();
        random.nextBytes(v.extraBits);
        // Values exactly representable in binary16, so a serialization round trip is lossless.
        v.scale = 0.5f + random.nextInt(64) / 64f;
        v.offset = -1f + random.nextInt(128) / 64f;
        v.landmark = (byte) random.nextInt(3);
        return v;
    }

    private static ASHCodes store(AsymmetricHashing.QuantizedVector[] vectors, int dims, int bits) {
        var codes = ASHCodes.allocate(vectors.length, dims, bits, SMALL_PAGE_BYTES);
        for (int i = 0; i < vectors.length; i++) codes.set(i, vectors[i]);
        return codes;
    }

    @Test
    public void storesExactlyWhatItIsGivenAcrossPages() {
        var random = new Random(4);
        for (int bits : BITS) for (int dims : new int[]{1, 33, 64, 200}) {
            var vectors = new AsymmetricHashing.QuantizedVector[301];
            for (int i = 0; i < vectors.length; i++) vectors[i] = randomVector(dims, bits, random);
            var codes = store(vectors, dims, bits);
            assertTrue("bits=" + bits + " dims=" + dims + " should span pages",
                    (codes.signPages != null ? codes.signPages.length : codes.bodyPages.length) > 1);
            var reused = AsymmetricHashing.QuantizedVector.createEmpty(dims, bits);
            for (int i = 0; i < vectors.length; i++) {
                assertEquals(vectors[i], codes.get(i));
                codes.copyTo(i, reused);
                assertEquals(vectors[i], reused);
            }
            // Same content in a store with different page sizes is equal and hashes equally.
            var defaultPages = ASHCodes.copyOf(vectors, dims, bits);
            assertTrue(codes.contentEquals(defaultPages));
            assertEquals(codes.contentHash(), defaultPages.contentHash());

            int[] newToOld = new int[vectors.length];
            for (int i = 0; i < newToOld.length; i++) newToOld[i] = (i * 7) % newToOld.length;
            var permuted = codes.permute(newToOld);
            for (int i = 0; i < vectors.length; i++) assertEquals(vectors[newToOld[i]], permuted.get(i));
        }
    }

    @Test
    public void rejectsVectorsWithTheWrongShape() {
        var codes = ASHCodes.allocate(2, 64, 4);
        var shortCode = AsymmetricHashing.QuantizedVector.createEmpty(64, 4);
        shortCode.extraBits = new byte[shortCode.extraBits.length - 1];
        assertThrows(IllegalArgumentException.class, () -> codes.set(0, shortCode));
        assertThrows(IndexOutOfBoundsException.class,
                () -> codes.set(2, AsymmetricHashing.QuantizedVector.createEmpty(64, 4)));
        assertThrows(IllegalArgumentException.class, () -> ASHCodes.allocate(1, 64, 4, 0));
    }

    @Test
    public void serializesInTheStandaloneVectorFormat() throws Exception {
        var random = new Random(5);
        for (int bits : BITS) for (int dims : new int[]{1, 33, 200}) {
            var vectors = new AsymmetricHashing.QuantizedVector[97];
            for (int i = 0; i < vectors.length; i++) vectors[i] = randomVector(dims, bits, random);
            int size = vectors.length * AsymmetricHashing.QuantizedVector.serializedSizeBytes(dims, bits);
            ByteBuffer expected, actual;
            try (var writer = ByteBufferIndexWriter.create(size, false)) {
                for (var v : vectors) v.write(writer, dims, bits);
                expected = writer.getWrittenData();
            }
            try (var writer = ByteBufferIndexWriter.create(size, false)) {
                store(vectors, dims, bits).write(writer);
                actual = writer.getWrittenData();
            }
            assertEquals("bits=" + bits + " dims=" + dims, expected, actual);
            var loaded = ASHCodes.load(new ByteBufferReader(actual.duplicate()), vectors.length, dims, bits);
            for (int i = 0; i < vectors.length; i++) assertEquals(vectors[i], loaded.get(i));
        }
    }

    private static VectorFloat<?>[] inputs(int n, int dim, long seed) {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(seed);
        VectorFloat<?>[] out = new VectorFloat<?>[n];
        for (int i = 0; i < n; i++) {
            float[] v = new float[dim];
            for (int j = 0; j < dim; j++) v[j] = (float) random.nextGaussian();
            out[i] = vts.createFloatVector(v);
        }
        return out;
    }

    /** Encoded vectors re-stored in small pages, plus standalone copies of the same codes. */
    private static ASHVectors smallPages(ASHVectors encoded) {
        var a = encoded.getCompressor();
        var codes = ASHCodes.allocate(encoded.count(), a.quantizedDim, a.bitsPerDimension, SMALL_PAGE_BYTES);
        for (int i = 0; i < encoded.count(); i++) codes.set(i, encoded.get(i));
        return new ASHVectors(a, codes, encoded.symmetricOffsets);
    }

    @Test
    public void asymmetricScoresMatchStandaloneVectors() throws Exception {
        for (int bits : new int[]{1, 2, 3, 4}) for (int dim : new int[]{40, 130}) {
            var input = inputs(260, dim, 17L * bits + dim);
            var values = MockVectorValues.fromValues(input);
            var ash = AsymmetricHashing.initialize(values, AsymmetricHashing.RANDOM,
                    AsymmetricHashing.HEADER_BITS + (dim / 2) * bits, 2, bits);
            var vectors = smallPages(ash.encodeAll(values, ForkJoinPool.commonPool()));
            var reference = new ASHScorer(ash);
            for (int q = 0; q < 5; q++) {
                var query = input[q * 31];
                var standalone = reference.scoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
                var inPlace = vectors.scoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
                for (int i = 0; i < vectors.count(); i++) {
                    float expected = ASHScorer.toSimilarity(standalone.similarityTo(vectors.get(i)));
                    float actual = inPlace.similarityTo(i);
                    if (bits == 1) {
                        // The 1-bit SIMD kernel accumulates in a different order.
                        assertEquals("bits=1 dim=" + dim + " i=" + i, expected, actual, 1e-5f * Math.max(1f, Math.abs(expected)));
                    } else {
                        assertEquals("bits=" + bits + " dim=" + dim + " i=" + i,
                                Float.floatToIntBits(expected), Float.floatToIntBits(actual));
                    }
                }
            }
        }
    }

    @Test
    public void symmetricScoresDoNotDependOnWhereTheQueryIsStored() throws Exception {
        for (int bits : new int[]{1, 2, 4}) for (int dim : new int[]{40, 96, 200}) {
            var input = inputs(300, dim, 31L * bits + dim);
            var values = MockVectorValues.fromValues(input);
            var ash = AsymmetricHashing.initialize(values, AsymmetricHashing.RANDOM,
                    AsymmetricHashing.HEADER_BITS + (dim / 2) * bits, 1, bits);
            var vectors = smallPages(ash.encodeAll(values, ForkJoinPool.commonPool()));
            var auto = new ASHSymmetricScorer(vectors, ASHSymmetricScorer.Kernel.AUTO);
            var scalar = new ASHSymmetricScorer(vectors, ASHSymmetricScorer.Kernel.SCALAR);
            var random = new Random(bits * 1000L + dim);
            for (int t = 0; t < 3000; t++) {
                int i = random.nextInt(input.length), j = random.nextInt(input.length);
                String where = "bits=" + bits + " dim=" + dim + " i=" + i + " j=" + j;
                int expected = Float.floatToIntBits(scalar.scoreFunctionFor(i).similarityTo(j));
                // Integer-exact pair kernels: SIMD and scalar agree bit for bit.
                assertEquals(where, expected, Float.floatToIntBits(auto.scoreFunctionFor(i).similarityTo(j)));
                // A standalone copy of a stored query scores the same as the stored query.
                assertEquals(where, expected, Float.floatToIntBits(auto.scoreFunctionFor(vectors.get(i)).similarityTo(j)));
            }
            var diversity = vectors.diversityFunctionFor(3, VectorSimilarityFunction.DOT_PRODUCT);
            for (int j = 0; j < input.length; j++) {
                assertEquals(Float.floatToIntBits(auto.scoreFunctionFor(3).similarityTo(j)),
                        Float.floatToIntBits(diversity.similarityTo(j)));
            }
        }
    }

    @Test
    public void reorderingMovesCodesWithTheirOrdinals() throws Exception {
        var input = inputs(200, 64, 9);
        var values = MockVectorValues.fromValues(input);
        var ash = AsymmetricHashing.initialize(values, AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + 32 * 4, 3, 4);
        var vectors = smallPages(ash.encodeAll(values, ForkJoinPool.commonPool()));
        var order = vectors.reorderByLandmarkFast();
        for (int n = 0; n < vectors.count(); n++) {
            assertEquals(vectors.get(order.newToOld[n]), order.vectors.get(n));
        }
        assertEquals(vectors.ramBytesUsed(), order.vectors.ramBytesUsed());
    }

    @Test
    public void offsetKernelsMatchZeroOffsetKernels() {
        var vu = VectorizationProvider.getInstance().getVectorUtilSupport();
        var random = new Random(20261109);
        for (int bits : new int[]{2, 4}) {
            // Odd and partial-word lengths exercise the vector tails.
            for (int dims : new int[]{1, 3, 7, 31, 32, 33, 63, 65, 127, 129, 384, 768, 1536}) {
                int bytes = (dims * bits + 7) / 8;
                for (int trial = 0; trial < 4; trial++) {
                    byte[] a = new byte[bytes], b = new byte[bytes];
                    random.nextBytes(a);
                    random.nextBytes(b);
                    // Embed the codes at arbitrary offsets among random bytes, as in a code page.
                    int aOff = random.nextInt(130), bOff = random.nextInt(130);
                    byte[] aPage = embed(a, aOff, random);
                    byte[] bPage = embed(b, bOff, random);
                    String where = "bits=" + bits + " dims=" + dims;
                    // Exact length: no kernel may read query elements past the dimension count.
                    float[] query = new float[dims];
                    for (int k = 0; k < dims; k++) query[k] = (float) random.nextGaussian();
                    assertEquals(where, Float.floatToIntBits(vu.ashProjectionDot(query, a, dims, bits)),
                            Float.floatToIntBits(vu.ashProjectionDot(query, aPage, aOff, dims, bits)));
                    assertEquals(where, Float.floatToIntBits(vu.ashProjectionDotTuned(query, a, dims, bits)),
                            Float.floatToIntBits(vu.ashProjectionDotTuned(query, aPage, aOff, dims, bits)));
                    int groups = FusedASHLayout.codeGroups(dims, bits);
                    float[] lut = new float[groups * 16];
                    for (int k = 0; k < lut.length; k++) lut[k] = (float) random.nextGaussian();
                    assertEquals(where, Float.floatToIntBits(vu.ashCanonicalLutDot(a, groups, lut)),
                            Float.floatToIntBits(vu.ashCanonicalLutDot(aPage, aOff, groups, lut)));
                    if (vu.supportsAshSymmetricDotAt()) {
                        assertEquals(where, Float.floatToIntBits(vu.ashSymmetricDot(null, a, null, b, dims, bits)),
                                Float.floatToIntBits(vu.ashSymmetricDotAt(aPage, aOff, bPage, bOff, dims, bits)));
                    }
                }
            }
        }
        if (vu.supportsAshSymmetricScoring()) {
            for (int dims : new int[]{1, 63, 64, 65, 384, 1000}) {
                int words = (dims + 63) / 64;
                long[] a = random.longs(words).toArray(), b = random.longs(words).toArray();
                long[] aPage = new long[words + 9], bPage = new long[words + 5];
                System.arraycopy(a, 0, aPage, 7, words);
                System.arraycopy(b, 0, bPage, 2, words);
                assertEquals("dims=" + dims, Float.floatToIntBits(vu.ashSymmetricDot(a, null, b, null, dims, 1)),
                        Float.floatToIntBits(vu.ashSymmetricBinaryDotAt(aPage, 7, bPage, 2, dims)));
            }
            long[] bitsPage = new long[4];
            assertThrows(IndexOutOfBoundsException.class, () -> vu.ashSymmetricBinaryDotAt(bitsPage, 3, bitsPage, 0, 128));
            assertThrows(IndexOutOfBoundsException.class, () -> vu.ashSymmetricBinaryDotAt(bitsPage, -1, bitsPage, 0, 64));
        }
        float[] query = new float[384];
        byte[] page = new byte[200];
        assertThrows(IndexOutOfBoundsException.class, () -> vu.ashProjectionDot(query, page, 150, 384, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> vu.ashProjectionDot(query, page, -1, 8, 2));
        // A code that ends exactly at the end of the array is in range.
        vu.ashProjectionDot(query, page, 104, 384, 2);
        if (vu.supportsAshSymmetricDotAt()) {
            assertThrows(IllegalArgumentException.class, () -> vu.ashSymmetricDotAt(page, 0, page, 0, 8, 1));
            assertThrows(IndexOutOfBoundsException.class, () -> vu.ashSymmetricDotAt(page, 150, page, 0, 384, 2));
            vu.ashSymmetricDotAt(page, 104, page, 0, 384, 2);
        }
    }

    private static byte[] embed(byte[] code, int offset, Random random) {
        byte[] page = new byte[offset + code.length + random.nextInt(70)];
        random.nextBytes(page);
        System.arraycopy(code, 0, page, offset, code.length);
        return page;
    }
}
