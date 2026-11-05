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

import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.lang.reflect.Proxy;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/** Base-vector encoding contracts shared by the native and Panama providers. */
public class TestASHBatchEncoding {
    private static final int DIMENSION = 16;

    private static VectorFloat<?>[] vectors(int count) {
        Random random = new Random(771023);
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        VectorFloat<?>[] result = new VectorFloat<?>[count];
        for (int i = 0; i < count; i++) {
            result[i] = types.createFloatVector(DIMENSION);
            for (int d = 0; d < DIMENSION; d++) result[i].set(d, random.nextFloat() - .5f);
        }
        return result;
    }

    private static AsymmetricHashing model(int bits, int landmarks) throws Exception {
        return AsymmetricHashing.initialize(MockVectorValues.fromValues(vectors(128)),
                AsymmetricHashing.RANDOM, AsymmetricHashing.HEADER_BITS + 9 * bits, landmarks, bits);
    }

    private static void assertCode(AsymmetricHashing.QuantizedVector expected,
                                   AsymmetricHashing.QuantizedVector actual) {
        assertEquals(expected.landmark, actual.landmark);
        assertEquals(expected.scale, actual.scale, 0f);
        assertEquals(expected.offset, actual.offset, 0f);
        assertArrayEquals(expected.binaryVector, actual.binaryVector);
        assertArrayEquals(expected.extraBits, actual.extraBits);
    }

    private static void assertHighPrecisionCodeCompatible(AsymmetricHashing.QuantizedVector expected,
                                                           AsymmetricHashing.QuantizedVector actual,
                                                           int dimensions) {
        assertEquals(expected.landmark, actual.landmark);
        assertEquals(expected.offset, actual.offset, 0f);
        assertArrayEquals(expected.binaryVector, actual.binaryVector);
        // Different FP32 reduction orders can cross a 9-bit code boundary.
        // Check the reconstructed component within one step from each code,
        // rather than requiring bit equality of that discontinuous quantizer.
        for (int d = 0; d < dimensions; d++) {
            boolean positive = AsymmetricHashing.QuantizedVector.getBit(expected.binaryVector, d);
            int e = AsymmetricHashing.QuantizedVector.readExtraCode(expected.extraBits, d, 8);
            int a = AsymmetricHashing.QuantizedVector.readExtraCode(actual.extraBits, d, 8);
            if (!positive) { e = (~e) & 255; a = (~a) & 255; }
            assertEquals(expected.scale * (e + .5), actual.scale * (a + .5),
                    expected.scale + actual.scale + 1e-7);
        }
    }

    private static final class Values implements RandomAccessVectorValues {
        private final VectorFloat<?>[] vectors;
        private final boolean shared;
        private final VectorFloat<?> scratch;
        private final AtomicInteger reads;
        Values(VectorFloat<?>[] vectors, boolean shared) {
            this(vectors, shared, null);
        }
        Values(VectorFloat<?>[] vectors, boolean shared, AtomicInteger reads) {
            this.reads = reads;
            this.vectors = vectors;
            this.shared = shared;
            scratch = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(DIMENSION);
        }
        @Override public int size() { return vectors.length; }
        @Override public int dimension() { return DIMENSION; }
        @Override public boolean isValueShared() { return shared; }
        @Override public RandomAccessVectorValues copy() { return new Values(vectors, shared, reads); }
        @Override public VectorFloat<?> getVector(int ordinal) {
            if (reads != null) reads.incrementAndGet();
            VectorFloat<?> vector = vectors[ordinal];
            if (vector == null || !shared) return vector;
            for (int d = 0; d < DIMENSION; d++) scratch.set(d, vector.get(d));
            return scratch;
        }
    }

    @Test
    public void bulkMatchesSingleForTailsMissingValuesAndBorrowedStorage() throws Exception {
        for (int bits : new int[]{1,2,3,4,9}) {
            var ash = model(bits, bits == 3 ? 3 : 1);
            for (int count : new int[]{0,1,31,32,33,65,97}) {
                VectorFloat<?>[] input = vectors(count);
                if (count > 32) {
                    input[0] = null;
                    input[31] = null;
                    input[count - 1] = ash.landmarks[0].copy(); // zero residual
                }
                ASHVectors firstBulkResult = null;
                for (boolean shared : new boolean[]{false,true}) {
                    for (int threads : new int[]{1,3}) {
                        ForkJoinPool pool = new ForkJoinPool(threads);
                        try {
                            var actual = ash.encodeAll(new Values(input, shared), pool);
                            assertEquals(count, actual.count());
                            if (firstBulkResult == null) firstBulkResult = actual;
                            for (int i = 0; i < count; i++) {
                                var expected = input[i] == null
                                        ? AsymmetricHashing.QuantizedVector.createEmpty(ash.quantizedDim, bits)
                                        : ash.encode(input[i]);
                                try {
                                    // A BLAS implementation may choose a different FP32
                                    // reduction for a worker's batch shape. Preserve exact
                                    // 1/2/3/4-bit checks; use the existing reconstruction
                                    // error bound for discontinuous 9-bit codes/half scales.
                                    if (bits == 9 && input[i] != null)
                                        assertHighPrecisionCodeCompatible(firstBulkResult.get(i), actual.get(i), ash.quantizedDim);
                                    else assertCode(firstBulkResult.get(i), actual.get(i));
                                    if (bits == 9 && input[i] != null)
                                        assertHighPrecisionCodeCompatible(expected, actual.get(i), ash.quantizedDim);
                                    else assertCode(expected, actual.get(i));
                                }
                                catch (AssertionError failure) {
                                    throw new AssertionError("bits=" + bits + ", count=" + count
                                            + ", shared=" + shared + ", threads=" + threads + ", ordinal=" + i,
                                            failure);
                                }
                            }
                        } finally { pool.shutdown(); }
                    }
                }
            }
        }
    }

    @Test
    public void correctionsAreComputedInTheEncodingPassAndSharedWithoutChangingCodesOrScores() throws Exception {
        for (int bits : new int[]{2,4}) for (int landmarks : new int[]{1,3}) {
            var ash = model(bits, landmarks);
            var input = vectors(97);
            input[31] = null;
            input[96] = ash.landmarks[0].copy();
            AtomicInteger reads = new AtomicInteger();
            var source = new Values(input, true, reads);
            ForkJoinPool pool = new ForkJoinPool(3);
            try {
                var bulk = ash.encodeAll(source, pool);
                assertEquals("Read each raw vector exactly once", input.length, reads.get());
                assertNotNull(bulk.symmetricOffsets);
                AsymmetricHashing.QuantizedVector[] codes = new AsymmetricHashing.QuantizedVector[input.length];
                for (int i = 0; i < codes.length; i++) {
                    codes[i] = input[i] == null ? AsymmetricHashing.QuantizedVector.createEmpty(ash.quantizedDim, bits) : ash.encode(input[i]);
                    assertCode(codes[i], bulk.get(i));
                    float expected = bulk.get(i).offset + bulk.get(i).scale * AsymmetricHashing.dotProjectionCode(
                            ash.landmarkProj[bulk.get(i).landmark & 255], bulk.get(i).extraBits, ash.quantizedDim, bits);
                    assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(bulk.symmetricOffsets[i]));
                }
                var uncached = new ASHVectors(ash, codes);
                var prepared = new ASHSymmetricScorer(bulk);
                var reference = new ASHSymmetricScorer(uncached);
                var offsets = ASHSymmetricScorer.class.getDeclaredField("centeredOffsets");
                offsets.setAccessible(true);
                assertSame(bulk.symmetricOffsets, offsets.get(prepared));
                assertSame(bulk.symmetricOffsets, offsets.get(new ASHSymmetricScorer(bulk)));
                assertEquals(io.github.jbellis.jvector.util.RamUsageEstimator.shallowSizeOf(bulk.symmetricOffsets),
                        bulk.ramBytesUsed() - uncached.ramBytesUsed());
                var reordered = bulk.reorderByLandmarkFast();
                var reorderedScorer = new ASHSymmetricScorer(reordered.vectors);
                for (int i = 0; i < codes.length; i++) for (int j = 0; j < codes.length; j++) {
                    assertEquals(reference.dotProduct(i, j), prepared.dotProduct(i, j), 0f);
                    assertEquals(prepared.dotProduct(reordered.newToOld[i], reordered.newToOld[j]),
                            reorderedScorer.dotProduct(i, j), 0f);
                }
                assertEquals("Scorer creation/reordering must not reread raw input", input.length, reads.get());
            } finally { pool.shutdown(); }
        }
    }

    @Test
    public void encodeToClearsReusedCodesAndEncodingIgnoresQueryPrecision() throws Exception {
        String old = System.getProperty("jvector.ash.queryProjection");
        try {
            // This would fail even on a capable machine if encoding requested a
            // query view. Query-only selection must never affect base encoding.
            System.setProperty("jvector.ash.queryProjection", "invalid-query-mode");
            var ash = model(4, 1);
            VectorFloat<?>[] input = vectors(65);
            var dest = ash.encode(input[0]);
            for (var vector : input) {
                var expected = ash.encode(vector);
                ash.encodeTo(vector, dest);
                assertCode(expected, dest);
            }
            ForkJoinPool pool = new ForkJoinPool(2);
            try {
                var bulk = ash.encodeAll(new Values(input, true), pool);
                for (int i = 0; i < input.length; i++) assertCode(ash.encode(input[i]), bulk.get(i));
            } finally { pool.shutdown(); }
        } finally {
            if (old == null) System.clearProperty("jvector.ash.queryProjection");
            else System.setProperty("jvector.ash.queryProjection", old);
        }
    }

    @Test
    public void allMissingSourceRetainsZeroRepresentations() throws Exception {
        var ash = model(2, 1);
        ForkJoinPool pool = new ForkJoinPool(3);
        try {
            var encoded = ash.encodeAll(new Values(new VectorFloat<?>[97], true), pool);
            var empty = AsymmetricHashing.QuantizedVector.createEmpty(ash.quantizedDim, 2);
            for (int i = 0; i < 97; i++) assertCode(empty, encoded.get(i));
        } finally { pool.shutdown(); }
    }

    @Test
    public void invalidArgumentsAndSourceFailureArePropagatedAfterWorkersFinish() throws Exception {
        var ash = model(2, 1);
        assertThrows(NullPointerException.class, () -> ash.encode(null));
        var shortVector = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(1);
        assertThrows(IllegalArgumentException.class, () -> ash.encode(shortVector));
        var input = vectors(128);
        AtomicInteger otherRangeReads = new AtomicInteger();
        RandomAccessVectorValues source = new RandomAccessVectorValues() {
            @Override public int size() { return 128; }
            @Override public int dimension() { return DIMENSION; }
            @Override public boolean isValueShared() { return false; }
            @Override public RandomAccessVectorValues copy() { return this; }
            @Override public VectorFloat<?> getVector(int ordinal) {
                if (ordinal == 0) throw new IllegalStateException("source failed");
                if (ordinal >= 64) otherRangeReads.incrementAndGet();
                return input[ordinal];
            }
        };
        ForkJoinPool pool = new ForkJoinPool(2);
        try {
            assertThrows(NullPointerException.class, () -> ash.encodeAll(null, pool));
            assertThrows(NullPointerException.class, () -> ash.encodeAll(source, null));
            assertThrows(IllegalStateException.class, () -> ash.encodeAll(source, pool));
            assertEquals("No source worker may outlive encodeAll failure", 64, otherRangeReads.get());
        } finally { pool.shutdown(); }
    }

    @Test
    public void backingArrayNormalizationMatchesOpaqueVectorsBitForBit() throws Exception {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        var normalize = AsymmetricHashing.class.getDeclaredMethod("normalizeResidual",
                VectorFloat.class,
                VectorFloat.class, float.class, float[].class);
        normalize.setAccessible(true);
        Random random = new Random(81231);
        for (int dimensions : new int[]{0,1,15,16,17,31,32,33,768,3072}) {
            var vector = types.createFloatVector(dimensions);
            var mean = types.createFloatVector(dimensions);
            for (int j = 0; j < dimensions; j++) {
                vector.set(j, j % 7 == 0 ? -0.0f : random.nextFloat() - .5f);
                mean.set(j, j % 7 == 0 ? 0.0f : random.nextFloat() - .5f);
            }
            var opaqueVector = (VectorFloat<?>) Proxy.newProxyInstance(
                    getClass().getClassLoader(), new Class<?>[]{VectorFloat.class},
                    (proxy, method, args) -> method.invoke(vector, args));
            var opaqueMean = (VectorFloat<?>) Proxy.newProxyInstance(
                    getClass().getClassLoader(), new Class<?>[]{VectorFloat.class},
                    (proxy, method, args) -> method.invoke(mean, args));
            for (float residualNorm : new float[]{0f,1f,3f,Float.MAX_VALUE}) {
                float[] expected = new float[dimensions];
                normalize.invoke(null, opaqueVector, opaqueMean, residualNorm, expected);
                for (var x : new VectorFloat<?>[]{vector, opaqueVector})
                    for (var mu : new VectorFloat<?>[]{mean, opaqueMean}) {
                        float[] actual = new float[dimensions];
                        normalize.invoke(null, x, mu, residualNorm, actual);
                        for (int j = 0; j < dimensions; j++)
                            assertEquals(Float.floatToRawIntBits(expected[j]), Float.floatToRawIntBits(actual[j]));
                    }
            }
        }
    }

}
