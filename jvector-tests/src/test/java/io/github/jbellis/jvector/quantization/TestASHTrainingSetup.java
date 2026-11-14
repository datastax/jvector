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

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/** Sampled C=1 setup, numeric order, RNG state and borrowed-source lifetime. */
public class TestASHTrainingSetup {
    private static final class Values implements RandomAccessVectorValues {
        final VectorFloat<?>[] input;
        final int dimensions;
        final boolean shared;
        final VectorFloat<?> scratch;
        final AtomicInteger reads;
        int failAt = -1;

        Values(VectorFloat<?>[] input, int dimensions, boolean shared) {
            this(input, dimensions, shared, new AtomicInteger());
        }

        private Values(VectorFloat<?>[] input, int dimensions, boolean shared, AtomicInteger reads) {
            this.input = input;
            this.dimensions = dimensions;
            this.shared = shared;
            this.reads = reads;
            scratch = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(dimensions);
        }

        @Override public int size() { return input.length; }
        @Override public int dimension() { return dimensions; }
        @Override public boolean isValueShared() { return shared; }
        @Override public RandomAccessVectorValues copy() {
            Values copy = new Values(input, dimensions, shared, reads);
            copy.failAt = failAt;
            return copy;
        }
        @Override public VectorFloat<?> getVector(int ordinal) {
            reads.incrementAndGet();
            if (ordinal == failAt) throw new IllegalStateException("source failure");
            if (!shared || input[ordinal] == null) return input[ordinal];
            scratch.copyFrom(input[ordinal], 0, 0, dimensions);
            return scratch;
        }
    }

    private static Object prepare(Values source, int dimensions, Random random, boolean learn) throws Exception {
        var method = AsymmetricHashing.class.getDeclaredMethod("prepareSingleLandmarkTraining",
                RandomAccessVectorValues.class, int.class, Random.class, boolean.class);
        method.setAccessible(true);
        try { return method.invoke(null, source, dimensions, random, learn); }
        catch (InvocationTargetException failure) {
            if (failure.getCause() instanceof Exception) throw (Exception) failure.getCause();
            throw (Error) failure.getCause();
        }
    }

    private static Object field(Object owner, String name) throws Exception {
        var field = owner.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(owner);
    }

    private static VectorFloat<?>[] vectors(int count, int dimensions) {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        Random random = new Random(19093);
        VectorFloat<?>[] values = new VectorFloat<?>[count];
        for (int i = 0; i < count; i++) {
            values[i] = types.createFloatVector(dimensions);
            for (int d = 0; d < dimensions; d++) {
                float value = random.nextFloat() - .5f;
                if (i % 11 == 0) value = d % 2 == 0 ? -0f : 0f;
                if (i % 11 == 1) value *= 1e-12f;
                values[i].set(d, value);
            }
        }
        return values;
    }

    private static final class Reference {
        final VectorFloat<?> mean;
        final double[] xCol;
        final int rows;

        Reference(VectorFloat<?>[] source, int dimensions, Random rng, boolean learn) {
            int sampleSize = Math.min(source.length, dimensions * 20);
            Random selection = learn ? rng : new Random(103);
            int[] ordinals = new int[sampleSize];
            for (int i = 0; i < sampleSize; i++) ordinals[i] = i;
            for (int i = sampleSize; i < source.length; i++) {
                int j = selection.nextInt(i + 1);
                if (j < sampleSize) ordinals[j] = i;
            }
            Arrays.sort(ordinals);
            VectorFloat<?>[] points = new VectorFloat<?>[sampleSize];
            for (int i = 0; i < sampleSize; i++) {
                points[i] = source[ordinals[i]].copy();
                float norm = (float) Math.sqrt(VectorUtil.dotProduct(points[i], points[i]));
                if (norm > 1e-10f) VectorUtil.scale(points[i], 1f / norm);
            }
            double[] accumulator = new double[dimensions];
            for (VectorFloat<?> point : points)
                for (int d = 0; d < dimensions; d++) accumulator[d] += point.get(d);
            mean = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(dimensions);
            double inverseCount = 1.0 / points.length;
            for (int d = 0; d < dimensions; d++) mean.set(d, (float) (accumulator[d] * inverseCount));
            rows = learn ? sampleSize : 0;
            xCol = learn ? new double[rows * dimensions] : null;
            if (!learn) return;
            double[] residual = new double[dimensions];
            for (int row = 0; row < rows; row++) {
                double norm2 = 0;
                for (int d = 0; d < dimensions; d++) {
                    double value = (double) points[row].get(d) - (double) mean.get(d);
                    residual[d] = value;
                    norm2 += value * value;
                }
                double inverse = 1.0 / Math.max(Math.sqrt(norm2), 1e-20);
                for (int d = 0; d < dimensions; d++) xCol[row + d * rows] = residual[d] * inverse;
            }
        }
    }

    @Test
    public void sampledMeanAndResidualsMatchOracleIncludingRngAndBorrowedStorage() throws Exception {
        for (int[] shape : new int[][]{{1,1},{1,20},{1,21},{1,65},{7,31},{7,32},{7,33},
                {7,64},{7,65},{7,139},{7,140},{7,141},{16,321},{33,67},{65,1311}}) {
            int dimensions = shape[0], count = shape[1];
            VectorFloat<?>[] values = vectors(count, dimensions);
            int[][] saved = new int[count][dimensions];
            for (int i = 0; i < count; i++) for (int d = 0; d < dimensions; d++)
                saved[i][d] = Float.floatToRawIntBits(values[i].get(d));
            for (boolean shared : new boolean[]{false,true}) for (boolean learn : new boolean[]{false,true}) {
                Random referenceRandom = new Random(103), actualRandom = new Random(103);
                Reference expected = new Reference(values, dimensions, referenceRandom, learn);
                Values source = new Values(values, dimensions, shared);
                Object actual = prepare(source, dimensions, actualRandom, learn);
                VectorFloat<?> mean = (VectorFloat<?>) field(actual, "landmark");
                assertEquals(Math.min(count, dimensions * 20), source.reads.get());
                for (int d = 0; d < dimensions; d++)
                    assertEquals(Float.floatToRawIntBits(expected.mean.get(d)), Float.floatToRawIntBits(mean.get(d)));
                Object training = field(actual, "trainingData");
                if (learn) {
                    assertEquals(expected.rows, field(training, "rows"));
                    assertEquals(dimensions, field(training, "columns"));
                    double[] observed = (double[]) field(training, "xCol");
                    assertEquals(expected.xCol.length, observed.length);
                    for (int i = 0; i < observed.length; i++)
                        assertEquals(Double.doubleToRawLongBits(expected.xCol[i]), Double.doubleToRawLongBits(observed[i]));
                } else assertNull(training);
                for (int i = 0; i < 8; i++) assertEquals(referenceRandom.nextLong(), actualRandom.nextLong());
                for (int i = 0; i < count; i++) for (int d = 0; d < dimensions; d++)
                    assertEquals(saved[i][d], Float.floatToRawIntBits(values[i].get(d)));
            }
        }
    }

    @Test
    public void initializationReadsTheSameSampleForEveryTrainingStage() throws Exception {
        int dimensions = 7, count = 141;
        for (int optimizer : new int[]{AsymmetricHashing.RANDOM, AsymmetricHashing.ITQ}) {
            for (int bits : new int[]{1,2,4,9}) {
                Values source = new Values(vectors(count, dimensions), dimensions, true);
                var model = AsymmetricHashing.initialize(source, optimizer,
                        AsymmetricHashing.HEADER_BITS + 3 * bits, 1, bits);
                Reference expected = new Reference(source.input, dimensions, new Random(103),
                        optimizer == AsymmetricHashing.ITQ);
                assertEquals(1, model.landmarks.length);
                assertEquals(3, model.quantizedDim);
                assertEquals(dimensions * 20, source.reads.get());
                for (int d = 0; d < dimensions; d++) {
                    assertEquals(Float.floatToRawIntBits(expected.mean.get(d)),
                            Float.floatToRawIntBits(model.landmarks[0].get(d)));
                }
            }
        }
    }

    @Test
    public void setupRejectsInvalidSourcesAndPropagatesReadFailures() throws Exception {
        assertThrows(IllegalArgumentException.class,
                () -> prepare(new Values(new VectorFloat<?>[0], 3, false), 3, new Random(103), true));
        Values source = new Values(vectors(65, 7), 7, false);
        source.failAt = 17;
        assertThrows(IllegalStateException.class, () -> prepare(source, 7, new Random(103), true));
        assertEquals(18, source.reads.get());
        Values ragged = new Values(vectors(3, 7), 7, false);
        ragged.input[1] = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(6);
        assertThrows(IllegalArgumentException.class, () -> prepare(ragged, 7, new Random(103), true));
        Values nullRow = new Values(vectors(3, 7), 7, false);
        nullRow.input[1] = null;
        assertThrows(NullPointerException.class, () -> prepare(nullRow, 7, new Random(103), true));
        Values empty = new Values(new VectorFloat<?>[0], 7, false);
        assertThrows(IllegalArgumentException.class, () -> AsymmetricHashing.initialize(empty,
                AsymmetricHashing.ITQ, AsymmetricHashing.HEADER_BITS + 6, 1, 2));
        assertEquals(0, empty.reads.get());
    }

    /** Metadata-only sources must fail validation before any vector access. */
    private static RandomAccessVectorValues metadataOnlySource(int count, int dimensions) {
        return new RandomAccessVectorValues() {
            @Override public int size() { return count; }
            @Override public int dimension() { return dimensions; }
            @Override public boolean isValueShared() { return false; }
            @Override public RandomAccessVectorValues copy() { return this; }
            @Override public VectorFloat<?> getVector(int ordinal) {
                throw new AssertionError("Invalid source/configuration must not fetch vectors");
            }
        };
    }

    @Test
    public void invalidMetadataAndConfigurationAreRejectedBeforeReading() {
        int budget = AsymmetricHashing.HEADER_BITS + 6;
        assertThrows(NullPointerException.class,
                () -> AsymmetricHashing.initialize(null, AsymmetricHashing.ITQ, budget, 1, 2));
        assertThrows(IllegalArgumentException.class,
                () -> AsymmetricHashing.initialize(metadataOnlySource(0, 7), AsymmetricHashing.ITQ, budget, 1, 2));
        assertThrows(IllegalArgumentException.class,
                () -> AsymmetricHashing.initialize(metadataOnlySource(-1, 7), AsymmetricHashing.ITQ, budget, 1, 2));
        for (int dimensions : new int[]{0,-1}) {
            assertThrows(IllegalArgumentException.class,
                    () -> AsymmetricHashing.initialize(metadataOnlySource(1, dimensions),
                            AsymmetricHashing.ITQ, budget, 1, 2));
        }
        for (int optimizer : new int[]{0,-1,3}) {
            assertThrows(IllegalArgumentException.class,
                    () -> AsymmetricHashing.initialize(metadataOnlySource(1, 7), optimizer, budget, 1, 2));
        }
        for (int bits : new int[]{0,10}) {
            assertThrows(IllegalArgumentException.class,
                    () -> AsymmetricHashing.initialize(metadataOnlySource(1, 7),
                            AsymmetricHashing.ITQ, budget, 1, bits));
        }
        for (int landmarks : new int[]{0,257}) {
            assertThrows(IllegalArgumentException.class,
                    () -> AsymmetricHashing.initialize(metadataOnlySource(1, 7),
                            AsymmetricHashing.ITQ, budget, landmarks, 2));
        }
        for (int invalidBudget : new int[]{AsymmetricHashing.HEADER_BITS,
                AsymmetricHashing.HEADER_BITS + 3, AsymmetricHashing.HEADER_BITS + 16}) {
            assertThrows(IllegalArgumentException.class,
                    () -> AsymmetricHashing.initialize(metadataOnlySource(1, 7),
                            AsymmetricHashing.ITQ, invalidBudget, 1, 2));
        }
    }

    @Test
    public void sampleSizingDoesNotOverflowAndRejectsUnrepresentableMatrices() throws Exception {
        var method = AsymmetricHashing.class.getDeclaredMethod("trainingSampleSize", int.class, int.class);
        method.setAccessible(true);
        assertEquals(Integer.valueOf(Integer.MAX_VALUE),
                method.invoke(null, Integer.MAX_VALUE, Integer.MAX_VALUE));
        assertEquals(Integer.valueOf(100), method.invoke(null, 100, Integer.MAX_VALUE));
        assertEquals(Integer.valueOf(140), method.invoke(null, Integer.MAX_VALUE, 7));
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> AsymmetricHashing.initialize(metadataOnlySource(50_000, 50_000),
                        AsymmetricHashing.ITQ, AsymmetricHashing.HEADER_BITS + 6, 1, 2));
        assertTrue(failure.getMessage().contains("array size limit"));
        IllegalArgumentException wideFailure = assertThrows(IllegalArgumentException.class,
                () -> AsymmetricHashing.initialize(metadataOnlySource(1, 50_000),
                        AsymmetricHashing.ITQ, AsymmetricHashing.HEADER_BITS + 6, 1, 2));
        assertTrue(wideFailure.getMessage().contains("PCA basis"));
    }

    @Test
    public void nonFiniteAndOverflowingSampleNormsAreRejected() {
        for (float invalid : new float[]{Float.NaN, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.MAX_VALUE}) {
            for (boolean shared : new boolean[]{false,true}) {
                for (boolean learn : new boolean[]{false,true}) {
                    Values source = new Values(vectors(3, 7), 7, shared);
                    source.input[1].set(3, invalid);
                    IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                            () -> prepare(source, 7, new Random(103), learn));
                    assertTrue(failure.getMessage().contains("ordinal 1"));
                    assertEquals(2, source.reads.get());
                    assertEquals(Float.floatToRawIntBits(invalid),
                            Float.floatToRawIntBits(source.input[1].get(3)));
                }
            }
        }
    }

    @Test
    public void sharedSourceReadFailuresPropagateDuringInitialization() {
        Values source = new Values(vectors(65, 7), 7, true);
        source.failAt = 17;
        IllegalStateException failure = assertThrows(IllegalStateException.class,
                () -> AsymmetricHashing.initialize(source, AsymmetricHashing.ITQ,
                        AsymmetricHashing.HEADER_BITS + 6, 1, 2));
        assertEquals("source failure", failure.getMessage());
        assertEquals(18, source.reads.get());
    }

    @Test
    public void zeroAndConstantSamplesProduceFiniteModels() throws Exception {
        for (boolean constant : new boolean[]{false,true}) {
            VectorFloat<?>[] input = vectors(65, 3);
            for (VectorFloat<?> value : input) {
                for (int d = 0; d < 3; d++) value.set(d, constant ? (d + 1) * .25f : 0f);
            }
            Object prepared = prepare(new Values(input, 3, true), 3, new Random(103), true);
            Object training = field(prepared, "trainingData");
            for (double value : (double[]) field(training, "xCol")) assertEquals(0.0, value, 0.0);
            Values source = new Values(input, 3, true);
            var model = AsymmetricHashing.initialize(source, AsymmetricHashing.ITQ,
                    AsymmetricHashing.HEADER_BITS + 4, 1, 2);
            for (double[] row : model.stiefelTransform.AData) {
                for (double value : row) assertTrue(Double.isFinite(value));
            }
            assertEquals(60, source.reads.get());
        }
    }

    @Test
    public void multipleLandmarksStillTrainThroughTheExistingPath() throws Exception {
        for (int optimizer : new int[]{AsymmetricHashing.ITQ, AsymmetricHashing.RANDOM}) {
            Values source = new Values(vectors(120, 7), 7, true);
            var model = AsymmetricHashing.initialize(source, optimizer,
                    AsymmetricHashing.HEADER_BITS + 6, 3, 2);
            assertEquals(3, model.landmarks.length);
            assertEquals(3, model.quantizedDim);
            assertEquals(120, source.reads.get());
        }
    }
}
