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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedASH;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.junit.Test;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.function.IntFunction;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

/** Retained ownership and lazy-cache deltas; all sizes follow the running JVM's layout. */
public class TestASHMemoryAccounting {
    private static volatile long accountingResult;

    private static AsymmetricHashing model(int bits, int landmarks) throws Exception {
        var type = VectorizationProvider.getInstance().getVectorTypeSupport();
        VectorFloat<?>[] values = new VectorFloat<?>[97];
        Random random = new Random(87614);
        for (int n = 0; n < values.length; n++) {
            values[n] = type.createFloatVector(64);
            for (int j = 0; j < 64; j++) values[n].set(j, random.nextFloat() - .5f);
            VectorUtil.l2normalize(values[n]);
        }
        return AsymmetricHashing.initialize(MockVectorValues.fromValues(values), AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + 33 * bits, landmarks, bits);
    }

    private static AsymmetricHashing.QuantizedVector[] codes(AsymmetricHashing ash, int count) {
        var result = new AsymmetricHashing.QuantizedVector[count];
        for (int i = 0; i < count; i++) {
            result[i] = AsymmetricHashing.QuantizedVector.createEmpty(ash.quantizedDim, ash.bitsPerDimension);
            result[i].scale = 1f; result[i].offset = 0f;
            result[i].landmark = (byte)(i % ash.landmarkCount);
        }
        return result;
    }

    private static Object field(Object object, String name) throws Exception {
        Field f = object.getClass().getDeclaredField(name);
        f.setAccessible(true);
        return f.get(object);
    }

    private static long blockBytes(byte[][] blocks) {
        long bytes = RamUsageEstimator.shallowSizeOf(blocks);
        for (byte[] block : blocks) bytes += RamUsageEstimator.sizeOf(block);
        return bytes;
    }

    @Test
    public void canonicalVectorsChargeHeapObjectsAndArrayHeaders() {
        for (int bits : new int[]{1,2,3,4,9}) for (int dims : new int[]{1,33,384,1536}) {
            var code = AsymmetricHashing.QuantizedVector.createEmpty(dims, bits);
            long expected = RamUsageEstimator.shallowSizeOf(code)
                    + RamUsageEstimator.shallowSizeOf(code.binaryVector)
                    + RamUsageEstimator.shallowSizeOf(code.extraBits);
            assertEquals(expected, code.ramBytesUsed());
            assertTrue(code.ramBytesUsed() > AsymmetricHashing.QuantizedVector.serializedSizeBytes(dims, bits));
            // Capacity, including padding beyond the logical code, remains retained.
            code.binaryVector = Arrays.copyOf(code.binaryVector, code.binaryVector.length + 2);
            code.extraBits = Arrays.copyOf(code.extraBits, code.extraBits.length + 3);
            assertEquals(RamUsageEstimator.shallowSizeOf(code)
                    + RamUsageEstimator.sizeOf(code.binaryVector)
                    + RamUsageEstimator.sizeOf(code.extraBits), code.ramBytesUsed());
        }
    }

    @Test
    public void collectionsChargeVectorsHeadersAndModelExactlyOnce() throws Exception {
        for (int bits : new int[]{1,2,3,4,9}) {
            var ash = model(bits, 2);
            var empty = new ASHVectors(ash, codes(ash, 0));
            for (int count : new int[]{1,17,1024}) {
                var data = codes(ash, count);
                var vectors = new ASHVectors(ash, data);
                long expectedDelta = RamUsageEstimator.shallowSizeOf(data)
                        - RamUsageEstimator.shallowSizeOf(new Object[0])
                        + 2*(RamUsageEstimator.sizeOf(new float[count])-RamUsageEstimator.sizeOf(new float[0]))
                        + RamUsageEstimator.sizeOf(new byte[count])-RamUsageEstimator.sizeOf(new byte[0]);
                for (var code : data) expectedDelta += code.ramBytesUsed();
                assertEquals(expectedDelta, vectors.ramBytesUsed()-empty.ramBytesUsed());
                long bytes = vectors.ramBytesUsed();
                for (int i=0;i<10;i++) assertEquals(bytes, vectors.ramBytesUsed());
                assertEquals(bytes, vectors.reorderByLandmarkFast().vectors.ramBytesUsed());
            }
        }
    }

    @Test
    public void sharedOrdinalsAreChargedConservativelyWithoutIdentityMaps() throws Exception {
        var ash = model(2, 1);
        var data = codes(ash, 2);
        var separate = new ASHVectors(ash, data);
        data[1] = data[0];
        assertEquals(separate.ramBytesUsed(), new ASHVectors(ash, data).ramBytesUsed());
    }

    @Test
    public void conservativeLargeCollectionTotalsUseLongArithmetic() throws Exception {
        var ash = model(4, 1);
        var code = codes(ash, 1)[0];
        code.extraBits = new byte[8192]; // Valid oversized capacity; only its code prefix is scored.
        var data = new AsymmetricHashing.QuantizedVector[300_000];
        Arrays.fill(data, code);
        var vectors = new ASHVectors(ash, data);
        assertTrue((long)data.length*code.ramBytesUsed() > Integer.MAX_VALUE);
        assertTrue(vectors.ramBytesUsed() > Integer.MAX_VALUE);
        long bytes = vectors.ramBytesUsed();
        assertEquals(bytes, vectors.ramBytesUsed());
    }

    @Test
    public void productionCodeLayoutsExposeThePreviouslyMissingPerVectorStorage() {
        for (int bits : new int[]{2,4}) for (int dimension : new int[]{384,512,768,1536}) {
            var code = AsymmetricHashing.QuantizedVector.createEmpty(dimension, bits);
            long missing = code.ramBytesUsed()+2L*Float.BYTES+Byte.BYTES
                    - AsymmetricHashing.QuantizedVector.serializedSizeBytes(dimension,bits);
            assertTrue(missing > 0);
            System.out.printf("ASH_RETAINED_LAYOUT refs=%d bits=%d projected=%d extra_per_million=%d%n",
                    RamUsageEstimator.NUM_BYTES_OBJECT_REF,bits,dimension,missing*1_000_000L);
        }
    }

    @Test
    public void allLandmarkBackingArraysAreIncluded() throws Exception {
        var one = model(2, 1); var four = model(2, 4);
        long expected = RamUsageEstimator.shallowSizeOf(four.landmarks)-RamUsageEstimator.shallowSizeOf(one.landmarks);
        for (int c=1;c<4;c++) expected += four.landmarks[c].ramBytesUsed();
        expected += 2*(RamUsageEstimator.sizeOf(new float[4])-RamUsageEstimator.sizeOf(new float[1]));
        expected += RamUsageEstimator.shallowSizeOf(four.landmarkProj)-RamUsageEstimator.shallowSizeOf(one.landmarkProj);
        expected += 3*RamUsageEstimator.sizeOf(new float[33]);
        assertEquals(expected, four.ramBytesUsed()-one.ramBytesUsed());
    }

    @Test
    public void queryRepresentationsAreChargedOnlyAfterCreationAndOnlyOnce() throws Exception {
        String previous = System.getProperty("jvector.ash.queryProjection");
        try {
            for (String mode : new String[]{"fp32", "auto"}) {
                System.setProperty("jvector.ash.queryProjection", mode);
                var ash = model(2, 1);
                long before = ash.ramBytesUsed();
                assertNull(field(ash.stiefelTransform, "queryProjection"));
                float[] input = new float[64], output = new float[33];
                ash.stiefelTransform.projectQuery(input, output);
                var primary = (io.github.jbellis.jvector.vector.VectorUtilSupport.AshProjection)
                        field(ash.stiefelTransform, "projection");
                var query = (io.github.jbellis.jvector.vector.VectorUtilSupport.AshProjection)
                        field(ash.stiefelTransform, "queryProjection");
                if (mode.equals("fp32")) assertSame(primary, query);
                if (mode.equals("auto") && Boolean.getBoolean("jvector.test.requireFp16")) {
                    assertNotSame(primary, query);
                    assertTrue(query.ramBytesUsed() >= 64L*33*Short.BYTES);
                }
                long added = query == primary ? 0 : RamUsageEstimator.shallowSizeOf(query)+query.ramBytesUsed();
                assertEquals(before+added, ash.ramBytesUsed());
                ash.stiefelTransform.projectQuery(input, output);
                assertEquals(before+added, ash.ramBytesUsed());
            }
        } finally {
            if (previous == null) System.clearProperty("jvector.ash.queryProjection");
            else System.setProperty("jvector.ash.queryProjection", previous);
        }
    }

    @Test
    public void blockCachesChargeAllocatedCapacityIncludingPaddedTails() throws Exception {
        for (int bits : new int[]{2,4}) {
            var ash = model(bits, 1); var vectors = new ASHVectors(ash, codes(ash, 17));
            for (int block : new int[]{8,16,32}) {
                long before = vectors.ramBytesUsed();
                byte[][] data = vectors.symmetricProjectionBlocks(block);
                long added = vectors.ramBytesUsed()-before;
                assertTrue(added >= blockBytes(data));
                assertTrue("Small conservative map allowance only", added <= blockBytes(data)+512);
                assertSame(data, vectors.symmetricProjectionBlocks(block));
                assertEquals(before+added, vectors.ramBytesUsed());
            }
        }
    }

    @Test
    public void concurrentCacheCreationIsChargedOnceAndFailuresDoNotPublish() throws Exception {
        var ash = model(4, 1); var data = codes(ash, 37);
        var serial = new ASHVectors(ash, data); var parallel = new ASHVectors(ash, data);
        for (int block : new int[]{8,16,32}) serial.symmetricProjectionBlocks(block);
        var pool = Executors.newFixedThreadPool(8);
        try {
            var jobs = new ArrayList<java.util.concurrent.Callable<Void>>();
            for (int i=0;i<32;i++) jobs.add(() -> {
                for (int block : new int[]{8,16,32}) parallel.symmetricProjectionBlocks(block);
                return null;
            });
            for (var result : pool.invokeAll(jobs)) result.get();
        } finally {
            pool.shutdownNow();
            assertTrue(pool.awaitTermination(30, java.util.concurrent.TimeUnit.SECONDS));
        }
        assertEquals(serial.ramBytesUsed(), parallel.ramBytesUsed());
        var bad = codes(ash, 17); bad[0].extraBits = new byte[0];
        var broken = new ASHVectors(ash, bad); long before = broken.ramBytesUsed();
        assertThrows(IllegalArgumentException.class, () -> broken.symmetricProjectionBlocks(8));
        assertEquals(before, broken.ramBytesUsed());
    }

    @Test
    public void symmetricCachesAreChargedWithoutCountingTheirSharedModelAgain() throws Exception {
        for (int bits : new int[]{1,2,4}) {
            var ash = model(bits, 2); var vectors = new ASHVectors(ash, codes(ash, 17));
            long before = vectors.ramBytesUsed();
            long[] previousFlat = (long[])field(vectors, "flatPackedVectors");
            vectors.diversityFunctionFor(0, VectorSimilarityFunction.DOT_PRODUCT);
            Object symmetric = field(vectors, "diversityScorer");
            float[][] dots = (float[][]) field(symmetric, "landmarkDots");
            long expected = RamUsageEstimator.shallowSizeOf(symmetric)+RamUsageEstimator.shallowSizeOf(dots);
            for (float[] row : dots) expected += RamUsageEstimator.sizeOf(row);
            expected += RamUsageEstimator.shallowSizeOf((float[])field(symmetric, "centeredOffsets"));
            expected += RamUsageEstimator.shallowSizeOf((long[])field(vectors, "flatPackedVectors"))
                    - RamUsageEstimator.shallowSizeOf(previousFlat);
            assertEquals(expected, vectors.ramBytesUsed()-before);
            long after = vectors.ramBytesUsed();
            vectors.diversityFunctionFor(1, VectorSimilarityFunction.DOT_PRODUCT);
            assertEquals(after, vectors.ramBytesUsed());
        }
    }

    @Test
    public void oneBitFlatAndBlockRepresentationsAreBothCharged() throws Exception {
        assumeTrue(VectorizationProvider.getInstance().getVectorUtilSupport().supportsAshMaskedLoad());
        var ash = model(1, 1); var vectors = new ASHVectors(ash, codes(ash, 17));
        long before = vectors.ramBytesUsed();
        long[] flat = vectors.getFlatPackedVectors();
        assertEquals(RamUsageEstimator.sizeOf(flat), vectors.ramBytesUsed()-before);
        var query = ash.landmarks[0];
        for (int block : new int[]{8,16,8}) {
            vectors.blockScorerFor(query, VectorSimilarityFunction.DOT_PRODUCT, block);
            long[] packed = (long[])field(vectors, "packedBits");
            // Query preparation may initialize provider weights; subtract model growth.
            long own = vectors.ramBytesUsed()-ash.ramBytesUsed();
            assertTrue(packed != null);
            assertTrue(own >= RamUsageEstimator.sizeOf(flat)+RamUsageEstimator.sizeOf(packed));
            long stable = vectors.ramBytesUsed();
            vectors.blockScorerFor(query, VectorSimilarityFunction.DOT_PRODUCT, block);
            assertEquals(stable, vectors.ramBytesUsed());
        }
    }

    @Test
    public void fusedAshSourcesChargeTheirOwnedVectorStorage() throws Exception {
        var ash = model(2, 1); var encoded = new ASHVectors(ash, codes(ash, 97));
        var graph = new TestUtil.RandomlyConnectedGraphIndex(97, 8, new Random(9182));
        var path = Files.createTempFile("ash-retained-source-", ".index");
        try {
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            try (var graphView = graph.getView(); var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                    .with(new FusedASH(8, ash, 8)).with(new InlineVectors(64)).build()) {
                suppliers.put(FeatureId.FUSED_ASH, node -> new FusedASH.State(graphView, encoded, node));
                suppliers.put(FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(ash.landmarks[0]));
                writer.write(suppliers);
            }
            try (var reader = new SimpleMappedReader.Supplier(path); var loaded = OnDiskGraphIndex.load(reader, 0);
                 var view = loaded.getView()) {
                var source = (FusedASH.FusedASHInlineSource)view.getInlineSourceFeatures().values().iterator().next();
                assertEquals(RamUsageEstimator.shallowSizeOf(source)+source.getVector().ramBytesUsed(), source.ramBytesUsed());
            }
        } finally { Files.deleteIfExists(path); }
    }


    @Test
    public void repeatedModelAndVectorAccountingDoesNotAllocate() throws Exception {
        java.lang.management.ThreadMXBean threadBean = java.lang.management.ManagementFactory.getThreadMXBean();
        assumeTrue(threadBean instanceof com.sun.management.ThreadMXBean);
        com.sun.management.ThreadMXBean bean = (com.sun.management.ThreadMXBean) threadBean;
        assumeTrue(bean.isThreadAllocatedMemorySupported());
        if (!bean.isThreadAllocatedMemoryEnabled()) bean.setThreadAllocatedMemoryEnabled(true);

        var ash = model(4, 1);
        var vectors = new ASHVectors(ash, codes(ash, 17));
        ash.stiefelTransform.projectQuery(new float[64], new float[33]);
        vectors.symmetricProjectionBlocks(8);
        vectors.diversityFunctionFor(0, VectorSimilarityFunction.DOT_PRODUCT);

        // Exercise exactly the measured call site before reading the allocation counter.
        for (int i = 0; i < 100; i++) accountingResult = readAccounting(ash, vectors, 2048);
        long thread = Thread.currentThread().getId();
        long before = bean.getThreadAllocatedBytes(thread);
        accountingResult = readAccounting(ash, vectors, 2048);
        long allocated = bean.getThreadAllocatedBytes(thread) - before;
        assertEquals("Polling retained ASH memory must not allocate", 0L, allocated);
    }

    private static long readAccounting(AsymmetricHashing ash, ASHVectors vectors, int count) {
        long bytes = 0;
        for (int i = 0; i < count; i++) bytes += ash.ramBytesUsed() + vectors.ramBytesUsed();
        return bytes;
    }

    @Test
    public void denseAndBlockedTransformBackingIsMeasuredWithoutMatrixCopies() throws Exception {
        for (int rows : new int[]{1,53,105}) {
            int cols = Math.min(rows, 53);
            var array = new AsymmetricHashing.StiefelTransform(new Array2DRowRealMatrix(rows, cols));
            var block = new AsymmetricHashing.StiefelTransform(new BlockRealMatrix(rows, cols));
            long arrayMatrix = RamUsageEstimator.shallowSizeOf(array.W)
                    + RamUsageEstimator.shallowSizeOf(((Array2DRowRealMatrix)array.W).getDataRef());
            for (double[] row : ((Array2DRowRealMatrix)array.W).getDataRef()) arrayMatrix += RamUsageEstimator.sizeOf(row);
            Field blocksField = BlockRealMatrix.class.getDeclaredField("blocks"); blocksField.setAccessible(true);
            double[][] blocks = (double[][])blocksField.get(block.W);
            long blockMatrix = RamUsageEstimator.shallowSizeOf(block.W)+RamUsageEstimator.shallowSizeOf(blocks);
            for (double[] data : blocks) blockMatrix += RamUsageEstimator.sizeOf(data);
            assertEquals(blockMatrix-arrayMatrix, (long)field(block,"heapBytes")-(long)field(array,"heapBytes"));
        }
    }
}
