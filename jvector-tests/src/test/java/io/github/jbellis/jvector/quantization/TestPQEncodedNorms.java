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
import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * A PQ model knows the largest reconstruction length of the vectors it holds codes for, keeps it through a save and
 * load at format version 7 or later, and reports it unknown when loaded from an older format.
 */
public class TestPQEncodedNorms {
    @Test
    public void largestLengthIsSavedWithTheModelAndUnknownWithoutIt() throws Exception {
        var random = new Random(29);
        int n = 300, dim = 16;
        List<VectorFloat<?>> base = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            var v = TestUtil.randomVector(random, dim);
            VectorUtil.scale(v, (float) Math.pow(10, -1 + 2 * random.nextDouble()));
            base.add(v);
        }
        var ravv = new ListRandomAccessVectorValues(base, dim);
        var pq = ProductQuantization.compute(ravv, 4, 32, false);
        var pqv = pq.encodeAll(ravv, ForkJoinPool.commonPool());
        float expected = 0;
        for (int i = 0; i < n; i++) expected = Math.max(expected, (float) Math.sqrt(pq.reconstructedSquaredNorm(pqv.get(i), 0)));
        assertEquals(expected, pq.maxEncodedNorm(), 0f);
        assertEquals(expected, pqv.maxReconstructedNorm(), 0f);

        // Saved and loaded at the current version, the model keeps it, and the header size accounts for it.
        var current = roundTrip(pq, OnDiskGraphIndex.CURRENT_VERSION);
        assertEquals(expected, current.maxEncodedNorm(), 0f);

        // An older format did not store it: unknown (infinite), and it stays unknown after encoding more, since the
        // lengths of the codes encoded before cannot be known. Dot-product searches then keep the standard rule.
        var older = roundTrip(pq, ProductQuantization.MAX_NORM_VERSION - 1);
        assertEquals(Float.POSITIVE_INFINITY, older.maxEncodedNorm(), 0f);
        older.encode(base.get(0));
        assertEquals(Float.POSITIVE_INFINITY, older.maxEncodedNorm(), 0f);
        var query = TestUtil.randomVector(random, dim);
        var olderVectors = older.encodeAll(ravv, ForkJoinPool.commonPool());
        assertTrue(!Float.isFinite(olderVectors.precomputedScoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT)
                .dotProductBound()));
        // Unknown survives a save and load at the current version too.
        assertEquals(Float.POSITIVE_INFINITY, roundTrip(older, OnDiskGraphIndex.CURRENT_VERSION).maxEncodedNorm(), 0f);

        // Codes wrapped with another instance of the model are counted in it.
        var fresh = roundTrip(pq, OnDiskGraphIndex.CURRENT_VERSION);
        var empty = new ProductQuantization(fresh.codebooks, fresh.getClusterCount(), fresh.subvectorSizesAndOffsets,
                fresh.globalCentroid, fresh.anisotropicThreshold);
        // Nothing recorded yet: no bound, so a model saved now does not claim one.
        assertTrue(Float.isNaN(empty.maxEncodedNorm()));
        assertTrue(Float.isNaN(roundTrip(empty, OnDiskGraphIndex.CURRENT_VERSION).maxEncodedNorm()));
        var codes = new ByteSequence<?>[n];
        for (int i = 0; i < n; i++) codes[i] = pqv.get(i).copy();
        empty.createCompressedVectors(codes);
        assertEquals(expected, empty.maxEncodedNorm(), 0f);
    }

    private static ProductQuantization roundTrip(ProductQuantization pq, int version) throws Exception {
        File file = File.createTempFile("pq-norm", ".bin");
        file.deleteOnExit();
        try (var writer = new SimpleWriter(file.toPath())) {
            pq.write(writer, version);
        }
        assertEquals("version " + version, pq.compressorSize(version), file.length());
        try (var readers = new SimpleMappedReader.Supplier(file.toPath())) {
            return ProductQuantization.load(readers.get());
        }
    }
}
