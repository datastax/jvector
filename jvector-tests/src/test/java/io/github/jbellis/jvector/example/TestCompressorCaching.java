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

package io.github.jbellis.jvector.example;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Test;

import java.nio.file.Files;
import java.util.List;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.*;

/** Grid's compressor cache must read each compressor back with its own format. */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestCompressorCaching extends RandomizedTest {
    private static VectorCompressor<?> roundTrip(CompressorParameters cp, VectorCompressor<?> compressor) throws Exception {
        var path = Files.createTempFile("compressor-cache", ".bin");
        try {
            // Grid.saveToCache writes with the current version.
            try (var writer = new BufferedRandomAccessWriter(path)) {
                compressor.write(writer, OnDiskGraphIndex.CURRENT_VERSION);
            }
            try (var readerSupplier = ReaderSupplierFactory.open(path);
                 var reader = readerSupplier.get()) {
                return cp.loadCompressor(reader);
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

    private static SimpleDataSet dataSet() {
        var base = createRandomVectors(256, 8);
        var queries = createRandomVectors(1, 8);
        return new SimpleDataSet("ds", VectorSimilarityFunction.EUCLIDEAN, base, queries, List.of(List.of(0)));
    }

    @Test
    public void pqLoadsAsPQ() throws Exception {
        var ds = dataSet();
        var cp = new CompressorParameters.PQParameters(2, 16, false, -1f);
        var pq = cp.computeCompressor(ds);
        var loaded = roundTrip(cp, pq);
        assertTrue(loaded.getClass().getName(), loaded instanceof ProductQuantization);
        assertEquals(pq, loaded);
    }

    @Test
    public void nvqLoadsAsNVQ() throws Exception {
        var ds = dataSet();
        var cp = new CompressorParameters.NVQParameters(2);
        assertEquals("NVQ_ds_2", cp.idStringFor(ds));
        var nvq = cp.computeCompressor(ds);
        var loaded = roundTrip(cp, nvq);
        assertTrue(loaded.getClass().getName(), loaded instanceof NVQuantization);
        assertEquals(nvq, loaded);
    }
    @Test
    public void ashLoadsAsASHAndCanEncodeVectors() throws Exception {
        var ds = dataSet();
        var cp = new CompressorParameters.ASHParameters(AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + 4 * 2, 1, 2);
        var ash = (AsymmetricHashing) cp.computeCompressor(ds);
        var loaded = roundTrip(cp, ash);
        assertTrue(loaded.getClass().getName(), loaded instanceof AsymmetricHashing);
        var restored = (AsymmetricHashing) loaded;
        assertEquals(ash.originalDimension, restored.originalDimension);
        assertEquals(ash.quantizedDim, restored.quantizedDim);
        assertEquals(ash.encodedBits, restored.encodedBits);
        assertEquals(ash.bitsPerDimension, restored.bitsPerDimension);
        assertEquals(ds.getBaseRavv().size(), restored.encodeAll(ds.getBaseRavv()).count());
    }

    @Test
    public void ashTrainingWindowTrainsOnlyOnTheWindow() {
        var base = createRandomVectors(256, 8);
        var ds = new SimpleDataSet("ds", VectorSimilarityFunction.DOT_PRODUCT, base,
                createRandomVectors(1, 8), List.of(List.of(0)));
        int start = 16, count = 64, bits = AsymmetricHashing.HEADER_BITS + 4 * 2;
        var cp = new CompressorParameters.ASHParameters(AsymmetricHashing.RANDOM, bits, 1, 2, start, count);
        cp.validateFor(ds, false, null);
        assertTrue(cp.idStringFor(ds), cp.idStringFor(ds).endsWith("_train16+64"));

        var ash = (AsymmetricHashing) cp.computeCompressor(ds);
        // One landmark is the mean of the normalized training vectors, here exactly the window.
        double[] mean = new double[8];
        for (int i = start; i < start + count; i++) {
            var v = base.get(i);
            double norm = Math.sqrt(io.github.jbellis.jvector.vector.VectorUtil.dotProduct(v, v));
            for (int d = 0; d < 8; d++) mean[d] += v.get(d) / norm / count;
        }
        for (int d = 0; d < 8; d++) assertEquals(mean[d], ash.landmarks[0].get(d), 1e-5);
        // The model cannot show which window trained it, so it is never matched as a stored model.
        assertFalse(cp.describes(ash));
        assertEquals(ds.getBaseRavv().size(), ash.encodeAll(ds.getBaseRavv()).count());

        var whole = new CompressorParameters.ASHParameters(AsymmetricHashing.RANDOM, bits, 1, 2);
        assertFalse(whole.idStringFor(ds).contains("_train"));
        assertThrows(IllegalArgumentException.class, () ->
                new CompressorParameters.ASHParameters(AsymmetricHashing.RANDOM, bits, 1, 2, 250, 64)
                        .validateFor(ds, false, null));
    }

}
