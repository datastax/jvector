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
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.nio.file.Files;
import java.util.EnumMap;
import java.util.Random;
import java.util.function.IntFunction;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Ownership and byte units for cached fused-PQ sources, independent of scoring kernels. */
public class TestFusedPQMemoryAccounting {
    @Test
    public void sourceIncludesItsObjectAndCodeStorage() {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        for (int size : new int[]{0, 1, 15, 16, 17, 31, 32, 33, 128, 384, 768}) {
            var code = types.createByteSequence(size);
            var source = new FusedPQ.FusedPQInlineSource(code);
            assertEquals("Code size " + size,
                    RamUsageEstimator.shallowSizeOf(source) + code.ramBytesUsed(), source.ramBytesUsed());
            assertTrue(source.ramBytesUsed() > code.length());
        }
    }

    @Test
    public void graphIncludesCachedSourceStorage() throws Exception {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        // Fixed codebooks avoid training: this test only serializes and loads source storage.
        var pq = new ProductQuantization(new VectorFloat<?>[]{
                types.createFloatVector(256 * 4), types.createFloatVector(256 * 4)},
                256, new int[][]{{4, 0}, {4, 4}}, null, KMeansPlusPlusClusterer.UNWEIGHTED);
        var graph = new TestUtil.RandomlyConnectedGraphIndex(37, 8, new Random(9182));
        var code = types.createByteSequence(2);
        var vector = types.createFloatVector(8);
        var path = Files.createTempFile("fused-pq-memory-", ".index");
        try {
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            try (var graphView = graph.getView();
                 var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                         .with(new FusedPQ(8, pq)).with(new InlineVectors(8)).build()) {
                suppliers.put(FeatureId.FUSED_PQ, node -> new FusedPQ.State(graphView, ignored -> code, node));
                suppliers.put(FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(vector));
                writer.write(suppliers);
            }
            try (var reader = new SimpleMappedReader.Supplier(path);
                 var loaded = OnDiskGraphIndex.load(reader, 0);
                 var view = loaded.getView()) {
                var sources = view.getInlineSourceFeatures();
                assertFalse("The entry point must have a cached source", sources.isEmpty());
                long sourceBytes = sources.values().stream().mapToLong(FusedFeature.InlineSource::ramBytesUsed).sum();
                long retained = loaded.ramBytesUsed();
                assertTrue(retained >= sourceBytes + RamUsageEstimator.shallowSizeOf(loaded));
                // Memory reporting is a cached read; it neither traverses nor changes the cache.
                for (int i = 0; i < 1000; i++) assertEquals(retained, loaded.ramBytesUsed());
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }
}
