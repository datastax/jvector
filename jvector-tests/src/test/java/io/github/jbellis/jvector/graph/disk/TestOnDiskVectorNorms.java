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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * On-disk graphs store the largest length of their full-precision vectors from format version 7, so dot-product
 * searches over them can stop adaptively; older formats leave it unknown, and those searches stop by the standard rule.
 */
public class TestOnDiskVectorNorms {
    private static final int N = 300, DIM = 16;

    @Test
    public void fullPrecisionHeadersStoreTheLargestLength() throws Exception {
        var random = new Random(37);
        List<VectorFloat<?>> vectors = new ArrayList<>();
        float maxNorm = 0;
        for (int i = 0; i < N; i++) {
            var v = TestUtil.randomVector(random, DIM);
            VectorUtil.scale(v, (float) Math.pow(10, -1 + 2 * random.nextDouble()));
            vectors.add(v);
            maxNorm = Math.max(maxNorm, (float) Math.sqrt(VectorUtil.dotProduct(v, v)));
        }
        var ravv = new ListRandomAccessVectorValues(vectors, DIM);
        var graph = new TestUtil.RandomlyConnectedGraphIndex(N, 8, random);
        var query = TestUtil.randomVector(random, DIM);
        float queryNorm = (float) Math.sqrt(VectorUtil.dotProduct(query, query));
        Path directory = Files.createTempDirectory("ondisk-norms");

        for (int version : new int[] {OnDiskGraphIndex.CURRENT_VERSION, 6}) {
            var path = directory.resolve("inline-" + version);
            try (var writer = new OnDiskGraphIndexWriter.Builder(graph, path).withVersion(version)
                    .with(new InlineVectors(DIM)).build()) {
                writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                        node -> new InlineVectors.State(ravv.getVector(node))));
            }
            try (var readers = new SimpleMappedReader.Supplier(path);
                 var index = OnDiskGraphIndex.load(readers);
                 var view = index.getView()) {
                float stored = ((InlineVectors) index.getFeatures().get(FeatureId.INLINE_VECTORS)).maxVectorNorm();
                float bound = view.rerankerFor(query, VectorSimilarityFunction.DOT_PRODUCT).dotProductBound();
                if (version >= 7) {
                    assertEquals(maxNorm, stored, maxNorm * 1e-6f);
                    assertEquals(queryNorm * maxNorm, bound, queryNorm * maxNorm * 1e-5f);
                } else {
                    assertTrue("version " + version + ": " + stored, Float.isNaN(stored));
                    assertTrue(Float.isNaN(bound));
                }
            }
        }

        var path = directory.resolve("separated");
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, path).with(new SeparatedVectors(DIM, 0L)).build()) {
            writer.write(Feature.singleStateFactory(FeatureId.SEPARATED_VECTORS,
                    node -> new InlineVectors.State(ravv.getVector(node))));
        }
        try (var readers = new SimpleMappedReader.Supplier(path);
             var index = OnDiskGraphIndex.load(readers)) {
            float stored = ((SeparatedVectors) index.getFeatures().get(FeatureId.SEPARATED_VECTORS)).maxVectorNorm();
            assertEquals(maxNorm, stored, maxNorm * 1e-6f);
        }
    }
}
