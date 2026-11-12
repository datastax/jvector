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

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/** Source graphs for the compaction tests that store their vectors, or their output, outside the graph file. */
final class CompactionSources {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private CompactionSources() {
    }

    /**
     * Builds a hierarchical graph over {@code vecs} and writes it to {@code path} either with its
     * vectors inline or topology-only (inline vectors of dimension 0), optionally with FUSED_PQ.
     */
    static Path writeSource(Path path, List<VectorFloat<?>> vecs, int dimension, VectorSimilarityFunction vsf,
                            boolean storeVectors, boolean fusedPQ) throws IOException {
        ForkJoinPool pool = ForkJoinPool.commonPool();
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vecs, dimension);
        var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(ravv, vsf), dimension, 16, 100, 1.2f, 1.2f,
                                            false, true, pool, pool);
        for (int i = 0; i < vecs.size(); i++)
            builder.addGraphNode(i, vecs.get(i));
        builder.cleanup();
        var graph = builder.getGraph();

        VectorFloat<?> empty = vectorTypeSupport.createFloatVector(0);
        var writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, path)
                .withMapper(new OrdinalMapper.IdentityMapper(vecs.size() - 1))
                .with(new InlineVectors(storeVectors ? dimension : 0));
        PQVectors pqv = null;
        if (fusedPQ) {
            ProductQuantization pq = ProductQuantization.compute(ravv, 8, 256, true, UNWEIGHTED, pool, pool);
            pqv = (PQVectors) pq.encodeAll(ravv, pool);
            writerBuilder.with(new FusedPQ(graph.maxDegree(), pq));
        }
        try (var writer = writerBuilder.build()) {
            Map<FeatureId, IntFunction<Feature.State>> suppliers = new EnumMap<>(FeatureId.class);
            suppliers.put(FeatureId.INLINE_VECTORS, ord -> new InlineVectors.State(storeVectors ? ravv.getVector(ord) : empty));
            if (fusedPQ) {
                final PQVectors codes = pqv;
                suppliers.put(FeatureId.FUSED_PQ, ord -> new FusedPQ.State(graph.getView(), codes, ord));
            }
            writer.write(suppliers);
        }
        return path;
    }
}
