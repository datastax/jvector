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

import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.FileNotFoundException;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;

/** The {@link FeatureProvider} for each {@link FeatureId} that {@link Grid} can benchmark. */
final class FeatureProviders {
    private static final Map<FeatureId, FeatureProvider> PROVIDERS = new EnumMap<>(FeatureId.class);

    static {
        register(new InlineVectorsProvider());
        register(new NVQProvider());
        register(new FusedPQProvider());
        register(new FusedASHProvider());
    }

    private FeatureProviders() {}

    private static void register(FeatureProvider provider) {
        if (PROVIDERS.putIfAbsent(provider.id(), provider) != null) {
            throw new IllegalStateException("Duplicate FeatureProvider for " + provider.id());
        }
    }

    /** The provider for {@code id}, or null if Grid cannot write that feature. */
    static FeatureProvider get(FeatureId id) {
        return PROVIDERS.get(id);
    }

    /**
     * The provider for {@code id}.
     *
     * @throws IllegalArgumentException if Grid has no provider for {@code id}, so results are never labeled
     * with a feature the index does not contain
     */
    static FeatureProvider require(FeatureId id) {
        var provider = PROVIDERS.get(id);
        if (provider == null) {
            throw new IllegalArgumentException("Grid has no FeatureProvider for " + id
                    + "; register one in FeatureProviders (supported: " + PROVIDERS.keySet() + ")");
        }
        return provider;
    }

    /** Whether any feature in {@code features} has a provider matching {@code test}; unknown features never match. */
    static boolean anyMatch(Set<FeatureId> features, Predicate<FeatureProvider> test) {
        for (var id : features) {
            var provider = PROVIDERS.get(id);
            if (provider != null && test.test(provider)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Creates the writer builder for one feature set and lets each feature's provider configure it.
     *
     * @param buildPQ PQ-encoded build vectors, or null for full-precision and non-PQ builds
     * @param metrics construction metrics sink, or null when not reporting
     */
    static FeatureWriteContext configureWriter(Set<FeatureId> features,
                                               ImmutableGraphIndex graph,
                                               Path outPath,
                                               RandomAccessVectorValues vectors,
                                               PQVectors buildPQ,
                                               ConstructionMetrics metrics) throws FileNotFoundException {
        return configureWriter(features, graph, outPath, vectors, buildPQ, null, null, metrics);
    }

    static FeatureWriteContext configureWriter(Set<FeatureId> features, ImmutableGraphIndex graph,
                                               Path outPath, RandomAccessVectorValues vectors,
                                               PQVectors buildPQ, ASHVectors buildASH,
                                               VectorFloat<?> inputMean, ConstructionMetrics metrics) throws FileNotFoundException {
        return configureWriter(features, graph, outPath, vectors, buildPQ, buildASH, inputMean, metrics, null);
    }

    static FeatureWriteContext configureWriter(Set<FeatureId> features, ImmutableGraphIndex graph,
                                               Path outPath, RandomAccessVectorValues vectors,
                                               PQVectors buildPQ, ASHVectors buildASH, VectorFloat<?> inputMean,
                                               ConstructionMetrics metrics, String writerMode) throws FileNotFoundException {
        var writer = GraphBuilderFactory.writerBuilder(graph, outPath, vectors.size(), writerMode);
        var ctx = new FeatureWriteContext(writer, graph, vectors, buildPQ, buildASH, inputMean, metrics);
        for (var id : features) {
            require(id).configure(ctx);
        }
        return ctx;
    }
}
