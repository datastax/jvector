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

package io.github.jbellis.jvector.example.diagnostics;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/** Benchmark-only adapter; asymmetric construction retains access to original vectors. */
public final class ASHBuildScoreExperiment {
    private ASHBuildScoreExperiment() {}
    public static BuildScoreProvider create(VectorSimilarityFunction vsf, ASHVectors encoded,
                                            RandomAccessVectorValues raw) {
        String mode=System.getProperty("jvector.bench.ashConstruction","symmetric");
        System.out.println("ASH construction scoring mode: " + mode);
        if (mode.equals("symmetric") || mode.equals("symmetric-all")) {
            return BuildScoreProvider.ashBuildScoreProvider(vsf,encoded);
        }
        if (mode.equals("symmetric-scalar")) {
            if (vsf != VectorSimilarityFunction.DOT_PRODUCT || encoded.getCompressor().landmarkCount != 1)
                throw new IllegalArgumentException("ASH construction requires DOT_PRODUCT and C=1");
            var symmetric = new io.github.jbellis.jvector.quantization.ASHSymmetricScorer(encoded,
                    io.github.jbellis.jvector.quantization.ASHSymmetricScorer.Kernel.SCALAR);
            return new BuildScoreProvider() {
                public boolean isExact() { return false; }
                public VectorFloat<?> approximateCentroid() { return encoded.getCompressor().landmarks[0].copy(); }
                public SearchScoreProvider searchProviderFor(VectorFloat<?> x) {
                    return new DefaultSearchScoreProvider(encoded.precomputedScoreFunctionFor(x,vsf),null,true);
                }
                public SearchScoreProvider searchProviderFor(int node) {
                    return new DefaultSearchScoreProvider(symmetric.scoreFunctionFor(node),null,true);
                }
                public SearchScoreProvider diversityProviderFor(int node) { return searchProviderFor(node); }
            };
        }
        if (!mode.equals("asymmetric") && !mode.equals("asymmetric-cached")) throw new IllegalArgumentException("Unknown ASH construction mode: " + mode);
        var views=raw.threadLocalSupplier();
        final SearchScoreProvider[] cached;
        if (mode.equals("asymmetric-cached")) {
            // ASH single-vector scorers own immutable query state and can be shared.
            cached = new SearchScoreProvider[encoded.count()];
            io.github.jbellis.jvector.util.PhysicalCoreExecutor.pool().submit(() ->
                    java.util.stream.IntStream.range(0, cached.length).parallel().forEach(node ->
                            cached[node] = new DefaultSearchScoreProvider(
                                    encoded.precomputedScoreFunctionFor(views.get().getVector(node), vsf), null, true)))
                    .join();
            long landmarks = encoded.getCompressor().landmarkCount;
            long floats = (long) cached.length * ((1L + landmarks) * encoded.getCompressor().quantizedDim + 2 * landmarks);
            System.out.printf("Cached asymmetric query arrays: %.3f MiB (array/object overhead additional)%n",
                    floats * Float.BYTES / 1048576.0);
        } else {
            cached = null;
        }
        return new BuildScoreProvider() {
            public boolean isExact() { return false; }
            public VectorFloat<?> approximateCentroid() { return encoded.getCompressor().landmarks[0].copy(); }
            public SearchScoreProvider searchProviderFor(VectorFloat<?> x) {
                return new DefaultSearchScoreProvider(encoded.precomputedScoreFunctionFor(x,vsf),null,true);
            }
            public SearchScoreProvider searchProviderFor(int node) { return cached == null ? searchProviderFor(views.get().getVector(node)) : cached[node]; }
            public SearchScoreProvider diversityProviderFor(int node) { return searchProviderFor(node); }
        };
    }
}
