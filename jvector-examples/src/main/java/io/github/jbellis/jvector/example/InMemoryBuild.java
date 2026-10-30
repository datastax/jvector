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

import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.IntFunction;

/**
 * Builds the graph on heap from full-precision vectors, then writes each feature set to the work directory.
 * Used when no build compressor is configured; the index cache does not apply.
 */
final class InMemoryBuild implements IndexBuildStrategy {
    private final Path workDirectory;

    InMemoryBuild(Path workDirectory) {
        this.workDirectory = workDirectory;
    }

    @Override
    public BuiltIndexes build(List<? extends Set<FeatureId>> featureSets, BuildConfig build, DataSet ds) throws IOException {
        var floatVectors = ds.getBaseRavv();
        Map<Set<FeatureId>, ImmutableGraphIndex> indexes = new LinkedHashMap<>();
        var builder = GraphBuilderFactory.forFullPrecision(floatVectors, ds.getSimilarityFunction(), build);
        long start = System.nanoTime();
        var onHeapGraph = builder.build(floatVectors);
        double buildTimeS = (System.nanoTime() - start) / 1_000_000_000.0;
        System.out.format("Build (%s) M=%d overflow=%.2f ef=%d in %.2fs%n",
                          "full res",
                          build.M,
                          build.neighborOverflow,
                          build.efConstruction,
                          buildTimeS);
        for (int i = 0; i <= onHeapGraph.getMaxLevel(); i++) {
            System.out.format("  L%d: %d nodes, %.2f avg degree%n",
                              i,
                              onHeapGraph.size(i),
                              onHeapGraph.getAverageDegree(i));
        }
        int n = 0;
        for (var features : featureSets) {
            if (FeatureProviders.anyMatch(features, p -> !p.supportsInMemoryBuild())) {
                System.out.println("Skipping " + features + " when building in memory: it needs a compressed build");
                continue;
            }
            var graphPath = workDirectory.resolve("graph" + n++);
            var ctx = FeatureProviders.configureWriter(features, onHeapGraph, graphPath, floatVectors, null, null);
            // The graph is already complete, so after-build suppliers can share one view for the write.
            try (var writer = ctx.buildWriter();
                 var view = ctx.afterBuildSuppliers().isEmpty() ? null : onHeapGraph.getView()) {
                start = System.nanoTime();
                writer.write(writeSuppliers(ctx, view));
                System.out.format("Wrote %s in %.2fs%n", features, (System.nanoTime() - start) / 1_000_000_000.0);
            }

            var index = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graphPath));
            indexes.put(features, index);
        }
        return new BuiltIndexes(indexes, new LinkedHashMap<>(), buildTimeS);
    }

    /** @param view a view of the completed graph, or null when no feature needs one */
    private static Map<FeatureId, IntFunction<Feature.State>> writeSuppliers(FeatureWriteContext ctx,
                                                                              ImmutableGraphIndex.View view) {
        if (view == null) {
            return ctx.ordinalSuppliers();
        }
        Map<FeatureId, IntFunction<Feature.State>> suppliers = new EnumMap<>(FeatureId.class);
        suppliers.putAll(ctx.ordinalSuppliers());
        ctx.afterBuildSuppliers().forEach((id, supplier) -> suppliers.put(id, supplier.apply(view)));
        return suppliers;
    }
}
