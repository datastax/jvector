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
import io.github.jbellis.jvector.example.IndexBuildStrategy.BuiltIndexes;
import io.github.jbellis.jvector.example.benchmarks.datasets.ByteDataSet;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Builds the graph on heap from int8 vectors with exact int8 scoring, then writes each feature set to the work
 * directory with the configured writer mode and page alignment. Int8 builds have no compression; the index cache
 * does not apply.
 */
final class Int8Build {
    private final Path workDirectory;
    private final ConstructionMetrics constructionMetrics;

    Int8Build(Path workDirectory, ConstructionMetrics constructionMetrics) {
        this.workDirectory = workDirectory;
        this.constructionMetrics = constructionMetrics;
    }

    BuiltIndexes build(List<? extends Set<FeatureId>> featureSets, BuildConfig build, ByteDataSet ds) throws IOException {
        var vectors = ds.getBaseRavv();
        var builder = GraphBuilderFactory.forByteVectors(vectors, ds.getSimilarityFunction(), build);
        long start = System.nanoTime();
        var graph = builder.build(vectors);
        double buildSeconds = (System.nanoTime() - start) / 1e9;
        System.out.format("Build (int8) M=%d overflow=%.2f ef=%d in %.2fs%n",
                          build.M, build.neighborOverflow, build.efConstruction, buildSeconds);
        for (int i = 0; i <= graph.getMaxLevel(); i++) {
            System.out.format("  L%d: %d nodes, %.2f avg degree%n", i, graph.size(i), graph.getAverageDegree(i));
        }

        Map<Set<FeatureId>, ImmutableGraphIndex> indexes = new LinkedHashMap<>();
        Map<Set<FeatureId>, Path> files = new LinkedHashMap<>();
        double flushSeconds = 0;
        int n = 0;
        for (var features : featureSets) {
            var path = workDirectory.resolve("graph" + n++);
            var ctx = FeatureProviders.configureWriter(features, graph, path, vectors, build.writerMode,
                    GraphBuilderFactory.constructionThreads(build), build.pageAlignment);
            start = System.nanoTime();
            try (var view = graph.getView(); var writer = ctx.buildWriter()) {
                writer.write(ctx.completedSuppliers(view, null));
            }
            flushSeconds += (System.nanoTime() - start) / 1e9;
            System.out.format("Wrote %s%n", features);
            indexes.put(features, OnDiskGraphIndex.load(ReaderSupplierFactory.open(path)));
            files.put(features, path);
        }
        constructionMetrics.graphBuildTimeS = buildSeconds;
        constructionMetrics.flushTimeS = flushSeconds;
        return new BuiltIndexes(indexes, files, buildSeconds + flushSeconds);
    }
}
