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
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.MultiConfiguredSystem;
import io.github.jbellis.jvector.example.util.MultiSharder;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.DirectoryNotEmptyException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

/**
 * Tests a grid of configurations against a dataset
 */
public class MultiGrid {

    private static final String pqCacheDir = "pq_cache";

    private static final String dirPrefix = "BenchGraphDir";

    static void runAll(int shards,
                       DataSet ds,
                       List<Integer> mGrid,
                       List<Integer> efConstructionGrid,
                       List<Float> neighborOverflowGrid,
                       List<Boolean> addHierarchyGrid,
                       List<Boolean> refineFinalGraphGrid,
                       List<? extends Set<FeatureId>> featureSets,
                       List<Function<DataSet, CompressorParameters>> buildCompressors,
                       List<Function<DataSet, CompressorParameters>> compressionGrid,
                       Map<Integer, List<Double>> topKGrid,
                       List<Boolean> usePruningGrid) throws IOException
    {
        var testDirectory = Files.createTempDirectory(dirPrefix);
        try {
            for (var addHierarchy :  addHierarchyGrid) {
                for (var refineFinalGraph : refineFinalGraphGrid) {
                    for (int M : mGrid) {
                        for (float neighborOverflow : neighborOverflowGrid) {
                            for (int efC : efConstructionGrid) {
                                for (var bc : buildCompressors) {
                                    runOneGraph(shards, featureSets, M, efC, neighborOverflow, addHierarchy, refineFinalGraph, bc, compressionGrid, topKGrid, usePruningGrid, ds, testDirectory);
                                }
                            }
                        }
                    }
                }
            }
        } finally {
            try
            {
                Files.delete(testDirectory);
            } catch (DirectoryNotEmptyException e) {
                // something broke, we're almost certainly in the middle of another exception being thrown,
                // so if we don't swallow this one it will mask the original exception
            }

            cachedCompressors.clear();
        }
    }

    static void runOneGraph(int shards,
                            List<? extends Set<FeatureId>> featureSets,
                            int M,
                            int efConstruction,
                            float neighborOverflow,
                            boolean addHierarchy,
                            boolean refineFinalGraph,
                            Function<DataSet, CompressorParameters> buildCompressorFunction,
                            List<Function<DataSet, CompressorParameters>> compressionGrid,
                            Map<Integer, List<Double>> topKGrid,
                            List<Boolean> usePruningGrid,
                            DataSet ds,
                            Path testDirectory) throws IOException
    {
        var sharder = new MultiSharder.ContiguousSharder(ds.getBaseRavv(), shards);

        List<String> graphNamePrefices = new ArrayList<>();
        for (int shard = 0; shard < shards; shard++) {
            graphNamePrefices.add("graph-shard" + shard + "-");
        }

        Map<Set<FeatureId>, List<GraphIndex>> shardedIndices = new HashMap<>();

        for (int shard = 0; shard < shards; shard++) {
            var shardRavv = sharder.getShard(shard);
            System.out.println("Building shard " + (shard + 1) + " of " + shards + " with " + shardRavv.size() + " vectors");
            Map<Set<FeatureId>, GraphIndex> indices;
            var buildCompressor = getCompressor(buildCompressorFunction, ds, shardRavv, shard);
            if (buildCompressor == null) {
                indices = Grid.buildInMemory(
                        shardRavv, ds.similarityFunction, featureSets,
                        M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph,
                        testDirectory, graphNamePrefices.get(shard)
                );
            } else {
                indices = Grid.buildOnDisk(
                        shardRavv, ds.similarityFunction, featureSets,
                        M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph, buildCompressor,
                        testDirectory, graphNamePrefices.get(shard)
                );
            }
            indices.forEach((features, index) ->
                shardedIndices.computeIfAbsent(features, k -> new ArrayList<>()).add(index)
            );
        }

        try {
            for (var cpSupplier : compressionGrid) {
                List<CompressedVectors> cvs = new ArrayList<>(shards);
                for (int shard = 0; shard < shards; shard++) {
                    var shardRavv = sharder.getShard(shard);

                    var compressor = getCompressor(cpSupplier, ds, shardRavv, shard);
                    CompressedVectors cv;
                    if (compressor == null) {
                        cv = null;
                        System.out.format("Uncompressed vectors%n");
                    } else {
                        long start = System.nanoTime();
                        cv = compressor.encodeAll(shardRavv);
                        System.out.format("%s encoded %d vectors [%.2f MB] in %.2fs%n", compressor, shardRavv.size(), (cv.ramBytesUsed() / 1024f / 1024f), (System.nanoTime() - start) / 1_000_000_000.0);
                    }

                    cvs.add(cv);
                }

                shardedIndices.forEach((features, indices) -> {
                    try (var cs = new MultiConfiguredSystem(ds, sharder, indices, cvs, sharder.getConverter())) {
                        Grid.testConfiguration(cs, topKGrid, usePruningGrid, M, efConstruction, neighborOverflow, addHierarchy);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                });
            }
            for (var indices : shardedIndices.values()) {
                for (var index : indices) {
                    index.close();
                }
            }
        } finally {
            for (int shard = 0; shard < shards; shard++) {
                for (int n = 0; n < featureSets.size(); n++) {
                    Files.deleteIfExists(testDirectory.resolve(graphNamePrefices.get(shard) + n));
                }
            }
        }
    }

    // avoid recomputing the compressor repeatedly (this is a relatively small memory footprint)
    static final Map<String, VectorCompressor<?>> cachedCompressors = new IdentityHashMap<>();

    private static VectorCompressor<?> getCompressor(Function<DataSet, CompressorParameters> cpSupplier,
                                                     DataSet ds,
                                                     RandomAccessVectorValues ravv,
                                                     int shard) {
        var cp = cpSupplier.apply(ds);
        if (!cp.supportsCaching()) {
            return cp.computeCompressor(ravv);
        }

        var fname = cp.idStringFor(ds.name + "-shard" + shard);
        return cachedCompressors.computeIfAbsent(fname, __ -> {
            var path = Paths.get(pqCacheDir).resolve(fname);
            if (path.toFile().exists()) {
                try {
                    try (var readerSupplier = ReaderSupplierFactory.open(path)) {
                        try (var rar = readerSupplier.get()) {
                            var pq = ProductQuantization.load(rar);
                            System.out.format("%s loaded from %s%n", pq, fname);
                            return pq;
                        }
                    }
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }

            var start = System.nanoTime();
            var compressor = cp.computeCompressor(ravv);
            System.out.format("%s build in %.2fs,%n", compressor, (System.nanoTime() - start) / 1_000_000_000.0);
            if (cp.supportsCaching()) {
                try {
                    Files.createDirectories(path.getParent());
                    try (var writer = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
                        compressor.write(writer, OnDiskGraphIndex.CURRENT_VERSION);
                    }
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
            return compressor;
        });
    }
}
