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
import io.github.jbellis.jvector.example.util.OnDiskGraphIndexCache;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.MutableASHVectors;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;

/**
 * Saves an in-memory index, its graph and its traversal codes, in the index cache, and reloads it into memory on a
 * later run with the same settings, so later JVMs can query the same index (for example with fewer cores). A cache
 * miss builds the index with the wrapped strategy and saves it.
 */
final class SavedInMemoryBuild implements IndexBuildStrategy {
    private final OnDiskGraphIndexCache cache;
    private final String identity;
    private final Supplier<IndexBuildStrategy> fresh;
    private final Supplier<CompressedVectors> freshCodes;
    private final boolean codesBuiltByIndex;

    /**
     * @param identity          what besides the graph settings determines the index, such as its compression
     * @param fresh             builds the index on a cache miss
     * @param freshCodes        the codes a fresh build encoded, when it does not report them itself; may return null
     * @param codesBuiltByIndex whether searches must use the saved codes as they are, as a streaming index's are
     */
    SavedInMemoryBuild(OnDiskGraphIndexCache cache, String identity, Supplier<IndexBuildStrategy> fresh,
                       Supplier<CompressedVectors> freshCodes, boolean codesBuiltByIndex) {
        this.cache = cache;
        this.identity = identity;
        this.fresh = fresh;
        this.freshCodes = freshCodes;
        this.codesBuiltByIndex = codesBuiltByIndex;
    }

    @Override
    @SuppressWarnings("deprecation") // OnHeapGraphIndex.save and load are the library's in-memory graph format
    public BuiltIndexes build(List<? extends Set<FeatureId>> featureSets, BuildConfig build, DataSet ds) throws IOException {
        var key = cache.keyWithCompressorId(ds.getName(), Set.of(), build.M, build.efConstruction,
                build.neighborOverflow, build.alpha, build.addHierarchy, build.refineFinalGraph, "inMemory:" + identity,
                ds.getBaseRavv().size(), ds.getDimension(), 0).withUpperLayerDegree(build.upperLayerDegree);
        var path = cache.resolve(key).finalPath;
        if (Files.exists(path)) {
            long start = System.nanoTime();
            OnHeapGraphIndex graph;
            try (var supplier = ReaderSupplierFactory.open(path); var in = supplier.get()) {
                graph = OnHeapGraphIndex.load(in, ds.getDimension(), build.neighborOverflow, null);
            }
            System.out.printf("%s: Using cached in-memory graph index, loaded in %.2f s%n", ds.getName(),
                    (System.nanoTime() - start) / 1e9);
            var codes = StoredCodes.load(path, graph.getIdUpperBound());
            Map<Set<FeatureId>, StoredCodes.Loaded> stored = codes == null ? Map.of()
                    : Map.of(Set.of(), codesBuiltByIndex ? codes.builtByIndex() : codes);
            return new BuiltIndexes(Map.of(Set.of(), graph), Map.of(), null, stored);
        }

        var built = fresh.get().build(featureSets, build, ds);
        var graph = (OnHeapGraphIndex) built.indexes.get(Set.of());
        var loaded = built.storedCodes.get(Set.of());
        CompressedVectors codes = loaded != null ? loaded.codes : freshCodes.get();
        if (codes instanceof MutableASHVectors) codes = ((MutableASHVectors) codes).snapshot();
        try (var handle = cache.beginWrite(key, OnDiskGraphIndexCache.Overwrite.ALLOW)) {
            try (var out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(handle.writePath())))) {
                graph.save(out);
            }
            // The codes are in place before the graph is committed, so a committed graph never sits beside codes
            // from an earlier build.
            if (codes != null && StoredCodes.supports(codes)) {
                StoredCodes.write(codes, handle.writePath());
                StoredCodes.move(handle.writePath(), handle.finalPath());
            } else {
                StoredCodes.delete(handle.finalPath());
            }
            handle.commit();
            System.out.printf("%s: Saved the in-memory graph index to the cache%n", ds.getName());
        }
        return built;
    }
}
