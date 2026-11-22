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

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.OnDiskGraphIndexCache;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/// A compressed build that a feature cannot be written with must fail before any index file is
/// opened: the harness promises that a failed run leaves no partial index files behind.
public class TestCompressedBuildPreflight {
    @Test
    public void fusedPqWithoutTwoHundredFiftySixClustersFailsBeforeWriting() throws Exception {
        var ds = dataset("preflight-" + UUID.randomUUID(), 16, 128);
        var work = Files.createTempDirectory("compressed-build-preflight");
        var cacheDir = work.resolve("saved");
        var pqFile = Path.of("compressor_cache").resolve(new CompressorParameters.PQParameters(4, 16, false, -1f).idStringFor(ds));
        try {
            var cache = OnDiskGraphIndexCache.initialize(cacheDir);
            var e = assertThrows(IllegalArgumentException.class, () -> Grid.runOneGraph(cache,
                    List.of(EnumSet.of(FeatureId.INLINE_VECTORS), EnumSet.of(FeatureId.INLINE_VECTORS, FeatureId.FUSED_PQ)),
                    8, 32, 1.2f, false, false,
                    ignored -> new CompressorParameters.PQParameters(4, 16, false, -1f),
                    List.of(ignored -> CompressorParameters.NONE), Map.of(10, List.of(1.0)), List.of(false),
                    RunArtifacts.disabled(), ds, work));
            // The check must run before any writer is opened, so neither feature set may have a file,
            // not even the cache's temp file for the feature set configured before the fused one.
            try (var files = Files.list(cacheDir)) {
                assertEquals("Index files written before the build was rejected",
                        List.of(), files.map(p -> p.getFileName().toString()).collect(Collectors.toList()));
            }
            assertTrue(e.getMessage(), e.getMessage().contains("256 clusters"));
        } finally {
            Grid.cachedCompressors.clear();
            Files.deleteIfExists(pqFile);
            try (var files = Files.walk(work)) {
                for (var f : files.sorted(Comparator.reverseOrder()).collect(Collectors.toList())) Files.deleteIfExists(f);
            }
        }
    }

    private static DataSet dataset(String name, int dimension, int count) {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(7381);
        var vectors = new ArrayList<VectorFloat<?>>(count);
        for (int i = 0; i < count; i++) {
            var vector = types.createFloatVector(dimension);
            for (int j = 0; j < dimension; j++) vector.set(j, random.nextFloat());
            vectors.add(vector);
        }
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);
        return new DataSet() {
            @Override public int getDimension() { return dimension; }
            @Override public String getName() { return name; }
            @Override public VectorSimilarityFunction getSimilarityFunction() { return VectorSimilarityFunction.EUCLIDEAN; }
            @Override public RandomAccessVectorValues getBaseRavv() { return ravv; }
            @Override public List<VectorFloat<?>> getBaseVectors() { return vectors; }
            @Override public List<VectorFloat<?>> getQueryVectors() { return List.of(vectors.get(0)); }
            @Override public List<? extends List<Integer>> getGroundTruth() { return List.of(List.of(0)); }
        };
    }
}
