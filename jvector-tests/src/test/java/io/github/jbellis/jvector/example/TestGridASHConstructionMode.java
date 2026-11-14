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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.OnDiskGraphIndexCache;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Assume;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.stream.Collectors;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;

/** Exercises Grid's insertion policy, not just the printed construction-mode label. */
public class TestGridASHConstructionMode {
    private static final String PROPERTY = "jvector.bench.ashConstruction";

    @Test(timeout = 120_000)
    public void defaultBuildMatchesExplicitSymmetricAllAndLegacyOverrideStillWorks() throws Exception {
        // Sequential insertion makes the serialized graph a deterministic comparison.
        Assume.assumeTrue("run with -Djvector.physical_core_count=1",
                PhysicalCoreExecutor.getPhysicalCoreCount() == 1);
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(42);
        var list = new ArrayList<VectorFloat<?>>();
        for (int n = 0; n < 256; n++) {
            var vector = vts.createFloatVector(32);
            for (int d = 0; d < 32; d++) vector.set(d, random.nextFloat() - 0.4f);
            VectorUtil.l2normalize(vector);
            list.add(vector);
        }
        var vectors = new ListRandomAccessVectorValues(list, 32);
        String name = "ash-construction-mode-" + UUID.randomUUID();
        DataSet ds = new DataSet() {
            public int getDimension() { return 32; }
            public String getName() { return name; }
            public VectorSimilarityFunction getSimilarityFunction() { return VectorSimilarityFunction.DOT_PRODUCT; }
            public RandomAccessVectorValues getBaseRavv() { return vectors; }
            public List<VectorFloat<?>> getBaseVectors() { return list; }
            public List<VectorFloat<?>> getQueryVectors() { return List.of(list.get(0)); }
            public List<List<Integer>> getGroundTruth() { return List.of(List.of(0)); }
        };
        var params = new CompressorParameters.ASHParameters(AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + 16 * 2, 1, 2);
        String previous = System.getProperty(PROPERTY);
        Path work = Files.createTempDirectory("ash-construction-mode");
        try {
            byte[] defaultGraph = build(ds, params, work.resolve("default"), null);
            byte[] explicitGraph = build(ds, params, work.resolve("explicit"), "symmetric-all");
            assertArrayEquals("unset property must use encoded queries for insertion", explicitGraph, defaultGraph);
            byte[] legacyGraph = build(ds, params, work.resolve("legacy"), "symmetric");
            assertFalse("the legacy override must still select raw-query insertion",
                    Arrays.equals(defaultGraph, legacyGraph));
        } finally {
            if (previous == null) System.clearProperty(PROPERTY);
            else System.setProperty(PROPERTY, previous);
            TestUtil.deleteQuietly(work);
            Files.deleteIfExists(Path.of("compressor_cache").resolve(params.idStringFor(ds)));
        }
    }

    private static byte[] build(DataSet ds, CompressorParameters params, Path work, String mode) throws Exception {
        if (mode == null) System.clearProperty(PROPERTY);
        else System.setProperty(PROPERTY, mode);
        Files.createDirectories(work);
        Path saved = work.resolve("index_cache");
        Grid.runOneGraph(OnDiskGraphIndexCache.initialize(saved),
                List.of(EnumSet.of(FeatureId.NVQ_VECTORS, FeatureId.FUSED_ASH)),
                8, 32, 1.2f, false, false, ignored -> params, List.of(),
                Map.of(10, List.of(1.0)), List.of(false), RunArtifacts.disabled(), ds, work);
        try (var files = Files.list(saved)) {
            var indexes = files.filter(p -> p.getFileName().toString().startsWith("graph_"))
                    .collect(Collectors.toList());
            if (indexes.size() != 1) throw new AssertionError("Expected one freshly built index: " + indexes);
            return Files.readAllBytes(indexes.get(0));
        }
    }
}
