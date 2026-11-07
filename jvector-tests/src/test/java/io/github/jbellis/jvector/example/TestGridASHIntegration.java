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

import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.yaml.RunConfig;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.SearchParameters;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import org.junit.Test;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import static org.junit.Assert.*;

/** The refactored strategy, feature provider, queued queries and CSV listener must work together. */
public class TestGridASHIntegration {
    @Rule public final TemporaryFolder temporary = new TemporaryFolder();

    @Test(timeout = 120_000)
    public void gridSpecBuildsFusedASHAndRunsQueuedQueriesWithPrecisionMetadata() throws Exception {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(19);
        var vectors = new ArrayList<io.github.jbellis.jvector.vector.types.VectorFloat<?>>();
        for (int i = 0; i < 128; i++) {
            var v = vts.createFloatVector(32);
            for (int d = 0; d < 32; d++) v.set(d, random.nextFloat() - 0.5f);
            vectors.add(v);
        }
        var data = new SimpleDataSet("grid-ash-merge-" + UUID.randomUUID(), VectorSimilarityFunction.DOT_PRODUCT,
                vectors, List.of(vectors.get(0)), List.of(List.of(0)));
        var parameters = new CompressorParameters.ASHParameters(AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + 16 * 2, 1, 2);
        var spec = GridSpec.fromGrids(List.of(false), List.of(false), List.of(8), List.of(1.2f), List.of(32),
                List.of(ignored -> parameters), List.of(EnumSet.of(FeatureId.INLINE_VECTORS, FeatureId.FUSED_ASH)),
                SearchConfig.builder().searchCompressors(List.of(ignored -> CompressorParameters.NONE))
                        .topKGrid(Map.of(1, List.of(1.0, 2.0))).usePruningGrid(List.of(false)).build(), false);
        var config = new RunConfig();
        config.benchmarks = Map.of("accuracy", List.of("recall"));
        config.queryExecution.mode = "queued";
        config.logging = new RunConfig.RunLogging();
        config.logging.type = "csv";
        config.logging.dir = temporary.getRoot().toString();
        config.logging.benchmarks = config.benchmarks;
        String previous = System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY);
        var originalOut = System.out;
        var output = new ByteArrayOutputStream();
        try {
            System.setProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY, "fp32");
            var indexConfig = new MultiConfig();
            indexConfig.search = new SearchParameters();
            indexConfig.search.topKOverquery = Map.of(1, List.of(1.0, 2.0));
            var artifacts = RunArtifacts.open(config, List.of(indexConfig));
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            Grid.run(data, spec, artifacts);
            var lines = Files.readAllLines(artifacts.experimentsWriter().path());
            assertEquals(3, lines.size());
            var headers = Arrays.asList(lines.get(0).split(","));
            for (var line : lines.subList(1, lines.size())) {
                var cells = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
                assertEquals(headers.size(), cells.length);
                assertEquals("fp32", cells[headers.indexOf("ash_query_projection_requested")]);
                assertEquals("fp32", cells[headers.indexOf("ash_query_projection_effective")]);
                assertTrue(cells[headers.indexOf("feature_set")].contains("FUSED_ASH"));
            }
            var console = output.toString(StandardCharsets.UTF_8);
            assertTrue(console, console.contains("executionMode") && console.contains("queued"));
            assertTrue(console, console.contains("batchPreparation"));
        } finally {
            System.setOut(originalOut);
            if (previous == null) System.clearProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY);
            else System.setProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY, previous);
            Files.deleteIfExists(Path.of("pq_cache").resolve(parameters.idStringFor(data)));
        }
    }
}
