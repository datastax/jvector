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
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.yaml.Compression;
import io.github.jbellis.jvector.example.yaml.ConstructionParameters;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.SearchParameters;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import org.junit.Test;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.junit.Assert.*;

public class TestGridSpec {
    private static SearchConfig search() {
        return SearchConfig.builder()
                .searchCompressors(List.of(ds -> CompressorParameters.NONE))
                .topKGrid(Map.of(10, List.of(1.0)))
                .usePruningGrid(List.of(true))
                .build();
    }

    private static String describe(BuildConfig b) {
        return b.addHierarchy + "/" + b.refineFinalGraph + "/" + b.M + "/" + b.neighborOverflow + "/"
                + b.efConstruction + "/" + b.buildCompressor;
    }

    @Test
    public void buildPointsMatchTheNestedLoopOrder() {
        List<Boolean> hierarchy = List.of(true, false);
        List<Boolean> refine = List.of(false, true);
        List<Integer> ms = List.of(8, 16, 32);
        List<Float> overflow = List.of(1.2f);
        List<Integer> efs = List.of(50, 100);
        Function<DataSet, CompressorParameters> none = ds -> CompressorParameters.NONE;
        Function<DataSet, CompressorParameters> bq = ds -> new CompressorParameters.BQParameters();
        List<Function<DataSet, CompressorParameters>> compressors = List.of(none, bq);

        var spec = GridSpec.builder()
                .sweep("addHierarchy", hierarchy, BuildConfig.Builder::addHierarchy)
                .sweep("refineFinalGraph", refine, BuildConfig.Builder::refineFinalGraph)
                .sweep("M", ms, BuildConfig.Builder::M)
                .sweep("neighborOverflow", overflow, BuildConfig.Builder::neighborOverflow)
                .sweep("efConstruction", efs, BuildConfig.Builder::efConstruction)
                .sweep("buildCompressor", compressors, BuildConfig.Builder::buildCompressor)
                .featureSets(List.of(EnumSet.of(FeatureId.INLINE_VECTORS)))
                .search(search())
                .build();

        // The loop nest Grid.runAll used before sweep dimensions.
        List<String> expected = new ArrayList<>();
        for (var h : hierarchy)
            for (var r : refine)
                for (var m : ms)
                    for (var o : overflow)
                        for (var ef : efs)
                            for (var bc : compressors)
                                expected.add(h + "/" + r + "/" + m + "/" + o + "/" + ef + "/" + bc);

        List<String> actual = new ArrayList<>();
        for (var point : spec.buildPoints()) {
            actual.add(describe(point));
            assertEquals(1.2f, point.alpha, 0f);
        }
        assertEquals(expected, actual);
    }

    @Test
    public void anEmptyDimensionProducesNoPoints() {
        var spec = GridSpec.builder()
                .sweep("M", List.of(16), BuildConfig.Builder::M)
                .sweep("efConstruction", List.<Integer>of(), BuildConfig.Builder::efConstruction)
                .featureSets(List.of(EnumSet.of(FeatureId.INLINE_VECTORS)))
                .search(search())
                .build();
        assertTrue(spec.buildPoints().isEmpty());
    }

    @Test
    public void unsetSweepValuesNameTheParameter() {
        var e = assertThrows(IllegalArgumentException.class,
                () -> GridSpec.builder().sweep("outDegree", null, BuildConfig.Builder::M));
        assertTrue(e.getMessage(), e.getMessage().contains("outDegree"));
    }

    @Test
    public void aMissingRequiredParameterNamesIt() {
        var spec = GridSpec.builder()
                .sweep("M", List.of(16), BuildConfig.Builder::M)
                .featureSets(List.of(EnumSet.of(FeatureId.INLINE_VECTORS)))
                .search(search())
                .build();
        var e = assertThrows(IllegalStateException.class, spec::buildPoints);
        assertTrue(e.getMessage(), e.getMessage().contains("efConstruction"));
    }

    @Test
    public void fromConfigMapsTheYamlSweep() {
        var none = new Compression();
        none.type = "None";
        var bq = new Compression();
        bq.type = "BQ";

        var construction = new ConstructionParameters();
        construction.outDegree = List.of(16, 32);
        construction.efConstruction = List.of(100);
        construction.neighborOverflow = List.of(1.5f);
        construction.addHierarchy = List.of(true, false);
        construction.refineFinalGraph = List.of(true);
        construction.reranking = List.of("NVQ", "FP");
        construction.fusedGraph = List.of(false);
        construction.compression = List.of(none, bq);
        construction.useSavedIndexIfExists = true;

        var searchParams = new SearchParameters();
        searchParams.compression = List.of(none);
        searchParams.topKOverquery = Map.of(10, List.of(1.0, 2.0));
        searchParams.useSearchPruning = List.of(false);

        var config = new MultiConfig();
        config.construction = construction;
        config.search = searchParams;

        var spec = GridSpec.fromConfig(config);
        assertTrue(spec.enableIndexCache);
        assertEquals(List.of(EnumSet.of(FeatureId.NVQ_VECTORS), EnumSet.of(FeatureId.INLINE_VECTORS)), spec.featureSets);
        assertEquals(Map.of(10, List.of(1.0, 2.0)), spec.search.topKGrid);
        assertEquals(List.of(false), spec.search.usePruningGrid);
        assertEquals(1, spec.search.searchCompressors.size());

        var points = spec.buildPoints();
        // addHierarchy varies slowest, then outDegree, then build compression fastest.
        List<String> actual = new ArrayList<>();
        for (var p : points) {
            actual.add(p.addHierarchy + "/" + p.M + "/" + p.buildCompressor.apply(null).getClass().getSimpleName());
            assertEquals(100, p.efConstruction);
            assertEquals(1.5f, p.neighborOverflow, 0f);
            assertTrue(p.refineFinalGraph);
        }
        assertEquals(List.of(
                "true/16/NoCompressionParameters", "true/16/BQParameters",
                "true/32/NoCompressionParameters", "true/32/BQParameters",
                "false/16/NoCompressionParameters", "false/16/BQParameters",
                "false/32/NoCompressionParameters", "false/32/BQParameters"), actual);
    }

    @Test
    public void fromConfigRejectsAnUnsetCacheFlag() {
        var none = new Compression();
        none.type = "None";
        var construction = new ConstructionParameters();
        construction.outDegree = List.of(16);
        construction.efConstruction = List.of(100);
        construction.neighborOverflow = List.of(1.2f);
        construction.addHierarchy = List.of(true);
        construction.refineFinalGraph = List.of(true);
        construction.reranking = List.of("FP");
        construction.fusedGraph = List.of(false);
        construction.compression = List.of(none);
        var searchParams = new SearchParameters();
        searchParams.compression = List.of(none);
        searchParams.topKOverquery = Map.of(10, List.of(1.0));
        searchParams.useSearchPruning = List.of(true);
        var config = new MultiConfig();
        config.construction = construction;
        config.search = searchParams;

        var e = assertThrows(NullPointerException.class, () -> GridSpec.fromConfig(config));
        assertTrue(e.getMessage(), e.getMessage().contains("useSavedIndexIfExists"));
    }
}
