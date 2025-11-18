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

import io.github.jbellis.jvector.example.util.*;
import io.github.jbellis.jvector.example.util.CompressorParameters.PQParameters;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * Tests GraphIndexes against vectors from a 2D dataset
 */
public class Bench2D {
    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        var mGrid = List.of(32); // List.of(16, 24, 32, 48, 64, 96, 128);
        var efConstructionGrid = List.of(100); // List.of(60, 80, 100, 120, 160, 200, 400, 600, 800);
        var topKGrid = Map.of(
                10, // topK
                List.of(1.0, 2.0, 5.0, 10.0, 20.0) // oq
        ); // rerankK = oq * topK
        var neighborOverflowGrid = List.of(1.2f); // List.of(1.2f, 2.0f);
        var addHierarchyGrid = List.of(true); // List.of(false, true);
        var refineFinalGraphGrid = List.of(true); // List.of(false, true);
        var usePruningGrid = List.of(false); // List.of(false, true);
        List<Function<DataSet, CompressorParameters>> buildCompression = Arrays.asList(__ -> CompressorParameters.NONE);
        List<Function<DataSet, CompressorParameters>> searchCompression = Arrays.asList(
                __ -> CompressorParameters.NONE,
                ds -> new PQParameters(ds.getDimension(), 256, true, UNWEIGHTED)
        );
        List<EnumSet<FeatureId>> featureSets = Arrays.asList(
                EnumSet.of(FeatureId.NVQ_VECTORS),
                EnumSet.of(FeatureId.INLINE_VECTORS)
        );

        // 2D grid, built and calculated at runtime
        var grid2d = DataSetCreator.create2DGrid(4_000_000, 10_000, 100);

        Grid.runAll(grid2d, mGrid, efConstructionGrid, neighborOverflowGrid, addHierarchyGrid, refineFinalGraphGrid,
                    featureSets, buildCompression, searchCompression, topKGrid, usePruningGrid);
    }
}
