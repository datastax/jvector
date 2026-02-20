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

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSetLoaderMFD;
import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.RunConfig;

import java.io.IOException;
import java.util.List;

public class HelloVectorWorld {
    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        String datasetName = "ada002-100k";

        // Dataset-tuned config (construction/search grids)
        MultiConfig config = MultiConfig.getDefaultConfig(datasetName);

        // Run-level policy config (benchmarks/console/logging + run metadata)
        RunConfig runCfg = RunConfig.loadDefault();

        // Load dataset
        var ds = new DataSetLoaderMFD().loadDataSet(datasetName)
                .orElseThrow(() -> new RuntimeException("dataset " + datasetName + " not found"))
                .getDataSet();

        // Run artifacts + selections (sys_info/dataset_info/experiments.csv)
        RunArtifacts artifacts = RunArtifacts.open(runCfg, List.of(config));
        artifacts.registerDataset(datasetName, ds);

        // Run
        Grid.runAll(ds,
                config.construction.useSavedIndexIfExists,
                config.construction.outDegree,
                config.construction.efConstruction,
                config.construction.neighborOverflow,
                config.construction.addHierarchy,
                config.construction.refineFinalGraph,
                config.construction.getFeatureSets(),
                config.construction.getCompressorParameters(),
                config.search.getCompressorParameters(),
                config.search.topKOverquery,
                config.search.useSearchPruning,
                artifacts);
    }
}
