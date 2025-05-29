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

import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.example.yaml.MultiConfig;

import java.io.IOException;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class HelloVectorWorld {
    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        String datasetName = "ada002-100k";

        var mfd = DownloadHelper.maybeDownloadFvecs(datasetName);
        DataSet ds = mfd.load();

        MultiConfig config = MultiConfig.getConfig(datasetName);

        Grid.runAll(ds, config.construction.outDegree, config.construction.efConstruction,
                config.construction.neighborOverflow, config.construction.addHierarchy,
                config.construction.getFeatureSets(), config.construction.getCompressorParameters(),
                config.search.getCompressorParameters(), config.search.topKOverquery, config.search.useSearchPruning);
    }
}
