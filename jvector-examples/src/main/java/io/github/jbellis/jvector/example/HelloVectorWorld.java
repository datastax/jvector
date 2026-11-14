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

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.util.Bits;

import java.io.IOException;

/** Builds a graph and searches it through the library API. */
public class HelloVectorWorld {
    public static void main(String[] args) throws IOException {
        String datasetName = "ada002-100k";
        var loaded = DataSets.loadDataSet(datasetName)
                .orElseThrow(() -> new IllegalArgumentException("Dataset not found: " + datasetName));
        try (var dataset = loaded.getDataSet()) {
            var vectors = dataset.getBaseRavv();
            var similarity = dataset.getSimilarityFunction();
            var query = dataset.getQueryVectors().get(0);

            try (var builder = GraphIndexBuilder.builder(vectors, similarity, 32)
                         .withBeamWidth(100)
                         .withNeighborOverflow(1.2f)
                         .withAlpha(1.2f)
                         .build();
                 var graph = builder.build(vectors)) {
                var result = GraphSearcher.search(query, 10, vectors, similarity, graph, Bits.ALL);
                System.out.printf("Nearest neighbors to the first query in %s (%d indexed vectors)%n",
                        datasetName, graph.size(0));
                for (var match : result.getNodes()) {
                    System.out.printf("vector %d  score %.6f%n", match.node, match.score);
                }
            }
        }
    }
}
