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
package io.github.jbellis.jvector.bench.compaction;

import io.github.jbellis.jvector.bench.compaction.Cli;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Compaction {
    private static final Logger log = LoggerFactory.getLogger(Compaction.class);

    public static void main(String[] args) throws Exception {
        
        VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.COSINE;
        var cli = new Cli(args);

        boolean enableRecall = cli.getBool("enableRecall", false);
        String datasetName = cli.get("datasetName", "");
        String baseVecPath = cli.get("baseVectorPath", "");
        String queriesPath = cli.get("queriesPath", "");
        String groundTruthPath = cli.get("groundTruthPath", "");
        int numSources = cli.getInt("numSources", 4);
        String segmentsDirStr = cli.get("segmentsDir", "target/segments");
        String outputPathStr = cli.get("outputPath", "target/compact-graph");

        String vectorizationProvider = cli.get("vectorizationProvider", "");
        if (vectorizationProvider != null && !vectorizationProvider.isBlank()) {
            System.setProperty("jvector.vectorization_provider", vectorizationProvider);
        }
        String resolvedVP = VectorizationProvider.getInstance().getClass().getSimpleName();

        int numParts = (numSources == 0) ? 1 : numSources;
        Path segmentsDir = Path.of(segmentsDirStr);
        Path outputPath = Path.of(outputPathStr);
        Files.createDirectories(outputPath.getParent());

        // load segments
        var graphs = new ArrayList<OnDiskGraphIndex>(numParts);
        var rss = new ArrayList<ReaderSupplier>(numParts);
        try {
            for (int i = 0; i < numParts; i++) {
                Path seg = segmentsDir.resolve("per-source-graph-" + i);
                if (!Files.exists(seg)) {
                    throw new IllegalStateException("Missing segment file: " + seg.toAbsolutePath());
                }
                log.info("Loading segment {}/{} from {}", i + 1, numParts, seg.toAbsolutePath());
                rss.add(ReaderSupplierFactory.open(seg.toAbsolutePath()));
                graphs.add(OnDiskGraphIndex.load(rss.get(i)));
            }

            log.info("Compacting {} segments -> {} (vp={}, sim={})",
                    numParts, outputPath.toAbsolutePath(), resolvedVP, similarityFunction);

            var compactor = new OnDiskGraphIndexCompactor(graphs);

            // ordinal remap: local [0..size-1] -> global increasing
            int globalOrdinal = 0;
            for (int n = 0; n < numParts; n++) {
                int size = graphs.get(n).size();
                Map<Integer, Integer> map = new HashMap<>(size * 2);
                for (int i = 0; i < size; i++) {
                    map.put(i, globalOrdinal++);
                }
                compactor.setRemapper(graphs.get(n), new OrdinalMapper.MapMapper(map));
            }

            compactor.compact(outputPath, similarityFunction);

            log.info("Done. Compacted index at {}", outputPath.toAbsolutePath());
        } finally {
            for (var g : graphs) try { g.close(); } catch (Exception ignored) {}
            for (var r : rss) try { r.close(); } catch (Exception ignored) {}
        }

        if(enableRecall) {
            log.info("Enable recall. Load index from {}", outputPath.toAbsolutePath());
            var baseVectors = SiftLoader.readFvecs(baseVecPath);
            var queryVectors = SiftLoader.readFvecs(queriesPath);
            var groundTruth = SiftLoader.readIvecs(groundTruthPath);
            int dimension = baseVectors.get(0).length();
            var ravv = new ListRandomAccessVectorValues(baseVectors, dimension);

            try (var rs = ReaderSupplierFactory.open(outputPath)) {
                var compactGraph = OnDiskGraphIndex.load(rs);
                List<SearchResult> compactedRetrieved = new ArrayList<>();
                for (int n = 0; n < queryVectors.size(); ++n) {
                    compactedRetrieved.add(GraphSearcher.search(queryVectors.get(n),
                            10,
                            ravv,
                            similarityFunction,
                            compactGraph,
                            Bits.ALL));
                }
                var recall = AccuracyMetrics.recallFromSearchResults(groundTruth, compactedRetrieved, 10, 10);
                log.info("Recall [dataset={}, numSources={}]: {}",
                        datasetName, numSources, recall);
            }
        }
    }
}
