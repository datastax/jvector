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
import io.github.jbellis.jvector.graph.disk.CompactOptions;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.disk.CompactOptions;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
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

    // CLI values for CompactOptions selection
    // compactMode = inline | pq_separate | fused_pq
    private enum CompactMode {
        INLINE,
        PQ_SEPARATE,
        FUSED_PQ;

        static CompactMode parse(String s) {
            if (s == null) return INLINE;
            if (s.equals("inline") || s.equals("withinlinevectors")) return INLINE;
            if (s.equals("pq_separate") || s.equals("pq-separate") || s.equals("withpqvectorsseparate")) return PQ_SEPARATE;
            if (s.equals("fused_pq") || s.equals("fused-pq") || s.equals("withfusedpq")) return FUSED_PQ;
            throw new IllegalArgumentException("Unknown compactMode: " + s +
                                               " (expected: inline | pq_separate | fused_pq)");
        }
    }


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
        String compactModeStr = cli.get("compactMode", "inline");
        CompactMode compactMode = CompactMode.parse(compactModeStr);

        // For pq_separate: provide pqPath, and optionally pqVecPath
        // pqPath can be a file that ProductQuantization.load(...) can read.
        String pqPathStr = cli.get("pqPath", "");
        String pqVecPathStr = cli.get("pqVecPath", ""); // default below if empty
        //int taskWindowSize = cli.getInt("taskWindowSize", 0);

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

            // Build CompactOptions based on CLI
            CompactOptions opts;
            switch (compactMode) {
                case INLINE: {
                    opts = CompactOptions.withInlineVectors();
                    break;
                }
                case PQ_SEPARATE: {
                    if (pqPathStr == null || pqPathStr.isBlank()) {
                        throw new IllegalArgumentException("compactMode=pq_separate requires pqPath=...");
                    }
                    Path pqPath = Path.of(pqPathStr);

                    // default pqVecPath if not provided
                    Path pqVecPath = (pqVecPathStr != null && !pqVecPathStr.isBlank())
                        ? Path.of(pqVecPathStr)
                        : Path.of(outputPathStr + ".pq");

                    // Load provided PQ codebook
                    ProductQuantization pq;
                    try (ReaderSupplier pqRs = ReaderSupplierFactory.open(pqPath.toAbsolutePath())) {
                        // ReaderSupplier#get() is expected to return a RandomAccessReader
                        pq = ProductQuantization.load(pqRs.get());
                    }

                    opts = CompactOptions.withPQVectorsSeparate(pq, pqVecPath);
                    break;
                }
                case FUSED_PQ: {
                    // Requires sources have FUSED_PQ feature (or compactor resolves codebook from sources)
                    opts = CompactOptions.withFusedPQ();
                    break;
                }
                default:
                    throw new IllegalStateException("Unhandled compactMode: " + compactMode);
            }

            //if (taskWindowSize > 0) {
                //// assuming you have builder/toBuilder
                //opts = opts.toBuilder()
                        //.taskWindowSize(taskWindowSize)
                        //.build();
            //}

            compactor.compact(outputPath, similarityFunction, opts);

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


            boolean recallUsePQ = cli.getBool(
                "recallUsePQ",
                compactMode == CompactMode.PQ_SEPARATE
            );
            PQVectors pqVectors = null;
            if (recallUsePQ) {
                Path pqVecPath = (pqVecPathStr != null && !pqVecPathStr.isBlank())
                    ? Path.of(pqVecPathStr)
                    : Path.of(outputPathStr + ".pq");
                log.info("Recall: using PQVectors for approximate scoring from {}", pqVecPath.toAbsolutePath());
                try (var pqRs = ReaderSupplierFactory.open(pqVecPath)) {
                    // pqVec file is in PQVectors format (pq + count + subspaceCount + chunks)
                    pqVectors = PQVectors.load(pqRs.get());
                }
            } else {
                log.info("Recall: using full precision (inline vectors) scoring");
            }

            try (var rs = ReaderSupplierFactory.open(outputPath)) {
                var compactGraph = OnDiskGraphIndex.load(rs);
                GraphSearcher searcher = new GraphSearcher(compactGraph);
                searcher.usePruning(false);
                List<SearchResult> compactedRetrieved = new ArrayList<>();
                for (int n = 0; n < queryVectors.size(); ++n) {
                    SearchResult sr;
                    if (pqVectors != null) {
                        // --- PQ approximate search path ---
                        // Score function uses PQVectors (approx) instead of full vectors
                        ScoreFunction.ApproximateScoreFunction asf = pqVectors.scoreFunctionFor(queryVectors.get(n), similarityFunction);
                        SearchScoreProvider ssp = new DefaultSearchScoreProvider(asf);

                        // Bits.ALL: everything is live for recall testing
                        sr = searcher.search(ssp, 10, 10, 0.0f, 0.0f, Bits.ALL);
                    }
                    else {
                        sr = GraphSearcher.search(queryVectors.get(n),
                            10,
                            ravv,
                            similarityFunction,
                            compactGraph,
                            Bits.ALL);
                    }

                    compactedRetrieved.add(sr);
                }
                var recall = AccuracyMetrics.recallFromSearchResults(groundTruth, compactedRetrieved, 10, 10);
                log.info("Recall [dataset={}, numSources={}]: {}",
                        datasetName, numSources, recall);
            }
        }
    }
}
