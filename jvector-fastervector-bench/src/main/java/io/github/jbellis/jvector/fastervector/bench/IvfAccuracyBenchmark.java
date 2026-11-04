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

package io.github.jbellis.jvector.fastervector.bench;

import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.fastervector.FvecThreadPool;
import io.github.jbellis.jvector.fastervector.IvfAshIndexSearcher;
import io.github.jbellis.jvector.fastervector.NodeScore;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.SearchPoint;
import io.github.jbellis.jvector.graph.SearchResult;

import java.util.ArrayList;
import java.util.List;

/**
 * Recall and MAP against the dataset's ground truth, computed with jvector-examples'
 * {@link AccuracyMetrics} so the numbers match the graph benchmarks'. Metric keys match
 * {@code AccuracyBenchmark}'s, so run-config.yml selections apply unchanged.
 */
public class IvfAccuracyBenchmark implements IvfQueryBenchmark {
    private static final String DEFAULT_FORMAT = ".2f";

    private boolean computeRecall;
    private boolean computeMAP;

    private IvfAccuracyBenchmark(boolean computeRecall, boolean computeMAP) {
        this.computeRecall = computeRecall;
        this.computeMAP = computeMAP;
    }

    public static IvfAccuracyBenchmark createDefault() {
        return new IvfAccuracyBenchmark(true, false);
    }

    public static IvfAccuracyBenchmark createEmpty() {
        return new IvfAccuracyBenchmark(false, false);
    }

    public IvfAccuracyBenchmark displayRecall() {
        computeRecall = true;
        return this;
    }

    public IvfAccuracyBenchmark displayMAP() {
        computeMAP = true;
        return this;
    }

    @Override
    public String getBenchmarkName() {
        return "IvfRecallBenchmark";
    }

    @Override
    public List<Metric> runBenchmark(IvfSystem system, SearchPoint point) {
        if (!(computeRecall || computeMAP)) {
            throw new IllegalStateException("At least one metric must be displayed");
        }
        List<SearchResult> results;
        try (FvecThreadPool pool = FvecThreadPool.create(point.searchThreads());
             IvfAshIndexSearcher searcher = system.searcher(point.topK(), point.nprobe(), pool)) {
            results = toSearchResults(searcher.search(system.queryVecs()));
        }

        int topK = point.topK();
        var groundTruth = system.dataSet().getGroundTruth();
        List<Metric> out = new ArrayList<>();
        if (computeRecall) {
            double recall = AccuracyMetrics.recallFromSearchResults(groundTruth, results, topK, topK);
            out.add(Metric.of("search.accuracy.recall_at_" + topK, "Recall@" + topK, DEFAULT_FORMAT, recall));
        }
        if (computeMAP) {
            double map = AccuracyMetrics.meanAveragePrecisionAtK(groundTruth, results, topK);
            out.add(Metric.of("search.accuracy.map_at_" + topK, "MAP@" + topK, DEFAULT_FORMAT, map));
        }
        return out;
    }

    /** One graph {@link SearchResult} per query, so jvector-examples' accuracy code can read IVF results. */
    static List<SearchResult> toSearchResults(io.github.jbellis.jvector.fastervector.SearchResult ivf) {
        List<SearchResult> out = new ArrayList<>(ivf.results().size());
        for (List<NodeScore> row : ivf.results()) {
            out.add(toSearchResult(row));
        }
        return out;
    }

    static SearchResult toSearchResult(List<NodeScore> row) {
        var nodes = new SearchResult.NodeScore[row.size()];
        for (int i = 0; i < nodes.length; i++) {
            NodeScore ns = row.get(i);
            nodes[i] = new SearchResult.NodeScore(Math.toIntExact(ns.label()), ns.score());
        }
        // IVF has no graph traversal, so the visit counts are zero
        float worst = nodes.length == 0 ? Float.NaN : nodes[nodes.length - 1].score;
        return new SearchResult(nodes, 0, 0, 0, 0, worst);
    }
}
