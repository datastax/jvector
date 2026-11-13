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

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * The search-side sweep {@link Grid} runs against every index it builds: each search compressor,
 * each topK with its overquery factors, and each pruning setting.
 *
 * <p>To add a search parameter, add a field, a {@link Builder} setter, and its use in {@link Grid}.</p>
 */
public final class SearchConfig {
    /** Compressors used for approximate scoring at search time; one returning NONE searches full precision. */
    public final List<Function<DataSet, CompressorParameters>> searchCompressors;
    /** topK mapped to the overquery factors to try; each factor gives rerankK = topK * factor. */
    public final Map<Integer, List<Double>> topKGrid;
    public final List<Boolean> usePruningGrid;
    private final List<Compression> entries;
    /** Search compression and delivery settings for a single schema 3 index case. */
    final Compression compression;
    /** Search worker counts to sweep over one index; null leaves query concurrency to JVM-wide settings. */
    public final List<Integer> workerGrid;
    /** direct or queued. */
    public final String queryExecution;
    /** Queued queries admitted per worker, or null for the default (the larger of 8 and the batch size). */
    public final Integer requestsPerWorker;

    private SearchConfig(Builder b) {
        this.searchCompressors = b.searchCompressors;
        this.topKGrid = b.topKGrid;
        this.usePruningGrid = b.usePruningGrid;
        this.entries = b.entries;
        this.compression = b.compression;
        this.workerGrid = b.workerGrid;
        this.queryExecution = b.queryExecution == null ? "direct" : b.queryExecution;
        this.requestsPerWorker = b.requestsPerWorker;
    }

    /** Batch size for queued execution with this case's compression: ASH's maxBatchSize, else 1. */
    int queuedBatchSize() {
        return compression == null ? 1 : compression.queuedBatchSize();
    }

    /** Queued queries admitted per worker for this case. */
    int resolvedRequestsPerWorker() {
        return requestsPerWorker != null ? requestsPerWorker : Math.max(8, queuedBatchSize());
    }

    /** The worker counts to query each index with; a single null means the JVM-wide setting. */
    List<Integer> workerCounts() {
        return workerGrid == null ? java.util.Collections.singletonList(null) : workerGrid;
    }

    /** One case per compression entry. Each case queries its index once per search worker count. */
    List<SearchConfig> cases() {
        if (entries == null) return List.of(this);
        return entries.stream().map(entry -> {
            var builder = builder().topKGrid(topKGrid).usePruningGrid(usePruningGrid)
                    .searchCompressors(List.of(entry.getCompressorParameters()));
            builder.compression = entry;
            builder.workerGrid = workerGrid;
            builder.queryExecution = queryExecution;
            builder.requestsPerWorker = requestsPerWorker;
            return builder.build();
        }).collect(Collectors.toList());
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private List<Function<DataSet, CompressorParameters>> searchCompressors;
        private Map<Integer, List<Double>> topKGrid;
        private List<Boolean> usePruningGrid;
        private List<Compression> entries;
        private Compression compression;
        private List<Integer> workerGrid;
        private String queryExecution;
        private Integer requestsPerWorker;

        private Builder() {}

        Builder queryExecution(String queryExecution) { this.queryExecution = queryExecution; return this; }
        Builder requestsPerWorker(Integer requestsPerWorker) { this.requestsPerWorker = requestsPerWorker; return this; }

        Builder workerGrid(List<Integer> workerGrid) {
            this.workerGrid = workerGrid == null ? null : List.copyOf(workerGrid);
            return this;
        }

        public Builder searchCompressors(List<Function<DataSet, CompressorParameters>> searchCompressors) { this.searchCompressors = searchCompressors; return this; }
        public Builder topKGrid(Map<Integer, List<Double>> topKGrid) { this.topKGrid = topKGrid; return this; }
        public Builder usePruningGrid(List<Boolean> usePruningGrid) { this.usePruningGrid = usePruningGrid; return this; }

        Builder compressionEntries(List<Compression> entries) {
            this.entries = List.copyOf(entries);
            this.searchCompressors = entries.stream().map(Compression::getCompressorParameters).collect(Collectors.toList());
            return this;
        }

        /** @throws IllegalStateException if a parameter was not set */
        public SearchConfig build() {
            require(searchCompressors, "searchCompressors");
            require(topKGrid, "topKGrid");
            require(usePruningGrid, "usePruningGrid");
            return new SearchConfig(this);
        }

        private static void require(Object value, String name) {
            if (value == null) {
                throw new IllegalStateException("SearchConfig." + name + " must be set");
            }
        }
    }
}
