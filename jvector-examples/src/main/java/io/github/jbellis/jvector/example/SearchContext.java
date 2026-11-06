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
import io.github.jbellis.jvector.example.benchmarks.diagnostics.SystemMonitor;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.util.Set;
import java.util.function.Function;

/**
 * One index searched with one search compressor, as reported to a {@link GridListener}.
 *
 * <p>To give listeners more data, add a field here and set it where {@link Grid} creates the context.</p>
 */
final class SearchContext {
    final DataSet ds;
    final BuildConfig build;
    final SearchConfig search;
    /** The feature set the index was requested with. */
    final Set<FeatureId> requestedFeatures;
    /** The features the written index actually has. */
    final Set<FeatureId> indexFeatures;
    /** The resolved build compressor, or "None". */
    final String buildCompressorLabel;
    final Function<DataSet, CompressorParameters> searchCompressor;
    /** The compressor used to encode vectors for search, or "None". */
    final String searchCompressorLabel;
    /** Seconds the most recent build of this dataset took, or null if none was recorded. */
    final Double indexBuildTimeS;
    /** On-disk size of the index file, or null if unknown. */
    final Long indexFileSizeBytes;
    /** Memory state captured right after the build, or null if none was captured. */
    final SystemMonitor.SystemSnapshot buildSnapshot;

    SearchContext(DataSet ds,
                  BuildConfig build,
                  SearchConfig search,
                  Set<FeatureId> requestedFeatures,
                  Set<FeatureId> indexFeatures,
                  String buildCompressorLabel,
                  Function<DataSet, CompressorParameters> searchCompressor,
                  String searchCompressorLabel,
                  Double indexBuildTimeS,
                  Long indexFileSizeBytes,
                  SystemMonitor.SystemSnapshot buildSnapshot) {
        this.ds = ds;
        this.build = build;
        this.search = search;
        this.requestedFeatures = requestedFeatures;
        this.indexFeatures = indexFeatures;
        this.buildCompressorLabel = buildCompressorLabel;
        this.searchCompressor = searchCompressor;
        this.searchCompressorLabel = searchCompressorLabel;
        this.indexBuildTimeS = indexBuildTimeS;
        this.indexFileSizeBytes = indexFileSizeBytes;
        this.buildSnapshot = buildSnapshot;
    }
}
