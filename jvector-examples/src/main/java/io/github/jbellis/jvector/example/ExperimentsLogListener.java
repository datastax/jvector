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

import io.github.jbellis.jvector.example.reporting.ReportingSelectionResolver;
import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.example.reporting.SearchSelection;

/**
 * Appends one experiments.csv row per measurement, showing the logging selection from run-config.yml.
 * Writes nothing when the run has logging disabled.
 */
final class ExperimentsLogListener implements GridListener {
    private final RunArtifacts artifacts;

    // Per (index, search compressor), so missing-metric warnings repeat for each configuration.
    private SearchSelection selection;
    private ReportingSelectionResolver.ResolvedSelection resolved;

    ExperimentsLogListener(RunArtifacts artifacts) {
        this.artifacts = artifacts;
    }

    @Override
    public void onSearchStart(SearchContext ctx) {
        selection = SearchSelection.forLogging(artifacts.benchmarksToLog(), artifacts.metricsToLog());
        selection.validate(artifacts.benchmarksToCompute());
    }

    @Override
    public void onTopKStart(SearchContext ctx, boolean usePruning, int topK) {
        resolved = selection.resolveForTopK(topK);
    }

    @Override
    public void onQueryResult(SearchContext ctx, QueryRow row) {
        selection.warnMissing(row.metrics, resolved);
        artifacts.logRow(
                ctx.ds.getName(),
                ctx.build.reportedParameters(),
                ctx.indexFeatures,
                ctx.buildCompressorLabel,
                ctx.searchCompressorLabel,
                row.usePruning,
                row.topK,
                row.overquery,
                row.rerankK,
                selection.apply(row.metrics, resolved)
        );
    }
}
