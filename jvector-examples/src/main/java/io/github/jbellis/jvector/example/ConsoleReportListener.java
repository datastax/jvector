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

import io.github.jbellis.jvector.example.benchmarks.BenchmarkTablePrinter;
import io.github.jbellis.jvector.example.reporting.ReportingSelectionResolver;
import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.example.reporting.SearchSelection;

import java.util.LinkedHashMap;
import java.util.Map;

/** Prints a results table per pruning setting and topK, showing the console selection from run-config.yml. */
final class ConsoleReportListener implements GridListener {
    private final RunArtifacts artifacts;

    // Per (index, search compressor), so missing-metric warnings repeat for each configuration.
    private SearchSelection selection;
    private BenchmarkTablePrinter printer;
    private ReportingSelectionResolver.ResolvedSelection resolved;

    ConsoleReportListener(RunArtifacts artifacts) {
        this.artifacts = artifacts;
    }

    @Override
    public void onSearchStart(SearchContext ctx) {
        selection = SearchSelection.forConsole(artifacts.benchmarksToDisplay(), artifacts.metricsToDisplay());
        selection.validate(artifacts.benchmarksToCompute());
    }

    @Override
    public void onPruningStart(SearchContext ctx, boolean usePruning) {
        printer = new BenchmarkTablePrinter();
        // Print the config once per index and query configuration (not per topK)
        Map<String, Object> indexConfig = new LinkedHashMap<>();
        indexConfig.put("featureSetForIndex", ctx.indexFeatures);
        indexConfig.putAll(ctx.build.reportedParameters());
        printer.printConfig(indexConfig, Map.of("usePruning", usePruning));
    }

    @Override
    public void onTopKStart(SearchContext ctx, boolean usePruning, int topK) {
        // Resolving selections depends on topK
        resolved = selection.resolveForTopK(topK);
        // Force header to re-print for this topK (columns change)
        printer.resetTable();
    }

    @Override
    public void onQueryResult(SearchContext ctx, QueryRow row) {
        // Best-effort runtime availability warnings (warn once per key)
        selection.warnMissing(row.metrics, resolved);
        printer.printRow(row.overquery, selection.apply(row.metrics, resolved));
    }

    @Override
    public void onTopKEnd(SearchContext ctx, boolean usePruning, int topK) {
        printer.printFooter();
    }
}
