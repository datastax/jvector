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

package io.github.jbellis.jvector.bench.benchtools;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * CLI utility that reads a CompactorBenchmark JSONL event log and reports
 * max concurrency and the cartesian parameter matrix with testIds and results.
 */
public final class EventLogAnalyzer {

    private static final List<String> PARAM_FIELDS = List.of(
            "params.dataset", "params.numSources", "params.graphDegree", "params.beamWidth",
            "params.storageDirectories", "params.storageClasses", "params.splitDistribution",
            "params.indexPrecision", "params.parallelWriteThreads", "params.vectorizationProvider",
            "params.datasetPortion"
    );

    private static final List<String> RESULT_FIELDS = List.of(
            "results.recall", "results.durationMs", "results.errorMessage"
    );

    private static final Path RESULTS_DIR = Path.of("target", "benchmark-results");
    private static final String RESULTS_FILENAME = "compactor-results.jsonl";

    /**
     * Finds the most recent compactor-results.jsonl under target/benchmark-results/
     * by selecting the compactor-* directory with the highest name (epoch-second suffix).
     */
    private static Path findLatestResultsFile() {
        if (!Files.isDirectory(RESULTS_DIR)) {
            return null;
        }
        try (Stream<Path> dirs = Files.list(RESULTS_DIR)) {
            return dirs.filter(Files::isDirectory)
                    .filter(d -> d.getFileName().toString().startsWith("compactor-"))
                    .sorted(Comparator.reverseOrder())
                    .map(d -> d.resolve(RESULTS_FILENAME))
                    .filter(Files::exists)
                    .findFirst()
                    .orElse(null);
        } catch (IOException e) {
            return null;
        }
    }

    public static void main(String[] args) throws IOException {
        Path inputFile = null;
        String startingAt = null, endingAt = null, startingTestId = null, endingTestId = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--starting-at":    startingAt = args[++i]; break;
                case "--ending-at":      endingAt = args[++i]; break;
                case "--starting-testid": startingTestId = args[++i]; break;
                case "--ending-testid":  endingTestId = args[++i]; break;
                case "--help":
                case "-h":
                    printUsage();
                    System.exit(0);
                    break;
                default:
                    if (args[i].startsWith("--")) {
                        System.err.println("Unknown option: " + args[i]);
                        System.exit(1);
                    }
                    if (inputFile != null) {
                        System.err.println("Multiple input files specified");
                        System.exit(1);
                    }
                    inputFile = Path.of(args[i]);
            }
        }

        if (inputFile == null) {
            inputFile = findLatestResultsFile();
            if (inputFile == null) {
                System.err.println("No results file found under " + RESULTS_DIR.toAbsolutePath());
                printUsage();
                System.exit(1);
            }
        }

        System.err.println("Using: " + inputFile.toAbsolutePath());

        // Read and parse all lines
        List<Map<String, String>> allEvents = new ArrayList<>();
        for (String line : Files.readAllLines(inputFile)) {
            line = line.trim();
            if (!line.isEmpty()) {
                allEvents.add(parseJsonLine(line));
            }
        }

        // Assign synthetic testIds where missing
        boolean hasSyntheticIds = false;
        String currentSyntheticId = null;
        int syntheticCounter = 0;
        for (var event : allEvents) {
            if (!event.containsKey("testId") || event.get("testId").isEmpty()) {
                hasSyntheticIds = true;
                if ("started".equals(event.get("event"))) {
                    currentSyntheticId = String.format("#%03d", ++syntheticCounter);
                }
                if (currentSyntheticId != null) {
                    event.put("testId", currentSyntheticId);
                }
            }
        }

        // Group all events by testId
        Map<String, List<Map<String, String>>> allByTestId = new LinkedHashMap<>();
        for (var event : allEvents) {
            String tid = event.get("testId");
            if (tid != null) {
                allByTestId.computeIfAbsent(tid, k -> new ArrayList<>()).add(event);
            }
        }

        // Determine which testIds are in scope
        Set<String> inScopeTestIds = new LinkedHashSet<>();
        for (var entry : allByTestId.entrySet()) {
            String tid = entry.getKey();

            // testId range filter (skip for synthetic IDs)
            if (!hasSyntheticIds) {
                if (startingTestId != null && tid.compareTo(startingTestId) < 0) continue;
                if (endingTestId != null && tid.compareTo(endingTestId) > 0) continue;
            }

            // Timestamp range filter: test is in scope if ANY event is within range
            if (startingAt != null || endingAt != null) {
                boolean anyInRange = false;
                for (var event : entry.getValue()) {
                    String ts = event.get("timestamp");
                    if (ts == null) continue;
                    boolean afterStart = startingAt == null || ts.compareTo(startingAt) >= 0;
                    boolean beforeEnd = endingAt == null || ts.compareTo(endingAt) <= 0;
                    if (afterStart && beforeEnd) {
                        anyInRange = true;
                        break;
                    }
                }
                if (!anyInRange) continue;
            }

            inScopeTestIds.add(tid);
        }

        if (inScopeTestIds.isEmpty()) {
            System.out.println("No matching events found.");
            return;
        }

        // Collect all events for in-scope testIds
        Map<String, List<Map<String, String>>> byTestId = new LinkedHashMap<>();
        for (String tid : inScopeTestIds) {
            byTestId.put(tid, allByTestId.get(tid));
        }

        // Compute max concurrency by sweeping start/end intervals
        List<String[]> intervals = new ArrayList<>();
        for (var entry : byTestId.entrySet()) {
            String startTs = null, endTs = null;
            for (var event : entry.getValue()) {
                String ev = event.get("event");
                String ts = event.get("timestamp");
                if (ts == null) continue;
                if ("started".equals(ev) && (startTs == null || ts.compareTo(startTs) < 0)) {
                    startTs = ts;
                }
                if (("completed".equals(ev) || "error".equals(ev))
                        && (endTs == null || ts.compareTo(endTs) > 0)) {
                    endTs = ts;
                }
            }
            if (startTs != null && endTs != null) {
                intervals.add(new String[]{startTs, endTs});
            }
        }

        System.out.println("Max concurrency: " + computeMaxConcurrency(intervals));
        System.out.println();

        // Extract params and results for each test.
        // Params come from the "started" event (which includes the "params" sub-object).
        // Results come from the "completed" or "error" event (which includes the "results" sub-object).
        Map<String, Map<String, String>> testParams = new LinkedHashMap<>();
        Map<String, Map<String, String>> testResults = new LinkedHashMap<>();
        Map<String, String> testStatus = new LinkedHashMap<>();
        for (var entry : byTestId.entrySet()) {
            String tid = entry.getKey();
            Map<String, String> params = null;
            Map<String, String> results = null;
            String status = "started";
            for (var event : entry.getValue()) {
                String ev = event.get("event");
                // Take params from whichever event has them (all events include params now)
                if (params == null && event.keySet().stream().anyMatch(k -> k.startsWith("params."))) {
                    params = event;
                }
                if ("completed".equals(ev)) {
                    status = "completed";
                    results = event;
                } else if ("error".equals(ev)) {
                    status = "error";
                    results = event;
                }
            }
            if (params != null) {
                testParams.put(tid, params);
            }
            testResults.put(tid, results != null ? results : Map.of());
            testStatus.put(tid, status);
        }

        // Classify parameters as static (single value) or varying (multiple values)
        List<String> varyingParams = new ArrayList<>();
        Map<String, String> staticParams = new LinkedHashMap<>();
        for (String field : PARAM_FIELDS) {
            Set<String> values = new HashSet<>();
            for (var params : testParams.values()) {
                values.add(params.getOrDefault(field, ""));
            }
            if (values.size() > 1) {
                varyingParams.add(field);
            } else if (values.size() == 1) {
                String value = values.iterator().next();
                if (!value.isEmpty()) {
                    staticParams.put(stripPrefix(field), value);
                }
            }
        }

        // Display static parameters at the top
        if (!staticParams.isEmpty()) {
            System.out.println("Static parameters:");
            for (var entry : staticParams.entrySet()) {
                System.out.println("  " + entry.getKey() + " = " + entry.getValue());
            }
            System.out.println();
        }

        if (varyingParams.isEmpty()) {
            System.out.println("No varying parameters found.");
        } else {
            // Display without "params." prefix for readability
            List<String> displayNames = new ArrayList<>();
            for (String p : varyingParams) {
                displayNames.add(stripPrefix(p));
            }
            System.out.println("Varying parameters: " + String.join(", ", displayNames));
        }
        System.out.println();

        // Determine which result fields have any non-empty values
        List<String> activeResultFields = new ArrayList<>();
        for (String field : RESULT_FIELDS) {
            for (var results : testResults.values()) {
                if (!results.getOrDefault(field, "").isEmpty()) {
                    activeResultFields.add(field);
                    break;
                }
            }
        }

        // Build and print table sorted by testId
        List<String> sortedTestIds = new ArrayList<>(testParams.keySet());
        Collections.sort(sortedTestIds);

        // Columns: testId, varying params (without prefix), status, active result fields (without prefix)
        List<String> columns = new ArrayList<>();
        columns.add("testId");
        for (String p : varyingParams) {
            columns.add(stripPrefix(p));
        }
        columns.add("status");
        for (String r : activeResultFields) {
            columns.add(stripPrefix(r));
        }

        List<List<String>> rows = new ArrayList<>();
        for (String tid : sortedTestIds) {
            List<String> row = new ArrayList<>();
            row.add(hasSyntheticIds ? "n/a" : tid);
            var params = testParams.get(tid);
            for (String field : varyingParams) {
                row.add(params != null ? params.getOrDefault(field, "") : "");
            }
            row.add(testStatus.getOrDefault(tid, ""));
            var results = testResults.getOrDefault(tid, Map.of());
            for (String field : activeResultFields) {
                row.add(results.getOrDefault(field, ""));
            }
            rows.add(row);
        }

        printTable(columns, rows);
    }

    private static String stripPrefix(String field) {
        int dot = field.indexOf('.');
        return dot >= 0 ? field.substring(dot + 1) : field;
    }

    private static void printTable(List<String> columns, List<List<String>> rows) {
        int[] widths = new int[columns.size()];
        for (int i = 0; i < columns.size(); i++) {
            widths[i] = columns.get(i).length();
        }
        for (var row : rows) {
            for (int i = 0; i < row.size(); i++) {
                widths[i] = Math.max(widths[i], row.get(i).length());
            }
        }

        StringBuilder header = new StringBuilder();
        for (int i = 0; i < columns.size(); i++) {
            if (i > 0) header.append("  ");
            header.append(String.format("%-" + widths[i] + "s", columns.get(i)));
        }
        System.out.println(header);

        for (var row : rows) {
            StringBuilder line = new StringBuilder();
            for (int i = 0; i < row.size(); i++) {
                if (i > 0) line.append("  ");
                line.append(String.format("%-" + widths[i] + "s", row.get(i)));
            }
            System.out.println(line);
        }
    }

    private static void printUsage() {
        System.err.println("Usage: EventLogAnalyzer [results.jsonl] [options]");
        System.err.println("  If no file is given, the latest results under target/benchmark-results/ are used.");
        System.err.println("Options:");
        System.err.println("  --starting-at <ISO-8601>     Include tests with events at or after this timestamp");
        System.err.println("  --ending-at <ISO-8601>       Include tests with events at or before this timestamp");
        System.err.println("  --starting-testid <id>       Include tests with testId >= this value");
        System.err.println("  --ending-testid <id>         Include tests with testId <= this value");
    }

    private static int computeMaxConcurrency(List<String[]> intervals) {
        if (intervals.isEmpty()) return 0;

        // Each sweep event is {timestamp, delta} where delta is +1 (start) or -1 (end)
        List<String[]> events = new ArrayList<>();
        for (var interval : intervals) {
            events.add(new String[]{interval[0], "+1"});
            events.add(new String[]{interval[1], "-1"});
        }
        // Sort by timestamp, then ends before starts at the same timestamp
        events.sort((a, b) -> {
            int cmp = a[0].compareTo(b[0]);
            if (cmp != 0) return cmp;
            return a[1].compareTo(b[1]); // "-1" < "+1" lexicographically
        });

        int max = 0, current = 0;
        for (var event : events) {
            current += Integer.parseInt(event[1]);
            max = Math.max(max, current);
        }
        return max;
    }

    /**
     * Parses a JSON line into key-value string pairs.
     * Nested objects are flattened with dot-prefixed keys (e.g., "params.dataset").
     * Handles quoted strings (with backslash escapes), numbers, booleans, and one level of nesting.
     */
    private static Map<String, String> parseJsonLine(String line) {
        Map<String, String> result = new LinkedHashMap<>();
        parseObject(line, new int[]{0}, "", result);
        return result;
    }

    private static void parseObject(String line, int[] pos, String prefix, Map<String, String> result) {
        int len = line.length();

        // Skip to opening brace
        while (pos[0] < len && line.charAt(pos[0]) != '{') pos[0]++;
        pos[0]++;

        while (pos[0] < len) {
            // Skip whitespace and commas
            while (pos[0] < len && (line.charAt(pos[0]) == ' ' || line.charAt(pos[0]) == ',' || line.charAt(pos[0]) == '\t')) pos[0]++;
            if (pos[0] >= len || line.charAt(pos[0]) == '}') {
                pos[0]++; // skip closing brace
                break;
            }

            // Parse key (quoted string)
            if (line.charAt(pos[0]) != '"') break;
            pos[0]++;
            int keyStart = pos[0];
            while (pos[0] < len && line.charAt(pos[0]) != '"') {
                if (line.charAt(pos[0]) == '\\') pos[0]++;
                pos[0]++;
            }
            String key = line.substring(keyStart, pos[0]);
            pos[0]++; // skip closing quote

            String fullKey = prefix.isEmpty() ? key : prefix + "." + key;

            // Skip colon and whitespace
            while (pos[0] < len && (line.charAt(pos[0]) == ':' || line.charAt(pos[0]) == ' ')) pos[0]++;

            // Parse value
            if (pos[0] < len && line.charAt(pos[0]) == '{') {
                // Nested object — recurse with dot-prefixed key
                parseObject(line, pos, fullKey, result);
            } else if (pos[0] < len && line.charAt(pos[0]) == '"') {
                // String value
                pos[0]++;
                StringBuilder sb = new StringBuilder();
                while (pos[0] < len && line.charAt(pos[0]) != '"') {
                    if (line.charAt(pos[0]) == '\\' && pos[0] + 1 < len) {
                        pos[0]++;
                        switch (line.charAt(pos[0])) {
                            case '"':  sb.append('"'); break;
                            case '\\': sb.append('\\'); break;
                            case 'n':  sb.append('\n'); break;
                            case 't':  sb.append('\t'); break;
                            default:   sb.append(line.charAt(pos[0])); break;
                        }
                    } else {
                        sb.append(line.charAt(pos[0]));
                    }
                    pos[0]++;
                }
                pos[0]++; // skip closing quote
                result.put(fullKey, sb.toString());
            } else {
                // Number, boolean, or null
                int valStart = pos[0];
                while (pos[0] < len && line.charAt(pos[0]) != ',' && line.charAt(pos[0]) != '}' && line.charAt(pos[0]) != ' ') pos[0]++;
                result.put(fullKey, line.substring(valStart, pos[0]));
            }
        }
    }
}
