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
package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.example.BenchResult;

import java.util.List;
import java.util.Map;

/**
 * Utility class for summarizing benchmark results by calculating average metrics
 * across all configurations.
 */
public class BenchmarkSummarizer {
    
    /**
     * Summary statistics for benchmark results
     */
    public static class SummaryStats {
        private final double avgRecall;
        private final double avgQps;
        private final double avgLatency;
        private final double indexConstruction;
        private final int totalConfigurations;
        private final double qpsStdDev;

        public SummaryStats(double avgRecall, double avgQps, double avgLatency, double indexConstruction, int totalConfigurations) {
            this(avgRecall, avgQps, avgLatency, indexConstruction, totalConfigurations, 0.0);
        }

        public SummaryStats(double avgRecall, double avgQps, double avgLatency, double indexConstruction, int totalConfigurations, double qpsStdDev) {
            this.avgRecall = avgRecall;
            this.avgQps = avgQps;
            this.avgLatency = avgLatency;
            this.indexConstruction = indexConstruction;
            this.totalConfigurations = totalConfigurations;
            this.qpsStdDev = qpsStdDev;
        }

        public double getAvgRecall() {
            return avgRecall;
        }

        public double getAvgQps() {
            return avgQps;
        }

        public double getAvgLatency() {
            return avgLatency;
        }

        public double getIndexConstruction() { return indexConstruction; }

        public int getTotalConfigurations() {
            return totalConfigurations;
        }

        public double getQpsStdDev() { return qpsStdDev; }

        @Override
        public String toString() {
            return String.format(
                "Benchmark Summary (across %d configurations):%n" +
                "  Average Recall@k: %.4f%n" +
                "  Average QPS: %.2f (± %.2f)%n" +
                "  Average Latency: %.2f ms%n" +
                "  Index Construction Time: %.2f",
                totalConfigurations, avgRecall, avgQps, qpsStdDev, avgLatency, indexConstruction);
        }
    }
    
    /**
     * Calculate summary statistics from a list of benchmark results
     * @param results List of benchmark results to summarize
     * @return SummaryStats containing average metrics
     */
    public static SummaryStats summarize(List<BenchResult> results) {
        if (results == null || results.isEmpty()) {
            return new SummaryStats(0, 0, 0, 0, 0, 0);
        }

        double totalRecall = 0;
        double totalQps = 0;
        double totalLatency = 0;
        double indexConstruction = 0;
        double totalQpsStdDev = 0;
        
        int recallCount = 0;
        int qpsCount = 0;
        int latencyCount = 0;
        int qpsStdDevCount = 0;

        for (BenchResult result : results) {
            if (result.metrics == null) continue;
            
            // Extract recall metrics (format is "Recall@N" where N is the topK value)
            Double recall = extractRecallMetric(result.metrics);
            if (recall != null) {
                totalRecall += recall;
                recallCount++;
            }
            
            // Extract QPS metric
            Double qps = extractQpsMetric(result.metrics);
            if (qps != null) {
                totalQps += qps;
                qpsCount++;
            }

            // Extract QPS StdDev metric (key from ThroughputBenchmark is "± Std Dev")
            Double qpsStdDev = extractQpsStdDevMetric(result.metrics);
            if (qpsStdDev != null) {
                totalQpsStdDev += qpsStdDev;
                qpsStdDevCount++;
            }
            
            // Extract latency metric (format is "Mean Latency (ms)")
            Double latency = extractLatencyMetric(result.metrics);
            if (latency != null) {
                totalLatency += latency;
                latencyCount++;
            }

            indexConstruction = extractIndexConstructionMetric(result.metrics);
        }

        // Calculate averages, handling cases where some metrics might not be present
        double avgRecall = recallCount > 0 ? totalRecall / recallCount : 0;
        double avgQps = qpsCount > 0 ? totalQps / qpsCount : 0;
        double avgLatency = latencyCount > 0 ? totalLatency / latencyCount : 0;
        double avgQpsStdDev = qpsStdDevCount > 0 ? totalQpsStdDev / qpsStdDevCount : 0;
        
        // Count total valid configurations as the maximum count of any metric
        int totalConfigurations = Math.max(Math.max(recallCount, qpsCount), latencyCount);

        return new SummaryStats(avgRecall, avgQps, avgLatency, indexConstruction, totalConfigurations, avgQpsStdDev);
    }

    private static Double extractIndexConstructionMetric(Map<String, Object> metrics) {
        // Look for keys starting with "Index Build Time"
        for (Map.Entry<String, Object> entry : metrics.entrySet()) {
            if (entry.getKey().startsWith("Index Build Time")) {
                return convertToDouble(entry.getValue());
            }
        }
        return 0.0;
    }
    
    /**
     * Extract a recall metric from the metrics map
     * @param metrics Map of metrics
     * @return The recall value as a Double, or null if not found
     */
    private static Double extractRecallMetric(Map<String, Object> metrics) {
        // Look for keys starting with "Recall@"
        for (Map.Entry<String, Object> entry : metrics.entrySet()) {
            if (entry.getKey().startsWith("Recall@")) {
                return convertToDouble(entry.getValue());
            }
        }
        return null;
    }
    
    /**
     * Extract a latency metric from the metrics map
     * @param metrics Map of metrics
     * @return The latency value as a Double, or null if not found
     */
    private static Double extractLatencyMetric(Map<String, Object> metrics) {
        // Try different variations of latency metrics
        Double value = extractMetric(metrics, "Mean Latency (ms)");
        if (value != null) return value;
        
        value = extractMetric(metrics, "Avg Runtime (s)");
        if (value != null) return value * 1000; // Convert seconds to milliseconds
        
        // Look for any key containing "latency" case insensitive
        for (Map.Entry<String, Object> entry : metrics.entrySet()) {
            if (entry.getKey().toLowerCase().contains("latency")) {
                return convertToDouble(entry.getValue());
            }
        }
        
        return null;
    }
    
    /**
     * Extract a QPS metric from the metrics map
     * @param metrics Map of metrics
     * @return The QPS value as a Double, or null if not found
     */
    private static Double extractQpsMetric(Map<String, Object> metrics) {
        // Try exact match first
        Double value = extractMetric(metrics, "QPS");
        if (value != null) return value;
        
        // Look for any key containing "QPS" case insensitive
        for (Map.Entry<String, Object> entry : metrics.entrySet()) {
            if (entry.getKey().contains("QPS")) {
                return convertToDouble(entry.getValue());
            }
        }
        
        return null;
    }

    /**
     * Extract a QPS standard deviation metric from the metrics map. We accept keys like "± Std Dev" or ones containing
     * "Std Dev" along with QPS context.
     */
    private static Double extractQpsStdDevMetric(Map<String, Object> metrics) {
        // First, look for the exact key used by ThroughputBenchmark
        Double value = extractMetric(metrics, "± Std Dev");
        if (value != null) return value;

        // Fallback: any key that contains "Std Dev" (case sensitive as produced) or case-insensitive match
        for (Map.Entry<String, Object> entry : metrics.entrySet()) {
            String k = entry.getKey();
            if (k.contains("Std Dev") || k.toLowerCase().contains("std dev")) {
                return convertToDouble(entry.getValue());
            }
        }
        return null;
    }
    
    /**
     * Extract a specific metric from the metrics map
     * @param metrics Map of metrics
     * @param metricName Name of the metric to extract
     * @return The metric value as a Double, or null if not found
     */
    private static Double extractMetric(Map<String, Object> metrics, String metricName) {
        Object value = metrics.get(metricName);
        return convertToDouble(value);
    }
    
    /**
     * Convert an object to Double
     * @param value Object to convert
     * @return Double value, or null if conversion fails
     */
    private static Double convertToDouble(Object value) {
        if (value == null) return null;
        
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        } else if (value instanceof String) {
            try {
                return Double.parseDouble((String) value);
            } catch (NumberFormatException e) {
                return null;
            }
        }
        return null;
    }
    
    /**
     * Calculate summary statistics grouped by dataset from a list of benchmark results
     * @param results List of benchmark results to summarize
     * @return Map of dataset names to their summary statistics
     */
    public static Map<String, SummaryStats> summarizeByDataset(List<BenchResult> results) {
        if (results == null || results.isEmpty()) {
            return Map.of();
        }

        // Group results by dataset
        Map<String, List<BenchResult>> resultsByDataset = new java.util.HashMap<>();
        for (BenchResult result : results) {
            if (result.dataset == null) continue;
            
            resultsByDataset.computeIfAbsent(result.dataset, k -> new java.util.ArrayList<>()).add(result);
        }
        
        // Calculate summary stats for each dataset
        Map<String, SummaryStats> statsByDataset = new java.util.HashMap<>();
        for (Map.Entry<String, List<BenchResult>> entry : resultsByDataset.entrySet()) {
            String dataset = entry.getKey();
            List<BenchResult> datasetResults = entry.getValue();
            
            SummaryStats stats = summarize(datasetResults);
            statsByDataset.put(dataset, stats);
        }
        
        return statsByDataset;
    }
}
