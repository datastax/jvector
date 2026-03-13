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
import io.github.jbellis.jvector.example.util.BenchmarkSummarizer.SummaryStats;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Simple test class to verify the BenchmarkSummarizer functionality.
 */
public class SummarizerTest {
    public static void main(String[] args) {
        System.out.println("Running BenchmarkSummarizer tests...");
        
        // Test with valid results
        testSummarizeWithValidResults();
        
        // Test with missing metrics
        testSummarizeWithMissingMetrics();
        
        // Test with empty list
        testSummarizeWithEmptyList();
        
        // Test toString method
        testSummaryStatsToString();
        
        System.out.println("All tests completed successfully!");
    }
    
    private static void testSummarizeWithValidResults() {
        System.out.println("\nTest: Summarize with valid results");
        
        // Create sample benchmark results
        List<BenchResult> results = new ArrayList<>();
        results.add(createBenchResult("dataset1", "config1", 0.85, 1200.0, 5.2));
        results.add(createBenchResult("dataset1", "config2", 0.78, 1500.0, 4.1));
        results.add(createBenchResult("dataset2", "config1", 0.92, 900.0, 7.3));
        results.add(createBenchResult("dataset2", "config2", 0.88, 1100.0, 6.5));
        
        // Calculate summary statistics
        SummaryStats stats = BenchmarkSummarizer.summarize(results);
        
        // Verify results
        assertEquals("Total configurations", 4, stats.getTotalConfigurations());
        assertEquals("Average Recall", 0.8575, stats.getAvgRecall(), 0.0001);
        assertEquals("Average QPS", 1175.0, stats.getAvgQps(), 0.01);
        assertEquals("Average Latency", 5.775, stats.getAvgLatency(), 0.001);
    }
    
    private static void testSummarizeWithMissingMetrics() {
        System.out.println("\nTest: Summarize with missing metrics");
        
        // Create sample benchmark results with missing metrics
        List<BenchResult> results = new ArrayList<>();
        results.add(createBenchResult("dataset1", "config1", 0.85, 1200.0, 5.2));
        
        // Add result with missing QPS
        BenchResult resultMissingQps = new BenchResult();
        resultMissingQps.dataset = "dataset2";
        Map<String, Object> params = new HashMap<>();
        params.put("config", "config2");
        resultMissingQps.parameters = params;
        resultMissingQps.metrics = new HashMap<>();
        resultMissingQps.metrics.put("Recall@10", 0.78);
        resultMissingQps.metrics.put("Mean Latency (ms)", 4.1);
        results.add(resultMissingQps);
        
        // Add result with missing latency
        BenchResult resultMissingLatency = new BenchResult();
        resultMissingLatency.dataset = "dataset3";
        Map<String, Object> paramsLatency = new HashMap<>();
        paramsLatency.put("config", "config3");
        resultMissingLatency.parameters = paramsLatency;
        resultMissingLatency.metrics = new HashMap<>();
        resultMissingLatency.metrics.put("Recall@10", 0.92);
        resultMissingLatency.metrics.put("QPS", 900.0);
        results.add(resultMissingLatency);
        
        // Calculate summary statistics
        SummaryStats stats = BenchmarkSummarizer.summarize(results);
        
        // Verify results
        assertEquals("Total configurations", 3, stats.getTotalConfigurations());
        assertEquals("Average Recall", 0.85, stats.getAvgRecall(), 0.0001);
        assertEquals("Average QPS", 1050.0, stats.getAvgQps(), 0.01);
        assertEquals("Average Latency", 4.65, stats.getAvgLatency(), 0.001);
    }
    
    private static void testSummarizeWithEmptyList() {
        System.out.println("\nTest: Summarize with empty list");
        
        // Test with empty list
        SummaryStats stats = BenchmarkSummarizer.summarize(new ArrayList<>());
        
        // Verify results
        assertEquals("Total configurations", 0, stats.getTotalConfigurations());
        assertEquals("Average Recall", 0.0, stats.getAvgRecall(), 0.0001);
        assertEquals("Average QPS", 0.0, stats.getAvgQps(), 0.01);
        assertEquals("Average Latency", 0.0, stats.getAvgLatency(), 0.001);
    }
    
    private static void testSummaryStatsToString() {
        System.out.println("\nTest: SummaryStats toString method");
        
        // Create a SummaryStats instance
        SummaryStats stats = new SummaryStats(0.85, 1200.0, 5.2, 1000000, 4);
        
        // Verify toString output
        String expected = String.format(
            "Benchmark Summary (across %d configurations):%n" +
            "  Average Recall@k: %.4f%n" +
            "  Average QPS: %.2f (± %.2f)%n" +
            "  Average Latency: %.2f ms",
            4, 0.85, 1200.0, 0.0, 5.2);
        
        assertEquals("toString output", expected, stats.toString());
    }
    
    private static BenchResult createBenchResult(String dataset, String config, 
                                               double recall, double qps, double latency) {
        BenchResult result = new BenchResult();
        result.dataset = dataset;
        
        Map<String, Object> params = new HashMap<>();
        params.put("config", config);
        result.parameters = params;
        
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("Recall@10", recall);
        metrics.put("QPS", qps);
        metrics.put("Mean Latency (ms)", latency);
        result.metrics = metrics;
        
        return result;
    }
    
    private static void assertEquals(String message, int expected, int actual) {
        if (expected != actual) {
            throw new AssertionError(message + " - Expected: " + expected + ", Actual: " + actual);
        }
        System.out.println("✓ " + message + " - Value: " + actual);
    }
    
    private static void assertEquals(String message, double expected, double actual, double delta) {
        if (Math.abs(expected - actual) > delta) {
            throw new AssertionError(message + " - Expected: " + expected + ", Actual: " + actual);
        }
        System.out.println("✓ " + message + " - Value: " + actual);
    }
    
    private static void assertEquals(String message, String expected, String actual) {
        if (!expected.equals(actual)) {
            throw new AssertionError(message + " - Expected: " + expected + ", Actual: " + actual);
        }
        System.out.println("✓ " + message + " - Value matches expected");
    }
}
