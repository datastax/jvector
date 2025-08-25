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
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the BenchmarkSummarizer class.
 */
public class BenchmarkSummarizerTest {

    @Test
    public void testSummarizeWithValidResults() {
        // Create sample benchmark results
        List<BenchResult> results = new ArrayList<>();
        results.add(createBenchResult("dataset1", "config1", 0.85, 1200.0, 5.2));
        results.add(createBenchResult("dataset1", "config2", 0.78, 1500.0, 4.1));
        results.add(createBenchResult("dataset2", "config1", 0.92, 900.0, 7.3));
        results.add(createBenchResult("dataset2", "config2", 0.88, 1100.0, 6.5));
        
        // Calculate summary statistics
        SummaryStats stats = BenchmarkSummarizer.summarize(results);
        
        // Verify results
        assertEquals(4, stats.getTotalConfigurations());
        assertEquals(0.8575, stats.getAvgRecall(), 0.0001);
        assertEquals(1175.0, stats.getAvgQps(), 0.01);
        assertEquals(5.775, stats.getAvgLatency(), 0.001);
    }
    
    @Test
    public void testSummarizeWithMissingMetrics() {
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
        assertEquals(3, stats.getTotalConfigurations());
        assertEquals(0.85, stats.getAvgRecall(), 0.0001);
        assertEquals(1050.0, stats.getAvgQps(), 0.01);
        assertEquals(4.65, stats.getAvgLatency(), 0.001);
    }
    
    @Test
    public void testSummarizeWithEmptyList() {
        // Test with empty list
        SummaryStats stats = BenchmarkSummarizer.summarize(new ArrayList<>());
        
        // Verify results
        assertEquals(0, stats.getTotalConfigurations());
        assertEquals(0.0, stats.getAvgRecall(), 0.0001);
        assertEquals(0.0, stats.getAvgQps(), 0.01);
        assertEquals(0.0, stats.getAvgLatency(), 0.001);
    }
    
    @Test
    public void testSummarizeWithNullList() {
        // Test with null list
        SummaryStats stats = BenchmarkSummarizer.summarize(null);
        
        // Verify results
        assertEquals(0, stats.getTotalConfigurations());
        assertEquals(0.0, stats.getAvgRecall(), 0.0001);
        assertEquals(0.0, stats.getAvgQps(), 0.01);
        assertEquals(0.0, stats.getAvgLatency(), 0.001);
    }
    
    @Test
    public void testSummaryStatsToString() {
        // Create a SummaryStats instance
        SummaryStats stats = new SummaryStats(0.85, 1200.0, 5.2, 1000000, 4);
        
        // Verify toString output
        String expected = String.format(
            "Benchmark Summary (across %d configurations):%n" +
            "  Average Recall@k: %.4f%n" +
            "  Average QPS: %.2f (± %.2f)%n" +
            "  Average Latency: %.2f ms%n" +
            "  Index Construction Time: %.2f",
            4, 0.85, 1200.0, 0.0, 5.2, 1000000.00);
        
        assertEquals(expected, stats.toString());
    }
    
    private BenchResult createBenchResult(String dataset, String config, 
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
}
