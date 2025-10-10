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

/**
 * JMH benchmarks for measuring the performance of JVector's core components.
 * <p>
 * This package contains Java Microbenchmark Harness (JMH) benchmarks that evaluate
 * various aspects of JVector's vector search functionality, including:
 * <ul>
 *   <li>Product Quantization (PQ) training and distance calculations</li>
 *   <li>Graph construction and search performance</li>
 *   <li>Recall quality measurement</li>
 *   <li>Performance comparisons between full-precision and quantized vectors</li>
 * </ul>
 *
 * <h2>Benchmark Categories</h2>
 *
 * <h3>Product Quantization Benchmarks</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.bench.PQDistanceCalculationBenchmark} - Compares distance
 *       calculation performance between full-precision and PQ-compressed vectors</li>
 *   <li>{@link io.github.jbellis.jvector.bench.PQTrainingWithRandomVectorsBenchmark} - Measures
 *       PQ training time on randomly generated vectors</li>
 *   <li>{@link io.github.jbellis.jvector.bench.PQTrainingWithSiftBenchmark} - Measures PQ training
 *       time on the SIFT Small dataset with real-world vectors</li>
 * </ul>
 *
 * <h3>Graph Search Benchmarks</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.bench.RecallWithRandomVectorsBenchmark} - Evaluates
 *       search performance and recall quality with and without PQ compression on random vectors</li>
 *   <li>{@link io.github.jbellis.jvector.bench.StaticSetVectorsBenchmark} - Measures pure search
 *       throughput on the SIFT Small dataset</li>
 * </ul>
 *
 * <h2>Running Benchmarks</h2>
 *
 * <p>
 * These benchmarks are packaged as a standalone executable JAR using Maven Shade Plugin.
 * To build and run the benchmarks:
 * </p>
 *
 * <pre>
 * # Build the shaded JAR
 * mvn clean package -pl benchmarks-jmh
 *
 * # Run all benchmarks
 * java -jar benchmarks-jmh/target/benchmarks-jmh-*.jar
 *
 * # Run a specific benchmark
 * java -jar benchmarks-jmh/target/benchmarks-jmh-*.jar PQDistanceCalculationBenchmark
 *
 * # Run with custom JMH options
 * java -jar benchmarks-jmh/target/benchmarks-jmh-*.jar -h  # Show help
 * </pre>
 *
 * <h2>Benchmark Configuration</h2>
 *
 * <p>
 * Most benchmarks use JMH annotations to configure:
 * </p>
 * <ul>
 *   <li><b>Mode:</b> Typically AverageTime (measures average execution time per operation)</li>
 *   <li><b>Time Units:</b> Typically MILLISECONDS for reporting results</li>
 *   <li><b>Warmup:</b> Multiple iterations to allow JVM warmup and JIT compilation</li>
 *   <li><b>Measurement:</b> Multiple iterations for statistically significant results</li>
 *   <li><b>Fork:</b> Usually 1 fork for faster execution, increase for production-grade results</li>
 *   <li><b>Parameters:</b> {@code @Param} annotations define benchmark variants (dimensions, vector counts, etc.)</li>
 * </ul>
 *
 * <h2>Data Sources</h2>
 *
 * <p>
 * The benchmarks use two types of vector datasets:
 * </p>
 * <ul>
 *   <li><b>Random Vectors:</b> Generated programmatically with uniform random values,
 *       useful for controlled testing and scalability evaluation</li>
 *   <li><b>SIFT Small Dataset:</b> Real-world image feature vectors (10,000 base vectors,
 *       100 queries, 128 dimensions), available at
 *       <a href="http://corpus-texmex.irisa.fr/">http://corpus-texmex.irisa.fr/</a></li>
 * </ul>
 *
 * <h2>Understanding Results</h2>
 *
 * <p>
 * JMH produces detailed output including:
 * </p>
 * <ul>
 *   <li><b>Score:</b> Average time per operation (or ops/sec depending on mode)</li>
 *   <li><b>Error:</b> 99.9% confidence interval (lower is better)</li>
 *   <li><b>Auxiliary Counters:</b> Some benchmarks report additional metrics like recall,
 *       visited nodes, etc.</li>
 * </ul>
 *
 * <h2>Best Practices</h2>
 *
 * <ul>
 *   <li>Run benchmarks on a quiet system with minimal background processes</li>
 *   <li>Use multiple forks ({@code -f 3}) for production-grade measurements</li>
 *   <li>Increase warmup and measurement iterations for stable results</li>
 *   <li>Be aware that vector incubator module performance may vary by platform</li>
 *   <li>Consider using JMH profilers ({@code -prof}) to understand hotspots</li>
 * </ul>
 *
 * @see <a href="https://github.com/openjdk/jmh">JMH Documentation</a>
 * @see <a href="http://corpus-texmex.irisa.fr/">SIFT Dataset</a>
 */
package io.github.jbellis.jvector.bench;
