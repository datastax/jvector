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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.vector.VectorizationProvider;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * Example demonstrating how to use parallel writes with OnDiskGraphIndexWriter.
 * <p>
 * Usage patterns:
 * <pre>
 * // Sequential (default):
 * var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
 *     .with(inlineVectors)
 *     .build();
 * writer.write(featureSuppliers);
 *
 * // Parallel:
 * var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
 *     .with(inlineVectors)
 *     .withParallelWrites(true)  // Enable parallel writes
 *     .build();
 * writer.write(featureSuppliers);
 * </pre>
 */
public class ParallelWriteExample {
    
    /**
     * Example: Write a graph index with parallel L0 writes enabled.
     */
    public static void writeGraphParallel(ImmutableGraphIndex graph,
                                          Path outputPath,
                                          IntFunction<float[]> vectorSupplier,
                                          int dimension) throws IOException {
        
        // Create inline vectors feature
        var inlineVectors = new InlineVectors(dimension);
        var vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
        
        // Build writer with parallel writes enabled
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                .withParallelWrites(true)
                .with(inlineVectors)
                .build()) {
            
            // Create feature state suppliers
            Map<FeatureId, IntFunction<Feature.State>> suppliers = 
                Feature.singleStateFactory(
                    FeatureId.INLINE_VECTORS,
                    ordinal -> new InlineVectors.State(vectorTypeSupport.createFloatVector(vectorSupplier.apply(ordinal)))
                );
            
            // Write the graph (L0 records will be built in parallel)
            writer.write(suppliers);
        }
    }
    
    /**
     * Example: Write a graph index with sequential writes (default behavior).
     */
    public static void writeGraphSequential(ImmutableGraphIndex graph,
                                            Path outputPath,
                                            IntFunction<float[]> vectorSupplier,
                                            int dimension) throws IOException {
        
        var inlineVectors = new InlineVectors(dimension);
        var vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
        
        // Default behavior is sequential
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                .withParallelWrites(false)
                .with(inlineVectors)
                .build()) {
            
            Map<FeatureId, IntFunction<Feature.State>> suppliers = 
                Feature.singleStateFactory(
                    FeatureId.INLINE_VECTORS,
                    ordinal -> new InlineVectors.State(vectorTypeSupport.createFloatVector(vectorSupplier.apply(ordinal)))
                );
            
            writer.write(suppliers);
        }
    }
    
    /**
     * Example: Benchmark comparison between sequential and parallel writes.
     */
    public static void benchmarkComparison(ImmutableGraphIndex graph,
                                          Path sequentialPath,
                                          Path parallelPath,
                                          IntFunction<float[]> vectorSupplier,
                                          int dimension) throws IOException {
        
        var inlineVectors = new InlineVectors(dimension);
        var vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
        Map<FeatureId, IntFunction<Feature.State>> suppliers = 
            Feature.singleStateFactory(
                FeatureId.INLINE_VECTORS,
                ordinal -> new InlineVectors.State(vectorTypeSupport.createFloatVector(vectorSupplier.apply(ordinal)))
            );
        
        // Sequential write
        long sequentialStart = System.nanoTime();
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, sequentialPath)
                .with(inlineVectors)
                .build()) {
            writer.write(suppliers);
        }
        long sequentialTime = System.nanoTime() - sequentialStart;
        
        // Parallel write
        long parallelStart = System.nanoTime();
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, parallelPath)
                .with(inlineVectors)
                .build()) {
            writer.write(suppliers);
        }
        long parallelTime = System.nanoTime() - parallelStart;
        
        System.out.printf("Sequential write: %.2f ms%n", sequentialTime / 1_000_000.0);
        System.out.printf("Parallel write:   %.2f ms%n", parallelTime / 1_000_000.0);
        System.out.printf("Speedup:          %.2fx%n", (double) sequentialTime / parallelTime);
    }
}
