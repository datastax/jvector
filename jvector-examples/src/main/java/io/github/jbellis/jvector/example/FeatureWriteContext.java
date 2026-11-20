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

import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.RandomAccessOnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Supplier;

/** What a {@link FeatureProvider} needs to add its feature to one graph index writer. */
final class FeatureWriteContext {
    private final RandomAccessOnDiskGraphIndexWriter.Builder writer;
    private final ImmutableGraphIndex graph;
    // Exactly one of vectors and byteVectors is set, according to the dataset's element type.
    private final RandomAccessVectorValues vectors;
    private final RandomAccessByteVectorValues byteVectors;
    // File-backed sources reuse scratch vectors; ordinal suppliers need a view per worker.
    private final Supplier<RandomAccessVectorValues> writerVectors;
    private final PQVectors buildPQ;
    private final ConstructionMetrics metrics;

    private final Map<FeatureId, IntFunction<Feature.State>> ordinalSuppliers = new EnumMap<>(FeatureId.class);
    private final Map<FeatureId, Function<VectorFloat<?>, Feature.State>> vectorSuppliers = new EnumMap<>(FeatureId.class);
    private final Map<FeatureId, Function<ImmutableGraphIndex.View, IntFunction<Feature.State>>> afterBuildSuppliers =
            new EnumMap<>(FeatureId.class);

    /**
     * @param buildPQ PQ-encoded build vectors, or null for full-precision and non-PQ builds
     * @param metrics construction metrics sink, or null when not reporting
     */
    FeatureWriteContext(RandomAccessOnDiskGraphIndexWriter.Builder writer,
                        ImmutableGraphIndex graph,
                        RandomAccessVectorValues vectors,
                        PQVectors buildPQ,
                        ConstructionMetrics metrics) {
        this.writer = writer;
        this.graph = graph;
        this.vectors = vectors;
        this.byteVectors = null;
        this.writerVectors = vectors.threadLocalSupplier();
        this.buildPQ = buildPQ;
        this.metrics = metrics;
    }

    /** A context for an int8 dataset: no float vectors, no PQ, no metrics. */
    FeatureWriteContext(RandomAccessOnDiskGraphIndexWriter.Builder writer,
                        ImmutableGraphIndex graph,
                        RandomAccessByteVectorValues byteVectors) {
        this.writer = writer;
        this.graph = graph;
        this.vectors = null;
        this.byteVectors = byteVectors;
        this.writerVectors = null;
        this.buildPQ = null;
        this.metrics = null;
    }

    /** The graph being written; complete only for in-memory builds. */
    ImmutableGraphIndex graph() { return graph; }

    /**
     * The full-precision base vectors.
     *
     * @throws IllegalStateException for an int8 dataset, whose features cannot be built from float vectors
     */
    RandomAccessVectorValues vectors() {
        if (vectors == null) {
            throw new IllegalStateException("This is an int8 dataset; the feature needs float vectors");
        }
        return vectors;
    }

    /**
     * The int8 base vectors.
     *
     * @throws IllegalStateException for a float dataset
     */
    RandomAccessByteVectorValues byteVectors() {
        if (byteVectors == null) {
            throw new IllegalStateException("This is a float dataset; the feature needs int8 vectors");
        }
        return byteVectors;
    }

    /** PQ-encoded build vectors, or null for full-precision and non-PQ builds. */
    PQVectors buildPQ() { return buildPQ; }

    void add(Feature feature) {
        writer.with(feature);
    }

    /**
     * Registers a feature whose state is encoded from a node's original vector. Grid applies it during
     * insertion, at the final write, or by ordinal, depending on the writer.
     */
    void supplyFromVector(FeatureId id, Function<VectorFloat<?>, Feature.State> encode) {
        vectorSuppliers.put(id, encode);
        ordinalSuppliers.put(id, ordinal -> encode.apply(writerVectors.get().getVector(ordinal)));
    }

    /** Registers a feature whose per-node state is computed from the node's ordinal alone. */
    void supplyByOrdinal(FeatureId id, IntFunction<Feature.State> supplier) {
        ordinalSuppliers.put(id, supplier);
    }

    /** Registers a feature whose state needs the completed graph, such as fused neighbor codes. */
    void supplyAfterBuild(FeatureId id, Function<ImmutableGraphIndex.View, IntFunction<Feature.State>> supplier) {
        afterBuildSuppliers.put(id, supplier);
    }

    /** Times a quantizer computed for this index under the given type label, when metrics are being recorded. */
    <T> T timeIndexCompute(String quantType, Supplier<T> compute) {
        return metrics == null ? compute.get() : metrics.index(quantType).timeCompute(compute);
    }

    RandomAccessOnDiskGraphIndexWriter buildWriter() throws IOException {
        return writer.build();
    }

    Map<FeatureId, IntFunction<Feature.State>> ordinalSuppliers() { return ordinalSuppliers; }

    Map<FeatureId, Function<VectorFloat<?>, Feature.State>> vectorSuppliers() { return vectorSuppliers; }

    Map<FeatureId, Function<ImmutableGraphIndex.View, IntFunction<Feature.State>>> afterBuildSuppliers() {
        return afterBuildSuppliers;
    }
}
