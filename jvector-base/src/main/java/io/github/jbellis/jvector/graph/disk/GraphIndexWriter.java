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

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.Closeable;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * Interface for writing graph indices to various storage targets.
 * <p>
 * Implementations support different strategies for writing graph data,
 * including random access, sequential, and parallel writing modes.
 * Use {@link #getBuilderFor(GraphIndexWriterTypes, ImmutableGraphIndex, Path)}
 * factory methods to obtain appropriate builder instances.
 *
 * @see GraphIndexWriterTypes
 * @see OnDiskGraphIndexWriter
 * @see OnDiskSequentialGraphIndexWriter
 */
public interface GraphIndexWriter extends Closeable {
    /**
     * Write the index header and completed edge lists to the given outputs.  Inline features given in
     * `featureStateSuppliers` will also be written.  (Features that do not have a supplier are assumed
     * to have already been written by calls to writeInline).
     * <p>
     * Each supplier takes a node ordinal and returns a FeatureState suitable for Feature.writeInline.
     *
     * @param featureStateSuppliers a map of FeatureId to a function that returns a Feature.State
     * @throws IOException if an I/O error occurs
     */
    void write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException;

    /**
     * Factory method to obtain a builder for the specified writer type.
     *
     * @param type the type of writer to create
     * @param graphIndex the graph index to write
     * @param out the Path to the output file
     * @return a builder for the specified writer type
     * @throws IllegalArgumentException if the type requires a specific writer type that wasn't provided
     */
    static AbstractGraphIndexWriter.Builder<? extends AbstractGraphIndexWriter<?>, ? extends RandomAccessWriter>
            getBuilderFor(GraphIndexWriterTypes type, ImmutableGraphIndex graphIndex, Path out) throws FileNotFoundException {
        switch (type) {
            case RANDOM_ACCESS:
                return new OnDiskGraphIndexWriter.Builder(graphIndex, out);
            case RANDOM_ACCESS_PARALLEL:
                return new OnDiskParallelGraphIndexWriter.Builder(graphIndex, out);
            default:
                throw new IllegalArgumentException("Unknown RandomAccess GraphIndexWriterType: " + type);
        }
    }

    static AbstractGraphIndexWriter.Builder<? extends AbstractGraphIndexWriter<?>, ? extends IndexWriter>
            getBuilderFor(GraphIndexWriterTypes type, ImmutableGraphIndex graphIndex, IndexWriter out) {
        switch (type) {
            case ON_DISK_SEQUENTIAL:
                return new OnDiskSequentialGraphIndexWriter.Builder(graphIndex, out);
            default:
                throw new IllegalArgumentException("Unknown GraphIndexWriterType: " + type);
        }
    }
}
