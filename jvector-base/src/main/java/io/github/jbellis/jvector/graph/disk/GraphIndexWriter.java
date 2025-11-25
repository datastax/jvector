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
 * Use {@link #getBuilderFor(GraphIndexWriterTypes, ImmutableGraphIndex, IndexWriter)}
 * or {@link #getBuilderFor(GraphIndexWriterTypes, ImmutableGraphIndex, Path)}
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
     * Factory method to obtain a builder for the specified writer type with an IndexWriter.
     * <p>
     * This overload accepts any IndexWriter but certain types have specific requirements:
     * <ul>
     *   <li>ON_DISK requires a RandomAccessWriter (will throw IllegalArgumentException otherwise)</li>
     *   <li>ON_DISK_SEQUENTIAL accepts any IndexWriter</li>
     *   <li>ON_DISK_PARALLEL is not supported via this method (use the Path overload instead)</li>
     * </ul>
     *
     * @param type the type of writer to create
     * @param graphIndex the graph index to write
     * @param out the output writer
     * @return a builder for the specified writer type
     * @throws IllegalArgumentException if the type requires a specific writer type that wasn't provided
     */
    static AbstractGraphIndexWriter.Builder<? extends AbstractGraphIndexWriter<?>, ? extends IndexWriter>
            getBuilderFor(GraphIndexWriterTypes type, ImmutableGraphIndex graphIndex, IndexWriter out) {
        switch (type) {
            case ON_DISK_PARALLEL:
                if (!(out instanceof RandomAccessWriter)) {
                    throw new IllegalArgumentException("ON_DISK_PARALLEL requires a RandomAccessWriter");
                }
                return new OnDiskGraphIndexWriter.Builder(graphIndex, (RandomAccessWriter) out);
            case ON_DISK_SEQUENTIAL:
                return new OnDiskSequentialGraphIndexWriter.Builder(graphIndex, out);
            default:
                throw new IllegalArgumentException("Unknown GraphIndexWriterType: " + type);
        }
    }

    /**
     * Factory method to obtain a builder for the specified writer type with a file Path.
     * <p>
     * This overload accepts a Path and is required for:
     * <ul>
     *   <li>ON_DISK_PARALLEL - enables async I/O for improved throughput</li>
     * </ul>
     * Other writer types should use the {@link #getBuilderFor(GraphIndexWriterTypes, ImmutableGraphIndex, IndexWriter)}
     * overload instead.
     *
     * @param type the type of writer to create (currently only ON_DISK_PARALLEL is supported)
     * @param graphIndex the graph index to write
     * @param out the output file path
     * @return a builder for the specified writer type
     * @throws FileNotFoundException if the file cannot be created or opened
     * @throws IllegalArgumentException if the type is not supported via this method
     */
    static AbstractGraphIndexWriter.Builder<? extends AbstractGraphIndexWriter<?>, ? extends IndexWriter>
    getBuilderFor(GraphIndexWriterTypes type, ImmutableGraphIndex graphIndex, Path out) throws FileNotFoundException {
        switch (type) {
            case ON_DISK_PARALLEL:
                return new OnDiskGraphIndexWriter.Builder(graphIndex, out);
            default:
                throw new IllegalArgumentException("Unknown GraphIndexWriterType: " + type);
        }
    }
}
