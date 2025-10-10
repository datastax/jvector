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
 * Provides low-level I/O abstractions for reading and writing binary data.
 * <p>
 * This package contains interfaces and implementations for efficient random access I/O operations
 * used throughout JVector. These abstractions support both memory-mapped and traditional buffered
 * I/O strategies, enabling optimal performance across different use cases and storage backends.
 *
 * <h2>Core Abstractions</h2>
 *
 * <h3>Reader Interfaces</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.disk.RandomAccessReader} - Interface for reading data
 *       with seek capability. Supports reading primitive types (int, long, float) and bulk reads
 *       into arrays and buffers. Designed for sequential reads after seeking to a position.</li>
 *   <li>{@link io.github.jbellis.jvector.disk.ReaderSupplier} - Factory interface for creating
 *       {@code RandomAccessReader} instances. Used to provide thread-local readers since
 *       {@code RandomAccessReader} implementations are stateful and not thread-safe.</li>
 *   <li>{@link io.github.jbellis.jvector.disk.ReaderSupplierFactory} - Factory for creating
 *       {@code ReaderSupplier} instances from files. Provides the recommended entry point for
 *       opening files for reading.</li>
 * </ul>
 *
 * <h3>Writer Interfaces</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.disk.IndexWriter} - Base interface for sequential data
 *       writing with position tracking.</li>
 *   <li>{@link io.github.jbellis.jvector.disk.RandomAccessWriter} - Extends {@code IndexWriter}
 *       with seek capability for random access writes and checksum computation.</li>
 * </ul>
 *
 * <h2>Implementations</h2>
 *
 * <h3>Readers</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.disk.SimpleReader} - Buffered file reader using
 *       {@code FileChannel}</li>
 *   <li>{@link io.github.jbellis.jvector.disk.SimpleMappedReader} - Memory-mapped file reader
 *       for optimal performance with large files</li>
 *   <li>{@link io.github.jbellis.jvector.disk.ByteBufferReader} - Reader backed by a
 *       {@code ByteBuffer}, useful for in-memory data</li>
 *   <li>{@link io.github.jbellis.jvector.disk.MappedChunkReader} - Chunked memory-mapped reader
 *       for handling files larger than the maximum mapping size</li>
 * </ul>
 *
 * <h3>Writers</h3>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.disk.SimpleWriter} - Basic file writer using
 *       {@code FileChannel}</li>
 *   <li>{@link io.github.jbellis.jvector.disk.BufferedRandomAccessWriter} - Buffered writer with
 *       random access and checksum support, recommended for most writing scenarios</li>
 * </ul>
 *
 * <h2>Usage Pattern</h2>
 * <p>
 * The recommended usage pattern for reading is:
 * <pre>{@code
 * // Open a file with a ReaderSupplierFactory
 * try (ReaderSupplier readerSupplier = ReaderSupplierFactory.open(path)) {
 *     // Get a thread-local reader
 *     try (RandomAccessReader reader = readerSupplier.get()) {
 *         reader.seek(offset);
 *         int value = reader.readInt();
 *         float[] vector = new float[dimension];
 *         reader.readFully(vector);
 *     }
 * }
 * }</pre>
 *
 * <p>
 * For writing:
 * <pre>{@code
 * try (RandomAccessWriter writer = new BufferedRandomAccessWriter(path)) {
 *     writer.writeInt(42);
 *     writer.writeFloat(3.14f);
 *     long position = writer.getPosition();
 *     writer.seek(0);  // Go back and update header
 *     writer.writeLong(position);
 * }
 * }</pre>
 *
 * <h2>Thread Safety</h2>
 * <ul>
 *   <li>{@code RandomAccessReader} implementations are <b>not thread-safe</b>. Use
 *       {@code ReaderSupplier} to create separate instances per thread.</li>
 *   <li>{@code RandomAccessWriter} implementations are <b>not thread-safe</b>. Coordinate
 *       access externally if needed.</li>
 *   <li>{@code ReaderSupplier} implementations are typically thread-safe and can be shared
 *       across threads to create per-thread readers.</li>
 * </ul>
 *
 * @see io.github.jbellis.jvector.disk.RandomAccessReader
 * @see io.github.jbellis.jvector.disk.ReaderSupplierFactory
 * @see io.github.jbellis.jvector.disk.RandomAccessWriter
 */
package io.github.jbellis.jvector.disk;
