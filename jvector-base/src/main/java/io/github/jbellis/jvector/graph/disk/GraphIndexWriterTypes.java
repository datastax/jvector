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

/**
 * Enum defining the available types of graph index writers.
 * <p>
 * Different writer types offer different tradeoffs between performance,
 * compatibility, and features.
 */
public enum GraphIndexWriterTypes {
    /**
     * Sequential on-disk writer ({@link OnDiskSequentialGraphIndexWriter}) optimized for write-once scenarios.
     * Writes all data sequentially without seeking back, making it suitable
     * for cloud storage or systems that optimize for sequential I/O.
     * Writes header as footer. Does not support incremental updates.
     * Accepts any IndexWriter.
     */
    ON_DISK_SEQUENTIAL,

    /**
     * Single-threaded random-access writer ({@link OnDiskGraphIndexWriter}); the default for
     * writing to a file. Writes a placeholder header, then the node records in ordinal order,
     * then seeks back to fill in the header. Supports writing at a start offset within an existing
     * file and writing individual nodes' inline features incrementally
     * ({@code writeFeaturesInline}). Accepts any RandomAccessWriter.
     */
    RANDOM_ACCESS,

    /**
     * Parallel random-access writer ({@link OnDiskParallelGraphIndexWriter}). Builds node records
     * in parallel across multiple threads and writes them asynchronously using
     * AsynchronousFileChannel, which is worthwhile when feature encoding (e.g. NVQ) dominates write
     * time. Requires a Path to be provided for async file channel access.
     */
    RANDOM_ACCESS_PARALLEL
}
