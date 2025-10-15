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
     * Sequential on-disk writer optimized for write-once scenarios.
     * Writes all data sequentially without seeking back, making it suitable
     * for cloud storage or systems that optimize for sequential I/O.
     * Writes header as footer. Does not support incremental updates.
     * Accepts any IndexWriter.
     */
    ON_DISK_SEQUENTIAL,

    /**
     * Parallel on-disk writer that uses asynchronous I/O for improved throughput.
     * Builds records in parallel across multiple threads and writes them
     * asynchronously using AsynchronousFileChannel.
     * Requires a Path to be provided for async file channel access.
     */
    ON_DISK_PARALLEL
}
