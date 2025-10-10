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

package io.github.jbellis.jvector.disk;

import java.io.IOException;

/**
 * A supplier of RandomAccessReaders that provides access to the underlying index file.
 * Implementations are responsible for opening new readers but not for managing their lifecycle
 * after creation.
 */
public interface ReaderSupplier extends AutoCloseable {
    /**
     * Creates and returns a new RandomAccessReader for accessing the index file. Each call returns
     * a distinct reader instance. Callers are responsible for managing the lifecycle of returned
     * readers, including closing them when done or reusing them as needed.
     * @return a new reader positioned at the beginning of the file
     * @throws IOException if an I/O error occurs while opening the reader
     */
    RandomAccessReader get() throws IOException;

    default void close() throws IOException {
    }
}
