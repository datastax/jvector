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
 * A supplier of RandomAccessReaders.
 */
public interface ReaderSupplier extends AutoCloseable {
    /**
     * Creates a new reader.
     * @return a new reader.  It is up to the caller to re-use these readers or close them,
     * the ReaderSupplier is not responsible for caching them.
     * @throws IOException if an I/O error occurs
     */
    RandomAccessReader get() throws IOException;

    default void close() throws IOException {
    }
}
