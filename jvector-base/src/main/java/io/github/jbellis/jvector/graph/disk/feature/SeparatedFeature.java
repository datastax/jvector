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

package io.github.jbellis.jvector.graph.disk.feature;

import java.io.DataOutput;
import java.io.IOException;

/**
 * A feature that is written separately from the main graph structure, with an offset
 * reference stored in the header to locate its data in a separate section of the file.
 */
public interface SeparatedFeature extends Feature {
    /**
     * Sets the file offset where this feature's data begins in the separated section.
     * @param offset the byte offset in the file
     */
    void setOffset(long offset);

    /**
     * Returns the file offset where this feature's data is stored.
     * @return the byte offset in the file
     */
    long getOffset();

    /**
     * Writes this feature's data to the separated section of the file.
     * @param out the output stream to write to
     * @param state the feature state containing the data to write
     * @throws IOException if an I/O error occurs
     */
    void writeSeparately(DataOutput out, State state) throws IOException;
}
