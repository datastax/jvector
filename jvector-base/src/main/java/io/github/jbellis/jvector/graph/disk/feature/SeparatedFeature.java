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
 * A feature whose data is stored separately from the main graph structure.
 * Separated features write their data to a separate location on disk, with only
 * the offset information stored in the graph header. This is useful for large
 * features that would make inline storage inefficient.
 */
public interface SeparatedFeature extends Feature {
    /**
     * Sets the file offset where this feature's data begins.
     *
     * @param offset the file offset in bytes
     */
    void setOffset(long offset);

    /**
     * Returns the file offset where this feature's data begins.
     *
     * @return the file offset in bytes
     */
    long getOffset();

    /**
     * Writes this feature's data to the specified output, separate from the graph structure.
     * This method is called during graph serialization to write feature data to its dedicated location.
     *
     * @param out the output to write to
     * @param state the feature state containing the data to write
     * @throws IOException if an I/O error occurs
     */
    void writeSeparately(DataOutput out, State state) throws IOException;
}
