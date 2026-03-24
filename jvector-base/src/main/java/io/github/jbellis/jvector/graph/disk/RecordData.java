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

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Value object representing a complete node record with all its components.
 * Each component consists of a ByteBuffer containing the data and the file offset where it should be written.
 * <p>
 * This class is used to separate record building (CPU-bound) from I/O operations (I/O-bound),
 * enabling true asynchronous writes where record building can proceed while previous writes are in flight.
 */
class RecordData {
    private final List<RecordComponent> components;

    /**
     * Creates a new RecordData with an empty component list.
     */
    public RecordData() {
        this.components = new ArrayList<>();
    }

    /**
     * Adds a component to this record.
     *
     * @param buffer the buffer containing the component data (must be ready to read)
     * @param fileOffset the file offset where this component should be written
     */
    public void addComponent(ByteBuffer buffer, long fileOffset) {
        components.add(new RecordComponent(buffer, fileOffset));
    }

    /**
     * Returns all components in this record.
     *
     * @return list of record components
     */
    public List<RecordComponent> getComponents() {
        return components;
    }

    /**
     * Returns the number of components in this record.
     *
     * @return component count
     */
    public int componentCount() {
        return components.size();
    }

    /**
     * Represents a single component of a node record.
     * A component is a contiguous piece of data (e.g., ordinal, features, neighbors)
     * that needs to be written to a specific file offset.
     */
    static class RecordComponent {
        private final ByteBuffer buffer;
        private final long fileOffset;

        /**
         * Creates a new record component.
         *
         * @param buffer the buffer containing the data (position should be at start of data)
         * @param fileOffset the file offset where this data should be written
         */
        public RecordComponent(ByteBuffer buffer, long fileOffset) {
            this.buffer = buffer;
            this.fileOffset = fileOffset;
        }

        /**
         * Returns the buffer containing this component's data.
         * The buffer is positioned at the start of the data to write.
         *
         * @return the data buffer
         */
        public ByteBuffer getBuffer() {
            return buffer;
        }

        /**
         * Returns the file offset where this component should be written.
         *
         * @return the file offset
         */
        public long getFileOffset() {
            return fileOffset;
        }
    }
}

// Made with Bob
