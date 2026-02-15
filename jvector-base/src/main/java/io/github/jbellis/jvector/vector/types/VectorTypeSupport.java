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

package io.github.jbellis.jvector.vector.types;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;

import java.io.IOException;

public interface VectorTypeSupport {
    /**
     * Create a vector from the given data.
     *
     * @param data the data to create the vector from. Supported data types are implementation-dependent.
     * @return the created vector.
     */
    VectorFloat<?> createFloatVector(Object data);

    /**
     * Create a zero-filled vector of the given length.
     * @param length the length of the vector to create.
     * @return the created vector.
     */
    VectorFloat<?> createFloatVector(int length);

    /**
     * Read a vector from the given RandomAccessReader.
     * @param r the reader to read the vector from.
     * @param size the size of the vector to read.
     * @return the vector.
     * @throws IOException
     */
    VectorFloat<?> readFloatVector(RandomAccessReader r, int size) throws IOException;

    /**
     * Read a vector from the given RandomAccessReader and store it in the given vector at the specified offset.
     * @param r the reader to read the vector from.
     * @param size the size of the vector to read.
     * @param vector the vector to store the read data in.
     * @param offset the offset in the vector to store the read data at.
     * @throws IOException
     */
    void readFloatVector(RandomAccessReader r, int size, VectorFloat<?> vector, int offset) throws IOException;

    /**
     * Write the given vector to the given IndexWriter.
     * @param out the output to write the vector to.
     * @param vector the vector to write.
     * @throws IOException
     */
    void writeFloatVector(IndexWriter out, VectorFloat<?> vector) throws IOException;

    /**
     * Create a sequence from the given data.
     *
     * @param data the data to create the sequence from. Supported data types are implementation-dependent.
     * @return the created vector.
     */
    ByteSequence<?> createByteSequence(Object data);

    /**
     * Create a zero-filled sequence of the given length.
     * @param length the length of the sequence to create.
     * @return the created sequence.
     */
    ByteSequence<?> createByteSequence(int length);

    ByteSequence<?> readByteSequence(RandomAccessReader r, int size) throws IOException;

    void readByteSequence(RandomAccessReader r, ByteSequence<?> sequence) throws IOException;

    void writeByteSequence(IndexWriter out, ByteSequence<?> sequence) throws IOException;
}
