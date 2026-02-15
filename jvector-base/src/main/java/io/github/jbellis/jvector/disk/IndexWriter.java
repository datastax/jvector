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

import java.io.Closeable;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Interface for writing index data.
 */
public interface IndexWriter extends DataOutput, Closeable {
    /**
     * Returns the current position in the output.
     * @return the current position in the output
     * @throws IOException if an I/O error occurs
     */
    long position() throws IOException;

    default void writeFloats(float[] floats, int offset, int count) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(count * Float.BYTES);
        // DataOutput specifies BIG_ENDIAN for float
        bb.order(ByteOrder.BIG_ENDIAN).asFloatBuffer().put(floats, offset, count);
        bb.rewind();
        write(bb.array());
    }
}
