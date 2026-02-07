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

import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;

import static org.junit.Assert.*;

public class TestByteBufferIndexWriter extends LuceneTestCase {

    @Test
    public void testWriteFloatsAdvancesPosition() throws IOException {
        ByteBufferIndexWriter writer = ByteBufferIndexWriter.create(1024, false);

        float[] floats = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        // Write floats
        writer.writeFloats(floats, 0, floats.length);

        // Verify position advanced correctly
        assertEquals(floats.length * Float.BYTES, writer.position());

        // Write more data to ensure position is correct
        writer.writeInt(42);
        assertEquals(floats.length * Float.BYTES + Integer.BYTES, writer.position());
    }

    @Test
    public void testWriteFloatsWithOffsetAndCount() throws IOException {
        ByteBufferIndexWriter writer = ByteBufferIndexWriter.create(1024, false);

        float[] floats = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        // Write only middle 3 elements
        writer.writeFloats(floats, 1, 3);

        // Verify position
        assertEquals(3 * Float.BYTES, writer.position());

        // Verify content
        ByteBuffer buffer = writer.getWrittenData();
        assertEquals(2.0f, buffer.getFloat(), 0.001f);
        assertEquals(3.0f, buffer.getFloat(), 0.001f);
        assertEquals(4.0f, buffer.getFloat(), 0.001f);
    }

    @Test
    public void testWriteFloatVectorIntegration() throws IOException {
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        ByteBufferIndexWriter writer = ByteBufferIndexWriter.create(1024, false);

        // Create a vector
        float[] data = {1.5f, 2.5f, 3.5f, 4.5f};
        VectorFloat<?> vector = vts.createFloatVector(data);

        // Write vector using VectorTypeSupport
        vts.writeFloatVector(writer, vector);

        // Verify position advanced
        assertEquals(data.length * Float.BYTES, writer.position());

        // Verify content
        ByteBuffer buffer = writer.getWrittenData();
        for (float expected : data) {
            assertEquals(expected, buffer.getFloat(), 0.001f);
        }
    }

    @Test
    public void testMultipleWriteFloatsCalls() throws IOException {
        ByteBufferIndexWriter writer = ByteBufferIndexWriter.create(1024, false);

        float[] floats1 = {1.0f, 2.0f};
        float[] floats2 = {3.0f, 4.0f, 5.0f};

        writer.writeFloats(floats1, 0, floats1.length);
        long pos1 = writer.position();
        assertEquals(floats1.length * Float.BYTES, pos1);

        writer.writeFloats(floats2, 0, floats2.length);
        long pos2 = writer.position();
        assertEquals((floats1.length + floats2.length) * Float.BYTES, pos2);

        // Verify all data was written correctly
        ByteBuffer buffer = writer.getWrittenData();
        assertEquals(1.0f, buffer.getFloat(), 0.001f);
        assertEquals(2.0f, buffer.getFloat(), 0.001f);
        assertEquals(3.0f, buffer.getFloat(), 0.001f);
        assertEquals(4.0f, buffer.getFloat(), 0.001f);
        assertEquals(5.0f, buffer.getFloat(), 0.001f);
    }

    @Test
    public void testWriteFloatsDoesNotOverwrite() throws IOException {
        ByteBufferIndexWriter writer = ByteBufferIndexWriter.create(1024, false);

        // Write an int first
        writer.writeInt(999);
        long posAfterInt = writer.position();

        // Write floats
        float[] floats = {1.0f, 2.0f, 3.0f};
        writer.writeFloats(floats, 0, floats.length);

        // Verify position
        assertEquals(posAfterInt + floats.length * Float.BYTES, writer.position());

        // Verify the int wasn't overwritten
        ByteBuffer buffer = writer.getWrittenData();
        assertEquals(999, buffer.getInt());
        assertEquals(1.0f, buffer.getFloat(), 0.001f);
        assertEquals(2.0f, buffer.getFloat(), 0.001f);
        assertEquals(3.0f, buffer.getFloat(), 0.001f);
    }

    @Test
    public void testWriteFloatsWithDirectBuffer() throws IOException {
        ByteBufferIndexWriter writer = ByteBufferIndexWriter.create(1024, true);

        float[] floats = {1.0f, 2.0f, 3.0f};
        writer.writeFloats(floats, 0, floats.length);

        assertEquals(floats.length * Float.BYTES, writer.position());

        ByteBuffer buffer = writer.getWrittenData();
        for (float expected : floats) {
            assertEquals(expected, buffer.getFloat(), 0.001f);
        }
    }

    @Test
    public void testBytesWritten() throws IOException {
        ByteBufferIndexWriter writer = ByteBufferIndexWriter.create(1024, false);

        assertEquals(0, writer.bytesWritten());

        float[] floats = {1.0f, 2.0f, 3.0f};
        writer.writeFloats(floats, 0, floats.length);

        assertEquals(floats.length * Float.BYTES, writer.bytesWritten());

        writer.writeInt(42);
        assertEquals(floats.length * Float.BYTES + Integer.BYTES, writer.bytesWritten());
    }

    @Test
    public void testArrayVectorFloatWriteTo() throws IOException {
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        ByteBufferIndexWriter writer = ByteBufferIndexWriter.create(1024, false);

        // Create an ArrayVectorFloat
        float[] data = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
        VectorFloat<?> vector = vts.createFloatVector(data);

        // Call writeTo which internally calls writeFloats
        vector.writeTo(writer);

        // Verify position advanced correctly
        assertEquals(data.length * Float.BYTES, writer.position());
        assertEquals(data.length * Float.BYTES, writer.bytesWritten());

        // Verify content was written correctly
        ByteBuffer buffer = writer.getWrittenData();
        for (float expected : data) {
            assertEquals(expected, buffer.getFloat(), 0.001f);
        }
    }

    @Test
    public void testArrayVectorFloatWriteToMultipleTimes() throws IOException {
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        ByteBufferIndexWriter writer = ByteBufferIndexWriter.create(1024, false);

        // Create two vectors
        float[] data1 = {1.0f, 2.0f, 3.0f};
        float[] data2 = {4.0f, 5.0f, 6.0f, 7.0f};
        VectorFloat<?> vector1 = vts.createFloatVector(data1);
        VectorFloat<?> vector2 = vts.createFloatVector(data2);

        // Write first vector
        vector1.writeTo(writer);
        long pos1 = writer.position();
        assertEquals(data1.length * Float.BYTES, pos1);

        // Write second vector
        vector2.writeTo(writer);
        long pos2 = writer.position();
        assertEquals((data1.length + data2.length) * Float.BYTES, pos2);

        // Verify all data was written correctly
        ByteBuffer buffer = writer.getWrittenData();
        assertEquals(1.0f, buffer.getFloat(), 0.001f);
        assertEquals(2.0f, buffer.getFloat(), 0.001f);
        assertEquals(3.0f, buffer.getFloat(), 0.001f);
        assertEquals(4.0f, buffer.getFloat(), 0.001f);
        assertEquals(5.0f, buffer.getFloat(), 0.001f);
        assertEquals(6.0f, buffer.getFloat(), 0.001f);
        assertEquals(7.0f, buffer.getFloat(), 0.001f);
    }

    @Test
    public void testArrayVectorFloatWriteToWithDirectBuffer() throws IOException {
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        ByteBufferIndexWriter writer = ByteBufferIndexWriter.create(1024, true);

        // Create a vector
        float[] data = {10.5f, 20.5f, 30.5f};
        VectorFloat<?> vector = vts.createFloatVector(data);

        // Write to direct buffer
        vector.writeTo(writer);

        // Verify position
        assertEquals(data.length * Float.BYTES, writer.position());

        // Verify content
        ByteBuffer buffer = writer.getWrittenData();
        for (float expected : data) {
            assertEquals(expected, buffer.getFloat(), 0.001f);
        }
    }
}

// Made with Bob
