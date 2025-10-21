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
package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.vector.types.ByteSequence;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TestArraySliceByteSequence {

    @Test
    void testConstructorValidation() {
        byte[] data = {1, 2, 3, 4, 5};
        ByteSequence<byte[]> baseSequence = new ArrayByteSequence(data);

        // Valid construction
        assertDoesNotThrow(() -> new ArraySliceByteSequence(baseSequence, 0, 5));
        assertDoesNotThrow(() -> new ArraySliceByteSequence(baseSequence, 1, 3));

        // Invalid constructions
        assertThrows(IllegalArgumentException.class, () -> new ArraySliceByteSequence(baseSequence, -1, 3));
        assertThrows(IllegalArgumentException.class, () -> new ArraySliceByteSequence(baseSequence, 0, -1));
        assertThrows(IllegalArgumentException.class, () -> new ArraySliceByteSequence(baseSequence, 0, 6));
        assertThrows(IllegalArgumentException.class, () -> new ArraySliceByteSequence(baseSequence, 4, 2));
    }

    @Test
    void testBasicOperations() {
        byte[] data = {1, 2, 3, 4, 5};
        ByteSequence<byte[]> baseSequence = new ArrayByteSequence(data);
        ArraySliceByteSequence slice = new ArraySliceByteSequence(baseSequence, 1, 3);

        // Test get()
        assertArrayEquals(data, slice.get());

        // Test offset()
        assertEquals(1, slice.offset());

        // Test length()
        assertEquals(3, slice.length());

        // Test get(n)
        assertEquals(2, slice.get(0));
        assertEquals(3, slice.get(1));
        assertEquals(4, slice.get(2));
    }

    @Test
    void testSetOperations() {
        byte[] data = {1, 2, 3, 4, 5};
        ByteSequence<byte[]> baseSequence = new ArrayByteSequence(data);
        ArraySliceByteSequence slice = new ArraySliceByteSequence(baseSequence, 1, 3);

        // Test set(n, value)
        slice.set(0, (byte) 10);
        assertEquals(10, slice.get(0));
        assertEquals(10, baseSequence.get(1));
    }

    @Test
    void testZero() {
        byte[] data = {1, 2, 3, 4, 5};
        ByteSequence<byte[]> baseSequence = new ArrayByteSequence(data);
        ArraySliceByteSequence slice = new ArraySliceByteSequence(baseSequence, 1, 3);

        slice.zero();
        assertEquals(0, slice.get(0));
        assertEquals(0, slice.get(1));
        assertEquals(0, slice.get(2));
        assertEquals(1, baseSequence.get(0)); // Verify surrounding data unchanged
        assertEquals(5, baseSequence.get(4));
    }

    @Test
    void testCopy() {
        byte[] data = {1, 2, 3, 4, 5};
        ByteSequence<byte[]> baseSequence = new ArrayByteSequence(data);
        ArraySliceByteSequence slice = new ArraySliceByteSequence(baseSequence, 1, 3);

        ByteSequence<byte[]> copy = slice.copy();
        assertEquals(3, copy.length());
        assertEquals(2, copy.get(0));
        assertEquals(3, copy.get(1));
        assertEquals(4, copy.get(2));
    }

    @Test
    void testSlice() {
        byte[] data = {1, 2, 3, 4, 5};
        ByteSequence<byte[]> baseSequence = new ArrayByteSequence(data);
        ArraySliceByteSequence slice = new ArraySliceByteSequence(baseSequence, 1, 3);

        // Test valid slices
        ByteSequence<byte[]> subSlice = slice.slice(1, 2);
        assertEquals(2, subSlice.length());
        assertEquals(3, subSlice.get(0));
        assertEquals(4, subSlice.get(1));

        // Test invalid slices
        assertThrows(IllegalArgumentException.class, () -> slice.slice(-1, 2));
        assertThrows(IllegalArgumentException.class, () -> slice.slice(0, 4));

        // Test full slice returns same instance
        assertSame(slice, slice.slice(0, slice.length()));
    }

    @Test
    void testCopyFrom() {
        byte[] sourceData = {10, 11, 12, 13, 14};
        byte[] destData = {1, 2, 3, 4, 5};
        ByteSequence<byte[]> sourceSeq = new ArrayByteSequence(sourceData);
        ByteSequence<byte[]> destSeq = new ArrayByteSequence(destData);

        ArraySliceByteSequence destSlice = new ArraySliceByteSequence(destSeq, 1, 3);

        // Test copying from another ArraySliceByteSequence
        ArraySliceByteSequence sourceSlice = new ArraySliceByteSequence(sourceSeq, 1, 3);
        destSlice.copyFrom(sourceSlice, 0, 0, 2);
        assertEquals(11, destSlice.get(0));
        assertEquals(12, destSlice.get(1));

        // Test copying from a regular ByteSequence
        destSlice.copyFrom(sourceSeq, 0, 1, 2);
        assertEquals(10, destSlice.get(1));
        assertEquals(11, destSlice.get(2));
    }

    @Test
    void testToString() {
        byte[] data = new byte[30];
        for (int i = 0; i < data.length; i++) {
            data[i] = (byte) i;
        }
        ByteSequence<byte[]> baseSequence = new ArrayByteSequence(data);
        ArraySliceByteSequence slice = new ArraySliceByteSequence(baseSequence, 0, 30);

        String result = slice.toString();
        assertTrue(result.startsWith("[0, 1, 2"));
        assertTrue(result.endsWith("...]"));
    }

    @Test
    void testEqualsAndHashCode() {
        byte[] data1 = {1, 2, 3, 4, 5};
        byte[] data2 = {1, 2, 3, 4, 5};
        byte[] data3 = {1, 2, 3, 4, 6};

        ByteSequence<byte[]> seq1 = new ArrayByteSequence(data1);
        ByteSequence<byte[]> seq2 = new ArrayByteSequence(data2);
        ByteSequence<byte[]> seq3 = new ArrayByteSequence(data3);

        ArraySliceByteSequence slice1 = new ArraySliceByteSequence(seq1, 2, 3);
        ArraySliceByteSequence slice2 = new ArraySliceByteSequence(seq2, 2, 3);
        ArraySliceByteSequence slice3 = new ArraySliceByteSequence(seq3, 2, 3);

        // Test equals
        assertEquals(slice1, slice2);
        assertNotEquals(slice1, slice3);

        // Test hashCode
        assertEquals(slice1.hashCode(), slice2.hashCode());
        assertNotEquals(slice1.hashCode(), slice3.hashCode());
    }
}