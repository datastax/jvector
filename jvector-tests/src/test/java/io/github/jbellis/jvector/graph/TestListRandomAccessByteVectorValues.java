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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class TestListRandomAccessByteVectorValues {

    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private ByteSequence<?> seq(byte... values) {
        return vts.createByteSequence(values);
    }

    @Test
    public void testSizeAndDimension() {
        int dim = 4;
        List<ByteSequence<?>> vectors = List.of(seq((byte) 1, (byte) 2, (byte) 3, (byte) 4),
                                                 seq((byte) 5, (byte) 6, (byte) 7, (byte) 8));
        var rabvv = new ListRandomAccessByteVectorValues(vectors, dim);
        assertEquals(2, rabvv.size());
        assertEquals(dim, rabvv.dimension());
    }

    @Test
    public void testGetVectorReturnsCorrectEntry() {
        var v0 = seq((byte) 10, (byte) 20);
        var v1 = seq((byte) -1, (byte) -2);
        var v2 = seq((byte) 127, (byte) -128);
        var rabvv = new ListRandomAccessByteVectorValues(List.of(v0, v1, v2), 2);

        assertSame(v0, rabvv.getVector(0));
        assertSame(v1, rabvv.getVector(1));
        assertSame(v2, rabvv.getVector(2));
    }

    @Test
    public void testIsValueShared() {
        var rabvv = new ListRandomAccessByteVectorValues(List.of(seq((byte) 0)), 1);
        assertFalse(rabvv.isValueShared());
    }

    @Test
    public void testCopyReturnsSelf() {
        var rabvv = new ListRandomAccessByteVectorValues(List.of(seq((byte) 1, (byte) 2)), 2);
        assertSame(rabvv, rabvv.copy());
    }

    @Test
    public void testMutableBackingListIsReflected() {
        // ListRandomAccessByteVectorValues documents that additions to the backing list are visible
        List<ByteSequence<?>> backing = new ArrayList<>();
        backing.add(seq((byte) 1));
        var rabvv = new ListRandomAccessByteVectorValues(backing, 1);
        assertEquals(1, rabvv.size());

        backing.add(seq((byte) 2));
        assertEquals(2, rabvv.size());
        assertSame(backing.get(1), rabvv.getVector(1));
    }
}
