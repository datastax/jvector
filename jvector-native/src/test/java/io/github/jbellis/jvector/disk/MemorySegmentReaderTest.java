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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class MemorySegmentReaderTest extends RandomizedTest {

    private Path tempFile;

    @Before
    public void setup() throws IOException {
        tempFile = Files.createTempFile(getClass().getSimpleName(), ".data");

        try (var out = new DataOutputStream(new FileOutputStream(tempFile.toFile()))) {
            out.write(new byte[] {1, 2, 3, 4, 5, 6, 7});
            for (int i = 0; i < 5; i++) {
                out.writeInt((i + 1) * 19);
            }
            for (int i = 0; i < 5; i++) {
                out.writeLong((i + 1) * 19L);
            }
            for (int i = 0; i < 5; i++) {
                out.writeFloat((i + 1) * 19);
            }
        }
    }

    @After
    public void tearDown() throws IOException {
        Files.deleteIfExists(tempFile);
    }

    @Test
    public void testReader() throws Exception {
        try (var supplier = new MemorySegmentReader.Supplier(tempFile)) {
            var r = supplier.get();
            verifyReader(r);

            // read 2nd time from beginning
            verifyReader(r);
        }
    }


    @Test
    public void testSupplierClose() throws Exception {
        var s = new MemorySegmentReader.Supplier(tempFile);
        var r1 = s.get();
        var r2 = s.get();

        // Close on supplied readers are nop.
        r1.close();
        r1.readInt();
        r2.close();
        r2.readInt();

        // Backing memory-map will be closed when supplier is closed.
        s.close();
        try {
            r1.readInt();
            fail("Should have thrown an exception");
        } catch (IllegalStateException _) {
        }
        try {
            r2.readInt();
            fail("Should have thrown an exception");
        } catch (IllegalStateException _) {
        }
    }

    private void verifyReader(MemorySegmentReader r) {
        r.seek(0);
        var bytes = new byte[7];
        r.readFully(bytes);
        for (int i = 0; i < bytes.length; i++) {
            assertEquals(i + 1, bytes[i]);
        }

        r.seek(0);
        var buff = ByteBuffer.allocate(6);
        r.readFully(buff);
        for (int i = 0; i < buff.remaining(); i++) {
            assertEquals(i + 1, buff.get(i));
        }

        r.seek(7);
        assertEquals(19, r.readInt());

        r.seek(7);
        var ints = new int[5];
        r.read(ints, 0, ints.length);
        for (int i = 0; i < ints.length; i++) {
            var k = ints[i];
            assertEquals((i + 1) * 19, k);
        }

        r.seek(7 + (4 * 5));
        var longs = new long[5];
        r.readFully(longs);
        for (int i = 0; i < longs.length; i++) {
            var l = longs[i];
            assertEquals((i + 1) * 19, l);
        }

        r.seek(7 + (4 * 5) + (8 * 5));
        assertEquals(19, r.readFloat(), 0.01);

        r.seek(7 + (4 * 5) + (8 * 5));
        var floats = new float[5];
        r.readFully(floats);
        for (int i = 0; i < floats.length; i++) {
            var f = floats[i];
            assertEquals((i + 1) * 19f, f, 0.01);
        }
    }
}
