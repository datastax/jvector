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

import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class TestSeekableSink {

    @Test
    public void writesAndReadsInRegionRelativeCoordinates() throws IOException {
        Path f = Files.createTempFile("sink", ".bin");
        try (FileChannel ch = FileChannel.open(f, StandardOpenOption.WRITE, StandardOpenOption.READ)) {
            long base = 100;
            SeekableSink sink = SeekableSink.over(ch, base);
            sink.writeAt(0, ByteBuffer.wrap("hello".getBytes(StandardCharsets.UTF_8)));
            sink.writeAt(5, ByteBuffer.wrap("WORLD".getBytes(StandardCharsets.UTF_8)));
            sink.force();

            // Region-relative read returns what was written.
            ByteBuffer dst = ByteBuffer.allocate(10);
            assertEquals(10, sink.readAt(0, dst));
            assertEquals("helloWORLD", new String(dst.array(), StandardCharsets.UTF_8));

            // The bytes actually land at the absolute base offset (region-relative -> absolute).
            ByteBuffer raw = ByteBuffer.allocate(10);
            ch.read(raw, base);
            assertEquals("helloWORLD", new String(raw.array(), StandardCharsets.UTF_8));

            // Nothing was written before the region.
            ByteBuffer before = ByteBuffer.allocate((int) base);
            ch.read(before, 0);
            for (byte b : before.array()) {
                assertEquals("region must not write before its base", 0, b);
            }

            // close() must NOT close the caller-owned channel.
            sink.close();
            assertTrue("sink.close() must not close the caller's channel", ch.isOpen());
        }
        Files.deleteIfExists(f);
    }

    @Test
    public void rejectsNegativeBaseAndPosition() throws IOException {
        Path f = Files.createTempFile("sink", ".bin");
        try (FileChannel ch = FileChannel.open(f, StandardOpenOption.WRITE, StandardOpenOption.READ)) {
            try { SeekableSink.over(ch, -1); fail("negative base"); } catch (IllegalArgumentException expected) { }
            SeekableSink sink = SeekableSink.over(ch, 0);
            try { sink.writeAt(-1, ByteBuffer.allocate(1)); fail("negative write pos"); } catch (IllegalArgumentException expected) { }
            try { sink.readAt(-1, ByteBuffer.allocate(1)); fail("negative read pos"); } catch (IllegalArgumentException expected) { }
        }
        Files.deleteIfExists(f);
    }

    @Test(expected = NullPointerException.class)
    public void rejectsNullChannel() {
        SeekableSink.over(null, 0);
    }
}
