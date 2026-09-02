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

import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assume.assumeTrue;

/** The native advisor links, hints a written range without error, and is what the base factory finds. */
public class SyncFileRangeAdvisorTest {
    @Test
    public void testHintsAWrittenRange() throws Exception {
        assumeTrue("sync_file_range not linked on this runtime", SyncFileRangeAdvisor.available());
        Path p = Files.createTempFile(getClass().getSimpleName(), ".out");
        try {
            byte[] data = new byte[1 << 20];
            try (FileChannel fc = FileChannel.open(p, StandardOpenOption.WRITE)) {
                fc.write(ByteBuffer.wrap(data), 0);
            }
            try (WritebackAdvisor advisor = WritebackAdvisor.open(p)) {
                assertNotNull("the base factory must find the native advisor", advisor);
                advisor.hint(0, data.length);
                advisor.hint(4096, 8192);
                advisor.hint(0, 0); // empty: ignored
                assertEquals(2, advisor.hints());
            }
        } finally {
            Files.deleteIfExists(p);
        }
    }
}
