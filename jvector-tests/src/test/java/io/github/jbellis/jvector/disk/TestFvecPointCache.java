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
import java.nio.ByteOrder;
import java.nio.file.Files;
import static org.junit.Assert.*;

public class TestFvecPointCache {
    @Test public void openingNarrowLargeFileKeepsMetadataBounded() throws Exception {
        // Only metadata and the first record are used; the remainder is a sparse extent.
        var file = Files.createTempFile("point-cache-metadata", ".fvecs");
        try {
            try (var channel = java.nio.channels.FileChannel.open(file,
                    java.nio.file.StandardOpenOption.WRITE)) {
                var header = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
                header.putInt(1).putFloat(123f).flip();
                while (header.hasRemaining()) channel.write(header);
                channel.position(8_000_000_000L - 1);
                channel.write(ByteBuffer.wrap(new byte[1]));
            }
            var bean = java.lang.management.ManagementFactory.getThreadMXBean();
            org.junit.Assume.assumeTrue(bean instanceof com.sun.management.ThreadMXBean);
            var allocations = (com.sun.management.ThreadMXBean) bean;
            allocations.setThreadAllocatedMemoryEnabled(true);
            long thread = Thread.currentThread().getId();
            long before = allocations.getThreadAllocatedBytes(thread);
            try (var source = FvecFileVectorValues.open(file,
                    FvecFileVectorValues.Options.defaults().withIoThreads(2))) {
                long bytes = allocations.getThreadAllocatedBytes(thread) - before;
                assertTrue("Constructor allocated " + bytes + " bytes", bytes < 8L << 20);
                assertEquals(1_000_000_000, source.size());
                assertEquals(123f, source.getVector(0).get(0), 0f);
                assertTrue(source.statistics().bufferBytes <= source.statistics().bufferLimitBytes);
            }
        } finally { Files.delete(file); }
    }

    @Test public void adjacentPointRequestsDoNotTriggerScanReadAhead() throws Exception {
        int rows = 10000, dimensions = 17;
        var file = Files.createTempFile("point-scan-separation", ".fvecs");
        var bytes = ByteBuffer.allocate(rows * (dimensions + 1) * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < rows; row++) {
            bytes.putInt(dimensions);
            for (int d = 0; d < dimensions; d++) bytes.putFloat(row + d / 32f);
        }
        Files.write(file, bytes.array());
        var source = FvecFileVectorValues.open(file, FvecFileVectorValues.Options.defaults()
                .withIoThreads(4).withBatchVectors(8).withReadAhead(3));
        try {
            for (int ordinal = 100; ordinal < 103; ordinal++)
                assertEquals(ordinal, source.getVector(ordinal).get(0), 0f);
        } finally {
            source.close();
            Files.delete(file);
        }
        // Closing drains speculative reads; exactly the three requested records should have been read.
        assertEquals(3, source.statistics().reads);
    }

    @Test public void pointFallbackKeepsScratchAndResourcesBounded() throws Exception {
        int rows = 10000, dimensions = 17;
        var file = Files.createTempFile("point-cache-check", ".fvecs");
        var bytes = ByteBuffer.allocate(rows * (dimensions + 1) * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < rows; row++) {
            bytes.putInt(dimensions);
            for (int d = 0; d < dimensions; d++) bytes.putFloat(row + d / 32f);
        }
        Files.write(file, bytes.array());
        try (var source = FvecFileVectorValues.open(file, FvecFileVectorValues.Options.defaults()
                .withIoThreads(4).withBatchVectors(8).withReadAhead(0).withMaxBufferBytes(65536))) {
            try (var other = source.copy()) {
                var first = source.getVector(100);
                long before = source.statistics().reads;
                var second = other.getVector(100);
                assertNotSame(first, second);
                assertEquals(before + 1, source.statistics().reads);
                source.getVector(101);
                assertEquals(100f, second.get(0), 0f);
                before = source.statistics().reads;
                assertEquals(100f, other.getVector(100).get(0), 0f);
                assertEquals(before + 1, source.statistics().reads);
                for (int i = 0; i < 20000; i++) {
                    int ordinal = (i * 71) % rows;
                    var vector = other.getVector(ordinal);
                    for (int d = 0; d < dimensions; d++)
                        assertEquals(ordinal + d / 32f, vector.get(d), 0f);
                }
                assertTrue(source.statistics().bufferBytes <= source.statistics().bufferLimitBytes);
            }
        } finally { Files.delete(file); }
    }
}
