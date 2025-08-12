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
import io.github.jbellis.jvector.TestUtil;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.jupiter.api.Assumptions;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Random;

import static org.junit.Assert.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestMappedChunkReader extends RandomizedTest {

    private Path testDirectory;
    private Path largeFilePath;
    private static final long CHUNK_SIZE = Integer.MAX_VALUE; // ~2GB
    private static final long FILE_SIZE = CHUNK_SIZE + (1024 * 1024 * 100); // ~2.1GB
    
    // Test data positions - before, at, and after the 2GB boundary
    private static final long POS_BEFORE_BOUNDARY = CHUNK_SIZE - 1024;
    private static final long POS_AT_BOUNDARY = CHUNK_SIZE;
    private static final long POS_AFTER_BOUNDARY = CHUNK_SIZE + 1024;
    
    // Test values to write and read
    private int testIntValue;
    private long testLongValue;
    private float testFloatValue;
    private byte[] testByteArray;
    
    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
        largeFilePath = testDirectory.resolve("large_test_file");
        
        // Generate random test values
        Random random = getRandom();
        testIntValue = random.nextInt();
        testLongValue = random.nextLong();
        testFloatValue = random.nextFloat();
        testByteArray = new byte[128];
        random.nextBytes(testByteArray);
    }
    
    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }
    
    /**
     * Creates a sparse file larger than 2GB with test data at specific positions.
     * This method uses sparse file allocation to avoid actually writing 2GB of data.
     */
    private void createLargeTestFile() throws IOException {
        try (RandomAccessFile file = new RandomAccessFile(largeFilePath.toFile(), "rw")) {
            // Set the file size to create a sparse file
            file.setLength(FILE_SIZE);
            
            // Write test data before the 2GB boundary
            file.seek(POS_BEFORE_BOUNDARY);
            file.writeInt(testIntValue);
            file.writeLong(testLongValue);
            file.writeFloat(testFloatValue);
            file.write(testByteArray);
            
            // Write test data at the 2GB boundary
            file.seek(POS_AT_BOUNDARY);
            file.writeInt(testIntValue + 1);
            file.writeLong(testLongValue + 1);
            file.writeFloat(testFloatValue + 1.0f);
            file.write(testByteArray);
            
            // Write test data after the 2GB boundary
            file.seek(POS_AFTER_BOUNDARY);
            file.writeInt(testIntValue + 2);
            file.writeLong(testLongValue + 2);
            file.writeFloat(testFloatValue + 2.0f);
            file.write(testByteArray);
        }
    }
    
    @Test
    public void testReadingLargeFile() throws IOException {
        // Skip the test if we can't allocate a large file
        // This is important for CI environments with limited disk space
        boolean fileCreated = true;
        try {
            createLargeTestFile();
        } catch (IOException e) {
            fileCreated = false;
        }
        Assumptions.assumeTrue(fileCreated, "Skipping test due to failure to create large file");
        // Open the file channel
        try (final FileChannel channel = FileChannel.open(largeFilePath, StandardOpenOption.READ)) {
            // Create a MappedChunkReader
            try (MappedChunkReader reader = new MappedChunkReader(channel, ByteOrder.BIG_ENDIAN)) {
                // Test reading before the 2GB boundary
                reader.seek(POS_BEFORE_BOUNDARY);
                assertEquals(testIntValue, reader.readInt());
                assertEquals(testLongValue, reader.readLong());
                assertEquals(testFloatValue, reader.readFloat(), 0.0001f);
                byte[] readBytes = new byte[testByteArray.length];
                reader.readFully(readBytes);
                assertArrayEquals(testByteArray, readBytes);
                
                // Test reading at the 2GB boundary
                reader.seek(POS_AT_BOUNDARY);
                assertEquals(testIntValue + 1, reader.readInt());
                assertEquals(testLongValue + 1, reader.readLong());
                assertEquals(testFloatValue + 1.0f, reader.readFloat(), 0.0001f);
                reader.readFully(readBytes);
                assertArrayEquals(testByteArray, readBytes);
                
                // Test reading after the 2GB boundary
                reader.seek(POS_AFTER_BOUNDARY);
                assertEquals(testIntValue + 2, reader.readInt());
                assertEquals(testLongValue + 2, reader.readLong());
                assertEquals(testFloatValue + 2.0f, reader.readFloat(), 0.0001f);
                reader.readFully(readBytes);
                assertArrayEquals(testByteArray, readBytes);
                
                // Test reading across the 2GB boundary
                ByteBuffer buffer = ByteBuffer.allocate(2048);
                reader.seek(CHUNK_SIZE - 1024);
                reader.readFully(buffer);
                assertEquals(2048, buffer.position());
            }
        }
    }
    
    @Test
    public void testReadingLargeFileWithSupplier() throws IOException {
        // Skip the test if we can't allocate a large file
        boolean fileCreated = true;
        try {
            createLargeTestFile();
        } catch (IOException e) {
            fileCreated = false;
        }
        Assumptions.assumeTrue(fileCreated, "Skipping test due to failure to create large file");
        
        // Test using the Supplier
        try (ReaderSupplier readerSupplier = new MappedChunkReader.Supplier(largeFilePath)) {
            try (RandomAccessReader reader = readerSupplier.get()) {
                // Test reading before the 2GB boundary
                reader.seek(POS_BEFORE_BOUNDARY);
                assertEquals(testIntValue, reader.readInt());
                assertEquals(testLongValue, reader.readLong());
                assertEquals(testFloatValue, reader.readFloat(), 0.0001f);
                byte[] readBytes = new byte[testByteArray.length];
                reader.readFully(readBytes);
                assertArrayEquals(testByteArray, readBytes);
                
                // Test reading after the 2GB boundary
                reader.seek(POS_AFTER_BOUNDARY);
                assertEquals(testIntValue + 2, reader.readInt());
                assertEquals(testLongValue + 2, reader.readLong());
                assertEquals(testFloatValue + 2.0f, reader.readFloat(), 0.0001f);
                reader.readFully(readBytes);
                assertArrayEquals(testByteArray, readBytes);
            }
        }
    }
    
    private static void assertArrayEquals(byte[] expected, byte[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals("Byte arrays differ at index " + i, expected[i], actual[i]);
        }
    }
}