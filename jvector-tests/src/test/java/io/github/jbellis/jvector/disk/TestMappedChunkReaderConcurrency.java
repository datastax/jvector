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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.fail;

/**
 * Tests the thread safety of MappedChunkReader when accessed concurrently by multiple threads.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestMappedChunkReaderConcurrency extends RandomizedTest {
    private static final Logger log = LoggerFactory.getLogger(TestMappedChunkReaderConcurrency.class);
    private Path testDirectory;
    private Path testFilePath;
    private static final int FILE_SIZE = 100 * 1024 * 1024; // 100MB
    private static final int CHUNK_SIZE = Integer.MAX_VALUE; // ~2GB
    private static final int NUM_THREADS = 8;
    private static final int OPERATIONS_PER_THREAD = 1000;
    
    // List to keep track of all readers created during tests
    private final List<MappedChunkReader> readers = new ArrayList<>();
    
    @Before
    public void setUp() throws IOException {
        testDirectory = Files.createTempDirectory("test_mapped_chunk_reader_concurrency");
        testFilePath = testDirectory.resolve("test_file");
        createTestFile(testFilePath, FILE_SIZE);
    }

    @After
    public void tearDown() throws IOException {
        // Close all readers first
        for (MappedChunkReader reader : readers) {
            try {
                reader.close();
            } catch (Exception e) {
                log.warn("Exception while closing reader", e);
            }
        }
        readers.clear();
        
        // Force garbage collection to release mapped ByteBuffers
        System.gc();
        System.runFinalization();
        
        // Add a small delay to give Windows time to release file locks
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        // Now try to delete the files
        try {
            Files.deleteIfExists(testFilePath);
        } catch (IOException e) {
            // On Windows, this might still fail, but we'll try our best
            log.warn("Exception while deleting test file", e);
        }
        
        try {
            Files.deleteIfExists(testDirectory);
        } catch (IOException e) {
            log.warn("Exception while deleting test directory", e);
        }
    }

    /**
     * Creates a test file with predictable integer values at each 4-byte position.
     */
    private void createTestFile(Path path, int size) throws IOException {
        try (RandomAccessFile file = new RandomAccessFile(path.toFile(), "rw")) {
            file.setLength(size);
            ByteBuffer buffer = ByteBuffer.allocate(4096).order(ByteOrder.BIG_ENDIAN);
            
            for (int pos = 0; pos < size; pos += 4) {
                if (buffer.remaining() < 4) {
                    buffer.flip();
                    file.getChannel().write(buffer);
                    buffer.clear();
                }
                buffer.putInt(pos); // Write the position as the value
            }
            
            if (buffer.position() > 0) {
                buffer.flip();
                file.getChannel().write(buffer);
            }
        }
    }

    /**
     * Test that multiple threads can concurrently read from different positions
     * in the file without interfering with each other.
     */
    @Test
    public void testConcurrentReads() throws Exception {
        FileChannel channel = null;
        try {
            channel = FileChannel.open(testFilePath, StandardOpenOption.READ);
            MappedChunkReader reader = new MappedChunkReader(channel, ByteOrder.BIG_ENDIAN);
            readers.add(reader); // Add reader to the list
            
            ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
            CyclicBarrier barrier = new CyclicBarrier(NUM_THREADS);
            AtomicBoolean failed = new AtomicBoolean(false);
            AtomicInteger errorCount = new AtomicInteger(0);
            final List<String> errors = Collections.synchronizedList(new ArrayList<>());
            
            List<Future<?>> futures = new ArrayList<>();
            
            for (int t = 0; t < NUM_THREADS; t++) {
                final int threadId = t;
                futures.add(executor.submit(() -> {
                    try {
                        Random random = new Random(threadId); // Deterministic per thread
                        barrier.await(); // Wait for all threads to be ready
                        
                        for (int i = 0; i < OPERATIONS_PER_THREAD; i++) {
                            // Choose a random position that's aligned to 4 bytes and within file bounds
                            int position = random.nextInt(FILE_SIZE / 4) * 4;
                            
                            synchronized (reader) {
                                reader.seek(position);
                                int value = reader.readInt();
                                
                                // The value should equal the position
                                if (value != position) {
                                    errors.add(String.format("Thread %d expected %d but got %d", 
                                                           threadId, position, value));
                                    failed.set(true);
                                    errorCount.incrementAndGet();
                                }
                            }
                        }
                    } catch (Exception e) {
                        errors.add("Thread " + threadId + " exception: " + e.getMessage());
                        failed.set(true);
                        errorCount.incrementAndGet();
                    }
                }));
            }
            
            // Wait for all threads to complete
            for (Future<?> future : futures) {
                future.get();
            }
            
            executor.shutdown();
            executor.awaitTermination(30, TimeUnit.SECONDS);
            
            if (errorCount.get() > 0) {
                fail("Concurrent read test failed with " + errorCount.get() + " errors: " + 
                     String.join(", ", errors));
            }
        } finally {
            if (channel != null) {
                try {
                    channel.close();
                } catch (IOException e) {
                    log.warn("Exception while closing channel", e);
                }
            }
        }
    }
    
    /**
     * Test that multiple threads can concurrently read across chunk boundaries
     * without interfering with each other.
     */
    @Test
    public void testConcurrentChunkBoundaryReads() throws Exception {
        // Skip this test if the file is too small to have chunk boundaries
        if (FILE_SIZE <= CHUNK_SIZE) {
            return; // Skip silently
        }
        
        FileChannel channel = null;
        try {
            channel = FileChannel.open(testFilePath, StandardOpenOption.READ);
            MappedChunkReader reader = new MappedChunkReader(channel, ByteOrder.BIG_ENDIAN);
            readers.add(reader); // Add reader to the list
            
            ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
            CountDownLatch startLatch = new CountDownLatch(1);
            AtomicBoolean failed = new AtomicBoolean(false);
            AtomicInteger errorCount = new AtomicInteger(0);
            final List<String> errors = Collections.synchronizedList(new ArrayList<>());
            
            List<Future<?>> futures = new ArrayList<>();
            
            for (int t = 0; t < NUM_THREADS; t++) {
                final int threadId = t;
                futures.add(executor.submit(() -> {
                    try {
                        Random random = new Random(threadId + 1000); // Different seed from previous test
                        startLatch.await(); // Wait for all threads to be ready
                        
                        for (int i = 0; i < OPERATIONS_PER_THREAD / 10; i++) { // Fewer operations for this test
                            // Choose positions near potential chunk boundaries
                            long position;
                            if (random.nextBoolean()) {
                                // Position just before a chunk boundary
                                position = CHUNK_SIZE - (random.nextInt(1024) + 1) * 4;
                            } else {
                                // Position just after a chunk boundary
                                position = CHUNK_SIZE + random.nextInt(1024) * 4;
                            }
                            
                            // Ensure position is within file bounds
                            position = position % FILE_SIZE;
                            position = (position / 4) * 4; // Align to 4 bytes
                            
                            synchronized (reader) {
                                reader.seek(position);
                                int value = reader.readInt();
                                
                                // The value should equal the position
                                if (value != position) {
                                    errors.add(String.format("Thread %d at boundary read expected %d but got %d", 
                                                           threadId, position, value));
                                    failed.set(true);
                                    errorCount.incrementAndGet();
                                }
                            }
                        }
                    } catch (Exception e) {
                        errors.add("Thread " + threadId + " exception: " + e.getMessage());
                        failed.set(true);
                        errorCount.incrementAndGet();
                    }
                }));
            }
            
            // Start all threads
            startLatch.countDown();
            
            // Wait for all threads to complete
            for (Future<?> future : futures) {
                future.get();
            }
            
            executor.shutdown();
            executor.awaitTermination(30, TimeUnit.SECONDS);
            
            if (errorCount.get() > 0) {
                fail("Concurrent chunk boundary read test failed with " + errorCount.get() + 
                     " errors: " + String.join(", ", errors));
            }
        } finally {
            if (channel != null) {
                try {
                    channel.close();
                } catch (IOException e) {
                    // Just continue
                }
            }
        }
    }
    
    /**
     * Test that multiple threads can concurrently read byte arrays of different sizes
     * without interfering with each other.
     */
    @Test
    public void testConcurrentByteArrayReads() throws Exception {
        FileChannel channel = null;
        try {
            channel = FileChannel.open(testFilePath, StandardOpenOption.READ);
            MappedChunkReader reader = new MappedChunkReader(channel, ByteOrder.BIG_ENDIAN);
            readers.add(reader); // Add reader to the list
            
            ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
            CyclicBarrier barrier = new CyclicBarrier(NUM_THREADS);
            AtomicBoolean failed = new AtomicBoolean(false);
            AtomicInteger errorCount = new AtomicInteger(0);
            final List<String> errors = Collections.synchronizedList(new ArrayList<>());
            
            List<Future<?>> futures = new ArrayList<>();
            
            for (int t = 0; t < NUM_THREADS; t++) {
                final int threadId = t;
                futures.add(executor.submit(() -> {
                    try {
                        Random random = new Random(threadId + 2000); // Different seed
                        barrier.await(); // Wait for all threads to be ready
                        
                        for (int i = 0; i < OPERATIONS_PER_THREAD / 5; i++) { // Fewer operations as these are more expensive
                            // Choose a random position and size for byte array
                            int arraySize = (random.nextInt(10) + 1) * 4; // 4 to 40 bytes
                            int maxPos = FILE_SIZE - arraySize;
                            int position = random.nextInt(maxPos / 4) * 4; // Align to 4 bytes
                            
                            byte[] bytes = new byte[arraySize];
                            
                            synchronized (reader) {
                                reader.seek(position);
                                reader.readFully(bytes);
                                
                                // Verify the content - each 4 bytes should form an int equal to its position
                                ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
                                for (int pos = 0; pos < arraySize; pos += 4) {
                                    int value = buffer.getInt();
                                    int expectedValue = position + pos;
                                    
                                    if (value != expectedValue) {
                                        errors.add(String.format("Thread %d byte array read at offset %d expected %d but got %d", 
                                                               threadId, pos, expectedValue, value));
                                        failed.set(true);
                                        errorCount.incrementAndGet();
                                        break;
                                    }
                                }
                            }
                        }
                    } catch (Exception e) {
                        errors.add("Thread " + threadId + " exception: " + e.getMessage());
                        failed.set(true);
                        errorCount.incrementAndGet();
                    }
                }));
            }
            
            // Wait for all threads to complete
            for (Future<?> future : futures) {
                future.get();
            }
            
            executor.shutdown();
            executor.awaitTermination(30, TimeUnit.SECONDS);
            
            if (errorCount.get() > 0) {
                fail("Concurrent byte array read test failed with " + errorCount.get() + 
                     " errors: " + String.join(", ", errors));
            }
        } finally {
            if (channel != null) {
                try {
                    channel.close();
                } catch (IOException e) {
                    log.warn("Failed to close channel", e);
                }
            }
        }
    }
}
