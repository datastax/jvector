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

import org.junit.Assume;
import org.junit.Test;

import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.ClosedChannelException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;


/// Closing the root while point reads are in flight must fail those reads cleanly, as the rest
/// of the source does ("Vector source is closed" or a closed-channel I/O error), and must leave
/// no file descriptors open on the input file.
///
/// The race is timing-dependent, so the test repeats close-under-load until it sees a
/// violation or the round limit is reached. A pass is not proof of absence.
public class TestFvecCloseDuringPointReads {
    private static final int ROWS = 10_000, DIMENSIONS = 16, READERS = 12, ROUNDS = 300;

    private static Path writeFvecs() throws Exception {
        var file = Files.createTempFile("close-during-point-reads", ".fvecs");
        var bytes = ByteBuffer.allocate(ROWS * (DIMENSIONS + 1) * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < ROWS; row++) {
            bytes.putInt(DIMENSIONS);
            for (int d = 0; d < DIMENSIONS; d++) bytes.putFloat(row);
        }
        Files.write(file, bytes.array());
        return file;
    }

    /// Expected failures after close: the source reports closure, or the leased channel was
    /// closed underneath the read.
    private static boolean isCleanClosureFailure(Throwable t) {
        if (t instanceof IllegalStateException) return true;
        return t instanceof UncheckedIOException && t.getCause() instanceof ClosedChannelException;
    }

    private static long openDescriptorsOn(Path file) throws Exception {
        var real = file.toRealPath();
        long count = 0;
        try (var fds = Files.list(Path.of("/proc/self/fd"))) {
            for (var fd : (Iterable<Path>) fds::iterator) {
                try { if (Files.readSymbolicLink(fd).equals(real)) count++; }
                catch (java.io.IOException ignored) { } // descriptor closed while listing
            }
        }
        return count;
    }

    @Test(timeout = 120_000)
    public void closeUnderPointLoadFailsCleanlyAndLeaksNoDescriptors() throws Exception {
        Assume.assumeTrue("needs /proc/self/fd", Files.isDirectory(Path.of("/proc/self/fd")));
        var file = writeFvecs();
        try {
            for (int round = 0; round < ROUNDS; round++) {
                var unexpected = new ConcurrentLinkedQueue<Throwable>();
                var root = FvecFileVectorValues.open(file);
                var started = new CountDownLatch(READERS);
                List<Thread> readers = new ArrayList<>();
                for (int r = 0; r < READERS; r++) {
                    var view = root.copy();
                    long seed = round * 1000L + r;
                    var thread = new Thread(() -> {
                        var random = new SplittableRandom(seed);
                        started.countDown();
                        try {
                            while (true) view.getVector(random.nextInt(ROWS));
                        } catch (Throwable t) {
                            if (!isCleanClosureFailure(t)) unexpected.add(t);
                        }
                    }, "point-reader-" + r);
                    readers.add(thread);
                    thread.start();
                }
                started.await();
                Thread.sleep(1);
                root.close();
                for (var thread : readers) thread.join();

                long leaked = openDescriptorsOn(file);
                if (!unexpected.isEmpty() || leaked != 0) {
                    var first = unexpected.peek();
                    var error = new AssertionError("round " + round + ": " + unexpected.size()
                            + " point read(s) failed with an unexpected exception after close"
                            + (first == null ? "" : " (first: " + first + ")")
                            + "; " + leaked + " file descriptor(s) still open on the input");
                    if (first != null) error.initCause(first);
                    throw error;
                }
            }
        } finally {
            Files.delete(file);
        }
    }
}
