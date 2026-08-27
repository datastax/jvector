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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;

/**
 * {@link WritebackAdvisor} over {@code sync_file_range(2)} with {@code SYNC_FILE_RANGE_WRITE}:
 * queues writeback of the dirty pages in the range and returns without waiting. Opens its own
 * descriptor on the path, so it works alongside whatever channel is doing the writing. Linux only;
 * where the symbol is missing every hint is a no-op.
 */
public final class SyncFileRangeAdvisor implements WritebackAdvisor {
    private static final Logger logger = LoggerFactory.getLogger(SyncFileRangeAdvisor.class);
    private static final int SYNC_FILE_RANGE_WRITE = 2;
    private static final MethodHandle OPEN_H;
    private static final MethodHandle CLOSE_H;
    private static final MethodHandle SYNC_FILE_RANGE_H;

    static {
        MethodHandle open = null, close = null, sfr = null;
        try {
            var linker = Linker.nativeLinker();
            var lookup = linker.defaultLookup();
            var openSym = lookup.find("open");
            var closeSym = lookup.find("close");
            var sfrSym = lookup.find("sync_file_range");
            if (openSym.isPresent() && closeSym.isPresent() && sfrSym.isPresent()) {
                open = linker.downcallHandle(openSym.get(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
                close = linker.downcallHandle(closeSym.get(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
                sfr = linker.downcallHandle(sfrSym.get(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
            }
        } catch (Throwable t) {
            logger.warn("native open/sync_file_range unavailable; writeback hints disabled", t);
            open = null;
            close = null;
            sfr = null;
        }
        OPEN_H = open;
        CLOSE_H = close;
        SYNC_FILE_RANGE_H = sfr;
    }

    private final int fd;
    private final AtomicLong hints = new AtomicLong();
    private volatile boolean warnedFailure;

    public SyncFileRangeAdvisor(Path path) throws IOException {
        if (OPEN_H == null || SYNC_FILE_RANGE_H == null) {
            throw new IOException("sync_file_range is not available on this runtime");
        }
        try (var confined = Arena.ofConfined()) {
            var cPath = confined.allocateFrom(path.toString());
            int f = (int) OPEN_H.invokeExact(cPath, 0); // O_RDONLY: sync_file_range acts on the page cache, not the descriptor's mode
            if (f < 0) {
                throw new IOException("open failed for " + path);
            }
            this.fd = f;
        } catch (IOException e) {
            throw e;
        } catch (Throwable t) {
            throw new IOException("opening " + path + " for writeback advice", t);
        }
    }

    /** Whether the native calls are linked on this runtime. */
    public static boolean available() {
        return OPEN_H != null && SYNC_FILE_RANGE_H != null;
    }

    @Override
    public void hint(long offset, long length) {
        if (length <= 0) {
            return;
        }
        try {
            int rc = (int) SYNC_FILE_RANGE_H.invokeExact(fd, Math.max(0, offset), length, SYNC_FILE_RANGE_WRITE);
            if (rc != 0 && !warnedFailure) {
                warnedFailure = true;
                logger.warn("sync_file_range returned {}; further failures not logged", rc);
            }
            hints.incrementAndGet();
        } catch (Throwable t) {
            if (!warnedFailure) {
                warnedFailure = true;
                logger.warn("sync_file_range failed; further failures not logged", t);
            }
        }
    }

    @Override
    public long hints() {
        return hints.get();
    }

    @Override
    public void close() {
        if (CLOSE_H != null) {
            try {
                int ignored = (int) CLOSE_H.invokeExact(fd);
            } catch (Throwable t) {
                // best effort
            }
        }
    }
}
