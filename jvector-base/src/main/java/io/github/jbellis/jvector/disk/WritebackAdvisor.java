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

import io.github.jbellis.jvector.annotations.Experimental;

import java.lang.reflect.Constructor;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Starts writeback of a byte range of a file without waiting for it — the host-interface
 * {@code durability_barrier} at the granularity a writer chooses. A durability barrier bounds
 * the dirty-page debt a long write leaves behind so that scattered reads elsewhere do not push
 * those pages out on the eviction path one at a time; {@code fdatasync} does that for the whole
 * file and blocks until it is done, while {@code sync_file_range(SYNC_FILE_RANGE_WRITE)} queues
 * exactly the range and returns. The native implementation lives in {@code jvector-native}
 * ({@code SyncFileRangeAdvisor}); {@link #open} finds it by reflection the way
 * {@link ReaderSupplierFactory} finds {@code MemorySegmentReader}, and returns {@code null} where
 * it is unavailable so callers can keep whatever coarser barrier they had.
 */
@Experimental
public interface WritebackAdvisor extends AutoCloseable {
    String NATIVE_CLASSNAME = "io.github.jbellis.jvector.disk.SyncFileRangeAdvisor";

    /** Initiates writeback of {@code [offset, offset + length)}; best-effort, never throws. */
    void hint(long offset, long length);

    /** Ranges hinted so far. */
    long hints();

    @Override
    void close();

    /** The native advisor for {@code path}, or {@code null} when none is available on this runtime. */
    static WritebackAdvisor open(Path path) {
        try {
            Class<?> cls = Class.forName(NATIVE_CLASSNAME);
            Constructor<?> ctor = cls.getConstructor(Path.class);
            return (WritebackAdvisor) ctor.newInstance(path);
        } catch (Throwable t) {
            Logger.getLogger(WritebackAdvisor.class.getName()).log(Level.FINE,
                    "native writeback advisor unavailable: {0}: {1}", new Object[] {t.getClass().getName(), t.getMessage()});
            return null;
        }
    }
}
