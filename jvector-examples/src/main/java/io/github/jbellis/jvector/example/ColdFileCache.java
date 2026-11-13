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

package io.github.jbellis.jvector.example;

import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

/** Linux file-specific cache eviction and residency verification; never drops global caches. */
final class ColdFileCache {
    private ColdFileCache() {}

    /** Pages of a file still in the page cache after eviction, out of its total. */
    static final class Residency {
        final long residentPages;
        final long totalPages;

        private Residency(long residentPages, long totalPages) {
            this.residentPages = residentPages;
            this.totalPages = totalPages;
        }
    }

    /** Evicts {@code file} and fails unless none of its pages remain cached. */
    static void evictAndVerify(Path file) throws IOException {
        var residency = evict(file);
        if (residency.residentPages != 0) throw new IOException("Cold-cache check failed: " + residency.residentPages + "/"
                + residency.totalPages + " input pages remain resident; stop other readers or choose --cache uncontrolled");
        System.out.printf("Cold cache verified: 0/%,d input pages resident%n", residency.totalPages);
    }

    /**
     * Writes back and drops {@code file}'s pages from the page cache, then reports how many remain: pages that a
     * process has mapped and touched stay resident.
     */
    static Residency evict(Path file) throws IOException {
        if (!System.getProperty("os.name").equals("Linux") || !System.getProperty("os.arch").matches("amd64|aarch64"))
            throw new IOException("File cache eviction requires 64-bit Linux");
        // Dirty pages cannot be dropped, so write back first.
        try (var channel = java.nio.channels.FileChannel.open(file, java.nio.file.StandardOpenOption.READ)) {
            channel.force(true);
        }
        try (var nativeCalls = new NativeCalls()) {
            var open = nativeCalls.function("open", "JAVA_INT", "ADDRESS", "JAVA_INT");
            var close = nativeCalls.function("close", "JAVA_INT", "JAVA_INT");
            var advise = nativeCalls.function("posix_fadvise", "JAVA_INT", "JAVA_INT", "JAVA_LONG", "JAVA_LONG", "JAVA_INT");
            var mmap = nativeCalls.function("mmap", "ADDRESS", "ADDRESS", "JAVA_LONG", "JAVA_INT", "JAVA_INT", "JAVA_INT", "JAVA_LONG");
            var mincore = nativeCalls.function("mincore", "JAVA_INT", "ADDRESS", "JAVA_LONG", "ADDRESS");
            var munmap = nativeCalls.function("munmap", "JAVA_INT", "ADDRESS", "JAVA_LONG");
            var getpagesize = nativeCalls.function("getpagesize", "JAVA_INT");
            int fd = (int) open.invokeWithArguments(nativeCalls.string(file.toAbsolutePath().toString()), 0);
            if (fd < 0) throw new IOException("Cannot open file for cache eviction: " + file);
            try {
                int error = (int) advise.invokeWithArguments(fd, 0L, 0L, 4); // POSIX_FADV_DONTNEED
                if (error != 0) throw new IOException("posix_fadvise failed: " + error);
                long length = Files.size(file);
                int pageSize = (int) getpagesize.invokeWithArguments();
                long pages = (length + pageSize - 1) / pageSize;
                Object resident = nativeCalls.allocate(pages);
                // Mapping alone does not fault in pages; mincore only inspects their residency.
                Object mapping = mmap.invokeWithArguments(nativeCalls.nullSegment(), length, 1, 1, fd, 0L);
                if (nativeCalls.address(mapping) == -1L) throw new IOException("mmap for residency check failed");
                try {
                    if ((int) mincore.invokeWithArguments(mapping, length, resident) != 0)
                        throw new IOException("mincore residency check failed");
                    return new Residency(nativeCalls.residentPages(resident, pages), pages);
                } finally {
                    if ((int) munmap.invokeWithArguments(mapping, length) != 0) throw new IOException("Residency mapping cleanup failed");
                }
            } finally {
                if ((int) close.invokeWithArguments(fd) != 0) throw new IOException("Cache-check descriptor cleanup failed");
            }
        } catch (IOException e) { throw e; }
        catch (Throwable e) { throw new IOException("File cache eviction requires JDK 23+ with --enable-native-access=ALL-UNNAMED", e); }
    }

    // Resolve FFM only for cold-cache setup, so ordinary examples retain their Java 11 target.
    // None of this reflection runs inside the measured interval or vector-processing loop.
    private static final class NativeCalls implements AutoCloseable {
        private final Class<?> arenaType = Class.forName("java.lang.foreign.Arena");
        private final Class<?> segmentType = Class.forName("java.lang.foreign.MemorySegment");
        private final Class<?> layoutType = Class.forName("java.lang.foreign.MemoryLayout");
        private final Class<?> valueType = Class.forName("java.lang.foreign.ValueLayout");
        private final Class<?> descriptorType = Class.forName("java.lang.foreign.FunctionDescriptor");
        private final Class<?> linkerType = Class.forName("java.lang.foreign.Linker");
        private final Class<?> optionType = Class.forName("java.lang.foreign.Linker$Option");
        private final Class<?> lookupType = Class.forName("java.lang.foreign.SymbolLookup");
        private final Object arena, linker, lookup;
        NativeCalls() throws Exception {
            arena = arenaType.getMethod("ofConfined").invoke(null);
            linker = linkerType.getMethod("nativeLinker").invoke(null);
            lookup = linkerType.getMethod("defaultLookup").invoke(linker);
        }
        private Object layout(String name) throws Exception { return valueType.getField(name).get(null); }
        MethodHandle function(String name, String result, String... parameters) throws Exception {
            var symbol = (Optional<?>) lookupType.getMethod("find", String.class).invoke(lookup, name);
            Object layouts = Array.newInstance(layoutType, parameters.length);
            for (int i = 0; i < parameters.length; i++) Array.set(layouts, i, layout(parameters[i]));
            Object descriptor = descriptorType.getMethod("of", layoutType, layouts.getClass()).invoke(null, layout(result), layouts);
            Object options = Array.newInstance(optionType, 0);
            return (MethodHandle) linkerType.getMethod("downcallHandle", segmentType, descriptorType, options.getClass())
                    .invoke(linker, symbol.orElseThrow(), descriptor, options);
        }
        Object string(String value) throws Exception { return arenaType.getMethod("allocateFrom", String.class).invoke(arena, value); }
        Object allocate(long bytes) throws Exception { return arenaType.getMethod("allocate", long.class).invoke(arena, bytes); }
        Object nullSegment() throws Exception { return segmentType.getField("NULL").get(null); }
        long address(Object segment) throws Exception { return (long) segmentType.getMethod("address").invoke(segment); }
        long residentPages(Object segment, long count) throws Exception {
            // Copy only the tiny residency bitmap, not file data. No per-page reflective calls.
            byte[] bitmap = (byte[]) segmentType.getMethod("toArray", Class.forName("java.lang.foreign.ValueLayout$OfByte"))
                    .invoke(segment, layout("JAVA_BYTE"));
            long remaining = 0;
            for (int i = 0; i < count; i++) if ((bitmap[i] & 1) != 0) remaining++;
            return remaining;
        }
        public void close() throws Exception { arenaType.getMethod("close").invoke(arena); }
    }
}
