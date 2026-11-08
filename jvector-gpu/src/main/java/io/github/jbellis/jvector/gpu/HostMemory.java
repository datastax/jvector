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

package io.github.jbellis.jvector.gpu;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * The host memory a GPU build needs, estimated from its shape alone, and the host memory available, so that
 * {@link TornadoGraphBuildAccelerator#supports} can decline a build that would not fit before anything is allocated.
 * A declined build runs on the CPU as usual.
 */
final class HostMemory {

    /** A fixed host budget for heap and native memory together ({@code -Djvector.gpu.hostMemory}, e.g. {@code 48GB}). */
    static final String PROPERTY = "jvector.gpu.hostMemory";

    private HostMemory() {
    }

    /** What a build of {@code n x dim} with max degree {@code maxDegree} holds at its peak, in bytes. */
    record Need(long heap, long nativeMemory) {
        long total() {
            return heap + nativeMemory;
        }
    }

    /**
     * Heap: the candidate lists and the pruned lists ({@link PruneKernels#SLOTS} ids and scores each) and the
     * builder's lists (with room for overflow). Native memory: the float32 rows when they are copied (not for
     * {@link MappedVectors}), the rows packed to FP16, the vectors of the upper layers, and cuVS's output
     * ({@code SLOTS + 1} 64-bit ids and float distances per row). The FP16 rows and cuVS's output are counted
     * together, although a batched build frees the one before it allocates the other, so the estimate errs high.
     */
    static Need need(long n, int dim, int maxDegree, boolean mapped) {
        int slots = PruneKernels.SLOTS;
        long lists = 2L * slots * (Integer.BYTES + Float.BYTES);
        long graph = (maxDegree + maxDegree / 2L) * (Integer.BYTES + Float.BYTES);
        long heap = n * (lists + graph);
        long rowBytes = (long) dim * Float.BYTES;
        long rows = mapped ? 0 : n * rowBytes;
        long upper = n / Math.max(maxDegree, 2) * rowBytes;
        long cuvs = n * (slots + 1L) * (Long.BYTES + Float.BYTES);
        return new Need(heap, rows + PackedVectors.bytes(n, dim) + upper + cuvs);
    }

    /** Why {@code need} does not fit, or null if it does. */
    static String shortfall(Need need, long freeHeap, long committedFreeHeap, long availableNative, String fixedBudget) {
        if (fixedBudget != null) {
            long budget = parseBytes(fixedBudget);
            return need.total() > budget
                    ? String.format(Locale.ROOT, "needs ~%.1f GB of host memory, -D%s=%s", need.total() / 1e9, PROPERTY, fixedBudget)
                    : null;
        }
        if (need.heap() > freeHeap) {
            return String.format(Locale.ROOT, "needs ~%.1f GB of heap, %.1f GB free (raise -Xmx, or set -D%s)",
                    need.heap() / 1e9, freeHeap / 1e9, PROPERTY);
        }
        // the heap the build grows into comes out of the same physical memory as the native allocations
        long heapGrowth = Math.max(0, need.heap() - committedFreeHeap);
        if (availableNative != Long.MAX_VALUE && need.nativeMemory() + heapGrowth > availableNative) {
            return String.format(Locale.ROOT, "needs ~%.1f GB of native memory and ~%.1f GB more heap, %.1f GB of memory available (set -D%s to override)",
                    need.nativeMemory() / 1e9, heapGrowth / 1e9, availableNative / 1e9, PROPERTY);
        }
        return null;
    }

    /** Why a build of this shape does not fit in this JVM and on this host now, or null if it does. */
    static String shortfall(long n, int dim, int maxDegree, boolean mapped) {
        return shortfall(need(n, dim, maxDegree, mapped), freeHeap(), Runtime.getRuntime().freeMemory(), availableNative(),
                System.getProperty(PROPERTY));
    }

    /** The heap this JVM can still grow into. */
    static long freeHeap() {
        var runtime = Runtime.getRuntime();
        return runtime.maxMemory() - (runtime.totalMemory() - runtime.freeMemory());
    }

    /** {@code MemAvailable} from {@code /proc/meminfo}; unlimited where it cannot be read. */
    static long availableNative() {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/meminfo"))) {
                if (line.startsWith("MemAvailable:")) {
                    return parseMemInfoLine(line);
                }
            }
        } catch (IOException | RuntimeException e) {
            // not Linux, or not readable: leave it to the allocation
        }
        return Long.MAX_VALUE;
    }

    /** A {@code /proc/meminfo} line such as {@code MemAvailable:   12345678 kB}, in bytes. */
    static long parseMemInfoLine(String line) {
        String[] parts = line.trim().split("\\s+");
        return Long.parseLong(parts[1]) * 1024L;
    }

    /** {@code 48GB}, {@code 512MB}, {@code 64KB} or a number of bytes. */
    static long parseBytes(String setting) {
        String size = setting.trim().toUpperCase(Locale.ROOT);
        long unit = size.endsWith("GB") ? 1L << 30 : size.endsWith("MB") ? 1L << 20 : size.endsWith("KB") ? 1L << 10 : 1L;
        String digits = size.replaceAll("[^0-9]", "");
        if (digits.isEmpty()) {
            throw new IllegalArgumentException("not a size: " + setting);
        }
        return Long.parseLong(digits) * unit;
    }
}
