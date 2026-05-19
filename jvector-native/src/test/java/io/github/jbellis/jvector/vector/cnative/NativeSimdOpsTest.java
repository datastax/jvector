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

package io.github.jbellis.jvector.vector.cnative;

import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class NativeSimdOpsTest {

    /**
     * Reads /proc/cpuinfo and returns true if all AVX-512 flags required by
     * check_avx512_compatibility() are present: avx512f, avx512cd, avx512dq,
     * avx512bw, avx512vl.
     */
    private static boolean cpuinfoReportsAvx512() throws IOException {
        List<String> lines = Files.readAllLines(Path.of("/proc/cpuinfo"));
        List<String> required = List.of("avx512f", "avx512cd", "avx512dq", "avx512bw", "avx512vl");
        for (String line : lines) {
            if (line.startsWith("flags")) {
                String[] flags = line.split("\\s+");
                List<String> flagList = List.of(flags);
                return flagList.containsAll(required);
            }
        }
        return false;
    }

    @Test
    public void testCheckAvx512CompatibilityMatchesCpuinfo() throws IOException {
        boolean libraryLoaded = LibraryLoader.loadJvector();
        Assume.assumeTrue("Native jvector library not available; skipping AVX-512 check", libraryLoaded);

        boolean expectedFromCpuinfo = cpuinfoReportsAvx512();
        boolean actualFromNative = NativeSimdOps.check_avx512_compatibility();

        assertEquals(
                "check_avx512_compatibility() should match AVX-512 flag presence in /proc/cpuinfo",
                expectedFromCpuinfo,
                actualFromNative);
    }
}
