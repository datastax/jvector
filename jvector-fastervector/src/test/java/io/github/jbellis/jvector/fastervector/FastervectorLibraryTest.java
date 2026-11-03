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

package io.github.jbellis.jvector.fastervector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FastervectorLibraryTest {

    @Test
    void testConstants() {
        assertEquals("libjvector.so.0", FastervectorLibrary.LIB_JVECTOR);
        assertEquals("libfvec_runtime.so", FastervectorLibrary.LIB_FVEC_RUNTIME);
        assertEquals("jvector.fastervector.native.dir", FastervectorLibrary.OVERRIDE_PROPERTY);
        assertEquals("native/linux-x86_64/", FastervectorLibrary.RESOURCE_PREFIX);
    }

    @Test
    void testSupportedPlatform() {
        assertTrue(FastervectorLibrary.isSupportedPlatform("Linux", "amd64"));
        assertTrue(FastervectorLibrary.isSupportedPlatform("Linux", "x86_64"));
        assertFalse(FastervectorLibrary.isSupportedPlatform("Linux", "aarch64"));
        assertFalse(FastervectorLibrary.isSupportedPlatform("Mac OS X", "x86_64"));
        assertFalse(FastervectorLibrary.isSupportedPlatform("Windows 11", "amd64"));
    }

    @Test
    void testCpuFlagsHaveAvx2() {
        assertTrue(FastervectorLibrary.cpuFlagsHaveAvx2("flags\t\t: fpu sse4_2 avx avx2 fma"));
        assertTrue(FastervectorLibrary.cpuFlagsHaveAvx2("flags\t\t: avx2"));
        assertFalse(FastervectorLibrary.cpuFlagsHaveAvx2("flags\t\t: fpu sse4_2 avx fma avx512f"));
        assertFalse(FastervectorLibrary.cpuFlagsHaveAvx2(""));
    }

    @Test
    void testExtractSuccess(@TempDir Path tempDir) throws IOException {
        ClassLoader cl = getClass().getClassLoader();
        List<Path> extracted = FastervectorLibrary.extract(tempDir, cl, List.of("fake_lib1.so", "fake_lib2.so"));

        assertEquals(2, extracted.size());
        Path file1 = tempDir.resolve("fake_lib1.so");
        Path file2 = tempDir.resolve("fake_lib2.so");

        assertTrue(Files.exists(file1), "fake_lib1.so should exist");
        assertTrue(Files.exists(file2), "fake_lib2.so should exist");

        assertEquals("dummy-content-1\n", Files.readString(file1));
        assertEquals("dummy-content-2\n", Files.readString(file2));
    }

    @Test
    void testExtractMissingResourceThrows(@TempDir Path tempDir) {
        ClassLoader cl = getClass().getClassLoader();
        UnsatisfiedLinkError error = assertThrows(UnsatisfiedLinkError.class, () ->
            FastervectorLibrary.extract(tempDir, cl, List.of("non_existent_lib.so"))
        );

        assertNotNull(error.getMessage());
        assertTrue(error.getMessage().contains("non_existent_lib.so"));
        assertTrue(error.getMessage().contains("stage-fastervector-native.sh"));
        assertTrue(error.getMessage().contains(FastervectorLibrary.OVERRIDE_PROPERTY));
    }

    @Test
    void testExtractOverwriteIdempotent(@TempDir Path tempDir) throws IOException {
        ClassLoader cl = getClass().getClassLoader();
        List<Path> first = FastervectorLibrary.extract(tempDir, cl, List.of("fake_lib1.so"));
        assertEquals(1, first.size());
        assertEquals("dummy-content-1\n", Files.readString(first.get(0)));

        // Run second time in same target directory
        List<Path> second = FastervectorLibrary.extract(tempDir, cl, List.of("fake_lib1.so"));
        assertEquals(1, second.size());
        assertEquals("dummy-content-1\n", Files.readString(second.get(0)));
    }
}
