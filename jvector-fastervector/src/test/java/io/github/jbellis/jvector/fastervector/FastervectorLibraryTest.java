package io.github.jbellis.jvector.fastervector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FastervectorLibraryTest {

    @Test
    void testConstants() {
        assertEquals("libjvector.so.0", FastervectorLibrary.LIB_JVECTOR);
        assertEquals("libfvec_runtime.so", FastervectorLibrary.LIB_FVEC_RUNTIME);
        assertEquals("fvec.native.dir", FastervectorLibrary.OVERRIDE_PROPERTY);
        assertEquals("native/", FastervectorLibrary.RESOURCE_PREFIX);
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
        assertTrue(error.getMessage().contains("scripts/build-and-run.sh"));
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
