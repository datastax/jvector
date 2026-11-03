package io.github.jbellis.jvector.fastervector;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Loads the fvec native libraries from the classpath resources or an override directory.
 */
public final class FastervectorLibrary {
    public static final String LIB_JVECTOR = "libjvector.so.0";
    public static final String LIB_FVEC_RUNTIME = "libfvec_runtime.so";
    public static final String OVERRIDE_PROPERTY = "fvec.native.dir";
    public static final String RESOURCE_PREFIX = "native/";

    private static volatile boolean loaded = false;

    private FastervectorLibrary() {}

    /**
     * Loads the bundled native libraries into the running JVM.
     * <p>
     * Loading order:
     * <ol>
     *   <li>{@code libjvector.so.0}</li>
     *   <li>{@code libfvec_runtime.so}</li>
     * </ol>
     * This method is thread-safe and idempotent.
     */
    public static synchronized void load() {
        if (loaded) {
            return;
        }

        String overrideDir = System.getProperty(OVERRIDE_PROPERTY);
        if (overrideDir != null && !overrideDir.isBlank()) {
            Path dir = Path.of(overrideDir);
            loadFromDirectory(dir);
            loaded = true;
            return;
        }

        try {
            Path tempDir = Files.createTempDirectory("fvec-native-");
            registerShutdownDelete(tempDir);
            List<Path> extracted = extract(tempDir, FastervectorLibrary.class.getClassLoader(), List.of(LIB_JVECTOR, LIB_FVEC_RUNTIME));
            for (Path lib : extracted) {
                System.load(lib.toAbsolutePath().toString());
            }
            loaded = true;
        } catch (IOException e) {
            throw new UnsatisfiedLinkError("Failed to extract native libraries: " + e.getMessage());
        }
    }

    /**
     * Checks if the libraries have been successfully loaded.
     */
    public static boolean isLoaded() {
        return loaded;
    }

    /**
     * Pure helper to extract specified classpath resources into the target directory.
     * Package-private for unit testing.
     */
    static List<Path> extract(Path targetDir, ClassLoader classLoader, List<String> libraryNames) throws IOException {
        Objects.requireNonNull(targetDir, "targetDir must not be null");
        ClassLoader loader = classLoader != null ? classLoader : FastervectorLibrary.class.getClassLoader();
        List<Path> extractedPaths = new ArrayList<>();

        for (String libName : libraryNames) {
            String resourcePath = RESOURCE_PREFIX + libName;
            Path destFile = targetDir.resolve(libName);

            try (InputStream in = loader.getResourceAsStream(resourcePath)) {
                if (in == null) {
                    throw new UnsatisfiedLinkError(
                        "Native library resource '" + resourcePath + "' not found on classpath. " +
                        "Ensure native libraries are built and staged into src/main/resources/native/ or run scripts/build-and-run.sh"
                    );
                }
                Files.copy(in, destFile, StandardCopyOption.REPLACE_EXISTING);
                destFile.toFile().deleteOnExit();
                extractedPaths.add(destFile);
            }
        }
        return extractedPaths;
    }

    private static void loadFromDirectory(Path dir) {
        if (!Files.isDirectory(dir)) {
            throw new UnsatisfiedLinkError(
                "Native library directory specified by " + OVERRIDE_PROPERTY + " does not exist: " + dir.toAbsolutePath()
            );
        }
        Path jvector = dir.resolve(LIB_JVECTOR);
        if (!Files.exists(jvector)) {
            Path alt = dir.resolve("libjvector.so");
            if (Files.exists(alt)) {
                jvector = alt;
            } else {
                throw new UnsatisfiedLinkError("Missing " + LIB_JVECTOR + " in directory " + dir.toAbsolutePath());
            }
        }
        Path fvecRuntime = dir.resolve(LIB_FVEC_RUNTIME);
        if (!Files.exists(fvecRuntime)) {
            throw new UnsatisfiedLinkError("Missing " + LIB_FVEC_RUNTIME + " in directory " + dir.toAbsolutePath());
        }

        System.load(jvector.toAbsolutePath().toString());
        System.load(fvecRuntime.toAbsolutePath().toString());
    }

    private static void registerShutdownDelete(Path dir) {
        dir.toFile().deleteOnExit();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                if (Files.exists(dir)) {
                    Files.walk(dir)
                        .sorted(Comparator.reverseOrder())
                        .forEach(p -> {
                            try {
                                Files.deleteIfExists(p);
                            } catch (IOException ignored) {}
                        });
                }
            } catch (Exception ignored) {}
        }, "fvec-native-cleanup"));
    }
}
