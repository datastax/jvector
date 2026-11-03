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

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.SymbolLookup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * Loads the fvec native libraries from the classpath resources or an override directory.
 * <p>
 * The libraries are built for Linux x86_64 with AVX2. {@link #load()} refuses to load them
 * anywhere else, and refuses to load them alongside jvector-native's own {@code libjvector},
 * which shares their SONAME and symbols.
 */
public final class FastervectorLibrary {
    public static final String LIB_JVECTOR = "libjvector.so.0";
    public static final String LIB_FVEC_RUNTIME = "libfvec_runtime.so";
    public static final String OVERRIDE_PROPERTY = "jvector.fastervector.native.dir";
    public static final String RESOURCE_PREFIX = "native/linux-x86_64/";

    /** Symbol exported by jvector-native's libjvector; its presence means that library is loaded. */
    static final String JVECTOR_NATIVE_PROBE_SYMBOL = "jvector_supports_abi";
    static final String DISABLE_JVECTOR_NATIVE_FLAG = "-Djvector.enable_native_vectorization=false";

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
     *
     * @throws UnsupportedOperationException if the platform is not Linux x86_64, the CPU lacks
     *         AVX2, or jvector-native's {@code libjvector} is already loaded
     */
    public static synchronized void load() {
        if (loaded) {
            return;
        }
        checkPlatform();
        checkJvectorNativeNotLoaded();

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

    private static void checkPlatform() {
        String os = System.getProperty("os.name", "");
        String arch = System.getProperty("os.arch", "");
        if (!isSupportedPlatform(os, arch)) {
            throw new UnsupportedOperationException(
                "fastervector native libraries are only available on Linux x86_64; this JVM reports "
                + os + " " + arch);
        }
        String cpuFlags;
        try (Stream<String> lines = Files.lines(Path.of("/proc/cpuinfo"))) {
            cpuFlags = lines.filter(l -> l.startsWith("flags")).findFirst().orElse("");
        } catch (IOException e) {
            throw new UnsupportedOperationException(
                "Cannot read /proc/cpuinfo to check for AVX2, which fastervector requires: " + e.getMessage(), e);
        }
        if (!cpuFlagsHaveAvx2(cpuFlags)) {
            throw new UnsupportedOperationException(
                "fastervector native libraries require a CPU with AVX2, and /proc/cpuinfo does not list avx2");
        }
    }

    private static void checkJvectorNativeNotLoaded() {
        if (SymbolLookup.loaderLookup().find(JVECTOR_NATIVE_PROBE_SYMBOL).isPresent()) {
            throw new UnsupportedOperationException(
                "jvector-native's libjvector is already loaded, and it conflicts with the libjvector.so.0 "
                + "bundled with fastervector (same SONAME and symbols). Start the JVM with "
                + DISABLE_JVECTOR_NATIVE_FLAG + " so jvector uses its Panama vector provider instead.");
        }
    }

    /** Package-private for unit testing. */
    static boolean isSupportedPlatform(String osName, String osArch) {
        return osName.toLowerCase(Locale.ROOT).startsWith("linux")
            && (osArch.equals("amd64") || osArch.equals("x86_64"));
    }

    /** Package-private for unit testing. {@code flagsLine} is the {@code flags} line of /proc/cpuinfo. */
    static boolean cpuFlagsHaveAvx2(String flagsLine) {
        int colon = flagsLine.indexOf(':');
        String flags = colon >= 0 ? flagsLine.substring(colon + 1) : flagsLine;
        return Arrays.asList(flags.trim().split("\\s+")).contains("avx2");
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
                        "Stage the libraries with jvector-fastervector/scripts/stage-fastervector-native.sh and rebuild, " +
                        "or set -D" + OVERRIDE_PROPERTY + " to a directory containing " +
                        LIB_JVECTOR + " and " + LIB_FVEC_RUNTIME
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
