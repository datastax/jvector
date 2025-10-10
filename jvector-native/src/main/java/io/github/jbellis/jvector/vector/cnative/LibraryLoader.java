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

import java.io.File;
import java.nio.file.Files;

/**
 * Utility class for loading the JVector native library, which provides SIMD-accelerated
 * vector operations through native code implementations.
 * <p>
 * This class implements a fallback loading strategy to maximize compatibility across
 * different deployment environments:
 * <ol>
 * <li>First attempts to load the library from the system library path using
 *     {@link System#loadLibrary(String)}, which checks standard locations like
 *     {@code java.library.path}</li>
 * <li>If that fails, attempts to load the library from the classpath by:
 *     <ul>
 *     <li>Extracting the platform-specific library (e.g., {@code libjvector.so},
 *         {@code libjvector.dylib}, or {@code jvector.dll}) from JAR resources</li>
 *     <li>Copying it to a temporary directory</li>
 *     <li>Loading it using {@link System#load(String)} with the absolute path</li>
 *     </ul>
 * </li>
 * </ol>
 * This dual-strategy approach allows the native library to be bundled within the JAR
 * for ease of distribution while still supporting system-installed libraries for
 * production deployments.
 * <p>
 * The class uses a private constructor to prevent instantiation, as it only provides
 * static utility methods.
 *
 * @see System#loadLibrary(String)
 * @see System#load(String)
 */
public class LibraryLoader {
    private LibraryLoader() {}

    /**
     * Attempts to load the JVector native library using a fallback strategy.
     * <p>
     * The method first tries to load {@code libjvector} from the system library path.
     * If that fails (typically when the library is not installed system-wide), it attempts
     * to extract and load the library from the classpath.
     * <p>
     * The classpath loading process:
     * <ol>
     * <li>Maps the library name to the platform-specific filename using
     *     {@link System#mapLibraryName(String)}</li>
     * <li>Creates a temporary file with the appropriate extension</li>
     * <li>Extracts the library resource from the JAR to the temporary file</li>
     * <li>Loads the library from the temporary file's absolute path</li>
     * </ol>
     * Any errors during the loading process are silently caught, making this method
     * suitable for optional native library loading where fallback implementations exist.
     *
     * @return {@code true} if the library was successfully loaded from either the system
     *         path or the classpath; {@code false} if both loading strategies failed or
     *         the library resource could not be found in the classpath
     */
    public static boolean loadJvector() {
        try {
            System.loadLibrary("jvector");
            return true;
        } catch (UnsatisfiedLinkError e) {
            // ignore
        }
        try {
            // reinventing the wheel instead of picking up deps, so we'll just use the classloader to load the library
            // as a resource and then copy it to a tmp directory and load it from there
            String libName = System.mapLibraryName("jvector");
            File tmpLibFile = File.createTempFile(libName.substring(0, libName.lastIndexOf('.')), libName.substring(libName.lastIndexOf('.')));
            try (var in = LibraryLoader.class.getResourceAsStream("/" + libName);
                 var out = Files.newOutputStream(tmpLibFile.toPath())) {
                if (in != null) {
                    in.transferTo(out);
                    out.flush();
                } else {
                    return false; // couldn't find library
                }
            }
            System.load(tmpLibFile.getAbsolutePath());
            return true;
        } catch (Exception | UnsatisfiedLinkError e) {
            // ignore
        }
        return false;
    }

}
