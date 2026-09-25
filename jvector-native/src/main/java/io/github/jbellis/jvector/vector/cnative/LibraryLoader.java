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
 * This class is used to load supporting native libraries. First, it tries to load the library from the system path.
 * If that fails, it tries to load the library from the classpath (using the usual copying to a tmp directory route).
 * <p>
 * Two resource names are bundled in the jar:
 * <ul>
 *   <li>{@code /libjvector-x86_64.so}  — built natively for x86_64</li>
 *   <li>{@code /libjvector-aarch64.so} — cross-compiled for aarch64</li>
 * </ul>
 * At runtime the correct file is chosen based on {@code os.arch}.
 */
public class LibraryLoader {
    private LibraryLoader() {}

    /**
     * Returns the classpath resource name for the native library appropriate for the
     * current CPU architecture, or {@code null} when the architecture is not supported.
     */
    static String resourceNameForArch() {
        String arch = System.getProperty("os.arch", "");
        if (arch.equals("aarch64") || arch.equals("arm64")) {
            return "/libjvector-aarch64.so";
        }
        if (arch.equals("amd64") || arch.equals("x86_64")) {
            return "/libjvector-x86_64.so";
        }
        return null;
    }

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
            String resourceName = resourceNameForArch();
            if (resourceName == null) {
                return false; // unsupported architecture
            }
            String baseName = resourceName.substring(1, resourceName.lastIndexOf('.'));   // e.g. "libjvector-aarch64"
            String ext      = resourceName.substring(resourceName.lastIndexOf('.'));      // e.g. ".so"
            File tmpLibFile = File.createTempFile(baseName, ext);
            try (var in = LibraryLoader.class.getResourceAsStream(resourceName);
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
