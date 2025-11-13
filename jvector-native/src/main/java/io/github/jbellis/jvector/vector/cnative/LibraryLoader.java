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
 */
public class LibraryLoader {
    private LibraryLoader() {}
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