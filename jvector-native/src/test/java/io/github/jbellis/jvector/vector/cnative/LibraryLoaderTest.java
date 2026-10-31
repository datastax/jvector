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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;
import static org.junit.jupiter.api.Assertions.*;

@ResourceLock("java.lang.System.properties")
class LibraryLoaderTest {
    @Test
    void optionalResourcesFollowTheExistingArchitectureConvention() {
        String previous = System.getProperty("os.arch");
        try {
            for (String arch : new String[]{"amd64", "x86_64"}) {
                System.setProperty("os.arch", arch);
                assertEquals("/libjvector-x86_64.so", LibraryLoader.resourceNameForArch());
                assertEquals("/libjvector_ash_fp16-x86_64.so", LibraryLoader.resourceNameForArch("jvector_ash_fp16"));
            }
            for (String arch : new String[]{"aarch64", "arm64"}) {
                System.setProperty("os.arch", arch);
                assertEquals("/libjvector-aarch64.so", LibraryLoader.resourceNameForArch());
                assertEquals("/libjvector_ash_fp16-aarch64.so", LibraryLoader.resourceNameForArch("jvector_ash_fp16"));
            }
            System.setProperty("os.arch", "unsupported");
            assertNull(LibraryLoader.resourceNameForArch());
            assertNull(LibraryLoader.resourceNameForArch("jvector_ash_fp16"));
        } finally {
            if (previous == null) System.clearProperty("os.arch");
            else System.setProperty("os.arch", previous);
        }
    }
}
