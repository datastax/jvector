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

package io.github.jbellis.jvector.util;

import java.io.IOException;

/**
 * Utility methods for exception handling.
 */
public class ExceptionUtils {
    /**
     * Private constructor to prevent instantiation.
     */
    private ExceptionUtils() {
    }

    /**
     * Rethrows the given throwable as an IOException or RuntimeException.
     * If the throwable is already an IOException, it is thrown directly.
     * If it's a RuntimeException or Error, it is also thrown directly.
     * Otherwise, it is wrapped in a RuntimeException.
     *
     * @param t the throwable to rethrow
     * @throws IOException if t is an IOException
     */
    public static void throwIoException(Throwable t) throws IOException {
        if (t instanceof RuntimeException) {
            throw (RuntimeException) t;
        } else if (t instanceof Error) {
            throw (Error) t;
        } else if (t instanceof IOException) {
            throw (IOException) t;
        } else {
            throw new RuntimeException(t);
        }
    }
}
