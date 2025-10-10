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
 * Exception utilities.
 */
public class ExceptionUtils {
    /**
     * Constructor.
     */
    public ExceptionUtils() {
    }

    /**
     * Throw IO exception.
     * @param t the t
     * @throws IOException if an error occurs
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
