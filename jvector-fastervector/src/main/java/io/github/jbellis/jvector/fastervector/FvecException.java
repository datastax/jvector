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

/**
 * Unchecked exception thrown when a native fvec C-API function returns a non-zero
 * result code.  The native side prints details to stderr; the message in this
 * exception names the C function that failed.
 */
public class FvecException extends RuntimeException {

    public FvecException(String message) {
        super(message);
    }

    public FvecException(String message, Throwable cause) {
        super(message, cause);
    }
}
