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

/**
 * Utility methods for mathematical operations.
 */
public class MathUtil {
    /** Private constructor to prevent instantiation. */
    private MathUtil() {
    }
    /**
     * Squares the given float value.
     * While this may look silly at first, it really does make code more readable.
     *
     * @param a the value to square
     * @return the square of a
     */
    public static float square(float a) {
        return a * a;
    }
}
