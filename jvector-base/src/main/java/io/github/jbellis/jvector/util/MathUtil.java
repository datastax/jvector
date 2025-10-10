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
 * Utility class providing common mathematical operations. This class cannot be instantiated.
 */
public class MathUtil {
    private MathUtil() {
        // Prevent instantiation
    }

    /**
     * Returns the square of a float value. While simple, this method improves code readability
     * by making the squaring operation explicit.
     * @param a the value to square
     * @return a * a
     */
    public static float square(float a) {
        return a * a;
    }
}
