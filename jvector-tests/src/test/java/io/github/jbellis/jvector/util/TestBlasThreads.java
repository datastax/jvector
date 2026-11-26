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

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/** BLAS thread scopes nest per thread and restore the enclosing count. */
public class TestBlasThreads {
    @Test
    public void scopesNestAndRestore() {
        assertEquals(0, BlasThreads.current());
        try (var outer = BlasThreads.limit(8)) {
            assertEquals(8, BlasThreads.current());
            try (var single = BlasThreads.singleUnlessLimited()) {
                assertEquals("An enclosing limit applies", 8, BlasThreads.current());
            }
            try (var inner = BlasThreads.limit(2)) {
                assertEquals(2, BlasThreads.current());
            }
            assertEquals(8, BlasThreads.current());
        }
        assertEquals(0, BlasThreads.current());
        try (var single = BlasThreads.singleUnlessLimited()) {
            assertEquals(1, BlasThreads.current());
        }
        assertEquals(0, BlasThreads.current());
    }

    @Test
    public void scopesAreThreadLocal() throws Exception {
        try (var scope = BlasThreads.limit(4)) {
            int[] seen = new int[1];
            var other = new Thread(() -> seen[0] = BlasThreads.current());
            other.start();
            other.join();
            assertEquals(0, seen[0]);
        }
    }

    @Test
    public void countMustBePositive() {
        assertThrows(IllegalArgumentException.class, () -> BlasThreads.limit(0));
    }
}
