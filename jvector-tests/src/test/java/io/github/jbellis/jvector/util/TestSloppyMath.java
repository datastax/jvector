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

import static org.junit.Assert.assertThrows;
public class TestSloppyMath {
    @Test
    public void testLatLonBoundaries() {
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(-91, 0, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(91, 0, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(0, -181, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(0, 181, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(0, 0,-91, 0));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(0, 0, 91, 0));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(0, 0, 0, -181));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(0, 0, 0, 181));
    }
}
