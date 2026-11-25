/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VisitedSetTest {
    @Test
    void marksDenseIdsOnceAndClears() {
        final VisitedSet set = new VisitedSet(128);

        assertTrue(set.mark(0));
        assertTrue(set.mark(63));
        assertTrue(set.mark(64));
        assertFalse(set.mark(0));
        assertFalse(set.mark(64));

        set.clear();
        assertTrue(set.mark(0));
        assertTrue(set.mark(64));
    }

    @Test
    void growsBeyondItsInitialBound() {
        final VisitedSet set = new VisitedSet(1);

        assertTrue(set.mark(0));
        assertTrue(set.mark(1_000));
        assertFalse(set.mark(1_000));
        set.clear();
        assertTrue(set.mark(1_000));
    }

    @Test
    void fallsBackForSparseIdSpaces() {
        final VisitedSet set = new VisitedSet(Integer.MAX_VALUE);

        assertTrue(set.mark(5));
        assertTrue(set.mark(1_000_000));
        assertFalse(set.mark(5));
        assertFalse(set.isDense());
        set.clear();
        assertTrue(set.mark(5));
    }

    @Test
    void largeOrdinalSpacesStaySparse() {
        final VisitedSet set = new VisitedSet(1_000_000);
        assertFalse(set.isDense());
        assertTrue(set.mark(999_999));
        assertFalse(set.isDense());
        assertFalse(set.mark(999_999));
        set.clear();
        assertTrue(set.mark(999_999));
    }

    @Test
    void growingPastDenseBudgetPreservesExistingMarks() {
        final VisitedSet set = new VisitedSet(128);
        assertTrue(set.mark(63));
        assertTrue(set.mark(64));
        assertTrue(set.mark(16_384));
        assertFalse(set.isDense());
        assertFalse(set.mark(63));
        assertFalse(set.mark(64));
        assertFalse(set.mark(16_384));

        set.clear();
        assertTrue(set.mark(63));
        assertTrue(set.mark(16_384));
    }
}
