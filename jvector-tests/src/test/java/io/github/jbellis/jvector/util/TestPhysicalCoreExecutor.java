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

import io.github.jbellis.jvector.LuceneTestCase;
import org.junit.Test;

import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

public class TestPhysicalCoreExecutor extends LuceneTestCase {

    @Test
    public void resolvedCountIsSaneAndCached() {
        int logical = Runtime.getRuntime().availableProcessors();
        int count = PhysicalCoreExecutor.getPhysicalCoreCount();
        // Whether resolved from the property, an OS probe, or the heuristic fallback, the count
        // must be a positive value that never exceeds the logical processors visible to the JVM.
        assertTrue("count must be >= 1, got " + count, count >= 1);
        assertTrue("count (" + count + ") must be <= availableProcessors (" + logical + ")",
                count <= logical);
        // Repeated calls return the cached value.
        assertEquals(count, PhysicalCoreExecutor.getPhysicalCoreCount());
    }

    @Test
    public void poolMatchesResolvedCountAndIsShared() {
        ForkJoinPool pool = PhysicalCoreExecutor.pool();
        assertNotNull(pool);
        assertEquals(PhysicalCoreExecutor.getPhysicalCoreCount(), pool.getParallelism());
        // pool()/instance() hand back the same lazily-created singleton.
        assertSame(pool, PhysicalCoreExecutor.pool());
        assertSame(PhysicalCoreExecutor.instance(), PhysicalCoreExecutor.instance());
    }

    @Test
    public void explicitPropertyOverrideIsHonored() {
        // The property is resolved once per JVM and cached; only assert against it when it is set,
        // so this test is meaningful under -Djvector.physical_core_count and inert otherwise.
        Integer override = Integer.getInteger("jvector.physical_core_count");
        if (override != null) {
            assertEquals(override.intValue(), PhysicalCoreExecutor.getPhysicalCoreCount());
        }
    }
}
