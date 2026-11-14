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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.EnumSet;
import java.util.Set;

import static org.junit.Assert.*;

/** Cache names must change exactly when something that shapes the index file changes. */
public class OnDiskGraphIndexCacheKeyTest {
    private static final Set<FeatureId> FUSED_NVQ = EnumSet.of(FeatureId.FUSED_ASH, FeatureId.NVQ_VECTORS);

    @Rule
    public TemporaryFolder dir = new TemporaryFolder();

    private static String name(OnDiskGraphIndexCache cache, String dataset, int count, int dim, int nvq) {
        var key = cache.keyWithCompressorId(dataset, FUSED_NVQ, 32, 100, 1.2f, 1.2f, true, true,
                "schema3:ASH_x:symmetric-all", count, dim, nvq);
        return cache.resolve(key).finalPath.getFileName().toString();
    }

    @Test
    public void signatureCarriesShapeFormatAndNvqLayout() throws Exception {
        var cache = OnDiskGraphIndexCache.initialize(dir.getRoot().toPath());
        String base = name(cache, "ds", 1_000_000, 768, 2);
        assertTrue(base, base.contains("_n1000000_d768_fmt" + OnDiskGraphIndex.CURRENT_VERSION + "_"));
        assertTrue(base, base.contains("_NVQ_VECTORS_nvq2_"));
        assertEquals("identical inputs share one graph", base, name(cache, "ds", 1_000_000, 768, 2));
        assertNotEquals("a larger profile is a different graph", base, name(cache, "ds", 100_000_000, 768, 2));
        assertNotEquals("a different dimension is a different graph", base, name(cache, "ds", 1_000_000, 384, 2));
        assertNotEquals("a different NVQ layout is a different file", base, name(cache, "ds", 1_000_000, 768, 1));
    }

    @Test
    public void nvqSubvectorsMustMatchTheFeatureSet() throws Exception {
        var cache = OnDiskGraphIndexCache.initialize(dir.getRoot().toPath());
        assertThrows(IllegalArgumentException.class, () -> name(cache, "ds", 10, 4, 0));
        assertThrows(IllegalArgumentException.class, () -> cache.keyWithCompressorId("ds",
                EnumSet.of(FeatureId.FUSED_ASH, FeatureId.INLINE_VECTORS), 32, 100, 1.2f, 1.2f, true, true,
                "id", 10, 4, 2));
        assertThrows(IllegalArgumentException.class, () -> name(cache, "ds", 0, 4, 2));
    }
}
