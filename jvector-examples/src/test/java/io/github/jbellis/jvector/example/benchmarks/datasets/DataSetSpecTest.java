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

package io.github.jbellis.jvector.example.benchmarks.datasets;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSetSpec.WrapperSpec;
import org.junit.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/// Tests for {@link DataSetSpec} parsing of the sugared and structured forms, including wrapper options.
public class DataSetSpecTest {

    private static List<String> names(DataSetSpec spec) {
        return spec.getWrappers().stream().map(WrapperSpec::getName).collect(java.util.stream.Collectors.toList());
    }

    @Test
    public void parsesNameOnly() {
        var spec = DataSetSpec.parse("cohere-english-v3-100k");
        assertEquals("cohere-english-v3-100k", spec.getName());
        assertEquals(DataSetSpec.DEFAULT_PROFILE, spec.getProfile());
        assertTrue(spec.isDefaultProfile());
        assertFalse(spec.hasWrappers());
        assertEquals("cohere-english-v3-100k", spec.toString());
    }

    @Test
    public void parsesProfileAndWrappers() {
        var spec = DataSetSpec.parse("cohere:fast(memory, mmap)");
        assertEquals("cohere", spec.getName());
        assertEquals("fast", spec.getProfile());
        assertFalse(spec.isDefaultProfile());
        assertEquals(List.of("memory", "mmap"), names(spec));
        assertTrue(spec.getWrappers().get(0).getOptions().isEmpty());
        assertEquals("cohere:fast(memory,mmap)", spec.toString());

        assertEquals(List.of("mmap"), names(DataSetSpec.parse("cohere(mmap)")));
        assertEquals("cohere(mmap)", DataSetSpec.parse("cohere(mmap)").toString());
    }

    @Test
    public void parsesWrapperOptionsAndRoundTrips() {
        var spec = DataSetSpec.parse("cohere(mmap, lru[grain=4096, capacityMb=512])");
        assertEquals(List.of("mmap", "lru"), names(spec));
        var lru = spec.getWrappers().get(1);
        assertEquals(Map.of("grain", "4096", "capacityMb", "512"), lru.getOptions());
        assertEquals(List.of("grain", "capacityMb"), List.copyOf(lru.getOptions().keySet()), "option order is preserved");
        assertEquals("cohere(mmap,lru[grain=4096,capacityMb=512])", spec.toString());
        assertEquals(spec, DataSetSpec.parse(spec.toString()));

        assertEquals(new WrapperSpec("lru", null), DataSetSpec.parse("x(lru[])").getWrappers().get(0));
        assertEquals("x(lru)", DataSetSpec.parse("x(lru[ ])").toString());
        assertEquals("x(a[k=])", DataSetSpec.parse("x(a[k=])").toString());
    }

    @Test
    public void defaultProfileIsCanonicalizedAway() {
        var explicit = DataSetSpec.parse("cohere:default(mmap)");
        var implicit = DataSetSpec.parse("cohere(mmap)");
        assertEquals(implicit, explicit);
        assertEquals(implicit.hashCode(), explicit.hashCode());
        assertEquals("cohere(mmap)", explicit.toString());
        assertEquals(DataSetSpec.DEFAULT_PROFILE, DataSetSpec.parse(" cohere : default ( ) ").getProfile());
        assertFalse(DataSetSpec.parse("cohere()").hasWrappers());
    }

    @Test
    public void rejectsMalformedStrings() {
        for (String bad : new String[] {"", "   ", "a(b", "a:b:c", "a)b", "(mmap)", ":fast", "a(b)c", "a((b))",
                "a(lru[grain)", "a(lru grain=1])", "a(lru[grain])", "a(lru[[x=1]])", "a(lru[x=1]])", "a[x=1]", "a(l r u)"}) {
            assertThrows(IllegalArgumentException.class, () -> DataSetSpec.parse(bad), "should reject '" + bad + "'");
        }
        assertThrows(IllegalArgumentException.class, () -> DataSetSpec.parse(null));
        assertThrows(IllegalArgumentException.class, () -> new DataSetSpec(" ", null, null));
        assertThrows(IllegalArgumentException.class, () -> new WrapperSpec(" ", null));
        assertThrows(IllegalArgumentException.class, () -> new WrapperSpec("lru", Map.of("grain", "1,2")));
        assertThrows(IllegalArgumentException.class, () -> new WrapperSpec("lru", Map.of("a=b", "1")));
        assertThrows(IllegalArgumentException.class, () -> new WrapperSpec("lru", Map.of(" ", "1")));
    }

    @Test
    public void fromAcceptsStringsAndMaps() {
        assertEquals(DataSetSpec.parse("cohere(mmap)"), DataSetSpec.from("cohere(mmap)"));

        var structured = DataSetSpec.from(Map.of("name", "cohere", "profile", "default", "wrappers", List.of("mmap")));
        assertEquals(DataSetSpec.parse("cohere(mmap)"), structured);

        var minimal = DataSetSpec.from(Map.of("name", "cohere"));
        assertEquals(DataSetSpec.parse("cohere"), minimal);

        var stringWrappers = DataSetSpec.from(Map.of("name", "cohere", "profile", "fast", "wrappers", "memory,mmap"));
        assertEquals(DataSetSpec.parse("cohere:fast(memory,mmap)"), stringWrappers);

        var same = DataSetSpec.parse("x");
        assertSame(same, DataSetSpec.from(same));
    }

    @Test
    public void fromAcceptsWrapperOptionsInEveryForm() {
        Map<String, Object> keyed = new LinkedHashMap<>();
        keyed.put("grain", 4096);
        keyed.put("capacityMb", 512);
        Map<String, Object> named = new LinkedHashMap<>();
        named.put("name", "lru");
        named.put("grain", 2048);
        Map<String, Object> bare = new LinkedHashMap<>();
        bare.put("mmap", null);

        var spec = DataSetSpec.from(Map.of("name", "cohere", "wrappers",
                List.of("memory", Map.of("lru", keyed), named, bare, "lru[grain=8]")));
        assertEquals("cohere(memory,lru[grain=4096,capacityMb=512],lru[grain=2048],mmap,lru[grain=8])", spec.toString());
        assertEquals(spec, DataSetSpec.parse(spec.toString()));
        assertEquals("4096", spec.getWrappers().get(1).getOptions().get("grain"));
    }

    @Test
    public void fromRejectsBadMaps() {
        assertThrows(IllegalArgumentException.class, () -> DataSetSpec.from(Map.of("profile", "fast")));
        assertThrows(IllegalArgumentException.class, () -> DataSetSpec.from(Map.of("name", "x", "loader", "y")));
        assertThrows(IllegalArgumentException.class, () -> DataSetSpec.from(Map.of("name", "x", "wrappers", 42)));
        assertThrows(IllegalArgumentException.class, () -> DataSetSpec.from(42));
        // a wrapper map with two keys and no name is ambiguous
        assertThrows(IllegalArgumentException.class, () -> DataSetSpec.from(Map.of("name", "x", "wrappers", List.of(Map.of("lru", Map.of(), "mmap", Map.of())))));
        // options must be a map
        assertThrows(IllegalArgumentException.class, () -> DataSetSpec.from(Map.of("name", "x", "wrappers", List.of(Map.of("lru", 4096)))));
        assertThrows(IllegalArgumentException.class, () -> DataSetSpec.from(Map.of("name", "x", "wrappers", List.of(42))));
    }
}
