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

import org.junit.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class BenchArgExpanderTest {

    @Test
    public void singleRangeWithLeadingZeros() {
        List<String> out = BenchArgExpander.expand("sift1m:label_[00..11]");
        assertEquals(12, out.size());
        assertEquals("sift1m:label_00", out.get(0));
        assertEquals("sift1m:label_01", out.get(1));
        assertEquals("sift1m:label_11", out.get(11));
    }

    @Test
    public void singleDigitRangeHasNoPadding() {
        List<String> out = BenchArgExpander.expand("foo[1..3]bar");
        assertEquals(List.of("foo1bar", "foo2bar", "foo3bar"), out);
    }

    @Test
    public void widthDerivesFromLongerEndpoint() {
        List<String> out = BenchArgExpander.expand("x[1..10]");
        assertEquals(10, out.size());
        assertEquals("x01", out.get(0));
        assertEquals("x10", out.get(9));
    }

    @Test
    public void reverseRangeCountsDown() {
        List<String> out = BenchArgExpander.expand("r[03..01]");
        assertEquals(List.of("r03", "r02", "r01"), out);
    }

    @Test
    public void tokenWithoutRangeIsPassedThrough() {
        assertEquals(List.of("plain-name"), BenchArgExpander.expand("plain-name"));
    }

    @Test
    public void multipleRangesExpandCombinatorially() {
        List<String> out = BenchArgExpander.expand("a[0..1]b[2..3]");
        assertEquals(List.of("a0b2", "a0b3", "a1b2", "a1b3"), out);
    }

    @Test
    public void expandAllSplitsWhitespaceAndFiltersEmpties() {
        String[] out = BenchArgExpander.expandAll(new String[]{"glove [00..02] nytimes", null, ""});
        assertArrayEquals(new String[]{"glove", "00", "01", "02", "nytimes"}, out);
    }

    @Test
    public void expandAllHandlesNullArgs() {
        assertArrayEquals(new String[0], BenchArgExpander.expandAll(null));
    }
}
