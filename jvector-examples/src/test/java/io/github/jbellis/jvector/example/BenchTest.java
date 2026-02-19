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
package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.yaml.TestDataPartition;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class BenchTest {

    @Test
    public void parseArgsDefaultsToUniformSingleSplit() {
        var parsed = Bench.parseArgs(new String[] {"glove"});

        assertEquals("(?:" + "glove" + ")", parsed.datasetPattern.pattern());
        assertEquals(List.of(1), parsed.partitions.numSplits);
        assertEquals(List.of(TestDataPartition.Distribution.UNIFORM), parsed.partitions.splitDistribution);
    }

    @Test
    public void parseArgsSupportsMultipleSplitDistributions() {
        var parsed = Bench.parseArgs(new String[] {
                "--num-splits", "2,4",
                "--split-distribution", "uniform,fibonacci,log2n",
                "ada"
        });

        assertEquals(List.of(2, 4), parsed.partitions.numSplits);
        assertEquals(
                List.of(
                        TestDataPartition.Distribution.UNIFORM,
                        TestDataPartition.Distribution.FIBONACCI,
                        TestDataPartition.Distribution.LOG2N
                ),
                parsed.partitions.splitDistribution
        );
        assertTrue(parsed.datasetPattern.matcher("ada002-100k").find());
    }

    @Test
    public void parseArgsRejectsInvalidSplitDistribution() {
        var ex = assertThrows(IllegalArgumentException.class, () ->
                Bench.parseArgs(new String[] {"--split-distribution", "not-a-mode"}));
        assertTrue(ex.getMessage().contains("Invalid --split-distribution value"));
    }
}
