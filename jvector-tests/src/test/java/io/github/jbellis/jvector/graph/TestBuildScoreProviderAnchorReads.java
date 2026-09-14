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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.disk.FvecFileVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertTrue;

/// A full-precision build score function compares one fixed anchor against many nodes. For a
/// value-shared file source it should read the anchor once, not once per comparison.
public class TestBuildScoreProviderAnchorReads {
    private static final int ROWS = 2000, DIMENSIONS = 16, COMPARISONS = 1000;

    private static Path writeFvecs() throws Exception {
        var file = Files.createTempFile("anchor-reads", ".fvecs");
        var bytes = ByteBuffer.allocate(ROWS * (DIMENSIONS + 1) * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < ROWS; row++) {
            bytes.putInt(DIMENSIONS);
            for (int d = 0; d < DIMENSIONS; d++) bytes.putFloat(row + d / 32f);
        }
        Files.write(file, bytes.array());
        return file;
    }

    private static void assertAnchorReadOnce(String kind, long reads) {
        // One read per compared node, plus one for the anchor and a little slack.
        long limit = COMPARISONS + 8;
        assertTrue(kind + " made " + reads + " source reads for " + COMPARISONS
                + " comparisons against one anchor; expected at most " + limit, reads <= limit);
    }

    @Test public void diversityScoreFunctionReadsAnchorOnce() throws Exception {
        var file = writeFvecs();
        try (var source = FvecFileVectorValues.open(file)) {
            var bsp = BuildScoreProvider.randomAccessScoreProvider(source, VectorSimilarityFunction.EUCLIDEAN);
            var sf = bsp.diversityScoreFunctionFor(0);
            long before = source.statistics().reads;
            for (int node = 1; node <= COMPARISONS; node++) sf.similarityTo(node);
            assertAnchorReadOnce("diversityScoreFunctionFor", source.statistics().reads - before);
        } finally {
            Files.delete(file);
        }
    }

    @Test public void searchProviderReadsAnchorOnce() throws Exception {
        var file = writeFvecs();
        try (var source = FvecFileVectorValues.open(file)) {
            var bsp = BuildScoreProvider.randomAccessScoreProvider(source, VectorSimilarityFunction.EUCLIDEAN);
            var sf = bsp.searchProviderFor(0).scoreFunction();
            long before = source.statistics().reads;
            for (int node = 1; node <= COMPARISONS; node++) sf.similarityTo(node);
            assertAnchorReadOnce("searchProviderFor", source.statistics().reads - before);
        } finally {
            Files.delete(file);
        }
    }
}
