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

package io.github.jbellis.jvector.fastervector.bench;

import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.fastervector.NodeScore;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** The parts of the IVF grid that run without the native libraries. */
class IvfGridTest {
    @TempDir
    Path dir;

    @Test
    void onlyEuclideanIsUnsupported() {
        assertTrue(IvfGrid.unsupportedReason(VectorSimilarityFunction.EUCLIDEAN).isPresent());
        assertTrue(IvfGrid.unsupportedReason(VectorSimilarityFunction.COSINE).isEmpty());
        assertTrue(IvfGrid.unsupportedReason(VectorSimilarityFunction.DOT_PRODUCT).isEmpty());
    }

    @Test
    void euclideanDatasetIsSkippedWithoutBuilding() throws IOException {
        VectorFloat<?> v = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(new float[]{1, 2});
        var ds = new SimpleDataSet("euclid", VectorSimilarityFunction.EUCLIDEAN, List.of(v), List.of(v), List.of(List.of(0)));
        boolean ran = IvfGrid.run(ds, IvfConfig.getDefaultConfig("euclid"), List.of(IvfAccuracyBenchmark.createDefault()),
                                  (system, point, metrics) -> fail("no rows expected"));
        assertFalse(ran);
    }

    @Test
    void ivfResultsWrapAsGraphSearchResults() {
        SearchResult sr = IvfAccuracyBenchmark.toSearchResult(List.of(new NodeScore(7, 0.9f), new NodeScore(3, 0.5f)));
        assertEquals(2, sr.getNodes().length);
        assertEquals(7, sr.getNodes()[0].node);
        assertEquals(0.9f, sr.getNodes()[0].score);
        assertEquals(3, sr.getNodes()[1].node);
        assertEquals(0, sr.getVisitedCount());
        assertEquals(0, IvfAccuracyBenchmark.toSearchResult(List.of()).getNodes().length);
    }

    @Test
    void labelsBeyondIntRangeAreRejected() {
        assertThrows(ArithmeticException.class,
                     () -> IvfAccuracyBenchmark.toSearchResult(List.of(new NodeScore(1L << 32, 1f))));
    }

    @Test
    void wrappedResultsScoreWithAccuracyMetrics() {
        var ivf = new io.github.jbellis.jvector.fastervector.SearchResult(List.of(
                List.of(new NodeScore(0, 1f), new NodeScore(1, 0.5f)),
                List.of(new NodeScore(5, 1f), new NodeScore(9, 0.5f))));
        var results = IvfAccuracyBenchmark.toSearchResults(ivf);
        // query 0 finds both true neighbors, query 1 finds one of two
        double recall = AccuracyMetrics.recallFromSearchResults(List.of(List.of(0, 1), List.of(5, 6)), results, 2, 2);
        assertEquals(0.75, recall, 1e-9);
    }

    @Test
    void indexLayoutMatchesTheBuilder() {
        List<Path> ssds = List.of(Path.of("/ssd0"), Path.of("/ssd1"));
        assertEquals(Path.of("/ssd0/ds/nlist4-sf1.0/dev0.ash.blk.manifest"), IvfSystem.manifestPath(ssds, "ds/nlist4-sf1.0"));
        assertEquals(List.of(Path.of("/ssd0/ds/nlist4-sf1.0/dev0.ash.blk"), Path.of("/ssd1/ds/nlist4-sf1.0/dev1.ash.blk")),
                     IvfSystem.devPaths(ssds, "ds/nlist4-sf1.0"));
    }

    @Test
    void indexBytesSumsEverySsd() throws IOException {
        List<Path> ssds = List.of(dir.resolve("a"), dir.resolve("b"));
        Files.createDirectories(ssds.get(0).resolve("idx/sub"));
        Files.createDirectories(ssds.get(1).resolve("idx"));
        Files.write(ssds.get(0).resolve("idx/f1"), new byte[10]);
        Files.write(ssds.get(0).resolve("idx/sub/f2"), new byte[5]);
        Files.write(ssds.get(1).resolve("idx/f3"), new byte[7]);
        Files.write(ssds.get(1).resolve("other"), new byte[100]);   // outside the index directory
        assertEquals(22, IvfSystem.indexBytes(ssds, "idx"));
        assertEquals(0, IvfSystem.indexBytes(List.of(dir.resolve("missing")), "idx"));
    }
}
