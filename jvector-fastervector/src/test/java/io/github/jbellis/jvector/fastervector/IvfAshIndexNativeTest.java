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

package io.github.jbellis.jvector.fastervector;

import io.github.jbellis.jvector.fastervector.utils.Array2DVecs;
import io.github.jbellis.jvector.fastervector.utils.FlatArrayVecs;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * End-to-end tests against the real fastervector native libraries. Skipped unless the
 * platform is Linux x86_64 with AVX2 and the staged libraries load (see
 * {@code scripts/stage-fastervector-native.sh}); with
 * {@code -Djvector.fastervector.test.requireNative=true} they fail instead.
 */
class IvfAshIndexNativeTest {
    private static final int N = 2000;
    private static final int DIM = 32;
    private static final int NQ = 20;
    private static final int NLIST = 16;
    private static final int TOPK = 10;
    // Probe every list, so recall reflects only the quantization error
    private static final int NPROBE = NLIST;
    // Several build chunks, the last one partial, so the streaming passes loop
    private static final long CHUNK_SIZE = 700;
    // Recall@10 against exact inner product over clustered data. ASH compresses
    // 32x by default, so this is a sanity floor, not a quality bar.
    private static final double MIN_RECALL = 0.5;

    private static float[][] data;
    private static float[][] queries;

    @TempDir
    Path dir;

    @BeforeAll
    static void loadNativeAndMakeData() {
        try {
            FastervectorLibrary.load();
        } catch (UnsupportedOperationException | UnsatisfiedLinkError e) {
            // On a machine that should run these, -Djvector.fastervector.test.requireNative=true
            // turns the silent skip into a failure
            if (Boolean.getBoolean("jvector.fastervector.test.requireNative")) {
                throw e;
            }
            assumeTrue(false, "fastervector native libraries unavailable: " + e.getMessage());
        }

        Random r = new Random(42);
        float[][] centers = new float[NLIST][];
        for (int c = 0; c < NLIST; c++) {
            centers[c] = gaussian(r, 1.0f);
        }
        data = new float[N][];
        for (int i = 0; i < N; i++) {
            data[i] = plus(centers[r.nextInt(NLIST)], gaussian(r, 0.3f));
        }
        queries = new float[NQ][];
        for (int q = 0; q < NQ; q++) {
            queries[q] = plus(centers[r.nextInt(NLIST)], gaussian(r, 0.3f));
        }
    }

    @BeforeEach
    void smallChunks() {
        BuildChunks.overrideChunkSize = CHUNK_SIZE;
    }

    @AfterEach
    void resetChunks() {
        BuildChunks.overrideChunkSize = 0;
    }

    @Test
    void buildAndSearchRecall() throws IOException {
        try (FvecThreadPool pool = FvecThreadPool.create(2);
             IvfAshIndex index = build(new Array2DVecs(data), pool)) {
            SearchResult result = index.search(queryVecs(), TOPK, NPROBE, pool);
            assertEquals(NQ, result.results().size());
            for (List<NodeScore> row : result.results()) {
                assertEquals(TOPK, row.size());
                for (int k = 1; k < row.size(); k++) {
                    assertTrue(row.get(k - 1).score() >= row.get(k).score(), "results must be best-first");
                }
            }
            assertRecall(result);
        }
    }

    @Test
    void reopenGivesIdenticalResults() throws IOException {
        SearchResult before;
        try (FvecThreadPool pool = FvecThreadPool.create(2)) {
            try (IvfAshIndex index = build(new Array2DVecs(data), pool)) {
                before = index.search(queryVecs(), TOPK, NPROBE, pool);
            }
            try (IvfAshIndex reopened = IvfAshIndex.prepareOpen()
                    .manifestPath(dir.resolve("dev0.ash.blk.manifest").toString())
                    .devPaths(List.of(dir.resolve("dev0.ash.blk").toString()))
                    .dim(DIM)
                    .buildThreadPool(pool)
                    .open()) {
                SearchResult after = reopened.search(queryVecs(), TOPK, NPROBE, pool);
                assertEquals(before, after);
            }
        }
    }

    @Test
    void buildFromRandomAccessVectorValues() throws IOException {
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        List<VectorFloat<?>> vectors = new ArrayList<>(N);
        for (float[] v : data) {
            vectors.add(vts.createFloatVector(v));
        }
        var ravv = new ListRandomAccessVectorValues(vectors, DIM);
        try (FvecThreadPool pool = FvecThreadPool.create(2);
             IvfAshIndex index = build(new RandomAccessVectorValuesVecs(ravv), pool)) {
            // Labels are jvector node ordinals, so recall against the same data checks the mapping
            assertRecall(index.search(queryVecs(), TOPK, NPROBE, pool));
        }
    }

    @Test
    void queryDimensionMismatchRejectedInJava() throws IOException {
        try (FvecThreadPool pool = FvecThreadPool.create(2);
             IvfAshIndex index = build(new Array2DVecs(data), pool);
             IvfAshIndexSearcher searcher = index.searcherBuilder().topk(TOPK).nprobe(NPROBE).build()) {
            Vecs wrongDim = new FlatArrayVecs(new float[DIM + 1], DIM + 1);
            assertThrows(IllegalArgumentException.class, () -> searcher.search(wrongDim));
        }
    }

    @Test
    void useAfterCloseThrows() throws IOException {
        try (FvecThreadPool pool = FvecThreadPool.create(2)) {
            IvfAshIndex index = build(new Array2DVecs(data), pool);
            IvfAshIndexSearcher searcher = index.searcherBuilder().topk(TOPK).nprobe(NPROBE).build();
            IvfAshIndexSearcher closedEarly = index.searcherBuilder().topk(TOPK).nprobe(NPROBE).build();
            closedEarly.close();
            assertThrows(IllegalStateException.class, () -> closedEarly.search(queryVecs()));

            index.close();
            // Closing the index closes its outstanding searchers
            assertThrows(IllegalStateException.class, () -> searcher.search(queryVecs()));
            assertThrows(IllegalStateException.class, () -> index.search(queryVecs(), TOPK, NPROBE, pool));
            assertThrows(IllegalStateException.class,
                         () -> index.searcherBuilder().topk(TOPK).nprobe(NPROBE).build());
            // close() is idempotent
            index.close();
            searcher.close();
        }
    }

    private IvfAshIndex build(Vecs vecs, FvecThreadPool pool) throws IOException {
        return IvfAshIndex.builder()
                .nlist(NLIST)
                .vectors(vecs)
                .ssdPaths(List.of(dir))
                .buildThreadPool(pool)
                .build();
    }

    private static Vecs queryVecs() {
        return new Array2DVecs(queries);
    }

    private static void assertRecall(SearchResult result) {
        int hits = 0;
        for (int q = 0; q < NQ; q++) {
            Set<Long> exact = exactTopK(queries[q]);
            for (NodeScore ns : result.results().get(q)) {
                assertTrue(ns.label() >= 0 && ns.label() < N, "label out of range: " + ns.label());
                if (exact.contains(ns.label())) {
                    hits++;
                }
            }
        }
        double recall = hits / (double) (NQ * TOPK);
        assertTrue(recall >= MIN_RECALL, "recall@" + TOPK + " = " + recall + ", expected >= " + MIN_RECALL);
    }

    private static Set<Long> exactTopK(float[] q) {
        Set<Long> top = new HashSet<>();
        IntStream.range(0, N).boxed()
                .sorted(Comparator.comparingDouble((Integer i) -> dot(q, data[i])).reversed())
                .limit(TOPK)
                .forEach(i -> top.add((long) i));
        return top;
    }

    private static float dot(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static float[] gaussian(Random r, float sigma) {
        float[] v = new float[DIM];
        for (int i = 0; i < DIM; i++) {
            v[i] = (float) r.nextGaussian() * sigma;
        }
        return v;
    }

    private static float[] plus(float[] a, float[] b) {
        float[] s = new float[DIM];
        for (int i = 0; i < DIM; i++) {
            s[i] = a[i] + b[i];
        }
        return s;
    }
}
