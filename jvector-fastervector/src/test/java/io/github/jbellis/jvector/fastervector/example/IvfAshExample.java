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

package io.github.jbellis.jvector.fastervector.example;

import io.github.jbellis.jvector.fastervector.FvecThreadPool;
import io.github.jbellis.jvector.fastervector.IvfAshIndex;
import io.github.jbellis.jvector.fastervector.IvfAshIndexSearcher;
import io.github.jbellis.jvector.fastervector.NodeScore;
import io.github.jbellis.jvector.fastervector.SearchResult;
import io.github.jbellis.jvector.fastervector.utils.Array2DVecs;
import io.github.jbellis.jvector.fastervector.utils.FlatArrayVecs;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

/**
 * Construct, populate, query, and reopen an IVF-ASH index.
 *
 * <p>Run with {@code ./mvnw -Pfastervector -pl jvector-fastervector -am test-compile exec:exec@example}.
 * It needs the staged native libraries (or {@code -Djvector.fastervector.native.dir}) and the
 * JVM flags described in the module README.
 */
public class IvfAshExample {
    public static void main(String[] args) throws IOException {
        int dim = 128, nBase = 50_000, nQuery = 10;
        Random r = new Random(42);

        // 1. Data. Any Vecs works: Array2DVecs wraps float[n][dim], and
        //    RandomAccessVectorValuesVecs wraps a jvector RandomAccessVectorValues.
        float[][] base = new float[nBase][dim];
        for (float[] v : base) for (int i = 0; i < dim; i++) v[i] = r.nextFloat();
        float[] queries = new float[nQuery * dim];
        for (int i = 0; i < queries.length; i++) queries[i] = r.nextFloat();

        Path ssd = Files.createTempDirectory("fvec-ivf");   // one directory per SSD in production
        String subfolder = "example";
        try {
            // 2. Construct and populate. build() trains on the first chunk (pass 0), assigns
            //    vectors to centroids (pass 1), and encodes them to the SSD paths (pass 2),
            //    streaming the vectors in chunks.
            try (FvecThreadPool buildPool = FvecThreadPool.create(8);
                 IvfAshIndex index = IvfAshIndex.builder()
                         .nlist(256)
                         .storageFactor(1.0f)
                         .vectors(new Array2DVecs(base))
                         .ssdPaths(List.of(ssd))
                         .subfolder(subfolder)
                         .buildThreadPool(buildPool)
                         .build();
                 FvecThreadPool searchPool = FvecThreadPool.create(4);
                 // 3. Query. A searcher has a fixed topk/nprobe and is reused across batches.
                 IvfAshIndexSearcher searcher = index.searcherBuilder()
                         .topk(10)
                         .nprobe(16)
                         .searchThreadPool(searchPool)
                         .build()) {

                SearchResult res = searcher.search(new FlatArrayVecs(queries, dim));
                for (int q = 0; q < nQuery; q++) {
                    for (NodeScore ns : res.results().get(q)) {
                        int ordinal = (int) ns.label();   // 0-based insertion ordinal = row in `base`
                        float score = ns.score();         // inner product, larger is better
                        System.out.printf("q%d -> %d (%.4f)%n", q, ordinal, score);
                    }
                }
            }

            // 4. Reopen the persisted index. With the current C API the caller spells out the
            //    on-disk layout: dev{i}.ash.blk per SSD, with the manifest beside dev0.
            Path dir = ssd.resolve(subfolder);
            try (IvfAshIndex reopened = IvfAshIndex.prepareOpen()
                         .manifestPath(dir.resolve("dev0.ash.blk.manifest").toString())
                         .devPaths(List.of(dir.resolve("dev0.ash.blk").toString()))
                         .dim(dim)
                         .open()) {
                // One-shot convenience: creates and closes a searcher internally.
                SearchResult again = reopened.search(new FlatArrayVecs(queries, dim), 10, 16, null);
                System.out.println("reopened, first hit: " + again.results().get(0).get(0));
            }
        } finally {
            deleteRecursively(ssd);
        }
    }

    private static void deleteRecursively(Path root) throws IOException {
        try (var walk = Files.walk(root)) {
            walk.toList().reversed().forEach(p -> {
                try {
                    Files.delete(p);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
        }
    }
}
