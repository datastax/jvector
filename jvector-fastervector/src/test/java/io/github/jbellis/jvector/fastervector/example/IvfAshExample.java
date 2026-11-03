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

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import io.github.jbellis.jvector.fastervector.FvecThreadPool;
import io.github.jbellis.jvector.fastervector.IvfAshIndex;
import io.github.jbellis.jvector.fastervector.utils.Array2DVecs;
import io.github.jbellis.jvector.fastervector.utils.FlatArrayVecs;

/**
 * Minimal example: build an IVF-ASH index from in-memory vectors and run a search.
 *
 * <p>Run with {@code mvn compile exec:java@example}.
 */
public class IvfAshExample {
    public static void main(String[] args) throws IOException {
        var vecs = new Array2DVecs(new float[][]{
            {0.2f, 0.3f, 0.4f},
            {1.2f, 1.3f, 1.4f},
            {2.2f, 2.3f, 2.4f},
        });

        Path path = Files.createTempDirectory("fv-jv-client-test");
        try (
            var buildPool = FvecThreadPool.create(1);
            var index = IvfAshIndex.builder()
                .nlist(2)
                .vectors(vecs)
                .ssdPaths(List.of(path))
                .build();
            var searchPool = FvecThreadPool.create(1);
        ) {
            var queries = new FlatArrayVecs(new float[] {
                0.1f, 0.2f, 0.3f,
                1.1f, 1.2f, 1.3f,
            }, 3);
            var topk = 3;
            var nprobe = 32;
            var searcher = index.searcherBuilder()
                .topk(topk)
                .nprobe(nprobe)
                .searchThreadPool(searchPool)
                .build();
            var res = searcher.search(queries);

            // convenience function to avoid searcher creation
            // var res = index.search(queries, topk, nprobe, searchPool);

            for (var row : res.results()) {
                System.out.println(row);
            }
        } finally {
            try (var walk = Files.walk(path)) {
                walk
                    .toList()
                    .reversed()
                    .forEach(p -> {
                        try {
                            Files.delete(p);
                        } catch (IOException e) {
                            throw new UncheckedIOException(e);
                        }
                    });
            }
            Files.deleteIfExists(path);
        }
    }
}
