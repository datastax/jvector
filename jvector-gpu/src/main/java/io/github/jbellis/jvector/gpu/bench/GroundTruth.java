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
package io.github.jbellis.jvector.gpu.bench;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.stream.IntStream;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.LongArray;
import uk.ac.manchester.tornado.cuvs.CuVS;
import uk.ac.manchester.tornado.cuvs.CuVSDistance;

/**
 * Exact top-k neighbours by inner product of every query, on the GPU (cuVS brute force through TornadoVM), written as
 * an .ivecs ground-truth file. The base file is streamed in chunks that fit one TornadoVM array and the GPU, and each
 * query's best k are merged on the host, so a 10M x 1024 base takes minutes instead of the CPU's quarter of an hour.
 * <p>
 * Usage: GroundTruth base.f32 query.f32 dim out.ivecs [k=100]
 */
public final class GroundTruth {
    /** Floats per chunk: under a TornadoVM array's int index, and about 6 GB of device memory. */
    private static final long CHUNK_FLOATS = 1_600_000_000L;

    private GroundTruth() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("usage: GroundTruth base.f32 query.f32 dim out.ivecs [k=100]");
            System.exit(1);
        }
        int dim = Integer.parseInt(args[2]);
        int k = args.length > 4 ? Integer.parseInt(args[4]) : 100;
        long t0 = System.nanoTime();
        try (Arena arena = Arena.ofShared();
             FileChannel baseFile = FileChannel.open(Path.of(args[0]), StandardOpenOption.READ);
             FileChannel queryFile = FileChannel.open(Path.of(args[1]), StandardOpenOption.READ)) {
            MemorySegment base = baseFile.map(FileChannel.MapMode.READ_ONLY, 0, baseFile.size(), arena);
            MemorySegment queryRows = queryFile.map(FileChannel.MapMode.READ_ONLY, 0, queryFile.size(), arena);
            long n = baseFile.size() / (4L * dim);
            int nq = (int) (queryFile.size() / (4L * dim));
            int chunk = (int) Math.min(n, CHUNK_FLOATS / dim);
            FloatArray queries = new FloatArray(nq * dim);
            MemorySegment.copy(queryRows, 0, queries.getSegment(), 0, (long) nq * dim * 4);

            // the best k so far per query: scores, and the base row ids
            float[] best = new float[nq * k];
            long[] ids = new long[nq * k];
            java.util.Arrays.fill(best, Float.NEGATIVE_INFINITY);
            FloatArray rows = new FloatArray(chunk * dim);
            LongArray neighbors = new LongArray(nq * k);
            FloatArray distances = new FloatArray(nq * k);
            TornadoExecutionPlan plan = null;
            int planRows = -1;
            Boolean largerIsCloser = null;
            for (long start = 0; start < n; start += chunk) {
                int r = (int) Math.min(chunk, n - start);
                MemorySegment.copy(base, start * dim * 4, rows.getSegment(), 0, (long) r * dim * 4);
                if (r != planRows) {
                    if (plan != null) {
                        plan.close();
                    }
                    ImmutableTaskGraph graph = new TaskGraph("gt")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, rows, queries)
                            .libraryTask("knn", CuVS::bruteForceKnn, rows, r, dim, queries, nq, Math.min(k, r),
                                    CuVSDistance.INNER_PRODUCT.value(), neighbors, distances)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, neighbors, distances)
                            .snapshot();
                    plan = new TornadoExecutionPlan(graph);
                    planRows = r;
                }
                plan.execute();
                if (largerIsCloser == null) {
                    // cuVS lists each query's neighbours nearest first; whether that means descending inner products
                    // is read off the first query rather than assumed
                    largerIsCloser = distances.get(0) >= distances.get(Math.min(k, r) - 1);
                }
                merge(best, ids, neighbors, distances, nq, k, Math.min(k, r), start, largerIsCloser);
                System.out.printf("%,d / %,d rows, %.0f s%n", start + r, n, (System.nanoTime() - t0) / 1e9);
            }
            if (plan != null) {
                plan.close();
            }
            write(Path.of(args[3]), ids, nq, k);
            System.out.printf("wrote %s: %,d queries x %d in %.1f s%n", args[3], nq, k, (System.nanoTime() - t0) / 1e9);
        }
    }

    /** Folds one chunk's candidates into each query's best k (by inner product, larger first). */
    private static void merge(float[] best, long[] ids, LongArray neighbors, FloatArray distances, int nq, int k, int got,
                              long offset, boolean largerIsCloser) {
        IntStream.range(0, nq).parallel().forEach(q -> {
            int base = q * k;
            for (int j = 0; j < got; j++) {
                float raw = distances.get(q * got + j);
                float score = largerIsCloser ? raw : -raw;
                if (score <= best[base + k - 1]) {
                    break; // the rest of this query's list is no closer
                }
                int p = k - 1;
                while (p > 0 && best[base + p - 1] < score) {
                    best[base + p] = best[base + p - 1];
                    ids[base + p] = ids[base + p - 1];
                    p--;
                }
                best[base + p] = score;
                ids[base + p] = offset + neighbors.get(q * got + j);
            }
        });
    }

    /** Writes .ivecs: per query, a little-endian k, then its k row ids. */
    private static void write(Path path, long[] ids, int nq, int k) throws IOException {
        try (var out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path), 1 << 20))) {
            for (int q = 0; q < nq; q++) {
                out.writeInt(Integer.reverseBytes(k));
                for (int j = 0; j < k; j++) {
                    out.writeInt(Integer.reverseBytes((int) ids[q * k + j]));
                }
            }
        }
    }
}
