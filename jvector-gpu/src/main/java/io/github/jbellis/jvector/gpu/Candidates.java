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

package io.github.jbellis.jvector.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import uk.ac.manchester.tornado.cuvs.CuVS;
import uk.ac.manchester.tornado.cuvs.CuVSAllNeighborsAlgo;
import uk.ac.manchester.tornado.cuvs.CuVSAllNeighborsOptions;
import uk.ac.manchester.tornado.cuvs.CuVSDistance;

/**
 * Candidate neighbors of every row, from cuVS all-neighbors through TornadoVM's cuVS provider, with JVector
 * similarity scores.
 */
final class Candidates {

    /** {@code k} candidates per row ({@code -1} padded), nearest first, as JVector similarity scores. */
    record TopK(int k, int[] ids, float[] scores) {

        /** The first {@code width} candidates of every row. */
        TopK first(int width) {
            if (width >= k) {
                return this;
            }
            int n = ids.length / k;
            int[] firstIds = new int[n * width];
            float[] firstScores = new float[n * width];
            IntStream.range(0, n).parallel().forEach(i -> {
                System.arraycopy(ids, i * k, firstIds, i * width, width);
                System.arraycopy(scores, i * k, firstScores, i * width, width);
            });
            return new TopK(width, firstIds, firstScores);
        }
    }

    /** NN-Descent iterations ({@code -Djvector.gpu.nndIterations}); cuVS's default (20) gives the same graphs, slower. */
    static final long ITERATIONS = Long.getLong("jvector.gpu.nndIterations", 8);
    /** NN-Descent intermediate graph degree: cuVS's default (128) costs more and gives the same graphs. */
    static final long INTERMEDIATE_DEGREE = 64;

    private Candidates() {
    }

    /** How the rows are scored: the cuVS metric and the conversion of its distances to JVector scores. */
    enum Metric {
        /** unit vectors (COSINE, normalised) or DOT_PRODUCT: (1 + dot) / 2 */
        INNER_PRODUCT(CuVSDistance.INNER_PRODUCT),
        /** COSINE over vectors that may not be unit length: cuVS gives 1 - cos */
        COSINE(CuVSDistance.COSINE_EXPANDED),
        /** EUCLIDEAN: cuVS gives the squared distance */
        EUCLIDEAN(CuVSDistance.L2_EXPANDED);

        final CuVSDistance distance;

        Metric(CuVSDistance distance) {
            this.distance = distance;
        }

        static Metric of(VectorSimilarityFunction similarity, boolean unitVectors) {
            switch (similarity) {
                case EUCLIDEAN:
                    return EUCLIDEAN;
                case DOT_PRODUCT:
                    return INNER_PRODUCT;
                default:
                    return unitVectors ? INNER_PRODUCT : COSINE;
            }
        }

        /** The JVector score of a cuVS distance. */
        float score(float distance) {
            switch (this) {
                case EUCLIDEAN:
                    return 1.0f / (1.0f + Math.max(distance, 0.0f));
                case COSINE:
                    return (2.0f - distance) / 2.0f;
                default:
                    return (1.0f + distance) / 2.0f;
            }
        }
    }

    /**
     * How cuVS splits a dataset that does not fit in device memory: it partitions the rows into {@code clusters}
     * batches, puts every row into its {@code overlap} nearest batches, builds each batch's graph on the GPU and
     * merges them. More overlap gives better graphs across batch borders, at the cost of building each row
     * {@code overlap} times. One cluster means a single batch, without overlap.
     */
    record Batching(long clusters, long overlap) {

        /** cuVS batches ({@code -Djvector.gpu.clusters}); 0 chooses from the device memory. */
        static final long CLUSTERS = Long.getLong("jvector.gpu.clusters", 0);
        /** Batches every row joins ({@code -Djvector.gpu.overlap}). */
        static final long OVERLAP = Long.getLong("jvector.gpu.overlap", 2);

        /**
         * One batch if the dataset fits in 70% of the device memory, else the fewest batches that each fit. A batch
         * needs its rows in FP32 and FP16 plus the NN-Descent graphs, about {@code 6 x dim + 1024} bytes per row, and
         * holds about {@code overlap / clusters} of the rows.
         */
        static Batching of(long n, int dim) {
            return of(n, dim, deviceMemory(), CLUSTERS, OVERLAP);
        }

        /** As {@link #of(long, int)}, for a device of {@code deviceMemory} bytes and the given settings. */
        static Batching of(long n, int dim, long deviceMemory, long clusters, long overlapSetting) {
            long overlap = Math.max(1, overlapSetting);
            if (clusters > 0) {
                return new Batching(clusters, clusters > 1 ? overlap : 1);
            }
            long perRow = (long) dim * 6 + 1024;
            long available = (long) (deviceMemory * 0.7);
            if (n * perRow <= available) {
                return new Batching(1, 1);
            }
            // cuVS needs more clusters than the overlap
            return new Batching(Math.max(overlap + 1, (n * perRow * overlap + available - 1) / available), overlap);
        }

        private static long deviceMemory() {
            try {
                return uk.ac.manchester.tornado.api.TornadoExecutionPlan.getDevice(0, 0).getMaxGlobalMemory();
            } catch (RuntimeException e) {
                return 8L << 30;
            }
        }
    }

    /**
     * Approximate candidates with NN-Descent.
     *
     * @param rows     {@code n x dim} row-major float32 in host memory
     * @param batching how cuVS splits the dataset
     */
    static TopK nnDescent(MemorySegment rows, int n, int dim, int k, Metric metric, Batching batching) {
        var options = new CuVSAllNeighborsOptions()
                .withIntermediateGraphDegree(Math.max(INTERMEDIATE_DEGREE, k + 1))
                .withMaxIterations(ITERATIONS);
        if (batching.clusters() > 1) {
            options.withClusters(batching.clusters(), batching.overlap());
        }
        return allNeighbors(rows, n, dim, k, metric, CuVSAllNeighborsAlgo.NN_DESCENT, options);
    }

    /** Exact candidates (for the small upper layers). */
    static TopK exact(MemorySegment rows, int n, int dim, int k, Metric metric) {
        return allNeighbors(rows, n, dim, k, metric, CuVSAllNeighborsAlgo.BRUTE_FORCE, null);
    }

    private static TopK allNeighbors(MemorySegment rows, int n, int dim, int k, Metric metric, CuVSAllNeighborsAlgo algo, CuVSAllNeighborsOptions options) {
        int kk = k + 1; // cuVS includes every row itself
        int[] ids = new int[n * k];
        float[] scores = new float[n * k];
        try (Arena arena = Arena.ofShared()) {
            MemorySegment neighbors = arena.allocate((long) n * kk * Long.BYTES, 64);
            MemorySegment distances = arena.allocate((long) n * kk * Float.BYTES, 64);
            CuVS.allNeighborsOnHost(rows, n, dim, kk, algo, metric.distance, neighbors, distances, options);
            IntStream.range(0, n).parallel().forEach(i -> {
                int c = 0;
                for (int j = 0; j < kk && c < k; j++) {
                    long id = neighbors.getAtIndex(ValueLayout.JAVA_LONG, (long) i * kk + j);
                    if (id == i || id < 0) {
                        continue;
                    }
                    ids[i * k + c] = (int) id;
                    scores[i * k + c] = metric.score(distances.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * kk + j));
                    c++;
                }
                for (; c < k; c++) {
                    ids[i * k + c] = -1;
                    scores[i * k + c] = -1;
                }
            });
        }
        return new TopK(k, ids, scores);
    }
}
