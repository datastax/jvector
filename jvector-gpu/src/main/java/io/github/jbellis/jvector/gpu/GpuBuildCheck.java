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

import java.io.DataInputStream;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Checks the GPU build against JVector's own, under the TornadoVM launcher. Both use M=32, efConstruction=100,
 * overflow 1.2, alpha 1.2 and a hierarchy; recall@10 is measured with the same search (topK 10, rerankK 50).
 * <ul>
 * <li>No arguments: clustered synthetic vectors for every similarity, at dimensions 128 and 100 (padded on the
 * GPU). Fails if the GPU graph's recall@10 is more than 0.01 below the CPU graph's.</li>
 * <li>{@code base.fvecs query.fvecs gt.ivecs COSINE|DOT_PRODUCT|EUCLIDEAN}: a real dataset, build times and recall
 * of both (skip the CPU build with {@code -Dcpu=false}).</li>
 * </ul>
 */
public final class GpuBuildCheck {

    private static final double TOLERANCE = 0.01;

    public static void main(String[] args) throws IOException {
        var accelerator = new TornadoGraphBuildAccelerator();
        if (args.length >= 4) {
            List<VectorFloat<?>> base = fvecs(Path.of(args[0]), Integer.MAX_VALUE);
            List<VectorFloat<?>> queries = fvecs(Path.of(args[1]), 1000);
            List<int[]> gt = ivecs(Path.of(args[2]), queries.size());
            var similarity = VectorSimilarityFunction.valueOf(args[3]);
            compare(accelerator, base, queries, gt, similarity, !"false".equals(System.getProperty("cpu")));
            return;
        }
        boolean ok = true;
        for (int dim : new int[] { 128, 100 }) {
            for (var similarity : VectorSimilarityFunction.values()) {
                var data = clustered(20_000 + 200, dim, similarity, new Random(dim));
                List<VectorFloat<?>> base = data.subList(0, 20_000);
                List<VectorFloat<?>> queries = data.subList(20_000, data.size());
                List<int[]> gt = exact(base, queries, similarity, 10);
                double[] recall = compare(accelerator, base, queries, gt, similarity, true);
                ok &= recall[0] >= recall[1] - TOLERANCE;
            }
        }
        System.out.println(ok ? "OK" : "FAILED: GPU recall below the CPU's");
        if (!ok) {
            System.exit(1);
        }
    }

    /** @return recall@10 of the GPU graph and (if built) of the CPU graph */
    static double[] compare(TornadoGraphBuildAccelerator accelerator, List<VectorFloat<?>> base, List<VectorFloat<?>> queries, List<int[]> gt,
                            VectorSimilarityFunction similarity, boolean cpu) {
        int dim = base.get(0).length();
        RandomAccessVectorValues vectors = new ListRandomAccessVectorValues(base, dim);
        var gpuBuilder = new GraphIndexBuilder(vectors, similarity, 32, 100, 1.2f, 1.2f, true);
        if (!accelerator.supports(gpuBuilder, vectors, similarity)) {
            throw new IllegalStateException("the GPU build is not available: run under the TornadoVM launcher with cuVS on the library path");
        }
        long t0 = System.nanoTime();
        accelerator.build(gpuBuilder, vectors, similarity);
        gpuBuilder.cleanup();
        double gpuSeconds = (System.nanoTime() - t0) / 1e9;
        double gpuRecall = recall(gpuBuilder.getGraph(), vectors, queries, gt, similarity);
        double cpuRecall = Double.NaN;
        double cpuSeconds = Double.NaN;
        if (cpu) {
            var cpuBuilder = new GraphIndexBuilder(vectors, similarity, 32, 100, 1.2f, 1.2f, true);
            long t1 = System.nanoTime();
            cpuBuilder.build(vectors);
            cpuSeconds = (System.nanoTime() - t1) / 1e9;
            cpuRecall = recall(cpuBuilder.getGraph(), vectors, queries, gt, similarity);
        }
        System.out.printf("%-12s %7d x %4d   GPU %7.2f s  recall@10 %.4f   CPU %7.2f s  recall@10 %.4f%n",
                similarity, base.size(), dim, gpuSeconds, gpuRecall, cpuSeconds, cpuRecall);
        return new double[] { gpuRecall, cpuRecall };
    }

    static double recall(ImmutableGraphIndex graph, RandomAccessVectorValues vectors, List<VectorFloat<?>> queries, List<int[]> gt,
                         VectorSimilarityFunction similarity) {
        double hits = 0;
        for (int q = 0; q < queries.size(); q++) {
            var result = GraphSearcher.search(queries.get(q), 10, 50, vectors, similarity, graph, Bits.ALL);
            var truth = new HashSet<Integer>();
            for (int k = 0; k < 10; k++) {
                truth.add(gt.get(q)[k]);
            }
            for (var ns : result.getNodes()) {
                if (truth.contains(ns.node)) {
                    hits++;
                }
            }
        }
        return hits / (10.0 * queries.size());
    }

    /** Gaussian clusters; unit vectors for COSINE and DOT_PRODUCT, as those are used. */
    static List<VectorFloat<?>> clustered(int n, int dim, VectorSimilarityFunction similarity, Random random) {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        int clusters = 64;
        float[][] centers = new float[clusters][dim];
        for (float[] c : centers) {
            for (int d = 0; d < dim; d++) {
                c[d] = (float) random.nextGaussian();
            }
        }
        List<VectorFloat<?>> data = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] c = centers[random.nextInt(clusters)];
            float[] v = new float[dim];
            double norm = 0;
            for (int d = 0; d < dim; d++) {
                v[d] = c[d] + 0.5f * (float) random.nextGaussian();
                norm += v[d] * v[d];
            }
            if (similarity != VectorSimilarityFunction.EUCLIDEAN) {
                for (int d = 0; d < dim; d++) {
                    v[d] /= (float) Math.sqrt(norm);
                }
            }
            data.add(vts.createFloatVector(v));
        }
        return data;
    }

    /** Exact top-k of every query by brute force. */
    static List<int[]> exact(List<VectorFloat<?>> base, List<VectorFloat<?>> queries, VectorSimilarityFunction similarity, int k) {
        return IntStream.range(0, queries.size()).parallel().mapToObj(q -> {
            Integer[] order = IntStream.range(0, base.size()).boxed().toArray(Integer[]::new);
            float[] scores = new float[base.size()];
            for (int i = 0; i < base.size(); i++) {
                scores[i] = similarity.compare(queries.get(q), base.get(i));
            }
            java.util.Arrays.sort(order, (a, b) -> Float.compare(scores[b], scores[a]));
            return IntStream.range(0, k).map(x -> order[x]).toArray();
        }).toList();
    }

    static List<VectorFloat<?>> fvecs(Path path, int limit) throws IOException {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        List<VectorFloat<?>> rows = new ArrayList<>();
        try (var in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path), 1 << 20))) {
            while (rows.size() < limit && in.available() > 0) {
                int dim = Integer.reverseBytes(in.readInt());
                byte[] bytes = in.readNBytes(dim * Float.BYTES);
                float[] v = new float[dim];
                ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(v);
                rows.add(vts.createFloatVector(v));
            }
        }
        return rows;
    }

    static List<int[]> ivecs(Path path, int limit) throws IOException {
        List<int[]> rows = new ArrayList<>();
        try (var in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path), 1 << 20))) {
            while (rows.size() < limit && in.available() > 0) {
                int k = Integer.reverseBytes(in.readInt());
                byte[] bytes = in.readNBytes(k * Integer.BYTES);
                int[] v = new int[k];
                ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(v);
                rows.add(v);
            }
        }
        return rows;
    }
}
