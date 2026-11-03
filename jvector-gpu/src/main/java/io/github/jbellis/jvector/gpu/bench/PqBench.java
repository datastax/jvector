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

import java.util.HashSet;
import java.util.List;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.gpu.GpuProductQuantization;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Product quantization, JVector on the CPU against {@link GpuProductQuantization}: train and encode times, the
 * recall@10 of an exhaustive scan by PQ distance, and how many codes the GPU reproduces when it encodes with
 * JVector's own codebooks.
 * <p>
 * Usage: {@code PqBench <dataset> [M=dim/8] [K=256]}, the dataset by name as JVector's {@code Bench} takes it
 * (downloaded on first use).
 */
public final class PqBench {
    private PqBench() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: PqBench <dataset> [M=dim/8] [K=256]");
            System.exit(1);
        }
        DataSet ds = DataSets.loadDataSet(args[0]).orElseThrow(() -> new IllegalArgumentException("unknown dataset " + args[0])).getDataSet();
        int dim = ds.getDimension();
        int M = args.length > 1 ? Integer.parseInt(args[1]) : dim / 8;
        int K = args.length > 2 ? Integer.parseInt(args[2]) : 256;
        RandomAccessVectorValues base = ds.getBaseRavv();
        List<VectorFloat<?>> queries = ds.getQueryVectors();
        List<? extends List<Integer>> gt = ds.getGroundTruth();
        VectorSimilarityFunction sim = ds.getSimilarityFunction();
        System.out.printf("%s: %,d x %d vectors, %s, PQ M=%d K=%d (unweighted)%n%n", ds.getName(), base.size(), dim, sim, M, K);

        // warm both paths on a slice, so the timed runs leave out JIT compilation
        int warmRows = Math.min(20_000, base.size());
        var warm = new ListRandomAccessVectorValues(IntStream.range(0, warmRows).mapToObj(base::getVector).toList(), dim);
        GpuProductQuantization.encodeAll(GpuProductQuantization.compute(warm, M, K, false), warm);
        ProductQuantization.compute(warm, M, K, false).encodeAll(warm);

        long t0 = System.nanoTime();
        var cpuPq = ProductQuantization.compute(base, M, K, false);
        long t1 = System.nanoTime();
        PQVectors cpuCodes = cpuPq.encodeAll(base, PhysicalCoreExecutor.pool());
        long t2 = System.nanoTime();
        var gpuPq = GpuProductQuantization.compute(base, M, K, false);
        long t3 = System.nanoTime();
        PQVectors gpuCodes = GpuProductQuantization.encodeAll(gpuPq, base);
        long t4 = System.nanoTime();
        PQVectors gpuCodesCpuBooks = GpuProductQuantization.encodeAll(cpuPq, base);
        long t5 = System.nanoTime();

        System.out.printf("%-34s %10s %10s %10s %10s%n", "", "train s", "encode s", "total s", "recall@10");
        System.out.printf("%-34s %10.2f %10.2f %10.2f %10.4f%n", "JVector (CPU)", (t1 - t0) / 1e9, (t2 - t1) / 1e9, (t2 - t0) / 1e9,
                recall(cpuCodes, queries, gt, sim));
        System.out.printf("%-34s %10.2f %10.2f %10.2f %10.4f%n", "GPU", (t3 - t2) / 1e9, (t4 - t3) / 1e9, (t4 - t2) / 1e9,
                recall(gpuCodes, queries, gt, sim));
        System.out.printf("%-34s %10s %10.2f %10s %10.4f%n", "GPU encode, JVector's codebooks", "-", (t5 - t4) / 1e9, "-",
                recall(gpuCodesCpuBooks, queries, gt, sim));

        long same = 0;
        long total = (long) base.size() * M;
        for (int i = 0; i < base.size(); i++) {
            var a = cpuCodes.get(i);
            var b = gpuCodesCpuBooks.get(i);
            for (int m = 0; m < M; m++) {
                if (a.get(m) == b.get(m)) {
                    same++;
                }
            }
        }
        System.out.printf("%nGPU encoding with JVector's codebooks: %.4f%% of %,d codes identical to JVector's%n", 100.0 * same / total, total);
    }

    /** Recall@10 of an exhaustive scan by PQ distance (JVector's asymmetric score), against the exact top 10. */
    static double recall(PQVectors codes, List<VectorFloat<?>> queries, List<? extends List<Integer>> gt, VectorSimilarityFunction sim) {
        return IntStream.range(0, queries.size()).parallel().mapToDouble(q -> {
            var score = codes.precomputedScoreFunctionFor(queries.get(q), sim);
            int[] ids = new int[10];
            float[] best = new float[10];
            java.util.Arrays.fill(best, -Float.MAX_VALUE);
            for (int i = 0; i < codes.count(); i++) {
                float s = score.similarityTo(i);
                if (s > best[9]) {
                    int p = 9;
                    while (p > 0 && best[p - 1] < s) {
                        best[p] = best[p - 1];
                        ids[p] = ids[p - 1];
                        p--;
                    }
                    best[p] = s;
                    ids[p] = i;
                }
            }
            var truth = new HashSet<>(gt.get(q).subList(0, 10));
            return IntStream.of(ids).filter(truth::contains).count() / 10.0;
        }).average().orElse(0);
    }
}
