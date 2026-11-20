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
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.gpu.GpuNvq;
import io.github.jbellis.jvector.quantization.NVQVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * NVQ encoding, JVector (CPU) vs GPU: time, the share of identical codes, and the recall@10 of an exhaustive scan
 * scored by NVQ (JVector's own score function) against the exact top 10. Usage: NvqBench &lt;dataset&gt;
 */
public final class NvqBench {
    private NvqBench() {}

    public static void main(String[] args) {
        DataSet ds = DataSets.loadDataSet(args[0]).orElseThrow(() -> new IllegalArgumentException("unknown dataset " + args[0])).getDataSet();
        RandomAccessVectorValues base = ds.getBaseRavv();
        int dim = ds.getDimension();
        var sim = ds.getSimilarityFunction();
        NVQuantization nvq = NVQuantization.compute(base, dim == 2 ? 1 : 2);
        System.out.printf("%s: %,d x %d, NVQ 8 bits, %d sub-vectors%n%n", ds.getName(), base.size(), dim, nvq.subvectorSizesAndOffsets.length);

        // warm both paths on a slice, so the timed runs leave out JIT compilation
        int warmRows = Math.min(20_000, base.size());
        var warm = new ListRandomAccessVectorValues(IntStream.range(0, warmRows).mapToObj(base::getVector).toList(), dim);
        nvq.encodeAll(warm, PhysicalCoreExecutor.pool());
        GpuNvq.encodeAll(nvq, warm);

        long t0 = System.nanoTime();
        NVQVectors cpu = nvq.encodeAll(base, PhysicalCoreExecutor.pool());
        long t1 = System.nanoTime();
        NVQVectors gpu = GpuNvq.encodeAll(nvq, base);
        long t2 = System.nanoTime();

        long same = 0;
        long growthSame = 0;
        long subVectors = 0;
        for (int i = 0; i < base.size(); i++) {
            var a = cpu.get(i).subVectors;
            var b = gpu.get(i).subVectors;
            for (int s = 0; s < a.length; s++) {
                subVectors++;
                if (a[s].growthRate == b[s].growthRate) {
                    growthSame++;
                }
                for (int d = 0; d < a[s].bytes.length(); d++) {
                    if (a[s].bytes.get(d) == b[s].bytes.get(d)) {
                        same++;
                    }
                }
            }
        }
        long total = (long) base.size() * dim;
        System.out.printf("%-16s %10s %10s%n", "", "encode s", "recall@10");
        System.out.printf("%-16s %10.2f %10.4f%n", "JVector (CPU)", (t1 - t0) / 1e9, recall(cpu, ds.getQueryVectors(), ds.getGroundTruth(), sim));
        System.out.printf("%-16s %10.2f %10.4f%n", "GPU", (t2 - t1) / 1e9, recall(gpu, ds.getQueryVectors(), ds.getGroundTruth(), sim));
        System.out.printf("%nidentical codes: %.4f%% of %,d; identical growth rates: %.4f%% of %,d sub-vectors%n",
                100.0 * same / total, total, 100.0 * growthSame / subVectors, subVectors);
    }

    /** Recall@10 of an exhaustive scan by NVQ score, against the exact top 10. */
    static double recall(NVQVectors codes, List<VectorFloat<?>> queries, List<? extends List<Integer>> gt, VectorSimilarityFunction sim) {
        return IntStream.range(0, queries.size()).parallel().mapToDouble(q -> {
            var score = codes.scoreFunctionFor(queries.get(q), sim);
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
