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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.RandomAccessOnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.gpu.GpuProductQuantization;
import io.github.jbellis.jvector.gpu.TornadoGraphBuildAccelerator;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;

/**
 * Where the time of JVector Bench's "Build and write" goes besides the graph: the same three on-disk formats
 * ([NVQ], [FUSED_PQ, NVQ], [INLINE]) written from a GPU-built graph, one phase at a time, to size what else is worth
 * moving to the GPU. Usage: WritePhases &lt;dataset&gt; [workDir]
 */
public final class WritePhases {
    private WritePhases() {}

    public static void main(String[] args) throws Exception {
        DataSet ds = DataSets.loadDataSet(args[0]).orElseThrow(() -> new IllegalArgumentException("unknown dataset " + args[0])).getDataSet();
        Path work = Files.createDirectories(Path.of(args.length > 1 ? args[1] : "target/write-phases"));
        RandomAccessVectorValues v = ds.getBaseRavv();
        int n = v.size();
        int dim = v.dimension();
        var sim = ds.getSimilarityFunction();
        System.out.printf("%s: %,d x %d%n", ds.getName(), n, dim);

        long t = System.nanoTime();
        var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(v, sim), dim, 32, 100, 1.2f, 1.2f, true, true);
        new TornadoGraphBuildAccelerator().build(builder, v, sim);
        builder.cleanup();
        ImmutableGraphIndex graph = builder.getGraph();
        report("GPU graph build", t);

        t = System.nanoTime();
        ProductQuantization pq = GpuProductQuantization.compute(v, dim / 8, 256, false);
        PQVectors pqv = GpuProductQuantization.encodeAll(pq, v);
        report("PQ train + encode (GPU)", t);

        t = System.nanoTime();
        NVQuantization nvq = NVQuantization.compute(v, dim == 2 ? 1 : 2);
        report("NVQ compute", t);
        t = System.nanoTime();
        var supplier = v.threadLocalSupplier();
        NVQuantization.QuantizedVector[] nvqCodes = PhysicalCoreExecutor.pool().submit(() -> IntStream.range(0, n).parallel()
                .mapToObj(i -> nvq.encode(supplier.get().getVector(i))).toArray(NVQuantization.QuantizedVector[]::new)).join();
        report("NVQ encode, all vectors (Bench does it twice)", t);

        writeAll(work.resolve("nvq"), graph, v, Map.of(FeatureId.NVQ_VECTORS, i -> new NVQ.State(nvqCodes[i])), new NVQ(nvq), null, null);
        writeAll(work.resolve("fused"), graph, v, Map.of(FeatureId.NVQ_VECTORS, i -> new NVQ.State(nvqCodes[i])), new NVQ(nvq),
                new FusedPQ(graph.maxDegree(), pq), pqv);
        writeAll(work.resolve("inline"), graph, v, Map.of(FeatureId.INLINE_VECTORS, i -> new InlineVectors.State(v.getVector(i))),
                new InlineVectors(dim), null, null);
    }

    private static void writeAll(Path path, ImmutableGraphIndex graph, RandomAccessVectorValues v, Map<FeatureId, IntFunction<Feature.State>> inline,
                                 Feature feature, FusedPQ fused, PQVectors pqv) throws Exception {
        var b = new RandomAccessOnDiskGraphIndexWriter.Builder(graph, path).withMapper(new OrdinalMapper.IdentityMapper(v.size() - 1)).with(feature);
        if (fused != null) {
            b.with(fused);
        }
        try (var writer = b.build()) {
            long t = System.nanoTime();
            PhysicalCoreExecutor.pool().submit(() -> IntStream.range(0, v.size()).parallel().forEach(i -> {
                var state = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
                inline.forEach((id, s) -> state.put(id, s.apply(i)));
                try {
                    writer.writeInline(i, state);
                } catch (java.io.IOException e) {
                    throw new java.io.UncheckedIOException(e);
                }
            })).join();
            report(path.getFileName() + ": inline writes", t);
            t = System.nanoTime();
            Map<FeatureId, IntFunction<Feature.State>> at = Map.of();
            if (fused != null) {
                var view = graph.getView();
                at = Map.of(FeatureId.FUSED_PQ, i -> new FusedPQ.State(view, pqv, i));
            }
            writer.write(at);
            report(path.getFileName() + ": final write (edges" + (fused != null ? " + fused PQ" : "") + ")", t);
        }
        System.out.printf("   %s: %.2f GB%n", path.getFileName(), Files.size(path) / 1e9);
    }

    private static void report(String what, long start) {
        System.out.printf("%-50s %8.2f s%n", what, (System.nanoTime() - start) / 1e9);
    }
}
