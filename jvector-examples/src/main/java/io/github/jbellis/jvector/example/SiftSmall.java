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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction.ApproximateScoreFunction;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction.ExactScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.MutablePQVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExceptionUtils;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.IntStream;

import static java.lang.Math.min;

// this class uses explicit typing instead of `var` for easier reading when excerpted for instructional use
public class SiftSmall {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    // hello world
    public static void siftInMemory(ArrayList<VectorFloat<?>> baseVectors) throws IOException {
        // infer the dimensionality from the first vector
        int originalDimension = baseVectors.get(0).length();
        // wrap the raw vectors in a RandomAccessVectorValues
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // score provider using the raw, in-memory vectors
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp,
                                                               ravv.dimension(),
                                                               16, // graph degree
                                                               100, // construction search depth
                                                               1.2f, // allow degree overflow during construction by this factor
                                                               1.2f, // relax neighbor diversity requirement by this factor
                                                               false))
        {
            // build the index (in memory)
            OnHeapGraphIndex index = builder.build(ravv);

            // search for a random vector
            VectorFloat<?> q = randomVector(originalDimension);
            SearchResult sr = GraphSearcher.search(q,
                                                   10, // number of results
                                                   ravv, // vectors we're searching, used for scoring
                                                   VectorSimilarityFunction.EUCLIDEAN, // how to score
                                                   index,
                                                   Bits.ALL); // valid ordinals to consider
            for (SearchResult.NodeScore ns : sr.getNodes()) {
                System.out.println(ns);
            }
        }
    }

    // show how to use explicit GraphSearcher objects
    public static void siftInMemoryWithSearcher(ArrayList<VectorFloat<?>> baseVectors) throws IOException {
        int originalDimension = baseVectors.get(0).length();
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f, false)) {
            OnHeapGraphIndex index = builder.build(ravv);

            // search for a random vector using a GraphSearcher and SearchScoreProvider
            VectorFloat<?> q = randomVector(originalDimension);
            try (GraphSearcher searcher = new GraphSearcher(index)) {
                SearchScoreProvider ssp = SearchScoreProvider.exact(q, VectorSimilarityFunction.EUCLIDEAN, ravv);
                SearchResult sr = searcher.search(ssp, 10, Bits.ALL);
                for (SearchResult.NodeScore ns : sr.getNodes()) {
                    System.out.println(ns);
                }
            }
        }
    }

    // call out to testRecall instead of doing manual searches
    public static void siftInMemoryWithRecall(List<VectorFloat<?>> baseVectors, List<VectorFloat<?>> queryVectors, List<Set<Integer>> groundTruth) throws IOException {
        int originalDimension = baseVectors.get(0).length();
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f, false)) {
            OnHeapGraphIndex index = builder.build(ravv);
            // measure our recall against the (exactly computed) ground truth
            Function<VectorFloat<?>, SearchScoreProvider> sspFactory = q -> SearchScoreProvider.exact(q, VectorSimilarityFunction.EUCLIDEAN, ravv);
            testRecall(index, queryVectors, groundTruth, sspFactory);
        }
    }

    // write and load index to and from disk
    public static void siftPersisted(List<VectorFloat<?>> baseVectors, List<VectorFloat<?>> queryVectors, List<Set<Integer>> groundTruth) throws IOException {
        int originalDimension = baseVectors.get(0).length();
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        Path indexPath = Files.createTempFile("siftsmall", ".inline");
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f, false)) {
            // build the index (in memory)
            OnHeapGraphIndex index = builder.build(ravv);
            // write the index to disk with default options
            OnDiskGraphIndex.write(index, ravv, indexPath);
        }

        // on-disk indexes require a ReaderSupplier (not just a Reader) because we will want it to
        // open additional readers for searching
        try (ReaderSupplier rs = ReaderSupplierFactory.open(indexPath)) {
            OnDiskGraphIndex index = OnDiskGraphIndex.load(rs);
            // measure our recall against the (exactly computed) ground truth
            Function<VectorFloat<?>, SearchScoreProvider> sspFactory = q -> SearchScoreProvider.exact(q, VectorSimilarityFunction.EUCLIDEAN, index.getView());
            testRecall(index, queryVectors, groundTruth, sspFactory);
        }
    }

    // diskann-style index with PQ
    public static void siftDiskAnn(List<VectorFloat<?>> baseVectors, List<VectorFloat<?>> queryVectors, List<Set<Integer>> groundTruth) throws IOException {
        int originalDimension = baseVectors.get(0).length();
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        Path indexPath = Files.createTempFile("siftsmall", ".inline");
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f, false)) {
            OnHeapGraphIndex index = builder.build(ravv);
            OnDiskGraphIndex.write(index, ravv, indexPath);
        }

        // compute and write compressed vectors to disk
        Path pqPath = Files.createTempFile("siftsmall", ".pq");
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(pqPath)))) {
            // Compress the original vectors using PQ. this represents a compression ratio of 128 * 4 / 16 = 32x
            ProductQuantization pq = ProductQuantization.compute(ravv,
                                                                 16, // number of subspaces
                                                                 256, // number of centroids per subspace
                                                                 true); // center the dataset
            var pqv = pq.encodeAll(ravv);
            // write the compressed vectors to disk
            pqv.write(out);
        }

        try (ReaderSupplier rs = ReaderSupplierFactory.open(indexPath)) {
            OnDiskGraphIndex index = OnDiskGraphIndex.load(rs);
            // load the PQVectors that we just wrote to disk
            try (ReaderSupplier pqSupplier = ReaderSupplierFactory.open(pqPath);
                 RandomAccessReader in = pqSupplier.get())
            {
                PQVectors pqv = PQVectors.load(in);
                // SearchScoreProvider that does a first pass with the loaded-in-memory PQVectors,
                // then reranks with the exact vectors that are stored on disk in the index
                Function<VectorFloat<?>, SearchScoreProvider> sspFactory = q -> {
                    ApproximateScoreFunction asf = pqv.precomputedScoreFunctionFor(q, VectorSimilarityFunction.EUCLIDEAN);
                    ExactScoreFunction reranker = index.getView().rerankerFor(q, VectorSimilarityFunction.EUCLIDEAN);
                    return new SearchScoreProvider(asf, reranker);
                };
                // measure our recall against the (exactly computed) ground truth
                testRecall(index, queryVectors, groundTruth, sspFactory);
            }
        }
    }

    public static void siftDiskAnnLTM(List<VectorFloat<?>> baseVectors, List<VectorFloat<?>> queryVectors, List<Set<Integer>> groundTruth) throws IOException {
        int originalDimension = baseVectors.get(0).length();
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // compute the codebook, but don't encode any vectors yet
        ProductQuantization pq = ProductQuantization.compute(ravv, 16, 256, true);

        // as we build the index we'll compress the new vectors and add them to this List backing a PQVectors;
        // this is used to score the construction searches
        var pqv = new MutablePQVectors(pq);
        BuildScoreProvider bsp = BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.EUCLIDEAN, pqv);

        Path indexPath = Files.createTempFile("siftsmall", ".inline");
        Path pqPath = Files.createTempFile("siftsmall", ".pq");
        // Builder creation looks mostly the same
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f, false);
             // explicit Writer for the first time, this is what's behind OnDiskGraphIndex.write
             OnDiskGraphIndexWriter writer = new OnDiskGraphIndexWriter.Builder(builder.getGraph(), indexPath)
                     .with(new InlineVectors(ravv.dimension()))
                     .withMapper(new OrdinalMapper.IdentityMapper(baseVectors.size() - 1))
                     .build();
             // output for the compressed vectors
             DataOutputStream pqOut = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(pqPath))))
        {
            // build the index vector-at-a-time (on disk)
            for (int ordinal = 0; ordinal < baseVectors.size(); ordinal++) {
                VectorFloat<?> v = baseVectors.get(ordinal);
                // compress the new vector and add it to the PQVectors
                pqv.encodeAndSet(ordinal, v);
                // write the full vector to disk
                writer.writeInline(ordinal, Feature.singleState(FeatureId.INLINE_VECTORS, new InlineVectors.State(v)));
                // now add it to the graph -- the previous steps must be completed first since the PQVectors
                // and InlineVectorValues are both used during the search that runs as part of addGraphNode construction
                builder.addGraphNode(ordinal, v);
            }

            // cleanup does a final enforcement of maxDegree and handles other scenarios like deleted nodes
            // that we don't need to worry about here
            builder.cleanup();

            // finish writing the index (by filling in the edge lists) and write our completed PQVectors
            writer.write(Map.of());
            pqv.write(pqOut);
        }

        // searching the index does not change
        ReaderSupplier rs = ReaderSupplierFactory.open(indexPath);
        OnDiskGraphIndex index = OnDiskGraphIndex.load(rs);
        try (ReaderSupplier pqSupplier = ReaderSupplierFactory.open(pqPath);
             RandomAccessReader in = pqSupplier.get())
        {
            var pqvSearch = PQVectors.load(in);
            Function<VectorFloat<?>, SearchScoreProvider> sspFactory = q -> {
                ApproximateScoreFunction asf = pqvSearch.precomputedScoreFunctionFor(q, VectorSimilarityFunction.EUCLIDEAN);
                ExactScoreFunction reranker = index.getView().rerankerFor(q, VectorSimilarityFunction.EUCLIDEAN);
                return new SearchScoreProvider(asf, reranker);
            };
            testRecall(index, queryVectors, groundTruth, sspFactory);
        }
    }

    public static void siftDiskAnnLTMWithNVQ(List<VectorFloat<?>> baseVectors, List<VectorFloat<?>> queryVectors, List<Set<Integer>> groundTruth) throws IOException {
        int originalDimension = baseVectors.get(0).length();
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // compute the codebook, but don't encode any vectors yet
        ProductQuantization pq = ProductQuantization.compute(ravv, 16, 256, true);

        var nvq = NVQuantization.compute(ravv, 2);

        // as we build the index we'll compress the new vectors and add them to this List backing a PQVectors;
        // this is used to score the construction searches
        var pqv = new MutablePQVectors(pq);
        BuildScoreProvider bsp = BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.EUCLIDEAN, pqv);

        Path indexPath = Files.createTempFile("siftsmall", ".inline");
        Path pqPath = Files.createTempFile("siftsmall", ".pq");
        // Builder creation looks mostly the same
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f, false);
             // explicit Writer for the first time, this is what's behind OnDiskGraphIndex.write
             OnDiskGraphIndexWriter writer = new OnDiskGraphIndexWriter.Builder(builder.getGraph(), indexPath)
                     .with(new NVQ(nvq))
                     .withMapper(new OrdinalMapper.IdentityMapper(baseVectors.size() - 1))
                     .build();
             // output for the compressed vectors
             DataOutputStream pqOut = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(pqPath))))
        {
            // build the index vector-at-a-time (on disk)
            for (int ordinal = 0; ordinal < baseVectors.size(); ordinal++) {
                VectorFloat<?> v = baseVectors.get(ordinal);
                // compress the new vector and add it to the PQVectors
                pqv.encodeAndSet(ordinal, v);
                // write the full vector to disk
                writer.writeInline(ordinal, Feature.singleState(FeatureId.NVQ_VECTORS, new NVQ.State(nvq.encode(v))));
                // now add it to the graph -- the previous steps must be completed first since the PQVectors
                // and InlineVectorValues are both used during the search that runs as part of addGraphNode construction
                builder.addGraphNode(ordinal, v);
            }

            // cleanup does a final enforcement of maxDegree and handles other scenarios like deleted nodes
            // that we don't need to worry about here
            builder.cleanup();

            // finish writing the index (by filling in the edge lists) and write our completed PQVectors
            writer.write(Map.of());
            pqv.write(pqOut);
        }

        // searching the index does not change
        ReaderSupplier rs = ReaderSupplierFactory.open(indexPath);
        OnDiskGraphIndex index = OnDiskGraphIndex.load(rs);
        try (ReaderSupplier pqSupplier = ReaderSupplierFactory.open(pqPath);
             RandomAccessReader in = pqSupplier.get())
        {
            var pqvSearch = PQVectors.load(in);
            Function<VectorFloat<?>, SearchScoreProvider> sspFactory = q -> {
                ApproximateScoreFunction asf = pqvSearch.precomputedScoreFunctionFor(q, VectorSimilarityFunction.EUCLIDEAN);
                ExactScoreFunction reranker = index.getView().rerankerFor(q, VectorSimilarityFunction.EUCLIDEAN);
                return new SearchScoreProvider(asf, reranker);
            };
            testRecall(index, queryVectors, groundTruth, sspFactory);
        }
    }

    //
    // Utilities and main() harness
    //

    public static VectorFloat<?> randomVector(int dim) {
        Random R = ThreadLocalRandom.current();
        VectorFloat<?> vec = vts.createFloatVector(dim);
        for (int i = 0; i < dim; i++) {
            vec.set(i, R.nextFloat());
            if (R.nextBoolean()) {
                vec.set(i, -vec.get(i));
            }
        }
        VectorUtil.l2normalize(vec);
        return vec;
    }

    private static void testRecall(GraphIndex graph,
                                   List<VectorFloat<?>> queryVectors,
                                   List<Set<Integer>> groundTruth,
                                   Function<VectorFloat<?>,
                                   SearchScoreProvider> sspFactory)
            throws IOException
    {
        AtomicInteger topKfound = new AtomicInteger(0);
        int topK = 100;
        String graphType = graph.getClass().getSimpleName();
        try (ExplicitThreadLocal<GraphSearcher> searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(graph))) {
            IntStream.range(0, queryVectors.size()).parallel().forEach(i -> {
                VectorFloat<?> queryVector = queryVectors.get(i);
                try (GraphSearcher searcher = searchers.get()) {
                    SearchScoreProvider ssp = sspFactory.apply(queryVector);
                    int rerankK = ssp.scoreFunction().isExact() ? topK : 2 * topK; // hardcoded overquery factor of 2x when reranking
                    SearchResult.NodeScore[] nn = searcher.search(ssp, rerankK, Bits.ALL).getNodes();

                    Set<Integer> gt = groundTruth.get(i);
                    long n = IntStream.range(0, min(topK, nn.length)).filter(j -> gt.contains(nn[j].node)).count();
                    topKfound.addAndGet((int) n);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
        } catch (Exception e) {
            ExceptionUtils.throwIoException(e);
        }
        System.out.printf("(%s) Recall: %.4f%n", graphType, (double) topKfound.get() / (queryVectors.size() * topK));
    }

    public static void main(String[] args) throws IOException {
        var siftPath = "siftsmall";
        var baseVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_base.fvecs", siftPath));
        var queryVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_query.fvecs", siftPath));
        var groundTruth = SiftLoader.readIvecs(String.format("%s/siftsmall_groundtruth.ivecs", siftPath));
        System.out.format("%d base and %d query vectors loaded, dimensions %d%n",
                          baseVectors.size(), queryVectors.size(), baseVectors.get(0).length());

        siftInMemory(baseVectors);
        siftInMemoryWithSearcher(baseVectors);
        siftInMemoryWithRecall(baseVectors, queryVectors, groundTruth);
        siftPersisted(baseVectors, queryVectors, groundTruth);
        siftDiskAnn(baseVectors, queryVectors, groundTruth);
        siftDiskAnnLTM(baseVectors, queryVectors, groundTruth);
        siftDiskAnnLTMWithNVQ(baseVectors, queryVectors, groundTruth);
    }
}
