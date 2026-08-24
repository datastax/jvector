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

package io.github.jbellis.jvector.example.tutorial;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriterTypes;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineByteVectors;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Demonstrates end-to-end INT8 (signed byte) vector support in JVector:
 *
 * <p><b>How to run:</b>
 * <pre>
 *   # 1. Download the siftsmall dataset from http://corpus-texmex.irisa.fr/
 *   #    and unzip it so that siftsmall/siftsmall_base.fvecs is present
 *   #    in the directory you run the command from.
 *
 *   # 2. Build and run — jdk22 is the default profile (active when no -P flag is given):
 *   ./mvnw compile -pl jvector-examples -am
 *   ./mvnw exec:exec@tutorial -pl jvector-examples -Dtutorial=int8
 *
 *   # Use -Pjdk20 or -Pjdk11 if you are on an older JDK:
 *   ./mvnw exec:exec@tutorial -pl jvector-examples -Pjdk20 -Dtutorial=int8
 * </pre>
 *
 * <ol>
 *   <li>Read the siftsmall dataset from .fvecs files (float32).</li>
 *   <li>Convert each float32 vector to a signed int8 {@link ByteSequence}.</li>
 *   <li>Build a graph index using only byte×byte distance calculations.</li>
 *   <li>Save the graph to disk with {@link InlineByteVectors} — 1 byte per component on disk.</li>
 *   <li>Load the index back from disk.</li>
 *   <li>Generate random int8 query vectors and search, scoring directly from disk byte vectors.</li>
 * </ol>
 *
 * <p>The siftsmall dataset must be present at {@code siftsmall/siftsmall_base.fvecs}
 * relative to the working directory. Download it from
 * <a href="http://corpus-texmex.irisa.fr/">http://corpus-texmex.irisa.fr/</a>.
 */
public class Int8Example {

    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static void main(String[] args) throws IOException {
        String siftDir = args.length > 0 ? args[0] : "siftsmall";

        // ── Step 1: Read siftsmall base vectors (.fvecs) ─────────────────────────────
        System.out.println("Loading siftsmall base vectors...");
        List<VectorFloat<?>> floatVectors = SiftLoader.readFvecs(siftDir + "/siftsmall_base.fvecs");
        int dimension = floatVectors.get(0).length();
        System.out.printf("Loaded %d vectors of dimension %d%n", floatVectors.size(), dimension);

        // ── Step 2: Convert float32 vectors to signed int8 (ByteSequence) ────────────
        // SIFT base vectors store gradient histogram bins as unsigned bytes in [0, 255].
        // We subtract 128 to shift them into the signed range [-128, 127] that
        // ByteVectorSimilarityFunction and the underlying SIMD routines expect.
        // For other float32 datasets you would typically scale by a dataset-specific
        // factor and then clamp before casting.
        System.out.println("Converting float32 vectors to int8...");
        List<ByteSequence<?>> byteVectors = new ArrayList<>(floatVectors.size());
        for (VectorFloat<?> fv : floatVectors) {
            ByteSequence<?> bv = vts.createByteSequence(dimension);
            for (int i = 0; i < dimension; i++) {
                // SIFT component in [0,255] → shift to signed [-128, 127]
                bv.set(i, (byte) ((int) fv.get(i) - 128));
            }
            byteVectors.add(bv);
        }

        // Wrap the list in a RandomAccessByteVectorValues (RABVV) —
        // the byte-vector analogue of RandomAccessVectorValues.
        RandomAccessByteVectorValues rabvv = new ListRandomAccessByteVectorValues(byteVectors, dimension);

        // ── Step 3: Build the graph index using int8 vectors ─────────────────────────
        // The GraphIndexBuilder convenience constructor for byte vectors automatically
        // wires up byte×byte scoring via ByteVectorSimilarityFunction —
        // no float32 round-trip occurs during construction.
        int M = 32;
        int efConstruction = 100;
        float neighborOverflow = 1.2f;
        float alpha = 1.2f;
        boolean addHierarchy = true;

        System.out.println("Building graph index from int8 vectors...");
        ImmutableGraphIndex heapGraph;
        try (GraphIndexBuilder builder = new GraphIndexBuilder(
                rabvv,
                ByteVectorSimilarityFunction.EUCLIDEAN,
                M,
                efConstruction,
                neighborOverflow,
                alpha,
                addHierarchy))
        {
            heapGraph = builder.build(rabvv);
        }
        System.out.printf("Graph built with %d nodes%n", heapGraph.size(0));

        // ── Step 4: Save the graph to disk with native int8 storage ──────────────────
        // InlineByteVectors stores each vector as `dimension` raw bytes on disk —
        // 4× more compact than the float32 InlineVectors alternative.
        Path graphPath = Files.createTempFile("jvector-int8-example", null);
        System.out.printf("Writing graph to disk (%d bytes/vector): %s%n", dimension, graphPath);
        try (GraphIndexWriter writer = GraphIndexWriter
                .getBuilderFor(GraphIndexWriterTypes.RANDOM_ACCESS_PARALLEL, heapGraph, graphPath)
                .with(new InlineByteVectors(dimension))
                .build())
        {
            writer.write(Map.of(
                FeatureId.INLINE_BYTE_VECTORS,
                nodeId -> new InlineByteVectors.State(rabvv.getVector(nodeId))
            ));
        }

        // ── Step 5: Load the index from disk ─────────────────────────────────────────
        System.out.println("Loading graph from disk...");
        ReaderSupplier readerSupplier = ReaderSupplierFactory.open(graphPath);
        OnDiskGraphIndex diskGraph = OnDiskGraphIndex.load(readerSupplier);

        // ── Step 6: Search with random int8 query vectors ────────────────────────────
        // The score function reads each candidate's byte vector directly from disk —
        // true int8 end-to-end, no float conversion anywhere in the search path.
        int numQueries = 10;
        int topK = 5;
        Random rng = new Random(42);

        System.out.printf("%nSearching with %d random int8 query vectors (top-%d):%n", numQueries, topK);

        try (GraphSearcher searcher = new GraphSearcher(diskGraph)) {
            OnDiskGraphIndex.View view = (OnDiskGraphIndex.View) searcher.getView();

            for (int q = 0; q < numQueries; q++) {
                ByteSequence<?> queryBytes = randomInt8Vector(dimension, rng);

                // byteVectorRerankerFor reads each candidate's int8 vector from disk
                // and scores it byte×byte — no float32 anywhere in the hot path.
                SearchScoreProvider ssp = new DefaultSearchScoreProvider(
                        view.byteVectorRerankerFor(queryBytes, ByteVectorSimilarityFunction.EUCLIDEAN));

                SearchResult result = searcher.search(ssp, topK, Bits.ALL);

                System.out.printf("Query %2d → top neighbors: ", q);
                for (SearchResult.NodeScore ns : result.getNodes()) {
                    System.out.printf("(id=%d, score=%.4f) ", ns.node, ns.score);
                }
                System.out.println();
            }
        }

        // cleanup
        readerSupplier.close();
        Files.deleteIfExists(graphPath);
    }

    /**
     * Returns a random signed-byte vector as a {@link ByteSequence}.
     * Each component is independently and uniformly drawn from [-128, 127].
     */
    private static ByteSequence<?> randomInt8Vector(int dimension, Random rng) {
        ByteSequence<?> v = vts.createByteSequence(dimension);
        for (int i = 0; i < dimension; i++) {
            // nextInt(256) gives [0, 255]; subtract 128 → [-128, 127]
            v.set(i, (byte) (rng.nextInt(256) - 128));
        }
        return v;
    }
}
