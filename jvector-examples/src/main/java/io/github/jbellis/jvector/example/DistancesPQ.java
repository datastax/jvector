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

import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.QuickerADCDecoder;
import io.github.jbellis.jvector.quantization.QuickerADCVectors;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.util.List;
import java.util.stream.IntStream;

import static java.lang.Math.abs;

// this class uses explicit typing instead of `var` for easier reading when excerpted for instructional use
public class DistancesPQ {
    public static void testNVQEncodings(String filenameBase, String filenameQueries, VectorSimilarityFunction vsf) throws IOException {
        List<VectorFloat<?>> vectors = SiftLoader.readFvecs(filenameBase);
        List<VectorFloat<?>> queries = SiftLoader.readFvecs(filenameQueries);

        int dimension = vectors.get(0).length();
        int nQueries = 100;
        int nVectors = vectors.size();

        vectors = vectors.subList(0, nVectors);

        System.out.format("\t%d base and %d query vectors loaded, dimensions %d%n",
                vectors.size(), queries.size(), vectors.get(0).length());

        int nSubspaces = 96;

        // Generate a NVQ for random vectors
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);
        var pq = ProductQuantization.compute(ravv, nSubspaces, 256, false);

        var pool = PhysicalCoreExecutor.pool();

        long startTime, endTime;
        double duration;

        // Compress the vectors
        startTime = System.nanoTime();
        QuickerADCVectors qadcVecs = QuickerADCVectors.encodeAndBuild(pq, ravv.size(), ravv, pool);
        endTime = System.nanoTime();
        duration = (double) (endTime - startTime) / 1_000_000_000;
        System.out.println("\tQuickerADC Encoding took " + duration + " seconds");

        startTime = System.nanoTime();
        PQVectors pqVecs = pq.encodeAll(ravv, pool);
        endTime = System.nanoTime();
        duration = (double) (endTime - startTime) / 1_000_000_000;
        System.out.println("\tPQ Encoding took " + duration + " seconds");

        // compare the encoded similarities to the raw
        float[] scoresQADC = new float[ravv.size()];
        double distanceErrorQADCvsFP = 0;
        double distanceErrorQADCvsPQ = 0;
        double distanceErrorPQvsFP = 0;
        for (int i = 0; i < nQueries; i++) {
            var q = queries.get(i);
            if (VectorUtil.dotProduct(q, q) == 0) {
                continue;
            }
            var f = QuickerADCDecoder.newDecoder(qadcVecs, q, vsf);
            f.warmup();
            f.similarities(scoresQADC, pool);

            var f2 = pqVecs.scoreFunctionFor(q, vsf);

            for (int j = 0; j < nVectors; j++) {
                var v = vectors.get(j);
                distanceErrorQADCvsFP += abs(scoresQADC[j] - vsf.compare(q, v));
                distanceErrorQADCvsPQ += abs(scoresQADC[j] - f2.similarityTo(j));
                distanceErrorPQvsFP += abs(f2.similarityTo(j) - vsf.compare(q, v));
            }


        }
        distanceErrorQADCvsFP /= nQueries * nVectors;
        distanceErrorQADCvsPQ /= nQueries * nVectors;
        distanceErrorPQvsFP /= nQueries * nVectors;

        System.out.println("\t" + vsf + " QADC error " + distanceErrorQADCvsFP + " wrt FP");
        System.out.println("\t" + vsf + " PQ error " + distanceErrorPQvsFP + " wrt FP");
        System.out.println("\t" + vsf + " QADC error " + distanceErrorQADCvsPQ + " wrt PQ");

        float dummyAccumulator = 0;

        startTime = System.nanoTime();
        duration = 0;
        for (int i = 0; i < nQueries; i++) {
            var q = queries.get(i);
            if (VectorUtil.dotProduct(q, q) == 0) {
                continue;
            }
            var f = QuickerADCDecoder.newDecoder(qadcVecs, q, vsf);
            f.warmup();

            startTime = System.nanoTime();
            f.similarities(scoresQADC, pool);
            endTime = System.nanoTime();
            duration += (double) (endTime - startTime) / 1_000_000_000;

            for (int j = 0; j < nVectors; j++) {
                dummyAccumulator += scoresQADC[j];
            }

        }
//        endTime = System.nanoTime();
//        duration = (double) (endTime - startTime) / 1_000_000_000;
        System.out.println("\tQuickerADC Distance computations took " + duration + " seconds");

//        startTime = System.nanoTime();
        duration  = 0;
        for (int i = 0; i < nQueries; i++) {
            var q = queries.get(i);
            if (VectorUtil.dotProduct(q, q) == 0) {
                continue;
            }

            var f = pqVecs.scoreFunctionFor(q, vsf);

            startTime = System.nanoTime();
//            pool.submit(() -> IntStream.range(0, nVectors)
//                    .parallel()
//                    .forEach(j -> {
//                        scoresQADC[j] = f.similarityTo(j);
//                    })).join();
            for (int j = 0; j < nVectors; j++) {
                dummyAccumulator += f.similarityTo(j);
            }
            endTime = System.nanoTime();
            duration += (double) (endTime - startTime) / 1_000_000_000;
        }
//        endTime = System.nanoTime();
//        duration = (double) (endTime - startTime) / 1_000_000_000;
        System.out.println("\tPQ Distance computations took " + duration + " seconds");

        System.out.println("\tdummyAccumulator: " + dummyAccumulator);
        System.out.println("--");
    }

    public static void runSIFT() throws IOException {
        System.out.println("Running siftsmall");

        var baseVectors = "siftsmall/siftsmall_base.fvecs";
        var queryVectors = "siftsmall/siftsmall_query.fvecs";
        testNVQEncodings(baseVectors, queryVectors, VectorSimilarityFunction.DOT_PRODUCT);
    }

    public static void runADA() throws IOException {
        System.out.println("Running ada_002");

        var baseVectors = "./fvec/wikipedia_squad/100k/ada_002_100000_base_vectors.fvec";
        var queryVectors = "./fvec/wikipedia_squad/100k/ada_002_100000_query_vectors_10000.fvec";
        testNVQEncodings(baseVectors, queryVectors, VectorSimilarityFunction.DOT_PRODUCT);
    }

    public static void runColbert() throws IOException {
        System.out.println("Running colbertv2");

        var baseVectors = "./fvec/wikipedia_squad/1M/colbertv2.0_128_base_vectors_1000000.fvec";
        var queryVectors = "./fvec/wikipedia_squad/1M/colbertv2.0_128_query_vectors_100000.fvec";
        testNVQEncodings(baseVectors, queryVectors, VectorSimilarityFunction.COSINE);
    }

    public static void runOpenai3072() throws IOException {
        System.out.println("Running text-embedding-3-large_3072");

        var baseVectors = "./fvec/wikipedia_squad/100k/text-embedding-3-large_3072_100000_base_vectors.fvec";
        var queryVectors = "./fvec/wikipedia_squad/100k/text-embedding-3-large_3072_100000_base_vectors.fvec";
        testNVQEncodings(baseVectors, queryVectors, VectorSimilarityFunction.COSINE);
    }

    public static void main(String[] args) throws IOException {
//        runSIFT();
        runADA();
//        runColbert();
//        runOpenai3072();
    }
}