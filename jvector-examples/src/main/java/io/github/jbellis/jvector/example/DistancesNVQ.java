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
import io.github.jbellis.jvector.graph.ListRandomAccessVectorRepresentations;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityType;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.util.*;

import static java.lang.Math.abs;

// this class uses explicit typing instead of `var` for easier reading when excerpted for instructional use
public class DistancesNVQ {
    public static void testNVQEncodings(String filenameBase, String filenameQueries, VectorSimilarityType vsf) throws IOException {
        List<VectorFloat<?>> vectors = SiftLoader.readFvecs(filenameBase);
        List<VectorFloat<?>> queries = SiftLoader.readFvecs(filenameQueries);

        int dimension = vectors.get(0).length();
        int nQueries = 100;
        int nVectors = 10_000;

        vectors = vectors.subList(0, nVectors);

        System.out.format("\t%d base and %d query vectors loaded, dimensions %d%n",
                vectors.size(), queries.size(), vectors.get(0).length());

        // Generate a NVQ for random vectors
        var ravv = new ListRandomAccessVectorRepresentations(vectors, dimension);
        var nvq = NVQuantization.compute(ravv, 2);

        // Compress the vectors
        long startTime = System.nanoTime();
        var nvqVecs = nvq.encodeAll(ravv);
        long endTime = System.nanoTime();
        double duration = (double) (endTime - startTime) / 1_000_000_000;
        System.out.println("\tEncoding took " + duration + " seconds");

        // compare the encoded similarities to the raw
        double distanceError = 0;
        for (int i = 0; i < nQueries; i++) {
            var q = queries.get(i);
            if (VectorUtil.dotProduct(q, q) == 0) {
                continue;
            }
            var f = nvqVecs.scoreFunctionFor(q, vsf);

            for (int j = 0; j < nVectors; j++) {
                var v = vectors.get(j);
                distanceError += abs(f.similarityTo(j) - vsf.compare(q, v));
            }
        }
        distanceError /= nQueries * nVectors;

        System.out.println("\t" + vsf + " error " + distanceError);


        float dummyAccumulator = 0;

        startTime = System.nanoTime();
        for (int i = 0; i < nQueries; i++) {
            var q = queries.get(i);
            if (VectorUtil.dotProduct(q, q) == 0) {
                continue;
            }
            var f = nvqVecs.scoreFunctionFor(q, vsf);

            for (int j = 0; j < nVectors; j++) {
                dummyAccumulator += f.similarityTo(j);
            }
        }
        endTime = System.nanoTime();
        duration = (double) (endTime - startTime) / 1_000_000_000;
        System.out.println("\tNVQ Distance computations took " + duration + " seconds");

        startTime = System.nanoTime();
        for (int i = 0; i < nQueries; i++) {
            var q = queries.get(i);
            if (VectorUtil.dotProduct(q, q) == 0) {
                continue;
            }

            for (int j = 0; j < nVectors; j++) {
                var v = vectors.get(j);
                dummyAccumulator += vsf.compare(q, v);
            }
        }
        endTime = System.nanoTime();
        duration = (double) (endTime - startTime) / 1_000_000_000;
        System.out.println("\tFloat Distance computations took " + duration + " seconds");

        System.out.println("\tdummyAccumulator: " + dummyAccumulator);
        System.out.println("--");
    }

    public static void runSIFT() throws IOException {
        System.out.println("Running siftsmall");

        var baseVectors = "siftsmall/siftsmall_base.fvecs";
        var queryVectors = "siftsmall/siftsmall_query.fvecs";
        testNVQEncodings(baseVectors, queryVectors, VectorSimilarityType.COSINE);
    }

    public static void runADA() throws IOException {
        System.out.println("Running ada_002");

        var baseVectors = "./fvec/wikipedia_squad/100k/ada_002_100000_base_vectors.fvec";
        var queryVectors = "./fvec/wikipedia_squad/100k/ada_002_100000_query_vectors_10000.fvec";
        testNVQEncodings(baseVectors, queryVectors, VectorSimilarityType.COSINE);
    }

    public static void runColbert() throws IOException {
        System.out.println("Running colbertv2");

        var baseVectors = "./fvec/wikipedia_squad/1M/colbertv2.0_128_base_vectors_1000000.fvec";
        var queryVectors = "./fvec/wikipedia_squad/1M/colbertv2.0_128_query_vectors_100000.fvec";
        testNVQEncodings(baseVectors, queryVectors, VectorSimilarityType.COSINE);
    }

    public static void runOpenai3072() throws IOException {
        System.out.println("Running text-embedding-3-large_3072");

        var baseVectors = "./fvec/wikipedia_squad/100k/text-embedding-3-large_3072_100000_base_vectors.fvec";
        var queryVectors = "./fvec/wikipedia_squad/100k/text-embedding-3-large_3072_100000_base_vectors.fvec";
        testNVQEncodings(baseVectors, queryVectors, VectorSimilarityType.COSINE);
    }

    public static void main(String[] args) throws IOException {
        runSIFT();
        runADA();
        runColbert();
        runOpenai3072();
    }
}