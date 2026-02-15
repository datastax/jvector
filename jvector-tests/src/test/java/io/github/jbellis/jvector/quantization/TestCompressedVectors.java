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

package io.github.jbellis.jvector.quantization;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.File;
import java.util.List;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.TestUtil.createNormalRandomVectors;
import static io.github.jbellis.jvector.TestUtil.nextInt;
import static java.lang.Math.abs;
import static java.lang.Math.log;
import static org.junit.jupiter.api.Assertions.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestCompressedVectors extends RandomizedTest {
    @Test
    public void testSaveLoadPQ() throws Exception {
        // Generate a PQ for random 2D vectors
        var vectors = createRandomVectors(512, 2);
        var ravv = new ListRandomAccessVectorValues(vectors, 2);
        var pq = ProductQuantization.compute(ravv, 1, 256, false);

        // Compress the vectors
        var cv = pq.encodeAll(ravv);
        assertEquals(2 * Float.BYTES, cv.getOriginalSize());
        assertEquals(1, cv.getCompressedSize());

        // Write compressed vectors
        File cvFile = File.createTempFile("pqtest", ".cv");
        try (var writer = new SimpleWriter(cvFile.toPath())) {
            cv.write(writer);
        }
        // Read compressed vectors
        try (var readerSupplier = new SimpleMappedReader.Supplier(cvFile.toPath())) {
            var cv2 = PQVectors.load(readerSupplier.get(), 0);
            assertEquals(cv.hashCode(), cv2.hashCode());
            assertEquals(cv, cv2);
        }
    }

    @Test
    public void testSaveLoadBQ() throws Exception {
        // Generate a PQ for random vectors
        var vectors = createRandomVectors(512, 64);
        var ravv = new ListRandomAccessVectorValues(vectors, 64);
        var bq = new BinaryQuantization(ravv.dimension());

        // Compress the vectors
        var cv = bq.encodeAll(ravv);
        assertEquals(64 * Float.BYTES, cv.getOriginalSize());
        assertEquals(8, cv.getCompressedSize());

        // Write compressed vectors
        File cvFile = File.createTempFile("bqtest", ".cv");
        try (var writer = new SimpleWriter(cvFile.toPath())) {
            cv.write(writer);
        }
        // Read compressed vectors
        try (var readerSupplier = new SimpleMappedReader.Supplier(cvFile.toPath())) {
            var cv2 = BQVectors.load(readerSupplier.get(), 0);
            assertEquals(cv, cv2);
        }
    }

    @Test
    public void testSaveLoadNVQ() throws Exception {

        int[][] testsConfigAndResults = {
                //Tuples of: nDimensions, nSubvectors, and the expected number of bytes
                {64, 1, 96},
                {64, 2, 124},
                {65, 1, 97},
        };

        for (int[] testConfigAndResult : testsConfigAndResults) {
            var nDimensions = testConfigAndResult[0];
            var nSubvectors = testConfigAndResult[1];
            var expectedSize = testConfigAndResult[2];

            // Generate an NVQ for random vectors
            var vectors = createRandomVectors(512, nDimensions);
            var ravv = new ListRandomAccessVectorValues(vectors, nDimensions);

            var nvq = NVQuantization.compute(ravv, nSubvectors);

            // Compress the vectors
            var cv = nvq.encodeAll(ravv);
            assertEquals(nDimensions * Float.BYTES, cv.getOriginalSize());
            assertEquals(expectedSize, cv.getCompressedSize());

            // Write compressed vectors
            File cvFile = File.createTempFile("bqtest", ".cv");
            try (var writer = new SimpleWriter(cvFile.toPath())) {
                cv.write(writer);
            }
            // Read compressed vectors
            try (var readerSupplier = new SimpleMappedReader.Supplier(cvFile.toPath())) {
                var cv2 = NVQVectors.load(readerSupplier.get(), 0);
                assertEquals(cv, cv2);
            }
        }
    }


    private void testPQEncodings(int dimension, int codebooks) {
        // Generate a PQ for random vectors
        var vectors = createRandomVectors(512, dimension);
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);
        var pq = ProductQuantization.compute(ravv, codebooks, 256, false);

        // Compress the vectors
        var cv = pq.encodeAll(ravv);

        // compare the encoded similarities to the raw
        for (var vsf : List.of(VectorSimilarityFunction.EUCLIDEAN, VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.COSINE)) {
            double delta = 0;
            for (int i = 0; i < 10; i++) {
                var q = TestUtil.randomVector(getRandom(), dimension);
                var f = cv.precomputedScoreFunctionFor(q, vsf);
                for (int j = 0; j < vectors.size(); j++) {
                    delta += abs(f.similarityTo(j) - vsf.compare(q, vectors.get(j)));
                }
            }
            // https://chat.openai.com/share/7ced3fc8-275a-4134-978c-c822275c3e1f
            // is there a better way to check for within-expected bounds?
            var expectedDelta = vsf == VectorSimilarityFunction.EUCLIDEAN
                    ? 96.98 * log(3.26 + dimension) / log(1.92 + codebooks) - 112.15
                    : 152.69 * log(3.76 + dimension) / log(1.95 + codebooks) - 180.86;
            // expected is accurate to within about 10% *on average*.  experimentally 25% is not quite enough
            // to avoid false positives, so we pad by 40%
            assert delta <= 1.4 * expectedDelta : String.format("%s > %s for %s with %d dimensions and %d codebooks", delta, expectedDelta, vsf, dimension, codebooks);
        }
    }

    @Test
    public void testPQEncodings() {
        // start with i=2 (dimension 4) b/c dimension 2 is an outlier for our error prediction
        for (int i = 2; i <= 8; i++) {
            for (int M = 1; M <= i; M++) {
                testPQEncodings(2 * i, M);
            }
        }
    }

    private void testNVQEncodings(List<VectorFloat<?>> vectors, List<VectorFloat<?>> queries, int nSubvectors,
                                  boolean learn) {
        int dimension = vectors.get(0).length();
        int nQueries = queries.size();

        // Generate a NVQ for random vectors
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);
        var nvq = NVQuantization.compute(ravv, nSubvectors);
        nvq.learn = learn;

        // Compress the vectors
        var cv = nvq.encodeAll(ravv);

        // compare the encoded similarities to the raw
        for (var vsf : List.of(VectorSimilarityFunction.EUCLIDEAN, VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.COSINE)) {
            double error = 0;
            for (int i = 0; i < nQueries; i++) {
                var q = queries.get(i);
                VectorUtil.l2normalize(q);
                var f = cv.scoreFunctionFor(q, vsf);
                for (int j = 0; j < vectors.size(); j++) {
                    var v = vectors.get(j);
                    vsf.compare(q, v);
                    if (vsf == VectorSimilarityFunction.DOT_PRODUCT) {
                        error += abs(f.similarityTo(j) - vsf.compare(q, v)) / abs(vsf.compare(v, v));
                    } else {
                        error += abs(f.similarityTo(j) - vsf.compare(q, v));
                    }
                }
            }
            error /= nQueries * vectors.size();

            float tolerance = 0.0005f * (dimension / 256.f);
            if (vsf == VectorSimilarityFunction.COSINE) {
                tolerance *= 10;
            } else if (vsf == VectorSimilarityFunction.DOT_PRODUCT) {
                tolerance *= 4;
            }
            // System.out.println(vsf + " error " + error + " tolerance " + tolerance);
            assert error <= tolerance : String.format("%s > %s for %s with %d dimensions and %d subvectors", error, tolerance, vsf, dimension, nSubvectors);
        }
        // System.out.println("--");
    }

    @Test
    public void testNVQEncodings() {
        for (int d = 256; d <= 2048; d += 256) {
            var vectors = createNormalRandomVectors(512, d);
            var queries = createNormalRandomVectors(10, d);

            for (var nSubvectors : List.of(1, 2, 4, 8)) {
                for (var learn : List.of(false, true)) {
                    // System.out.println("dimensions: " + d + " subvectors: " + nSubvectors + " learn: " + learn);
                    testNVQEncodings(vectors, queries, nSubvectors, learn);
                }
            }
        }
    }

    @Test
    public void testRawEqualsPrecomputed() {
        // Generate a PQ for random vectors
        int dimension = nextInt(getRandom(), 4, 2048);
        int codebooks = nextInt(getRandom(), 1, dimension / 2);
        var vectors = createRandomVectors(512, dimension);
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);
        for (boolean center : new boolean[] {true, false}) {
            var pq = ProductQuantization.compute(ravv, codebooks, 256, center);

            // Compress the vectors
            var cv = pq.encodeAll(ravv);

            // compare the precomputed similarities to the raw
            for (int i = 0; i < 10; i++) {
                var q = TestUtil.randomVector(getRandom(), dimension);
                for (var vsf : List.of(VectorSimilarityFunction.EUCLIDEAN, VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.COSINE)) {
                    var precomputed = cv.precomputedScoreFunctionFor(q, vsf);
                    var raw = cv.scoreFunctionFor(q, vsf);
                    for (int j = 0; j < 10; j++) {
                        var target = getRandom().nextInt(vectors.size());
                        assertEquals(raw.similarityTo(target), precomputed.similarityTo(target), 1e-6);
                    }
                }
            }
        }
    }

    @Test
    public void testCenteringDisturbance() {

    }
}
