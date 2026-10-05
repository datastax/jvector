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
import io.github.jbellis.jvector.graph.HnswIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.index.Indexes;
import io.github.jbellis.jvector.management.CompressionType;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.List;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

/**
 * Checks that the PQ training options of {@link HnswIndexBuilder} reach the trained
 * {@link ProductQuantization}, and that the defaults match Cassandra's training (no centering,
 * unweighted) for every similarity function. Lives in this package to read the trained quantizer's
 * settings.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class HnswBuilderPqOptionsTest extends RandomizedTest {
    private static final int DIMENSION = 16;

    private static ListRandomAccessVectorValues unitVectors(int n) {
        List<VectorFloat<?>> vectors = createRandomVectors(n, DIMENSION);
        vectors.forEach(VectorUtil::l2normalize);
        return new ListRandomAccessVectorValues(vectors, DIMENSION);
    }

    private static ProductQuantization trainedPq(HnswIndexBuilder builder) {
        builder.build();
        return ((PQVectors) builder.getCompressedVectors()).getCompressor();
    }

    @Test
    public void defaultsMatchCassandraForEverySimilarity() {
        var ravv = unitVectors(1_000);
        for (var vsf : VectorSimilarityFunction.values()) {
            ProductQuantization pq = trainedPq(Indexes.hnswBuilder(ravv, vsf).withCompressionType(CompressionType.PQ));
            assertNull(vsf + ": no centering by default", pq.globalCentroid);
            assertEquals(vsf + ": unweighted by default", KMeansPlusPlusClusterer.UNWEIGHTED, pq.anisotropicThreshold, 0.0f);
        }
    }

    @Test
    public void explicitValuesReachTraining() {
        var ravv = unitVectors(1_000);
        ProductQuantization pq = trainedPq(Indexes.hnswBuilder(ravv, VectorSimilarityFunction.DOT_PRODUCT)
                .withCompressionType(CompressionType.PQ)
                .withPqGlobalCentering(true)
                .withPqAnisotropicThreshold(0.2f));
        assertNotNull(pq.globalCentroid);
        assertEquals(0.2f, pq.anisotropicThreshold, 0.0f);
    }
}
