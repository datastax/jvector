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
package io.github.jbellis.jvector.example.yaml;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/// Verifies that {@link Compression#getCompressorParameters()} expands list-valued
/// PQ parameters into a Cartesian product of {@link CompressorParameters}.
public class CompressionTest {

    @Test
    public void noneReturnsSingleton() {
        Compression c = new Compression();
        c.type = "None";
        assertEquals(1, c.getCompressorParameters().size());
        assertSame(CompressorParameters.NONE, c.getCompressorParameters().get(0).apply(stub(128)));
    }

    @Test
    public void bqReturnsSingleton() {
        Compression c = new Compression();
        c.type = "BQ";
        assertEquals(1, c.getCompressorParameters().size());
        assertNotNull(c.getCompressorParameters().get(0).apply(stub(128)));
    }

    @Test
    public void pqScalarsProduceSingleConfig() {
        Compression c = new Compression();
        c.type = "PQ";
        c.parameters = map(
                "mFactor", 2,
                "k", 256,
                "centerData", "No",
                "anisotropicThreshold", -1.0);
        var list = c.getCompressorParameters();
        assertEquals(1, list.size());
        // mFactor=2 against dim=128 -> m=64
        assertEquals("PQ_ds_64_256_false_-1.0", list.get(0).apply(stub(128)).idStringFor(stub(128)));
    }

    @Test
    public void pqListParametersCartesian() {
        Compression c = new Compression();
        c.type = "PQ";
        c.parameters = map(
                "mFactor", Arrays.asList(2, 4),
                "k", Arrays.asList(128, 256),
                "centerData", Arrays.asList(true, false),
                "anisotropicThreshold", Arrays.asList(-1.0, 0.2));
        var list = c.getCompressorParameters();
        assertEquals(2 * 2 * 2 * 2, list.size());
    }

    @Test
    public void pqMFactorListExpands() {
        Compression c = new Compression();
        c.type = "PQ";
        c.parameters = map("mFactor", Arrays.asList(2, 4));
        var list = c.getCompressorParameters();
        assertEquals(2, list.size());
        // Verify both mFactor values are represented
        var ids = list.stream().map(f -> f.apply(stub(128)).idStringFor(stub(128))).collect(Collectors.toList());
        assertTrue(ids.stream().anyMatch(s -> s.contains("_64_")), ids.toString());
        assertTrue(ids.stream().anyMatch(s -> s.contains("_32_")), ids.toString());
    }

    @Test
    public void pqCenterDataAbsentUsesDatasetDefault() {
        Compression c = new Compression();
        c.type = "PQ";
        c.parameters = map("mFactor", 2);

        Function<DataSet, CompressorParameters> f = c.getCompressorParameters().get(0);
        // EUCLIDEAN -> centered=true
        assertTrue(f.apply(stub(128, VectorSimilarityFunction.EUCLIDEAN))
                .idStringFor(stub(128, VectorSimilarityFunction.EUCLIDEAN)).contains("_true_"));
        // DOT_PRODUCT -> centered=false
        assertTrue(f.apply(stub(128, VectorSimilarityFunction.DOT_PRODUCT))
                .idStringFor(stub(128, VectorSimilarityFunction.DOT_PRODUCT)).contains("_false_"));
    }

    @Test
    public void pqMPrecedenceOverMFactor() {
        // Both provided -> 'm' wins, mFactor list is ignored (matches prior behavior).
        Compression c = new Compression();
        c.type = "PQ";
        c.parameters = map(
                "m", 192,
                "mFactor", Arrays.asList(2, 4));
        var list = c.getCompressorParameters();
        assertEquals(1, list.size());
        assertTrue(list.get(0).apply(stub(128)).idStringFor(stub(128)).contains("_192_"));
    }

    @Test
    public void pqRejectsMissingMAndMFactor() {
        Compression c = new Compression();
        c.type = "PQ";
        c.parameters = map("k", 256);
        assertThrows(IllegalArgumentException.class, c::getCompressorParameters);
    }

    @Test
    public void pqAcceptsYesNoStrings() {
        Compression c = new Compression();
        c.type = "PQ";
        c.parameters = map("mFactor", 2, "centerData", "Yes");
        assertTrue(c.getCompressorParameters().get(0).apply(stub(128)).idStringFor(stub(128)).contains("_true_"));
    }

    @Test
    public void unsupportedTypeThrows() {
        Compression c = new Compression();
        c.type = "ZQ";
        assertThrows(IllegalArgumentException.class, c::getCompressorParameters);
    }

    // ------------------------------------------------------------------------

    private static Map<String, Object> map(Object... kv) {
        assertEquals(0, kv.length % 2);
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) {
            m.put((String) kv[i], kv[i + 1]);
        }
        return m;
    }

    private static DataSet stub(int dim) {
        return stub(dim, VectorSimilarityFunction.EUCLIDEAN);
    }

    private static DataSet stub(int dim, VectorSimilarityFunction sim) {
        return new DataSet() {
            @Override public int getDimension() { return dim; }
            @Override public String getName() { return "ds"; }
            @Override public VectorSimilarityFunction getSimilarityFunction() { return sim; }
            @Override public RandomAccessVectorValues getBaseRavv() { throw new UnsupportedOperationException(); }
            @Override public List<VectorFloat<?>> getBaseVectors() { throw new UnsupportedOperationException(); }
            @Override public List<VectorFloat<?>> getQueryVectors() { throw new UnsupportedOperationException(); }
            @Override public List<? extends List<Integer>> getGroundTruth() { throw new UnsupportedOperationException(); }
        };
    }
}
