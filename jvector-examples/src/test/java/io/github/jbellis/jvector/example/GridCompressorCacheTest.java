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

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Test;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/// Grid's compressor cache, exercised through the same private entry point the benchmarks use:
/// a compressor saved under its cache key must load back as the type that saved it, and the
/// in-memory memo must hand back the same instance for the same key.
public class GridCompressorCacheTest {
    private final List<Path> cacheFiles = new ArrayList<>();

    @After
    public void cleanUp() throws Exception {
        Grid.cachedCompressors.clear();
        for (Path file : cacheFiles) Files.deleteIfExists(file);
    }

    @Test
    public void cachedNvqCompressorLoadsBackAsNvq() throws Exception {
        var ds = dataset("grid-cache-nvq-" + System.nanoTime(), 8, 64);
        Function<DataSet, CompressorParameters> params = d -> new CompressorParameters.NVQParameters(2);
        var file = cacheFile(params, ds);

        Grid.cachedCompressors.clear();
        var computed = compressor(params, ds);
        assertTrue(String.valueOf(computed), computed instanceof NVQuantization);
        assertTrue("Compute must save under the cache key", Files.exists(file));

        Grid.cachedCompressors.clear(); // as a fresh JVM would, forcing the on-disk path
        var loaded = compressor(params, ds);
        assertTrue("Loaded " + loaded.getClass().getSimpleName() + " where NVQuantization was saved",
                loaded instanceof NVQuantization);
    }

    @Test
    public void sameKeyReturnsTheMemoizedCompressor() throws Exception {
        var ds = dataset("grid-cache-memo-" + System.nanoTime(), 8, 64);
        Function<DataSet, CompressorParameters> params = d -> new CompressorParameters.NVQParameters(2);
        cacheFile(params, ds);

        Grid.cachedCompressors.clear();
        var first = compressor(params, ds);
        var second = compressor(params, ds);
        assertSame("The same cache key must hit the in-memory memo instead of reloading", first, second);
        assertEquals("One entry per key", 1, Grid.cachedCompressors.size());
    }

    private Path cacheFile(Function<DataSet, CompressorParameters> params, DataSet ds) {
        var file = Paths.get("pq_cache").resolve(params.apply(ds).idStringFor(ds));
        cacheFiles.add(file);
        return file;
    }

    /// `Grid.getCompressor(Function, DataSet)` is private; it is the only way the benchmarks
    /// resolve a compressor, so the cache is tested through it rather than through a copy.
    private static VectorCompressor<?> compressor(Function<DataSet, CompressorParameters> params, DataSet ds) throws Exception {
        Method method = Grid.class.getDeclaredMethod("getCompressor", Function.class, DataSet.class);
        method.setAccessible(true);
        try {
            return (VectorCompressor<?>) method.invoke(null, params, ds);
        } catch (InvocationTargetException e) {
            throw e.getCause() instanceof Exception ? (Exception) e.getCause() : e;
        }
    }

    /// A small in-memory dataset with random vectors; only what the compressor path reads.
    private static DataSet dataset(String name, int dimension, int count) {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(7381);
        var vectors = new ArrayList<VectorFloat<?>>(count);
        for (int i = 0; i < count; i++) {
            var vector = types.createFloatVector(dimension);
            for (int j = 0; j < dimension; j++) vector.set(j, random.nextFloat());
            vectors.add(vector);
        }
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);
        return new DataSet() {
            @Override public int getDimension() { return dimension; }
            @Override public String getName() { return name; }
            @Override public VectorSimilarityFunction getSimilarityFunction() { return VectorSimilarityFunction.EUCLIDEAN; }
            @Override public RandomAccessVectorValues getBaseRavv() { return ravv; }
            @Override public List<VectorFloat<?>> getBaseVectors() { return vectors; }
            @Override public List<VectorFloat<?>> getQueryVectors() { throw new UnsupportedOperationException(); }
            @Override public List<? extends List<Integer>> getGroundTruth() { throw new UnsupportedOperationException(); }
        };
    }
}
