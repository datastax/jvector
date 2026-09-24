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

package io.github.jbellis.jvector.example.benchmarks.datasets;

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/// A {@link DataSetWrapper} whose base vectors are fully cached in heap memory.
///
/// On construction the origin's base {@link RandomAccessVectorValues} is read in its entirety
/// through {@link RandomAccessVectorValues#range(int, int)} views, one chunk per task, in
/// parallel on the common pool. Each vector is copied into a fresh heap vector, so the cache is
/// independent of the origin's storage (a memory-mapped file, for instance) and serves reads at
/// plain list-lookup cost.
///
/// An origin whose base RAVV is already a {@link ListRandomAccessVectorValues} is adopted as-is:
/// it is heap-resident already, and copying it would only double the footprint.
///
/// Registered in {@link DataSets#wrapperProviders} under the name {@value #WRAPPER_NAME}.
public final class InMemoryCachedDataSet implements DataSetWrapper {
    private static final Logger logger = LoggerFactory.getLogger(InMemoryCachedDataSet.class);
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    /// Symbolic wrapper name, as used in `dataset(memory)` and `wrappers: [memory]`.
    public static final String WRAPPER_NAME = "memory";

    /// Provider that caches a dataset's base vectors in memory; idempotent for already-cached datasets.
    public static final DataSetWrapper.Provider PROVIDER = InMemoryCachedDataSet::of;

    /// Registry entry for {@value #WRAPPER_NAME}; this wrapper takes no options.
    public static final DataSetWrapper.Factory FACTORY = DataSetWrapper.Factory.optionless(WRAPPER_NAME, PROVIDER);

    /// Vectors copied per parallel task.
    static final int DEFAULT_CHUNK_SIZE = 8 * 1024;

    private final DataSet origin;
    private final RandomAccessVectorValues baseRavv;

    /// Returns `origin` itself if it is already an {@link InMemoryCachedDataSet}, otherwise a new cache over it.
    ///
    /// @param origin the dataset whose base vectors should be heap-resident
    /// @return an in-memory view of `origin`
    public static DataSetWrapper of(DataSet origin) {
        if (origin instanceof InMemoryCachedDataSet) {
            return (InMemoryCachedDataSet) origin;
        }
        return new InMemoryCachedDataSet(origin);
    }

    private InMemoryCachedDataSet(DataSet origin) {
        this.origin = origin;
        RandomAccessVectorValues source = origin.getBaseRavv();
        if (source instanceof ListRandomAccessVectorValues) {
            this.baseRavv = source;
            logger.info("Base vectors of '{}' are already heap-resident; adopting {} vectors as-is", origin.getName(), source.size());
            return;
        }
        long start = System.nanoTime();
        List<VectorFloat<?>> vectors = readAllVectors(source);
        this.baseRavv = new ListRandomAccessVectorValues(vectors, source.dimension());
        double mb = (double) vectors.size() * source.dimension() * Float.BYTES / (1024.0 * 1024.0);
        logger.info("Cached {} base vectors ({} MB) of '{}' in memory in {}s",
                vectors.size(), String.format("%.1f", mb), origin.getName(),
                String.format("%.2f", (System.nanoTime() - start) / 1e9));
    }

    /// Copies every vector of `source` into a new heap-resident list, reading through ranged views
    /// in parallel chunks of {@value #DEFAULT_CHUNK_SIZE} vectors.
    ///
    /// @param source the vectors to copy; may be value-shared, since each chunk reads through its own {@link RandomAccessVectorValues#copy()}
    /// @return a fixed-size list with one independent vector per ordinal of `source`
    public static List<VectorFloat<?>> readAllVectors(RandomAccessVectorValues source) {
        return readAllVectors(source, DEFAULT_CHUNK_SIZE);
    }

    static List<VectorFloat<?>> readAllVectors(RandomAccessVectorValues source, int chunkSize) {
        int count = source.size();
        int dimension = source.dimension();
        VectorFloat<?>[] out = new VectorFloat<?>[count];
        int chunks = (count + chunkSize - 1) / chunkSize;
        IntStream.range(0, chunks).parallel().forEach(chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(count, start + chunkSize);
            RandomAccessVectorValues slice = source.range(start, end).copy();
            for (int i = 0; i < slice.size(); i++) {
                VectorFloat<?> v = vts.createFloatVector(dimension);
                slice.getVectorInto(i, v, 0);
                out[start + i] = v;
            }
        });
        return Arrays.asList(out);
    }

    @Override
    public DataSet getOrigin() {
        return origin;
    }

    @Override
    public RandomAccessVectorValues getBaseRavv() {
        return baseRavv;
    }

    @Override
    public int getDimension() {
        return baseRavv.dimension();
    }
}
