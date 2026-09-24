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

import io.github.jbellis.jvector.example.util.MappedFvecsRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/// Tests for the two built-in {@link DataSetWrapper}s and for how {@link DataSets} applies wrappers,
/// resolves symbolic wrapper names, and enforces the loader profile rule.
public class DataSetWrapperTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int DIMENSION = 4;

    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    private static final float[][] BASE = {
            {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0.5f, 0.5f, 0.5f, 0.5f}, {0.25f, 0.5f, 0.75f, 1},
    };

    private static List<VectorFloat<?>> heapVectors(float[][] values) {
        List<VectorFloat<?>> out = new ArrayList<>();
        for (float[] v : values) out.add(vts.createFloatVector(v.clone()));
        return out;
    }

    private static List<VectorFloat<?>> queries() {
        return heapVectors(new float[][] {{1, 0, 0, 0}, {0, 0, 1, 0}});
    }

    private static List<List<Integer>> groundTruth() {
        return List.of(List.of(0, 4), List.of(2, 4));
    }

    private Path writeBaseFvecs() throws IOException {
        Path file = tempFolder.newFile("base.fvecs").toPath();
        var buf = ByteBuffer.allocate(BASE.length * (Integer.BYTES + DIMENSION * Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
        for (float[] v : BASE) {
            buf.putInt(DIMENSION);
            for (float f : v) buf.putFloat(f);
        }
        Files.write(file, buf.array());
        return file;
    }

    private DataSet mappedDataSet() throws IOException {
        return new SimpleDataSet("mapped-ds", VectorSimilarityFunction.COSINE,
                new MappedFvecsRandomAccessVectorValues(writeBaseFvecs()), queries(), groundTruth());
    }

    private static DataSet heapDataSet() {
        return new SimpleDataSet("heap-ds", VectorSimilarityFunction.DOT_PRODUCT, heapVectors(BASE), queries(), groundTruth());
    }

    private static void assertBaseMatches(RandomAccessVectorValues ravv) {
        assertEquals(BASE.length, ravv.size());
        assertEquals(DIMENSION, ravv.dimension());
        for (int i = 0; i < BASE.length; i++) {
            VectorFloat<?> v = ravv.getVector(i);
            for (int d = 0; d < DIMENSION; d++) {
                assertEquals(BASE[i][d], v.get(d), 0f);
            }
        }
    }

    private static void assertDelegates(DataSetWrapper wrapper, DataSet origin) {
        assertSame(origin, wrapper.getOrigin());
        assertEquals(origin.getName(), wrapper.getName());
        assertEquals(origin.getSimilarityFunction(), wrapper.getSimilarityFunction());
        assertSame(origin.getQueryVectors(), wrapper.getQueryVectors());
        assertSame(origin.getGroundTruth(), wrapper.getGroundTruth());
        assertEquals(origin.getDimension(), wrapper.getDimension());
    }

    // ------------------------------------------------------------------ InMemoryCachedDataSet

    @Test
    public void inMemoryCacheCopiesMappedBaseVectorsToHeap() throws IOException {
        DataSet origin = mappedDataSet();
        DataSetWrapper cached = InMemoryCachedDataSet.of(origin);

        assertTrue(cached instanceof InMemoryCachedDataSet);
        assertDelegates(cached, origin);
        assertTrue(cached.getBaseRavv() instanceof ListRandomAccessVectorValues);
        assertFalse(cached.getBaseRavv().isValueShared());
        assertBaseMatches(cached.getBaseRavv());
        assertNotSame(cached.getBaseRavv().getVector(0), cached.getBaseRavv().getVector(1));

        assertSame(cached, InMemoryCachedDataSet.of(cached));
    }

    @Test
    public void inMemoryCacheAdoptsHeapResidentBaseVectors() {
        DataSet origin = heapDataSet();
        DataSetWrapper cached = InMemoryCachedDataSet.of(origin);
        assertSame(origin.getBaseRavv(), cached.getBaseRavv());
        assertDelegates(cached, origin);
    }

    // ------------------------------------------------------------------ MMapCachedDataSet

    @Test
    public void mmapCacheSpillsHeapBaseVectorsToFile() throws IOException {
        DataSet origin = heapDataSet();
        Path cacheDir = tempFolder.getRoot().toPath().resolve("mmap-cache");
        var mapped = new MMapCachedDataSet(origin, cacheDir);

        assertDelegates(mapped, origin);
        assertTrue(mapped.getBaseRavv() instanceof MappedFvecsRandomAccessVectorValues);
        assertBaseMatches(mapped.getBaseRavv());
        Path spill = ((MappedFvecsRandomAccessVectorValues) mapped.getBaseRavv()).getPath();
        assertEquals(cacheDir.resolve("heap-ds-6x4.fvecs"), spill);
        assertEquals(BASE.length * (Integer.BYTES + DIMENSION * Float.BYTES), Files.size(spill));

        assertSame(mapped, MMapCachedDataSet.of(mapped));
    }

    @Test
    public void mmapCacheAdoptsAlreadyMappedBaseVectors() throws IOException {
        DataSet origin = mappedDataSet();
        Path cacheDir = tempFolder.getRoot().toPath().resolve("unused-cache");
        var mapped = new MMapCachedDataSet(origin, cacheDir);
        assertSame(origin.getBaseRavv(), mapped.getBaseRavv());
        assertFalse(Files.exists(cacheDir));
    }

    @Test
    public void wrappersCompose() throws IOException {
        DataSet origin = heapDataSet();
        Path cacheDir = tempFolder.getRoot().toPath().resolve("compose");
        DataSetWrapper mapped = new MMapCachedDataSet(origin, cacheDir);
        DataSetWrapper rehydrated = InMemoryCachedDataSet.of(mapped);
        assertSame(mapped, rehydrated.getOrigin());
        assertTrue(rehydrated.getBaseRavv() instanceof ListRandomAccessVectorValues);
        assertNotSame(origin.getBaseRavv(), rehydrated.getBaseRavv());
        assertBaseMatches(rehydrated.getBaseRavv());
    }

    // ------------------------------------------------------------------ LruCachedDataSet

    @Test
    public void lruWrapperServesOriginThroughBoundedGrainCache() throws IOException {
        DataSet origin = mappedDataSet();
        var lru = new LruCachedDataSet(origin, 2, 2L * DIMENSION * Float.BYTES * 2); // two grains of two vectors
        assertDelegates(lru, origin);
        assertEquals(2, lru.getBaseRavv().grainSize());
        assertEquals(2, lru.getBaseRavv().maxGrains());
        assertEquals(3, lru.getBaseRavv().grainCount());
        assertBaseMatches(lru.getBaseRavv());
        assertEquals(3, lru.getBaseRavv().misses());
        assertTrue(lru.getBaseRavv().evictions() >= 1);
        assertFalse(lru.getBaseRavv().isValueShared());

        assertSame(lru, LruCachedDataSet.of(lru));
        DataSetWrapper viaDefaults = LruCachedDataSet.of(origin);
        assertTrue(viaDefaults instanceof LruCachedDataSet);
        assertEquals(LruCachedDataSet.DEFAULT_GRAIN_SIZE, ((LruCachedDataSet) viaDefaults).getBaseRavv().grainSize());

        DataSetWrapper tiny = LruCachedDataSet.provider(3, 1).wrap(origin);
        assertEquals(1, ((LruCachedDataSet) tiny).getBaseRavv().maxGrains(), "capacity below one grain still keeps one grain");
    }

    // ------------------------------------------------------------------ DataSets facade

    /// A loader that serves {@code test-ds} from a heap dataset and counts materialisations.
    private static class HeapLoader implements DataSetLoader {
        final AtomicInteger loads = new AtomicInteger();

        @Override
        public Optional<DataSetInfo> loadDataSet(String dataSetName) {
            if (!dataSetName.equals("test-ds")) return Optional.empty();
            var props = new DataSetProperties.PropertyMap(Map.of(
                    DataSetProperties.KEY_NAME, "test-ds",
                    DataSetProperties.KEY_SIMILARITY_FUNCTION, VectorSimilarityFunction.DOT_PRODUCT,
                    DataSetProperties.KEY_LOAD_BEHAVIOR, DataSetProperties.LoadBehavior.NO_SCRUB));
            return Optional.of(new DataSetInfo(props, () -> {
                loads.incrementAndGet();
                return heapDataSet();
            }));
        }
    }

    /// A loader that understands profiles and records the one it was asked for.
    private static final class ProfileLoader extends HeapLoader {
        String profileSeen;

        @Override
        public Optional<DataSetInfo> loadDataSet(DataSetSpec spec) {
            profileSeen = spec.getProfile();
            return loadDataSet(spec.getName());
        }
    }

    @Test
    public void defaultWrappersCacheInMemoryLazily() {
        var loader = new HeapLoader();
        var info = DataSets.loadDataSet("test-ds", List.of(loader)).orElseThrow();
        assertEquals("test-ds", info.getName());
        assertEquals(DataSetProperties.LoadBehavior.NO_SCRUB, info.loadBehavior());
        assertEquals(0, loader.loads.get(), "wrapping must not materialise the dataset");

        DataSet ds = info.getDataSet();
        assertEquals(1, loader.loads.get());
        assertTrue(ds instanceof InMemoryCachedDataSet);
        assertSame(ds, info.getDataSet());
        assertEquals(1, loader.loads.get());
    }

    @Test
    public void symbolicWrappersReplaceTheDefaults() {
        var loader = new HeapLoader();
        DataSet memory = DataSets.loadDataSet("test-ds(memory)", List.of(loader)).orElseThrow().getDataSet();
        assertTrue(memory instanceof InMemoryCachedDataSet);

        Path cacheDir = tempFolder.getRoot().toPath().resolve("facade");
        DataSetWrapper.Provider tempMmap = origin -> new MMapCachedDataSet(origin, cacheDir);
        DataSets.wrapperProviders.put("mmap-tmp", DataSetWrapper.Factory.optionless("mmap-tmp", tempMmap));
        try {
            DataSet mapped = DataSets.loadDataSet("test-ds:default(mmap-tmp)", List.of(loader)).orElseThrow().getDataSet();
            assertTrue(mapped instanceof MMapCachedDataSet);
            assertTrue(mapped.getBaseRavv() instanceof MappedFvecsRandomAccessVectorValues);

            DataSet both = DataSets.loadDataSet("test-ds(mmap-tmp, memory)", List.of(loader)).orElseThrow().getDataSet();
            assertTrue(both instanceof InMemoryCachedDataSet);
            assertTrue(((DataSetWrapper) both).getOrigin() instanceof MMapCachedDataSet);
            assertBaseMatches(both.getBaseRavv());
        } finally {
            DataSets.wrapperProviders.remove("mmap-tmp");
        }

        assertSame(MMapCachedDataSet.PROVIDER, DataSets.resolveWrappers(DataSetSpec.parse("x(mmap)").getWrappers()).get(0));
        assertSame(InMemoryCachedDataSet.PROVIDER, DataSets.resolveWrappers(DataSetSpec.parse("x(memory)").getWrappers()).get(0));
        assertSame(LruCachedDataSet.PROVIDER, DataSets.resolveWrappers(DataSetSpec.parse("x(lru)").getWrappers()).get(0));
        DataSet lru = DataSets.loadDataSet("test-ds(lru)", List.of(loader)).orElseThrow().getDataSet();
        assertTrue(lru instanceof LruCachedDataSet);
        assertBaseMatches(lru.getBaseRavv());
        assertThrows(IllegalArgumentException.class, () -> DataSets.loadDataSet("test-ds(bogus)", List.of(loader)));
    }

    @Test
    public void wrapperOptionsReachTheFactory() {
        var loader = new HeapLoader();
        DataSet tuned = DataSets.loadDataSet("test-ds(lru[grain=2,capacityMb=1])", List.of(loader)).orElseThrow().getDataSet();
        var cache = ((LruCachedDataSet) tuned).getBaseRavv();
        assertEquals(2, cache.grainSize());
        assertEquals((1024 * 1024) / (2 * DIMENSION * Float.BYTES), cache.maxGrains());
        assertBaseMatches(cache);

        var structured = DataSetSpec.from(Map.of("name", "test-ds", "wrappers", List.of(Map.of("lru", Map.of("grain", 3)))));
        DataSet fromYaml = DataSets.loadDataSet(structured, List.of(loader)).orElseThrow().getDataSet();
        assertEquals(3, ((LruCachedDataSet) fromYaml).getBaseRavv().grainSize());

        assertThrows(IllegalArgumentException.class, () -> DataSets.loadDataSet("test-ds(lru[grain=0])", List.of(loader)));
        assertThrows(IllegalArgumentException.class, () -> DataSets.loadDataSet("test-ds(lru[grain=many])", List.of(loader)));
        assertThrows(IllegalArgumentException.class, () -> DataSets.loadDataSet("test-ds(lru[pages=4])", List.of(loader)));
        assertThrows(IllegalArgumentException.class, () -> DataSets.loadDataSet("test-ds(memory[grain=4])", List.of(loader)));
        assertThrows(IllegalArgumentException.class, () -> DataSets.loadDataSet("test-ds(mmap[dir=x])", List.of(loader)));
        assertSame(LruCachedDataSet.PROVIDER, LruCachedDataSet.provider(Map.of()), "no options yields the shared default provider");
    }

    @Test
    public void explicitProvidersReplaceTheDefaults() {
        var loader = new HeapLoader();
        DataSet raw = DataSets.loadDataSet("test-ds", List.of(loader), List.of()).orElseThrow().getDataSet();
        assertTrue(raw instanceof SimpleDataSet);

        var applied = new ArrayList<String>();
        DataSetWrapper.Provider first = origin -> { applied.add("first"); return InMemoryCachedDataSet.of(origin); };
        DataSetWrapper.Provider second = origin -> { applied.add("second"); return InMemoryCachedDataSet.of(origin); };
        DataSet wrapped = DataSets.loadDataSet("test-ds", List.of(loader), List.of(first, second)).orElseThrow().getDataSet();
        assertEquals(List.of("first", "second"), applied);
        assertTrue(wrapped instanceof InMemoryCachedDataSet);

        assertThrows(IllegalArgumentException.class,
                () -> DataSets.loadDataSet("test-ds(memory)", List.of(loader), List.of(first)));
    }

    @Test
    public void profileRuleForLoadersWithoutProfileSupport() {
        var loader = new HeapLoader();
        assertTrue(DataSets.loadDataSet("test-ds:default", List.of(loader)).isPresent());
        assertThrows(IllegalArgumentException.class, () -> DataSets.loadDataSet("test-ds:fast", List.of(loader)));
        // a non-default profile for a dataset this loader does not have is not this loader's error
        assertTrue(DataSets.loadDataSet("other-ds:fast", List.of(loader)).isEmpty());
    }

    @Test
    public void profileAwareLoadersSeeTheProfile() {
        var loader = new ProfileLoader();
        assertTrue(DataSets.loadDataSet("test-ds:fast(memory)", List.of(loader)).isPresent());
        assertEquals("fast", loader.profileSeen);
        assertTrue(DataSets.loadDataSet("test-ds", List.of(loader)).isPresent());
        assertEquals(DataSetSpec.DEFAULT_PROFILE, loader.profileSeen);
    }

    @Test
    public void unknownDatasetIsEmptyAndHdf5NamesAreRejected() {
        assertTrue(DataSets.loadDataSet("nope", List.of(new HeapLoader())).isEmpty());
        assertThrows(java.security.InvalidParameterException.class,
                () -> DataSets.loadDataSet("test-ds.hdf5", List.of(new HeapLoader())));
    }
}
