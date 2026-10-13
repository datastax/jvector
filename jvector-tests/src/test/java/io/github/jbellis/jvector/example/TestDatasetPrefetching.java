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

import io.github.jbellis.jvector.example.benchmarks.datasets.*;
import io.github.jbellis.jvector.example.yaml.*;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.OnDiskGraphIndexCache;
import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.vector.*;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.Assert.*;

public class TestDatasetPrefetching {
    // Track the current versions so a format or schema bump doesn't break these tests.
    private static final String HEADER =
            "yamlSchemaVersion: 2\nonDiskIndexVersion: " + OnDiskGraphIndex.CURRENT_VERSION + "\n";

    private MultiConfig config(String build, String search, boolean fused, String rerank) {
        var config = new MultiConfig();
        config.construction = new ConstructionParameters();
        config.search = new SearchParameters();
        var bc = new Compression(); bc.type = build;
        var sc = new Compression(); sc.type = search;
        config.construction.compression = List.of(bc);
        config.construction.fusedGraph = List.of(fused);
        config.construction.reranking = List.of(rerank);
        config.search.compression = List.of(sc);
        return config;
    }
    @Test public void yamlDefaultsToResidentAndOptInIsParsed() throws Exception {
        var file = Files.createTempFile("prefetch-flag", ".yml");
        try {
            Files.writeString(file, HEADER);
            assertFalse(MultiConfig.getConfig(file.toString()).useDatasetPrefetching);
            Files.writeString(file, HEADER + "useDatasetPrefetching: false\n");
            assertFalse(MultiConfig.getConfig(file.toString()).useDatasetPrefetching);
            Files.writeString(file, HEADER + "useDatasetPrefetching: true\n");
            assertTrue(MultiConfig.getConfig(file.toString()).useDatasetPrefetching);
        } finally { Files.delete(file); }
    }
    @Test public void datasetPrefetchingAndCompactionOptionsAreIndependent() throws Exception {
        var file = Files.createTempFile("prefetch-compaction-flags", ".yml");
        try {
            String common = HEADER + "useDatasetPrefetching: true\n";
            Files.writeString(file, common);
            var config = MultiConfig.getConfig(file.toString());
            assertTrue(config.useDatasetPrefetching);
            assertFalse(config.isCompactionEnabled());
            Files.writeString(file, common + "withCompaction: [Yes]\n");
            config = MultiConfig.getConfig(file.toString());
            assertTrue(config.useDatasetPrefetching);
            assertTrue(config.isCompactionEnabled());
            Files.writeString(file, common + "withCompaction: [No]\n");
            config = MultiConfig.getConfig(file.toString());
            assertTrue(config.useDatasetPrefetching);
            assertFalse(config.isCompactionEnabled());
        } finally { Files.delete(file); }
    }
    @Test public void versionedYamlRejectsUnknownSettings() throws Exception {
        var file = Files.createTempFile("prefetch-unknown-setting", ".yml");
        try {
            Files.writeString(file, HEADER + "useDatasetPrefetching: false\nunknownSetting: true\n");
            assertThrows(org.yaml.snakeyaml.error.YAMLException.class,
                    () -> MultiConfig.getConfig(file.toString()));
        } finally { Files.delete(file); }
    }
    @Test public void sourceRoutingPreservesRawComparisonWorkloadsWithoutRejectingThem() {
        String previous = System.getProperty("jvector.dataset_loader");
        try {
            System.clearProperty("jvector.dataset_loader");
            var c = config("PQ", "PQ", true, "NVQ");
            assertEquals(BaseVectorLoading.PRELOAD, BaseVectorLoading.forBenchmark(c));
            c.useDatasetPrefetching = true;
            assertEquals(BaseVectorLoading.BUFFERED, BaseVectorLoading.forBenchmark(c));
            c.construction.compression.get(0).type = "None";
            assertEquals(BaseVectorLoading.PRELOAD, BaseVectorLoading.forBenchmark(c));
            c = config("PQ", "None", false, "NVQ"); c.useDatasetPrefetching = true;
            assertEquals(BaseVectorLoading.PRELOAD, BaseVectorLoading.forBenchmark(c));
            c.construction.reranking = List.of("FP");
            assertEquals(BaseVectorLoading.PRELOAD, BaseVectorLoading.forBenchmark(c));
            System.setProperty("jvector.dataset_loader", "preload");
            assertEquals(BaseVectorLoading.PRELOAD, BaseVectorLoading.forBenchmark(c));
        } finally {
            if (previous == null) System.clearProperty("jvector.dataset_loader");
            else System.setProperty("jvector.dataset_loader", previous);
        }
    }
    @Test(timeout=60000) public void gridFeedsOriginalBulkVectorsToInsertionAndBothStorageFormats() throws Exception {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var vectors = new ArrayList<VectorFloat<?>>(); var random = new Random(42);
        for (int n=0;n<128;n++) {
            var vector = vts.createFloatVector(17);
            for (int d=0;d<17;d++) vector.set(d,random.nextFloat()-0.5f);
            VectorUtil.l2normalize(vector); vectors.add(vector);
        }
        var resident = new ListRandomAccessVectorValues(vectors,17);
        var ranges = new AtomicInteger(); var selections = new AtomicInteger();
        BatchedVectorValues input = new BatchedVectorValues() {
            public int size(){return 128;}
            public int dimension(){return 17;}
            public boolean isValueShared(){return false;}
            public RandomAccessVectorValues copy(){throw new AssertionError("Unshared source copied");}
            public VectorFloat<?> getVector(int ordinal){throw new AssertionError("Grid reread a point instead of consuming its input batch");}
            public VectorCursor openRange(int a,int b){ranges.incrementAndGet();return VectorAccess.openRange(resident,a,b);}
            public VectorCursor openSelection(int[] ordinals,int a,int n){selections.incrementAndGet();return VectorAccess.openSelection(resident,ordinals,a,n);}
        };
        // A previous test run must not supply this run's training codebook.
        var datasetName = "bulk-wiring-" + UUID.randomUUID();
        DataSet ds = new DataSet() {
            public boolean useDatasetPrefetching(){return true;}
            public int getDimension(){return 17;}
            public String getName(){return datasetName;}
            public VectorSimilarityFunction getSimilarityFunction(){return VectorSimilarityFunction.DOT_PRODUCT;}
            public RandomAccessVectorValues getBaseRavv(){return input;}
            public List<VectorFloat<?>> getBaseVectors(){return vectors;}
            public List<VectorFloat<?>> getQueryVectors(){return List.of(vectors.get(0));}
            public List<List<Integer>> getGroundTruth(){return List.of(List.of(0));}
        };
        var cacheFile = Path.of("pq_cache").resolve(new CompressorParameters.PQParameters(4,16,false,-1f).idStringFor(ds));
        var work = Files.createTempDirectory("bulk-grid-wiring");
        try {
            Grid.runOneGraph(OnDiskGraphIndexCache.initialize(work.resolve("saved")), List.of(EnumSet.of(FeatureId.INLINE_VECTORS),EnumSet.of(FeatureId.NVQ_VECTORS)),
                    8,32,1.2f,true,false, ignored -> new CompressorParameters.PQParameters(4,16,false,-1f),
                    List.of(),Map.of(10,List.of(1.0)),List.of(false),RunArtifacts.disabled(),ds,work);
            assertTrue(selections.get()>0); assertTrue(ranges.get()>0);
            int seen=0;
            try(var files=Files.list(work.resolve("saved"))) {
                for(var file:files.collect(java.util.stream.Collectors.toList())) {
                    try(var graph=OnDiskGraphIndex.load(ReaderSupplierFactory.open(file));var view=graph.getView()) {
                        assertEquals(128,graph.size(0));
                        if(graph.getFeatures().containsKey(FeatureId.INLINE_VECTORS)) {
                            var stored=(RandomAccessVectorValues)view;
                            for(int n=0;n<128;n++) {
                                var actual=stored.getVector(n);
                                for(int d=0;d<17;d++)assertEquals(vectors.get(n).get(d),actual.get(d),0f);
                            }
                        } else assertTrue(graph.getFeatures().containsKey(FeatureId.NVQ_VECTORS));
                        seen++;
                    }
                }
            }
            assertEquals(2,seen);
        } finally {
            try { Files.deleteIfExists(cacheFile); }
            finally {
                try(var paths=Files.walk(work)){for(var path:paths.sorted(Comparator.reverseOrder()).collect(java.util.stream.Collectors.toList()))Files.delete(path);}
            }
        }
    }
}
