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

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.nosqlbench.nbdatatools.api.concurrent.ProgressIndicatingFuture;
import io.nosqlbench.vectordata.VectorTestData;
import io.nosqlbench.vectordata.discovery.ProfileSelector;
import io.nosqlbench.vectordata.discovery.TestDataGroup;
import io.nosqlbench.vectordata.discovery.TestDataSources;
import io.nosqlbench.vectordata.discovery.TestDataView;
import io.nosqlbench.vectordata.downloader.Catalog;
import io.nosqlbench.vectordata.downloader.DatasetEntry;
import io.nosqlbench.vectordata.downloader.DatasetProfileSpec;
import io.nosqlbench.vectordata.spec.datasets.types.DistanceFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.function.LongFunction;
import java.util.stream.Collectors;

/**
 * A DataSetLoader that uses the io.nosqlbench.datatools-vectordata library to load datasets.
 */
public class DataSetLoaderVectordata implements DataSetLoader {
    private static final Logger logger = LoggerFactory.getLogger(DataSetLoaderVectordata.class);
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final boolean prebuffer;

    public DataSetLoaderVectordata() {
        this(false);
    }

    public DataSetLoaderVectordata(boolean prebuffer) {
        this.prebuffer = prebuffer;
    }

    @Override
    public String getName() {
        return "VECTORDATA";
    }

    @Override
    public Optional<DataSet> loadDataSet(String dataSetName) {
        try {
            logger.info("Attempting to load dataset '{}' via vectordata", dataSetName);

            DatasetProfileSpec spec = DatasetProfileSpec.parse(dataSetName);

            TestDataSources tds1 = VectorTestData.catalogs();
            TestDataSources tds2 = tds1.configure();
            Catalog c1 = tds2.catalog();
            Optional<DatasetEntry> entryOpt = c1.findExact(spec.dataset());

            TestDataView view;
            if (entryOpt.isPresent()) {
                DatasetEntry entry = entryOpt.get();
                logger.info("Found dataset '{}' in catalog. URL: {}", spec.dataset(), entry.url());
                ProfileSelector selector = entry.select();
                view = spec.profile().map(selector::profile).orElseGet(selector::profile);
            } else {
                // Fallback to local load
                logger.debug("Dataset '{}' not found in catalog, attempting local load", spec.dataset());
                try {
                    TestDataGroup group = VectorTestData.load(spec.dataset());
                    view = spec.profile().map(group::profile).orElseGet(group::getDefaultProfile);
                } catch (Exception e) {
                    logger.debug("Local load failed for '{}'", spec.dataset());
                    return Optional.empty();
                }
            }

            if (view == null) {
                return Optional.empty();
            }

            if (prebuffer) {
                logger.info("Prebuffering dataset '{}'...", dataSetName);
                CompletableFuture<Void> f = view.prebuffer();
                if (f instanceof ProgressIndicatingFuture) {
                    System.out.println("blocking until prebuffer completes, with progress reporting...");
                    ((ProgressIndicatingFuture<?>) f).monitorProgress(System.out, 5000);
                } else {
                    System.out.println("blocking until prebuffer completes...");
                }
                f.get();
                // Block until data is ready/cached
            }

            return Optional.of(new VectordataDataSet(dataSetName, mapDistanceFunction(view.getDistanceFunction()), view));
        } catch (Exception e) {
            logger.error("Error loading dataset '{}' via vectordata", dataSetName, e);
            return Optional.empty();
        }
    }

    private static class VectordataRavv implements RandomAccessVectorValues {
        private final int dimension;
        private final int size;
        private final LongFunction<float[]> getter;

        public VectordataRavv(int dimension, int size, LongFunction<float[]> getter) {
            this.dimension = dimension;
            this.size = size;
            this.getter = getter;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public VectorFloat<?> getVector(int nodeId) {
            return vts.createFloatVector(getter.apply((long) nodeId));
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            return this;
        }
    }

    private static class VectordataDataSet implements DataSet {
        private final String name;
        private final VectorSimilarityFunction vsf;
        private final int dimension;
        private final RandomAccessVectorValues baseRavv;
        private final List<VectorFloat<?>> baseVectors;
        private final List<VectorFloat<?>> queryVectors;
        private final List<? extends List<Integer>> groundTruth;

        public VectordataDataSet(String name, VectorSimilarityFunction vsf, TestDataView view) {
            this.name = name;
            this.vsf = vsf;

            var bv = view.getBaseVectors().orElseThrow(() -> new RuntimeException("Base vectors missing in dataset " + name));
            int bSize = (int) bv.getCount();
            float[] firstVector = bv.get(0L);
            this.dimension = firstVector.length;

            this.baseRavv = new VectordataRavv(dimension, bSize, bv::get);
            this.baseVectors = new AbstractList<VectorFloat<?>>() {
                @Override
                public VectorFloat<?> get(int index) {
                    return vts.createFloatVector(bv.get((long) index));
                }

                @Override
                public int size() {
                    return bSize;
                }
            };

            var qv = view.getQueryVectors().orElseThrow(() -> new RuntimeException("Query vectors missing in dataset " + name));
            int qSize = (int) qv.getCount();
            this.queryVectors = new AbstractList<VectorFloat<?>>() {
                @Override
                public VectorFloat<?> get(int index) {
                    return vts.createFloatVector(qv.get((long) index));
                }

                @Override
                public int size() {
                    return qSize;
                }
            };

            var niOpt = view.getNeighborIndices();
            if (niOpt.isPresent()) {
                var ni = niOpt.get();
                int niSize = (int) ni.getCount();
                this.groundTruth = new AbstractList<List<Integer>>() {
                    @Override
                    public List<Integer> get(int index) {
                        int[] indices = ni.get((long) index);
                        return new AbstractList<Integer>() {
                            @Override
                            public Integer get(int i) {
                                return indices[i];
                            }

                            @Override
                            public int size() {
                                return indices.length;
                            }
                        };
                    }

                    @Override
                    public int size() {
                        return niSize;
                    }
                };
            } else {
                logger.warn("Ground truth missing in dataset {}, recall metrics will not be available", name);
                this.groundTruth = Collections.nCopies(queryVectors.size(), Collections.emptyList());
            }

            System.out.format("%n%s: %d base and %d query vectors loaded via Vectordata, dimensions %d%n",
                    name, baseVectors.size(), queryVectors.size(), dimension);
        }

        @Override
        public int getDimension() {
            return dimension;
        }

        @Override
        public RandomAccessVectorValues getBaseRavv() {
            return baseRavv;
        }

        @Override
        public String getName() {
            return name;
        }

        @Override
        public VectorSimilarityFunction getSimilarityFunction() {
            return vsf;
        }

        @Override
        public List<VectorFloat<?>> getBaseVectors() {
            return baseVectors;
        }

        @Override
        public List<VectorFloat<?>> getQueryVectors() {
            return queryVectors;
        }

        @Override
        public List<? extends List<Integer>> getGroundTruth() {
            return groundTruth;
        }
    }

    private VectorSimilarityFunction mapDistanceFunction(DistanceFunction df) {
        if (df == null) return VectorSimilarityFunction.COSINE;
        switch (df) {
            case COSINE:
                return VectorSimilarityFunction.COSINE;
            case DOT_PRODUCT:
                return VectorSimilarityFunction.DOT_PRODUCT;
            case EUCLIDEAN:
            case L2:
                return VectorSimilarityFunction.EUCLIDEAN;
            default:
                logger.warn("Unknown distance function {}, defaulting to COSINE", df);
                return VectorSimilarityFunction.COSINE;
        }
    }
}
