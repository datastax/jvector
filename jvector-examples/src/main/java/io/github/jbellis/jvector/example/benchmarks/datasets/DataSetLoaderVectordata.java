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
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.nosqlbench.nbdatatools.api.concurrent.ProgressIndicatingFuture;
import io.nosqlbench.vectordata.VectorTestData;
import io.nosqlbench.vectordata.discovery.ProfileSelector;
import io.nosqlbench.vectordata.discovery.TestDataSources;
import io.nosqlbench.vectordata.discovery.vector.VectorTestDataView;
import io.nosqlbench.vectordata.downloader.Catalog;
import io.nosqlbench.vectordata.downloader.DatasetEntry;
import io.nosqlbench.vectordata.downloader.DatasetProfileSpec;
import io.nosqlbench.vectordata.spec.datasets.types.BaseVectors;
import io.nosqlbench.vectordata.spec.datasets.types.DistanceFunction;
import io.nosqlbench.vectordata.spec.datasets.types.NeighborIndices;
import io.nosqlbench.vectordata.spec.datasets.types.QueryVectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
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
    public Optional<DataSetInfo> loadDataSet(String dataSetName) {
        try {
            logger.info("Attempting to load dataset '{}' via vectordata", dataSetName);

            DatasetProfileSpec spec = DatasetProfileSpec.parse(dataSetName);

            TestDataSources tds1 = VectorTestData.catalogs();
            TestDataSources tds2 = tds1.configure();
            Catalog c1 = tds2.catalog();
            Optional<DatasetEntry> entryOpt = c1.findExact(spec.dataset());

            Optional<VectorTestDataView> viewOption = Optional.empty();
            if (entryOpt.isPresent()) {
                DatasetEntry entry = entryOpt.get();
                logger.info("Found dataset '{}' in catalog. URL: {}", spec.dataset(), entry.url());
                ProfileSelector selector = entry.select();
                viewOption = spec.profile().map(selector::profile);
            }
            if (viewOption.isEmpty()) {
                return Optional.empty();
            }
            VectorTestDataView view = viewOption.get();

            if (prebuffer) {
                logger.info("Prebuffering dataset '{}'...", dataSetName);
                CompletableFuture<Void> f = view.prebuffer();
                if (f instanceof ProgressIndicatingFuture) {
                    ((ProgressIndicatingFuture<?>) f).monitorProgress(System.out, 5000);
                }
                f.get();
            }

            return Optional.of(new DataSetInfo(new VectorDataSetProperties(view), () -> new VectordataDataSet(view)));

        } catch (Exception e) {
            logger.error("Error loading dataset '{}' via vectordata", dataSetName, e);
            return Optional.empty();
        }
    }

    private static class VectorDataSetProperties implements DataSetProperties {

        private final VectorTestDataView view;
        public VectorDataSetProperties(VectorTestDataView view) {
            this.view = view;
        }

        @Override
        public Optional<VectorSimilarityFunction> similarityFunction() {
            return Optional.ofNullable(view.getDistanceFunction())
                    .map(df -> {
                        switch(df) {
                            case COSINE:
                                return VectorSimilarityFunction.COSINE;
                            case DOT_PRODUCT:
                                return VectorSimilarityFunction.DOT_PRODUCT;
                            case EUCLIDEAN:
                            case L2:
                                return VectorSimilarityFunction.EUCLIDEAN;
                            default:
                                throw new RuntimeException("Unknown distance function: " + df);
                        }
                    });
        }

        @Override
        public int numVectors() {
            return view.getBaseVectors().map(bv -> bv.getCount()).orElseThrow(
                    () -> new RuntimeException("count wasn't defined for base vectors in dataset " + getName()));
        }

        @Override
        public String getName() {
            return view.getName();
        }

        @Override
        public boolean isNormalized() {
            return view.isNormalized().orElse(false);
        }

        @Override
        public boolean isZeroVectorFree() {
            return view.isZeroVectorFree().orElse(false);
        }

        @Override
        public boolean isDuplicateVectorFree() {
            return view.isDuplicateVectorFree().orElse(false);
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

    public static class VectordataDataSet implements DataSet {
        private final VectorTestDataView datasetView;
        private RandomAccessVectorValues baseRavv;
        private List<VectorFloat<?>> baseVectors;
        private List<VectorFloat<?>> queryVectors;
        private List<List<Integer>> groundTruthIndices;

        public VectordataDataSet(VectorTestDataView view) {
            this.datasetView = view;
//            System.out.format("%n%s: %d base and %d query vectors loaded via Vectordata, dimensions %d%n",
//                    name, baseVectors.size(), queryVectors.size(), dimension);
        }

        @Override
        public int getDimension() {
            return datasetView.getBaseVectors().orElseThrow(
                    () -> new RuntimeException("Unable to get base vectors for " + datasetView.getName())
            ).getVectorDimensions();
        }

        @Override
        public synchronized RandomAccessVectorValues getBaseRavv() {
            if (baseRavv==null) {
                int vectorDimensions = datasetView.getBaseVectors().get().getVectorDimensions();
                this.baseRavv = new ListRandomAccessVectorValues(this.getBaseVectors(),vectorDimensions);
            }
            return baseRavv;
        }

        @Override
        public String getName() {
            return datasetView.getName();
        }

        @Override
        public VectorSimilarityFunction getSimilarityFunction() {
            switch(datasetView.getDistanceFunction()) {
                case COSINE: return VectorSimilarityFunction.COSINE;
                case DOT_PRODUCT: return VectorSimilarityFunction.DOT_PRODUCT;
                case L2:
                case EUCLIDEAN: return VectorSimilarityFunction.EUCLIDEAN;
                default: throw new RuntimeException("Unknown distance function: " + datasetView.getDistanceFunction());
            }
        }

        @Override
        public synchronized List<VectorFloat<?>> getBaseVectors() {
            if (this.baseVectors==null) {
                BaseVectors baseVectorsView = datasetView.getBaseVectors().get();
                this.baseVectors = new ArrayList<>(baseVectorsView.getCount());
                baseVectorsView.forEach(floatAry -> this.baseVectors.add(vts.createFloatVector(floatAry)));
            }
            return baseVectors;
       }

        @Override
        public synchronized List<VectorFloat<?>> getQueryVectors() {
            if (this.queryVectors==null) {
                QueryVectors queryVectorsView = datasetView.getQueryVectors().get();
                this.queryVectors = new ArrayList<>(queryVectorsView.getCount());
                queryVectorsView.forEach(floatAry -> this.queryVectors.add(vts.createFloatVector(floatAry)));
            }
            return queryVectors;
        }

        @Override
        public synchronized List<? extends List<Integer>> getGroundTruth() {
            if (this.groundTruthIndices == null) {
                NeighborIndices neighborIndices = datasetView.getNeighborIndices().get();
                this.groundTruthIndices = new ArrayList<>();
                for (int[] neighborIndex : neighborIndices) {
                    List<Integer> ordinalIndex = Arrays.stream(neighborIndex).mapToObj(Integer::new).collect(Collectors.toList());
                    groundTruthIndices.add(ordinalIndex);
                }
            }
            return this.groundTruthIndices;
        }
    }

}
