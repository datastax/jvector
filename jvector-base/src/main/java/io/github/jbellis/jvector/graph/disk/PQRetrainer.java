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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Handles Product Quantization retraining for graph index compaction.
 * Performs balanced sampling across multiple source indexes and trains
 * a new PQ codebook optimized for the combined dataset.
 */
public class PQRetrainer {
    private static final Logger log = LoggerFactory.getLogger(PQRetrainer.class);
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int MIN_SAMPLES_PER_SOURCE = 1000;

    private final List<OnDiskGraphIndex> sources;
    private final List<FixedBitSet> liveNodes;
    private final List<Integer> numLiveNodesPerSource;
    private final int dimension;
    private final int numTotalNodes;

    public PQRetrainer(List<OnDiskGraphIndex> sources, List<FixedBitSet> liveNodes, int dimension) {
        this.sources = sources;
        this.liveNodes = liveNodes;
        this.dimension = dimension;

        this.numLiveNodesPerSource = new ArrayList<>(sources.size());
        int total = 0;
        for (int s = 0; s < sources.size(); s++) {
            int numLiveNodes = liveNodes.get(s).cardinality();
            total += numLiveNodes;
            this.numLiveNodesPerSource.add(numLiveNodes);
        }
        this.numTotalNodes = total;
    }

    /**
     * Trains a new Product Quantization codebook using balanced sampling across all source indexes.
     * Note: ProductQuantization.compute() internally calls extractTrainingVectors() which prefetches
     * all vectors into memory, so random I/O only happens once during the initial extraction.
     */
    public ProductQuantization retrain(VectorSimilarityFunction similarityFunction) {
        log.info("Training PQ using balanced sampling across sources");

        List<SampleRef> samples = sampleBalanced(ProductQuantization.MAX_PQ_TRAINING_SET_SIZE);

        log.info("Collected {} training samples", samples.size());

        RandomAccessVectorValues ravv = new SampledRAVV(sources, samples, dimension);

        FusedPQ fpq = (FusedPQ) sources.get(0).getFeatures().get(FeatureId.FUSED_PQ);
        ProductQuantization basePQ = fpq.getPQ();

        boolean center = similarityFunction == VectorSimilarityFunction.EUCLIDEAN;

        return ProductQuantization.compute(
                ravv,
                basePQ.getSubspaceCount(),
                basePQ.getClusterCount(),
                center
        );
    }

    /**
     * Performs balanced sampling across all source indexes to ensure proportional representation.
     * Guarantees minimum samples per source while respecting total sample budget.
     */
    private List<SampleRef> sampleBalanced(int totalSamples) {
        // If total live nodes <= totalSamples, return ALL
        if (numTotalNodes <= totalSamples) {
            List<SampleRef> all = new ArrayList<>(numTotalNodes);

            for (int s = 0; s < sources.size(); s++) {
                FixedBitSet live = liveNodes.get(s);

                for (int node = live.nextSetBit(0);
                     node != DocIdSetIterator.NO_MORE_DOCS;
                     node = live.nextSetBit(node + 1)) {
                    all.add(new SampleRef(s, node));
                }
            }

            return all;
        }

        final int MIN_PER_SOURCE = Math.min(MIN_SAMPLES_PER_SOURCE, totalSamples / sources.size());

        int[] quota = new int[sources.size()];
        int assigned = 0;

        // Proportional allocation
        for (int s = 0; s < sources.size(); s++) {
            quota[s] = Math.max(
                    MIN_PER_SOURCE,
                    (int) ((long) totalSamples * numLiveNodesPerSource.get(s) / numTotalNodes)
            );
            assigned += quota[s];
        }

        // Normalize down
        while (assigned > totalSamples) {
            for (int s = 0; s < sources.size() && assigned > totalSamples; s++) {
                if (quota[s] > MIN_PER_SOURCE) {
                    quota[s]--;
                    assigned--;
                }
            }
        }

        // Normalize up
        while (assigned < totalSamples) {
            for (int s = 0; s < sources.size() && assigned < totalSamples; s++) {
                quota[s]++;
                assigned++;
            }
        }

        List<SampleRef> samples = new ArrayList<>(totalSamples);
        ThreadLocalRandom rand = ThreadLocalRandom.current();

        for (int s = 0; s < sources.size(); s++) {
            FixedBitSet live = liveNodes.get(s);
            int max = live.length();

            int count = 0;
            while (count < quota[s]) {
                int node = rand.nextInt(max);
                if (live.get(node)) {
                    samples.add(new SampleRef(s, node));
                    count++;
                }
            }
        }

        return samples;
    }

    /**
     * Reference to a sampled vector from a specific source index.
     */
    private static final class SampleRef {
        final int source;
        final int node;

        SampleRef(int source, int node) {
            this.source = source;
            this.node = node;
        }
    }

    /**
     * Random access vector values implementation backed by sampled references from source indexes.
     * Used for PQ training on a representative subset of the compacted data.
     *
     * Note: Random I/O only happens once when ProductQuantization.extractTrainingVectors()
     * prefetches all vectors into memory at the start of training.
     */
    private static final class SampledRAVV implements RandomAccessVectorValues {
        private final List<OnDiskGraphIndex> sources;
        private final List<SampleRef> samples;
        private final int dimension;

        SampledRAVV(List<OnDiskGraphIndex> sources, List<SampleRef> samples, int dimension) {
            this.sources = sources;
            this.samples = samples;
            this.dimension = dimension;
        }

        @Override
        public int size() {
            return samples.size();
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public VectorFloat<?> getVector(int nodeId) {
            VectorFloat<?> vec = vectorTypeSupport.createFloatVector(dimension);
            getVectorInto(nodeId, vec, 0);
            return vec;
        }

        @Override
        public void getVectorInto(int i, VectorFloat<?> dest, int offset) {
            SampleRef ref = samples.get(i);
            var view = (OnDiskGraphIndex.View) sources.get(ref.source).getView();
            view.getVectorInto(ref.node, dest, offset);
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            return null;
        }
    }
}
