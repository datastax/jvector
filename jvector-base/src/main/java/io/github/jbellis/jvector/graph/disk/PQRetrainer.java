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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

/**
 * Handles Product Quantization retraining for graph index compaction.
 * Performs balanced sampling across multiple source indexes and trains
 * a new PQ codebook optimized for the combined dataset.
 */
public class PQRetrainer {
    private static final Logger log = LoggerFactory.getLogger(PQRetrainer.class);
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int MIN_SAMPLES_PER_SOURCE = 1000;
    // Number of consecutive nodes to read per chunk before jumping to another location.
    // Keeping reads sequential within each chunk lets the OS read-ahead cover them,
    // avoiding the random I/O that would happen with per-node random sampling.
    private static final int SAMPLE_CHUNK_SIZE = 32;
    private static final int MIN_PARALLEL_EXTRACT_CHUNK = 1024;

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
     * The base PQ parameters are taken from the FUSED_PQ feature on the first source.
     * All sampled vectors are read into memory up front, so ProductQuantization.compute() itself
     * performs no I/O.
     */
    public ProductQuantization retrain(VectorSimilarityFunction similarityFunction) {
        FusedPQ fpq = (FusedPQ) sources.get(0).getFeatures().get(FeatureId.FUSED_PQ);
        return retrain(similarityFunction, fpq.getPQ());
    }

    /**
     * Trains a new Product Quantization codebook using balanced sampling across all source indexes
     * and the supplied base PQ for subspace/cluster parameters. Used when the base PQ comes from a
     * non-fused source (e.g. a sidecar {@code CompressedVectors}) rather than the FUSED_PQ feature.
     */
    public ProductQuantization retrain(VectorSimilarityFunction similarityFunction, ProductQuantization basePQ) {
        log.info("Training PQ using balanced sampling across sources");

        List<SampleRef> samples = sampleBalanced(ProductQuantization.MAX_PQ_TRAINING_SET_SIZE);

        // Sort by (source, node) so each extraction task reads its source's file in ascending
        // order, enabling OS read-ahead instead of random page faults.
        samples.sort(Comparator.comparingInt((SampleRef r) -> r.source).thenComparingInt(r -> r.node));

        log.info("Collected {} training samples", samples.size());

        List<VectorFloat<?>> trainingVectors = extractSampledVectors(samples);
        var ravv = new ListRandomAccessVectorValues(trainingVectors, dimension);

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
            int numChunks = (max + SAMPLE_CHUNK_SIZE - 1) / SAMPLE_CHUNK_SIZE;

            // Build a shuffled chunk order so samples are representative but
            // each chunk is read sequentially to minimize page faults.
            // Fisher-Yates shuffle
            int[] chunkOrder = new int[numChunks];
            for (int i = 0; i < numChunks; i++) chunkOrder[i] = i;
            for (int i = numChunks - 1; i > 0; i--) {
                int j = rand.nextInt(i + 1);
                int tmp = chunkOrder[i];
                chunkOrder[i] = chunkOrder[j];
                chunkOrder[j] = tmp;
            }

            int count = 0;
            outer:
            for (int ci = 0; ci < numChunks; ci++) {
                int start = chunkOrder[ci] * SAMPLE_CHUNK_SIZE;
                int end = Math.min(max, start + SAMPLE_CHUNK_SIZE);
                for (int node = start; node < end; node++) {
                    if (live.get(node)) {
                        samples.add(new SampleRef(s, node));
                        if (++count >= quota[s]) break outer;
                    }
                }
            }
        }

        return samples;
    }

    /**
     * Reads sampled vectors in the order provided. The caller must pre-sort {@code samples}
     * by (source, node). The sorted list is split into contiguous ranges that are read
     * concurrently across {@link PhysicalCoreExecutor#pool()}, so the scattered page faults
     * that dominate single-threaded extraction are overlapped, while each range still reads a
     * contiguous run so OS read-ahead applies within it. Each task uses its own per-source
     * {@code View} (and reader) and closes them when done.
     */
    private List<VectorFloat<?>> extractSampledVectors(List<SampleRef> samples) {
        int n = samples.size();
        VectorFloat<?>[] out = new VectorFloat<?>[n];
        if (n == 0) {
            return Arrays.asList(out);
        }

        int parallelism = Math.max(1, PhysicalCoreExecutor.getPhysicalCoreCount());
        int numTasks = Math.max(1, Math.min(parallelism, n / MIN_PARALLEL_EXTRACT_CHUNK));
        int chunk = (n + numTasks - 1) / numTasks;

        ForkJoinPool pool = PhysicalCoreExecutor.pool();
        final int taskCount = numTasks;
        final int chunkSize = chunk;
        pool.submit(() -> IntStream.range(0, taskCount).parallel().forEach(t -> {
            int start = t * chunkSize;
            if (start >= n) {
                return;
            }
            int end = Math.min(n, start + chunkSize);
            OnDiskGraphIndex.View[] views = new OnDiskGraphIndex.View[sources.size()];
            VectorFloat<?> tmp = vectorTypeSupport.createFloatVector(dimension);
            try {
                for (int i = start; i < end; i++) {
                    SampleRef ref = samples.get(i);
                    OnDiskGraphIndex.View view = views[ref.source];
                    if (view == null) {
                        view = (OnDiskGraphIndex.View) sources.get(ref.source).getView();
                        views[ref.source] = view;
                    }
                    view.getVectorInto(ref.node, tmp, 0);
                    out[i] = tmp.copy();
                }
            } finally {
                for (OnDiskGraphIndex.View view : views) {
                    if (view != null) {
                        try {
                            view.close();
                        } catch (IOException e) {
                            log.warn("Failed to close source view during PQ sample extraction", e);
                        }
                    }
                }
            }
        })).join();

        return Arrays.asList(out);
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

}
