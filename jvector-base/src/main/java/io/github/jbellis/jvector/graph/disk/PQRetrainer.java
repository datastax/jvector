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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
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
    // Number of consecutive nodes to read per chunk before jumping to another location.
    // Keeping reads sequential within each chunk lets the OS read-ahead cover them,
    // avoiding the random I/O that would happen with per-node random sampling.
    private static final int SAMPLE_CHUNK_SIZE = 32;

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
        return retrain(similarityFunction, PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    /**
     * As {@link #retrain(VectorSimilarityFunction)}, but runs the PQ refinement on the supplied
     * pools instead of the default {@link PhysicalCoreExecutor#pool()} / {@code commonPool()} — so a
     * compaction that supplies its own bounded pool keeps all retrain work on it rather than leaking
     * to an all-core pool.
     */
    public ProductQuantization retrain(VectorSimilarityFunction similarityFunction, ForkJoinPool simdExecutor, ForkJoinPool parallelExecutor) {
        FusedPQ fpq = (FusedPQ) sources.get(0).getFeatures().get(FeatureId.FUSED_PQ);
        return retrain(similarityFunction, fpq.getPQ(), simdExecutor, parallelExecutor);
    }

    /**
     * Trains a new Product Quantization codebook using balanced sampling across all source indexes
     * and the supplied base PQ for subspace/cluster parameters. Used when the base PQ comes from a
     * non-fused source (e.g. a sidecar {@code CompressedVectors}) rather than the FUSED_PQ feature.
     */
    public ProductQuantization retrain(VectorSimilarityFunction similarityFunction, ProductQuantization basePQ) {
        return retrain(similarityFunction, basePQ, PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    /**
     * As {@link #retrain(VectorSimilarityFunction, ProductQuantization)}, but runs the PQ refinement
     * on the supplied pools. This is the overload that keeps a pool-bounded compaction from leaking
     * PQ-retrain work to {@link PhysicalCoreExecutor#pool()} / {@code commonPool()}.
     */
    public ProductQuantization retrain(VectorSimilarityFunction similarityFunction, ProductQuantization basePQ, ForkJoinPool simdExecutor, ForkJoinPool parallelExecutor) {
        log.info("Training PQ using balanced sampling across sources");

        List<SampleRef> samples = sampleBalanced(ProductQuantization.MAX_PQ_TRAINING_SET_SIZE);

        // Sort by (source, node) so extractVectorsSequential reads each source's file
        // in ascending order, enabling OS read-ahead instead of random page faults.
        samples.sort(Comparator.comparingInt((SampleRef r) -> r.source).thenComparingInt(r -> r.node));

        log.info("Collected {} training samples", samples.size());

        List<VectorFloat<?>> trainingVectors = extractVectorsSequential(samples);

        long t0 = System.nanoTime();
        log.info("Extracted {} vectors in {}ms; starting PQ refinement",
                 trainingVectors.size(), (System.nanoTime() - t0) / 1_000_000L);

        var ravv = new ListRandomAccessVectorValues(trainingVectors, dimension);

        // Warm-start from the existing codebook via Lloyd's-only refinement rather than
        // re-running k-means++ from scratch. k-means++ initialization visits every point
        // once per centroid (256 passes for k=256), which dominates training time.
        // Since the source codebooks are already trained on data from the same underlying
        // distribution, this warm-start converges in far fewer passes with no recall loss.
        long t1 = System.nanoTime();
        ProductQuantization result = basePQ.refine(ravv,
                                                   ProductQuantization.K_MEANS_ITERATIONS,
                                                   -1.0f, // UNWEIGHTED / isotropic
                                                   simdExecutor,
                                                   parallelExecutor);
        log.info("PQ refinement complete in {}ms", (System.nanoTime() - t1) / 1_000_000L);
        return result;
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
     * by (source, node) so reads within each source are ascending, letting the OS read-ahead
     * cover them efficiently. Each source's view is opened once and reused for all its samples.
     */
    private List<VectorFloat<?>> extractVectorsSequential(List<SampleRef> samples) {
        prefetchSampleRanges(samples);

        OnDiskGraphIndex.View[] views = new OnDiskGraphIndex.View[sources.size()];
        for (int s = 0; s < sources.size(); s++) {
            views[s] = (OnDiskGraphIndex.View) sources.get(s).getView();
        }

        List<VectorFloat<?>> vectors = new ArrayList<>(samples.size());
        VectorFloat<?> tmp = vectorTypeSupport.createFloatVector(dimension);
        for (SampleRef ref : samples) {
            views[ref.source].getVectorInto(ref.node, tmp, 0);
            vectors.add(tmp.copy());
        }
        return vectors;
    }

    // Merging samples closer than this many ordinals into one range keeps the prefetch mostly
    // sequential without dragging in long runs of unsampled records.
    private static final int PREFETCH_MERGE_GAP = 256;

    /**
     * Streams the records of the (source, node)-sorted sample list into the page cache before
     * extraction. The mappings advise {@code MADV_RANDOM}, so without this the extraction loop
     * faults one page at a time on a cold cache; total demand is bounded by the training-set
     * size, not the file size.
     */
    private void prefetchSampleRanges(List<SampleRef> samples) {
        int i = 0;
        while (i < samples.size()) {
            SampleRef first = samples.get(i);
            int last = first.node;
            int j = i + 1;
            while (j < samples.size()
                    && samples.get(j).source == first.source
                    && samples.get(j).node - last <= PREFETCH_MERGE_GAP) {
                last = samples.get(j).node;
                j++;
            }
            sources.get(first.source).prefetchL0Records(first.node, last);
            i = j;
        }
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
