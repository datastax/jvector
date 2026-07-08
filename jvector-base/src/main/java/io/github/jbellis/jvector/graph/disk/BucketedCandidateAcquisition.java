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

import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.locks.ReentrantLock;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer;
import io.github.jbellis.jvector.quantization.PQSymmetricDistanceTables;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Cross-source candidate acquisition for compaction via PQ-bucketed brute-force scanning, replacing
 * per-node cross-source graph searches at the base layer.
 * <p>
 * All live nodes (across all sources) are clustered into spatial buckets of roughly
 * {@code targetBucketSize} members using k-means over vectors decoded from the fused-PQ code cache;
 * each node is assigned to its 2 nearest buckets (overlapping assignment, so true neighbor pairs
 * straddling one bucket boundary still co-occur). Within each bucket, all cross-source pairs are
 * scored with symmetric PQ distance ({@link PQSymmetricDistanceTables}) — a sequential table-lookup
 * scan with no graph traversal and no float-vector reads. Every scored pair updates <em>both</em>
 * endpoints' top-{@code topT} candidate heaps, so reverse candidates ("u found v, therefore v knows
 * u") are structural rather than dependent on search luck.
 * <p>
 * Scores are approximate; callers are expected to exactly rescore candidates before diversity
 * selection. Build is parallel on the supplied executor; after {@link #build} returns, the
 * read-side accessors are immutable and thread-safe.
 */
@Experimental
final class BucketedCandidateAcquisition {
    private static final Logger log = LoggerFactory.getLogger(BucketedCandidateAcquisition.class);
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    static final int DEFAULT_TARGET_BUCKET_SIZE = 1024;
    private static final int KMEANS_ITERATIONS = 6;
    private static final int MAX_KMEANS_SAMPLE = 131072;
    /** Buckets larger than this get split once by re-clustering their members. */
    private static final int MAX_BUCKET_MULTIPLE = 4;
    /** Power of two. */
    private static final int LOCK_STRIPES = 1024;

    private final int topT;
    /** newOrdinal -> source index, -1 for holes in the remapped ordinal space. */
    private final int[] srcOfNewOrd;
    /** newOrdinal -> original ordinal within its source. */
    private final int[] oldOfNewOrd;
    /** Per-node bounded min-heaps, laid out as slices of length {@link #topT} at newOrdinal * topT. */
    private final int[] heapNodes;
    private final float[] heapScores;
    private final short[] heapSizes;
    private final ReentrantLock[] stripes;

    interface CandidateConsumer {
        void accept(int sourceIdx, int oldOrdinal, float approxScore);
    }

    /** Candidate budget per node; sized relative to the graph degree like the search path's topK. */
    static int candidatesPerNode(int baseDegree) {
        return Math.max(16, Math.min(64, baseDegree));
    }

    private BucketedCandidateAcquisition(int ordinalUpperBound, int topT) {
        this.topT = topT;
        this.srcOfNewOrd = new int[ordinalUpperBound];
        this.oldOfNewOrd = new int[ordinalUpperBound];
        this.heapNodes = new int[ordinalUpperBound * topT];
        this.heapScores = new float[ordinalUpperBound * topT];
        this.heapSizes = new short[ordinalUpperBound];
        this.stripes = new ReentrantLock[LOCK_STRIPES];
        for (int i = 0; i < LOCK_STRIPES; i++) {
            stripes[i] = new ReentrantLock();
        }
    }

    /**
     * Builds the acquisition structure: ordinal maps, bucket clustering + assignment, and the
     * pairwise scan. Blocks until the scan completes.
     *
     * @param codeCache fused-PQ pre-encode cache indexed by new ordinal (see
     *                  {@code FusedCompactionStrategy.precomputeCodes})
     * @param codeSize  bytes per code in the cache
     */
    static BucketedCandidateAcquisition build(CompactionContext ctx,
                                              ProductQuantization pq,
                                              MappedByteBuffer codeCache,
                                              int codeSize,
                                              VectorSimilarityFunction vsf,
                                              int topT,
                                              int targetBucketSize) {
        long t0 = System.nanoTime();
        var acq = new BucketedCandidateAcquisition(ctx.maxOrdinal + 1, topT);
        int[] liveNewOrds = acq.buildOrdinalMaps(ctx);
        var tables = new PQSymmetricDistanceTables(pq, vsf);

        List<int[]> buckets = acq.assignBuckets(ctx, pq, codeCache, codeSize, liveNewOrds, targetBucketSize);
        long tAssign = System.nanoTime();

        acq.scanBuckets(ctx, buckets, tables, codeCache, codeSize);
        long tScan = System.nanoTime();

        log.info("Bucketed acquisition: {} live nodes in {} buckets (target size {}), topT={}, " +
                        "SDC tables {} KB, heaps {} MB; assign {} ms, scan {} ms",
                liveNewOrds.length, buckets.size(), targetBucketSize, topT,
                tables.tableBytes() / 1024,
                ((long) acq.heapNodes.length * 8) / (1024 * 1024),
                (tAssign - t0) / 1_000_000, (tScan - tAssign) / 1_000_000);
        return acq;
    }

    /** Number of cross-source candidates collected for the given new ordinal. */
    int candidateCount(int newOrd) {
        return heapSizes[newOrd];
    }

    /**
     * Visits the collected candidates for {@code newOrd} in unspecified order. Rare duplicates are
     * possible (a pair sharing both overlapping buckets); callers dedupe. Only valid after
     * {@link #build} has returned; not synchronized.
     */
    void forEachCandidate(int newOrd, CandidateConsumer consumer) {
        int base = newOrd * topT;
        int size = heapSizes[newOrd];
        for (int i = 0; i < size; i++) {
            int cand = heapNodes[base + i];
            consumer.accept(srcOfNewOrd[cand], oldOfNewOrd[cand], heapScores[base + i]);
        }
    }

    /** Populates srcOfNewOrd/oldOfNewOrd and returns the sorted-by-construction list of live new ordinals. */
    private int[] buildOrdinalMaps(CompactionContext ctx) {
        java.util.Arrays.fill(srcOfNewOrd, -1);
        int live = 0;
        for (int s = 0; s < ctx.sources.size(); s++) {
            live += ctx.liveNodes.get(s).cardinality();
        }
        int[] liveNewOrds = new int[live];
        int n = 0;
        for (int s = 0; s < ctx.sources.size(); s++) {
            FixedBitSet alive = ctx.liveNodes.get(s);
            OrdinalMapper mapper = ctx.remappers.get(s);
            for (int oldOrd = 0; oldOrd < alive.length(); oldOrd++) {
                if (!alive.get(oldOrd)) {
                    continue;
                }
                int newOrd = mapper.oldToNew(oldOrd);
                srcOfNewOrd[newOrd] = s;
                oldOfNewOrd[newOrd] = oldOrd;
                liveNewOrds[n++] = newOrd;
            }
        }
        return liveNewOrds;
    }

    /**
     * Clusters live nodes into buckets and assigns each node to its 2 nearest. Returns per-bucket
     * member arrays (new ordinals). Falls back to a single bucket when the population is small.
     */
    private List<int[]> assignBuckets(CompactionContext ctx,
                                      ProductQuantization pq,
                                      MappedByteBuffer codeCache,
                                      int codeSize,
                                      int[] liveNewOrds,
                                      int targetBucketSize) {
        int numBuckets = (liveNewOrds.length + targetBucketSize - 1) / targetBucketSize;
        var result = new ArrayList<int[]>();
        if (numBuckets <= 1) {
            result.add(liveNewOrds);
            return result;
        }

        // Train two-level centroids on a decoded sample: B1 superclusters routing to ~B/B1
        // children each, so per-node assignment is O(B1 + 2*B/B1) instead of O(B).
        VectorFloat<?>[] sample = decodeSample(pq, codeCache, codeSize, liveNewOrds, ctx.dimension);
        int b1 = Math.max(1, Math.min((int) Math.ceil(Math.sqrt(numBuckets)), sample.length));
        VectorFloat<?> superCentroids = new KMeansPlusPlusClusterer(sample, b1).cluster(KMEANS_ITERATIONS, 0);

        // Partition the sample by nearest supercluster, then cluster each partition into children.
        List<List<VectorFloat<?>>> partitions = new ArrayList<>(b1);
        for (int i = 0; i < b1; i++) {
            partitions.add(new ArrayList<>());
        }
        for (VectorFloat<?> v : sample) {
            partitions.get(nearestCentroid(v, superCentroids, b1, ctx.dimension)).add(v);
        }
        int childrenPerSuper = (numBuckets + b1 - 1) / b1;
        VectorFloat<?>[] childCentroids = new VectorFloat<?>[b1];
        int[] childCounts = new int[b1];
        int[] bucketBase = new int[b1 + 1];
        for (int s = 0; s < b1; s++) {
            var part = partitions.get(s);
            if (part.isEmpty()) {
                // No sample landed here, but real nodes still can: use the supercentroid itself
                // as this super's single child so assignment stays total.
                var single = vectorTypeSupport.createFloatVector(ctx.dimension);
                single.copyFrom(superCentroids, s * ctx.dimension, 0, ctx.dimension);
                childCentroids[s] = single;
                childCounts[s] = 1;
            } else {
                int k = Math.max(1, Math.min(childrenPerSuper, part.size()));
                childCentroids[s] = new KMeansPlusPlusClusterer(part.toArray(new VectorFloat<?>[0]), k)
                        .cluster(KMEANS_ITERATIONS, 0);
                childCounts[s] = k;
            }
            bucketBase[s + 1] = bucketBase[s] + childCounts[s];
        }
        int totalBuckets = bucketBase[b1];

        // Assign every live node to its 2 nearest buckets, in parallel over ordinal chunks.
        int[] assign1 = new int[srcOfNewOrd.length];
        int[] assign2 = new int[srcOfNewOrd.length];
        int targetTasks = Math.max(ctx.taskWindowSize * 4, 16);
        int chunk = Math.max(1024, (liveNewOrds.length + targetTasks - 1) / targetTasks);
        List<Callable<Void>> tasks = new ArrayList<>();
        for (int start = 0; start < liveNewOrds.length; start += chunk) {
            final int cStart = start;
            final int cEnd = Math.min(start + chunk, liveNewOrds.length);
            tasks.add(() -> {
                byte[] code = new byte[codeSize];
                ByteSequence<?> codeSeq = vectorTypeSupport.createByteSequence(code);
                VectorFloat<?> vec = vectorTypeSupport.createFloatVector(ctx.dimension);
                for (int i = cStart; i < cEnd; i++) {
                    int newOrd = liveNewOrds[i];
                    readCode(codeCache, newOrd, codeSize, code);
                    pq.decode(codeSeq, vec);
                    assignTop2(vec, superCentroids, b1, childCentroids, childCounts, bucketBase,
                            ctx.dimension, assign1, assign2, newOrd);
                }
                return null;
            });
        }
        invokeAll(ctx, tasks);

        // Group members by bucket (each node appears in up to 2).
        int[] counts = new int[totalBuckets];
        for (int newOrd : liveNewOrds) {
            counts[assign1[newOrd]]++;
            if (assign2[newOrd] >= 0) {
                counts[assign2[newOrd]]++;
            }
        }
        int[][] members = new int[totalBuckets][];
        int[] fill = new int[totalBuckets];
        for (int b = 0; b < totalBuckets; b++) {
            members[b] = new int[counts[b]];
        }
        for (int newOrd : liveNewOrds) {
            int b = assign1[newOrd];
            members[b][fill[b]++] = newOrd;
            int b2 = assign2[newOrd];
            if (b2 >= 0) {
                members[b2][fill[b2]++] = newOrd;
            }
        }

        // Split pathologically large buckets once; the O(size^2) scan cost is the concern, not
        // correctness, so a still-oversized bucket after one split is kept and logged.
        int maxBucket = MAX_BUCKET_MULTIPLE * targetBucketSize;
        for (int[] bucket : members) {
            if (bucket.length > maxBucket) {
                result.addAll(splitBucket(ctx, pq, codeCache, codeSize, bucket, targetBucketSize));
            } else if (bucket.length > 0) {
                result.add(bucket);
            }
        }
        return result;
    }

    /** Decodes a stride sample of live nodes for centroid training. */
    private VectorFloat<?>[] decodeSample(ProductQuantization pq,
                                          MappedByteBuffer codeCache,
                                          int codeSize,
                                          int[] liveNewOrds,
                                          int dimension) {
        int sampleSize = Math.min(liveNewOrds.length, MAX_KMEANS_SAMPLE);
        var sample = new VectorFloat<?>[sampleSize];
        byte[] code = new byte[codeSize];
        ByteSequence<?> codeSeq = vectorTypeSupport.createByteSequence(code);
        double stride = (double) liveNewOrds.length / sampleSize;
        for (int i = 0; i < sampleSize; i++) {
            int newOrd = liveNewOrds[(int) (i * stride)];
            readCode(codeCache, newOrd, codeSize, code);
            var vec = vectorTypeSupport.createFloatVector(dimension);
            pq.decode(codeSeq, vec);
            sample[i] = vec;
        }
        return sample;
    }

    /** Re-clusters an oversized bucket's members into sub-buckets (single level). */
    private List<int[]> splitBucket(CompactionContext ctx,
                                    ProductQuantization pq,
                                    MappedByteBuffer codeCache,
                                    int codeSize,
                                    int[] bucket,
                                    int targetBucketSize) {
        int k = Math.max(2, (bucket.length + targetBucketSize - 1) / targetBucketSize);
        log.debug("Splitting oversized bucket of {} members into {} sub-buckets", bucket.length, k);

        int sampleSize = Math.min(bucket.length, 4096);
        var sample = new VectorFloat<?>[sampleSize];
        byte[] code = new byte[codeSize];
        ByteSequence<?> codeSeq = vectorTypeSupport.createByteSequence(code);
        double stride = (double) bucket.length / sampleSize;
        for (int i = 0; i < sampleSize; i++) {
            readCode(codeCache, bucket[(int) (i * stride)], codeSize, code);
            var vec = vectorTypeSupport.createFloatVector(ctx.dimension);
            pq.decode(codeSeq, vec);
            sample[i] = vec;
        }
        k = Math.min(k, sampleSize);
        VectorFloat<?> centroids = new KMeansPlusPlusClusterer(sample, k).cluster(KMEANS_ITERATIONS, 0);

        var subBuckets = new ArrayList<int[]>(k);
        int[] assignment = new int[bucket.length];
        int[] counts = new int[k];
        VectorFloat<?> vec = vectorTypeSupport.createFloatVector(ctx.dimension);
        for (int i = 0; i < bucket.length; i++) {
            readCode(codeCache, bucket[i], codeSize, code);
            pq.decode(codeSeq, vec);
            int c = nearestCentroid(vec, centroids, k, ctx.dimension);
            assignment[i] = c;
            counts[c]++;
        }
        int[][] subs = new int[k][];
        int[] fill = new int[k];
        for (int c = 0; c < k; c++) {
            subs[c] = new int[counts[c]];
        }
        for (int i = 0; i < bucket.length; i++) {
            subs[assignment[i]][fill[assignment[i]]++] = bucket[i];
        }
        for (int[] sub : subs) {
            if (sub.length > 0) {
                subBuckets.add(sub);
            }
        }
        return subBuckets;
    }

    /** Scans each bucket's cross-source pairs with SDC scoring, updating both endpoints' heaps. */
    private void scanBuckets(CompactionContext ctx,
                             List<int[]> buckets,
                             PQSymmetricDistanceTables tables,
                             MappedByteBuffer codeCache,
                             int codeSize) {
        List<Callable<Void>> tasks = new ArrayList<>(buckets.size());
        for (int[] members : buckets) {
            tasks.add(() -> {
                scanOneBucket(members, tables, codeCache, codeSize);
                return null;
            });
        }
        invokeAll(ctx, tasks);
    }

    private void scanOneBucket(int[] members,
                               PQSymmetricDistanceTables tables,
                               MappedByteBuffer codeCache,
                               int codeSize) {
        int n = members.length;
        if (n < 2) {
            return;
        }
        // Gather codes contiguously so the O(n^2) inner loop runs over a compact local array
        // instead of scattered mmap pages.
        byte[] codes = new byte[n * codeSize];
        int[] srcs = new int[n];
        for (int i = 0; i < n; i++) {
            readCode(codeCache, members[i], codeSize, codes, i * codeSize);
            srcs[i] = srcOfNewOrd[members[i]];
        }
        // Per-code correction terms, computed once per member instead of once per pair.
        float[] gAux = null;
        float[] normAux = null;
        if (tables.needsCentroidDot() || tables.needsNorm()) {
            gAux = new float[n];
            normAux = tables.needsNorm() ? new float[n] : null;
            for (int i = 0; i < n; i++) {
                gAux[i] = tables.centroidDotTerm(codes, i * codeSize);
                if (normAux != null) {
                    normAux[i] = tables.normSquaredTerm(codes, i * codeSize, gAux[i]);
                }
            }
        }

        for (int i = 0; i < n; i++) {
            int srcI = srcs[i];
            float gI = gAux == null ? 0f : gAux[i];
            float nI = normAux == null ? 0f : normAux[i];
            for (int j = i + 1; j < n; j++) {
                if (srcs[j] == srcI) {
                    continue;
                }
                float sim = tables.approximateSimilarity(codes, i * codeSize, codes, j * codeSize,
                        gI, gAux == null ? 0f : gAux[j], nI, normAux == null ? 0f : normAux[j]);
                offer(members[i], members[j], sim);
                offer(members[j], members[i], sim);
            }
        }
    }

    /**
     * Offers a candidate to {@code owner}'s bounded min-heap. The unlocked pre-check against the
     * heap root is racy by design: a stale read can only cause a borderline candidate to be
     * skipped or a redundant lock acquisition, never corruption, and it keeps the common
     * fully-populated-heap case lock-free.
     */
    private void offer(int owner, int candidate, float score) {
        int base = owner * topT;
        if (heapSizes[owner] == topT && score <= heapScores[base]) {
            return;
        }
        ReentrantLock lock = stripes[owner & (LOCK_STRIPES - 1)];
        lock.lock();
        try {
            int size = heapSizes[owner];
            if (size < topT) {
                int i = size;
                heapNodes[base + i] = candidate;
                heapScores[base + i] = score;
                while (i > 0) {
                    int parent = (i - 1) >> 1;
                    if (heapScores[base + parent] <= heapScores[base + i]) {
                        break;
                    }
                    swap(base + parent, base + i);
                    i = parent;
                }
                heapSizes[owner] = (short) (size + 1);
            } else if (score > heapScores[base]) {
                heapNodes[base] = candidate;
                heapScores[base] = score;
                siftDown(base, topT);
            }
        } finally {
            lock.unlock();
        }
    }

    private void siftDown(int base, int size) {
        int i = 0;
        while (true) {
            int left = 2 * i + 1;
            if (left >= size) {
                break;
            }
            int smallest = left;
            int right = left + 1;
            if (right < size && heapScores[base + right] < heapScores[base + left]) {
                smallest = right;
            }
            if (heapScores[base + i] <= heapScores[base + smallest]) {
                break;
            }
            swap(base + i, base + smallest);
            i = smallest;
        }
    }

    private void swap(int a, int b) {
        int tn = heapNodes[a];
        heapNodes[a] = heapNodes[b];
        heapNodes[b] = tn;
        float ts = heapScores[a];
        heapScores[a] = heapScores[b];
        heapScores[b] = ts;
    }

    /** Absolute-position read of one code from the cache; safe for concurrent readers. */
    private static void readCode(MappedByteBuffer codeCache, int newOrd, int codeSize, byte[] out) {
        readCode(codeCache, newOrd, codeSize, out, 0);
    }

    private static void readCode(MappedByteBuffer codeCache, int newOrd, int codeSize, byte[] out, int outOff) {
        int pos = newOrd * codeSize;
        for (int i = 0; i < codeSize; i++) {
            out[outOff + i] = codeCache.get(pos + i);
        }
    }

    private static int nearestCentroid(VectorFloat<?> vec, VectorFloat<?> centroids, int k, int dimension) {
        int best = 0;
        float bestDist = Float.MAX_VALUE;
        for (int c = 0; c < k; c++) {
            float d = VectorUtil.squareL2Distance(vec, 0, centroids, c * dimension, dimension);
            if (d < bestDist) {
                bestDist = d;
                best = c;
            }
        }
        return best;
    }

    /**
     * Finds the 2 nearest buckets for {@code vec} by routing through the 2 nearest superclusters
     * and comparing all of their children. Bucketing always uses squared L2 on decoded vectors
     * regardless of the similarity metric — buckets only need spatial coherence, and the PQ
     * codebooks are L2-trained.
     */
    private static void assignTop2(VectorFloat<?> vec,
                                   VectorFloat<?> superCentroids, int b1,
                                   VectorFloat<?>[] childCentroids, int[] childCounts, int[] bucketBase,
                                   int dimension,
                                   int[] assign1, int[] assign2, int newOrd) {
        int bestSuper = -1, secondSuper = -1;
        float bestSuperDist = Float.MAX_VALUE, secondSuperDist = Float.MAX_VALUE;
        for (int s = 0; s < b1; s++) {
            float d = VectorUtil.squareL2Distance(vec, 0, superCentroids, s * dimension, dimension);
            if (d < bestSuperDist) {
                secondSuper = bestSuper;
                secondSuperDist = bestSuperDist;
                bestSuper = s;
                bestSuperDist = d;
            } else if (d < secondSuperDist) {
                secondSuper = s;
                secondSuperDist = d;
            }
        }

        int bestBucket = -1, secondBucket = -1;
        float bestDist = Float.MAX_VALUE, secondDist = Float.MAX_VALUE;
        for (int pass = 0; pass < 2; pass++) {
            int s = pass == 0 ? bestSuper : secondSuper;
            if (s < 0) {
                continue;
            }
            VectorFloat<?> children = childCentroids[s];
            for (int c = 0; c < childCounts[s]; c++) {
                float d = VectorUtil.squareL2Distance(vec, 0, children, c * dimension, dimension);
                int bucket = bucketBase[s] + c;
                if (d < bestDist) {
                    secondBucket = bestBucket;
                    secondDist = bestDist;
                    bestBucket = bucket;
                    bestDist = d;
                } else if (d < secondDist) {
                    secondBucket = bucket;
                    secondDist = d;
                }
            }
        }
        assign1[newOrd] = bestBucket;
        assign2[newOrd] = secondBucket;
    }

    private static void invokeAll(CompactionContext ctx, List<Callable<Void>> tasks) {
        try {
            for (Future<Void> f : ctx.executor.invokeAll(tasks)) {
                f.get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Bucketed acquisition interrupted", e);
        } catch (ExecutionException e) {
            throw new RuntimeException("Bucketed acquisition failed", e.getCause());
        }
    }
}
