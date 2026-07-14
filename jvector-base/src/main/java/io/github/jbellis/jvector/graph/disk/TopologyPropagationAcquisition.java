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
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.quantization.ImmutablePQVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Entry-point propagation for cross-source candidate acquisition during graph compaction: instead
 * of descending each other source's hierarchy from its global entry node for every node, base-layer
 * searches are warm-started from entry points that the compaction itself produces as it runs.
 * <p>
 * For a node {@code I} in partition {@code s} searching partition {@code t}, the entry-point pool
 * is fed from four sources, in decreasing order of specificity:
 * <ol>
 *   <li><b>Pull</b>: {@code I}'s same-source neighbors that already finished compaction retain
 *       their diversity-selected merged adjacency here; its partition-{@code t} members are near
 *       {@code I} by triangle inequality.</li>
 *   <li><b>Reverse</b>: earlier searches by partition-{@code t} nodes that found {@code I} pushed
 *       themselves into {@code I}'s reverse slot ("J found I, therefore I should know J").</li>
 *   <li><b>Chaining</b>: results from partitions {@code I} already searched this round, when
 *       finished, contribute their partition-{@code t} adjacency — so only the first cross-source
 *       search per node can ever be cold.</li>
 *   <li><b>Descent</b>: when the pool is empty, a greedy descent through {@code t}'s
 *       <em>in-memory</em> upper layers scored against the fused-PQ code cache — no disk reads —
 *       lands seeds near {@code I} anyway.</li>
 * </ol>
 * Pools and the descent are scored with the same asymmetric PQ scoring (ADC) the compaction
 * search path uses — a per-node precomputed query table ({@link PQVectors#precomputedScoreFunctionFor})
 * over a heap copy of the pre-encode code cache — so acquisition introduces no new distance
 * approximation. Pools are capped at {@link #SEEDS_PER_PARTITION} seeds. Seed scores are
 * approximate; the compactor exactly rescores all search results before diversity selection, and
 * falls back to the full search path when no seeds exist or warm results fail its quality floor,
 * so the worst case is bounded at search-based acquisition.
 * <p>
 * Everything is opportunistic — a node whose neighbors aren't finished yet just gets fewer pool
 * sources — so the base layer's embarrassing parallelism is preserved: the only synchronization is
 * a volatile done-flag per node, striped locks on the bounded reverse slots, and safe publication
 * of retained adjacency via the done flag.
 * <p>
 * Memory: retained adjacency is {@code (maxOrdinal+1) * degree * 4} bytes, reverse slots are
 * {@code (maxOrdinal+1) * numSources * } {@link #REVERSE_CAPACITY} {@code * 8} bytes, and the
 * code-cache heap copy is {@code (maxOrdinal+1) * codeSize} bytes; all are released before
 * refinement. Streaming/partitioned variants are the designed fix if this bites at large scale.
 */
@Experimental
final class TopologyPropagationAcquisition {
    private static final io.github.jbellis.jvector.vector.types.VectorTypeSupport vectorTypeSupport =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    /** Max entry points handed to one warm search; the ADC-ranked best of the pool. */
    static final int SEEDS_PER_PARTITION = 4;
    /** Per (node, partition) bounded reverse-candidate slot size. */
    static final int REVERSE_CAPACITY = 8;
    /** Per-partition pool cap while gathering one node's candidates; extras are dropped. */
    static final int POOL_CAPACITY = 64;
    /** How many top results per partition search feed reverse pushes and chaining. */
    static final int PROPAGATE_TOP_RESULTS = 8;
    /** Power of two. */
    private static final int LOCK_STRIPES = 1024;

    private final VectorSimilarityFunction vsf;
    /** Heap copy of the fused-PQ pre-encode cache wrapped for the stock ADC decoders. */
    private final PQVectors cacheVectors;
    private final int numSources;
    private final int degree;
    private final List<OrdinalMapper> remappers;
    private final List<FixedBitSet> liveNodes;
    /** newOrdinal -> source index, -1 for holes in the remapped ordinal space. */
    private final int[] srcOfNewOrd;
    /** newOrdinal -> original ordinal within its source. */
    private final int[] oldOfNewOrd;

    // Finished nodes' diversity-selected merged adjacency (new ordinals), published by the
    // volatile doneFlag write: readers must observe doneFlag == 1 before touching the slice.
    private final AtomicIntegerArray doneFlag;
    private final int[] doneAdj;
    private final short[] doneDeg;

    // Reverse-candidate slots, laid out as REVERSE_CAPACITY-length slices at
    // (owner * numSources + partition) * REVERSE_CAPACITY; slice (owner, t) holds partition-t
    // nodes whose searches found `owner`, best-scored kept.
    private final int[] revNodes;
    private final float[] revScores;
    private final short[] revSizes;
    private final ReentrantLock[] stripes;

    final AtomicLong warmSearches = new AtomicLong();
    final AtomicLong descentSeeded = new AtomicLong();
    final AtomicLong fallbackSearches = new AtomicLong();

    private final ThreadLocal<NodeContext> contexts = ThreadLocal.withInitial(NodeContext::new);

    private TopologyPropagationAcquisition(CompactionContext ctx,
                                           ProductQuantization pq,
                                           MappedByteBuffer codeCache,
                                           int codeSize,
                                           VectorSimilarityFunction vsf,
                                           int baseDegree) {
        this.vsf = vsf;
        this.numSources = ctx.sources.size();
        this.degree = baseDegree;
        this.remappers = ctx.remappers;
        this.liveNodes = ctx.liveNodes;

        int bound = ctx.maxOrdinal + 1;
        // Copy the mmap'd pre-encode cache to the heap once and wrap it as PQVectors, so seed
        // scoring goes through the same ADC decoders the rest of the codebase uses (and the hot
        // ranking loop reads heap arrays instead of byte-at-a-time mmap gets).
        if (codeSize != pq.getSubspaceCount()) {
            throw new IllegalArgumentException("code cache entry size " + codeSize
                    + " != PQ subspace count " + pq.getSubspaceCount());
        }
        byte[] codes = new byte[Math.toIntExact((long) bound * codeSize)];
        java.nio.ByteBuffer cacheView = codeCache.duplicate();
        cacheView.position(0);
        cacheView.get(codes, 0, codes.length);
        ByteSequence<?> chunk = vectorTypeSupport.createByteSequence(codes);
        this.cacheVectors = new ImmutablePQVectors(pq, new ByteSequence<?>[] { chunk }, bound, bound);

        this.srcOfNewOrd = new int[bound];
        this.oldOfNewOrd = new int[bound];
        Arrays.fill(srcOfNewOrd, -1);
        for (int s = 0; s < numSources; s++) {
            FixedBitSet alive = ctx.liveNodes.get(s);
            OrdinalMapper mapper = ctx.remappers.get(s);
            for (int oldOrd = 0; oldOrd < alive.length(); oldOrd++) {
                if (!alive.get(oldOrd)) {
                    continue;
                }
                int newOrd = mapper.oldToNew(oldOrd);
                srcOfNewOrd[newOrd] = s;
                oldOfNewOrd[newOrd] = oldOrd;
            }
        }

        this.doneFlag = new AtomicIntegerArray(bound);
        this.doneAdj = new int[bound * degree];
        this.doneDeg = new short[bound];
        this.revNodes = new int[bound * numSources * REVERSE_CAPACITY];
        this.revScores = new float[bound * numSources * REVERSE_CAPACITY];
        this.revSizes = new short[bound * numSources];
        this.stripes = new ReentrantLock[LOCK_STRIPES];
        for (int i = 0; i < LOCK_STRIPES; i++) {
            stripes[i] = new ReentrantLock();
        }

    }

    static TopologyPropagationAcquisition create(CompactionContext ctx,
                                                 ProductQuantization pq,
                                                 MappedByteBuffer codeCache,
                                                 int codeSize,
                                                 VectorSimilarityFunction vsf,
                                                 int baseDegree) {
        return new TopologyPropagationAcquisition(ctx, pq, codeCache, codeSize, vsf, baseDegree);
    }

    /** Per-thread reusable state for gathering one node's cross-source candidates. */
    final class NodeContext {
        int newOrd;
        int sourceIdx;
        /** ADC scorer with this node's precomputed query table; rebuilt by each beginNode. */
        ScoreFunction.ApproximateScoreFunction decoder;
        final int[][] pools = new int[numSources][POOL_CAPACITY];
        final int[] poolSizes = new int[numSources];
        /** Selected beam entry points as ORIGINAL ordinals within the target partition. */
        final int[] seedNodes = new int[SEEDS_PER_PARTITION];
        /** ADC scores parallel to {@link #seedNodes}, descending. */
        final float[] seedScores = new float[SEEDS_PER_PARTITION];
        private final int[] seedNews = new int[SEEDS_PER_PARTITION];
        private final int[] topIdx = new int[PROPAGATE_TOP_RESULTS];
    }

    /**
     * Starts gathering for one node: builds its precomputed ADC query table, then fills the
     * per-partition pools from finished same-source neighbors (pull) and this node's reverse
     * slots. {@code sameSourceOldOrds} is the node's live same-source neighbor list as collected
     * by the same-source gather. The returned context is thread-local; valid until the next
     * {@code beginNode} on this thread (the query table shares the PQ's thread-local scratch).
     */
    NodeContext beginNode(int newOrd, int sourceIdx, VectorFloat<?> queryVec, int[] sameSourceOldOrds, int count) {
        NodeContext ctx = contexts.get();
        ctx.newOrd = newOrd;
        ctx.sourceIdx = sourceIdx;
        ctx.decoder = cacheVectors.precomputedScoreFunctionFor(queryVec, vsf);
        Arrays.fill(ctx.poolSizes, 0);

        OrdinalMapper mapper = remappers.get(sourceIdx);
        for (int k = 0; k < count; k++) {
            int nbNew = mapper.oldToNew(sameSourceOldOrds[k]);
            if (doneFlag.get(nbNew) == 0) {
                continue;
            }
            int base = nbNew * degree;
            int cnt = doneDeg[nbNew];
            for (int j = 0; j < cnt; j++) {
                int m = doneAdj[base + j];
                int t = srcOfNewOrd[m];
                if (t != sourceIdx) {
                    poolAdd(ctx, t, m);
                }
            }
        }

        ReentrantLock lock = stripes[newOrd & (LOCK_STRIPES - 1)];
        lock.lock();
        try {
            for (int t = 0; t < numSources; t++) {
                if (t == sourceIdx) {
                    continue;
                }
                int slot = newOrd * numSources + t;
                int base = slot * REVERSE_CAPACITY;
                int size = revSizes[slot];
                for (int j = 0; j < size; j++) {
                    poolAdd(ctx, t, revNodes[base + j]);
                }
            }
        } finally {
            lock.unlock();
        }
        return ctx;
    }

    /**
     * Ranks the pool for {@code targetPartition} by ADC similarity to the context node and fills
     * {@code ctx.seedNodes}/{@code ctx.seedScores} with the best {@link #SEEDS_PER_PARTITION}
     * (as original ordinals within the target). When the pool is empty, the seeds come from a
     * greedy descent through the target's in-memory hierarchy instead. Returns the seed count;
     * 0 means the caller should use the full search path.
     */
    int selectSeeds(NodeContext ctx, int targetPartition, OnDiskGraphIndex.View targetView) {
        int n = ctx.poolSizes[targetPartition];
        if (n == 0) {
            int count = descentSeeds(ctx, targetPartition, targetView);
            if (count > 0) {
                descentSeeded.incrementAndGet();
            }
            return count;
        }
        int[] pool = ctx.pools[targetPartition];
        Arrays.sort(pool, 0, n);
        int ranked = 0;
        int prev = -1;
        for (int i = 0; i < n; i++) {
            int m = pool[i];
            if (m == prev) {
                continue;
            }
            prev = m;
            ranked = seedInsert(ctx, ranked, m, ctx.decoder.similarityTo(m));
        }
        for (int j = 0; j < ranked; j++) {
            ctx.seedNodes[j] = oldOfNewOrd[ctx.seedNews[j]];
        }
        warmSearches.incrementAndGet();
        return ranked;
    }

    /**
     * Post-search hook for one partition: pushes this node into the top results' reverse slots and
     * chains finished results' adjacency into the pools of partitions not yet searched this round
     * ({@code > targetPartition}). Call with the final candidate range for the partition, whether
     * it came from the warm search or the fallback path.
     */
    void afterPartition(NodeContext ctx, int targetPartition,
                        int[] candNode, float[] candScore, int from, int to) {
        int count = Math.min(PROPAGATE_TOP_RESULTS, to - from);
        if (count <= 0) {
            return;
        }
        // partial selection of the top-`count` result indices by score
        int selected = 0;
        for (int i = from; i < to; i++) {
            int pos = selected;
            while (pos > 0 && candScore[ctx.topIdx[pos - 1]] < candScore[i]) {
                pos--;
            }
            if (pos < count) {
                int limit = Math.min(selected, count - 1);
                for (int k = limit; k > pos; k--) {
                    ctx.topIdx[k] = ctx.topIdx[k - 1];
                }
                ctx.topIdx[pos] = i;
                if (selected < count) {
                    selected++;
                }
            }
        }

        OrdinalMapper mapper = remappers.get(targetPartition);
        for (int r = 0; r < selected; r++) {
            int i = ctx.topIdx[r];
            int cNew = mapper.oldToNew(candNode[i]);
            if (doneFlag.get(cNew) == 0) {
                // Not gathered yet: this node is a useful future entry point for it. (A finished
                // node's reverse slot can never be read again, so offering there is pure waste.)
                reverseOffer(cNew, ctx.sourceIdx, ctx.newOrd, candScore[i]);
                continue;
            }
            // Finished: chain its merged adjacency into the pools of partitions still to search.
            int base = cNew * degree;
            int cnt = doneDeg[cNew];
            for (int j = 0; j < cnt; j++) {
                int m = doneAdj[base + j];
                int u = srcOfNewOrd[m];
                if (u != ctx.sourceIdx && u > targetPartition) {
                    poolAdd(ctx, u, m);
                }
            }
        }
    }

    /**
     * Records a finished node's diversity-selected merged adjacency (new ordinals) and publishes
     * it via the done flag. Safe to call before the record is physically written — the content is
     * final once diversity selection has run.
     */
    void recordDone(int newOrd, int[] selectedNewOrds, int count) {
        int cnt = Math.min(count, degree);
        System.arraycopy(selectedNewOrds, 0, doneAdj, newOrd * degree, cnt);
        doneDeg[newOrd] = (short) cnt;
        doneFlag.set(newOrd, 1);
    }

    String statsSummary() {
        return String.format("%,d warm searches, %,d descent-seeded, %,d full-search fallbacks",
                warmSearches.get(), descentSeeded.get(), fallbackSearches.get());
    }

    private void poolAdd(NodeContext ctx, int partition, int candNew) {
        int size = ctx.poolSizes[partition];
        if (size < POOL_CAPACITY) {
            ctx.pools[partition][size] = candNew;
            ctx.poolSizes[partition] = size + 1;
        }
    }

    /** Inserts (candNew, score) into the bounded descending seed arrays; returns the new count. */
    private int seedInsert(NodeContext ctx, int seedCount, int candNew, float score) {
        if (seedCount == SEEDS_PER_PARTITION && score <= ctx.seedScores[seedCount - 1]) {
            return seedCount;
        }
        int pos = seedCount == SEEDS_PER_PARTITION ? seedCount - 1 : seedCount;
        while (pos > 0 && ctx.seedScores[pos - 1] < score) {
            ctx.seedNews[pos] = ctx.seedNews[pos - 1];
            ctx.seedScores[pos] = ctx.seedScores[pos - 1];
            pos--;
        }
        ctx.seedNews[pos] = candNew;
        ctx.seedScores[pos] = score;
        return Math.min(seedCount + 1, SEEDS_PER_PARTITION);
    }

    /**
     * Cold-start seeder: greedy first-improvement descent from the target's entry node through its
     * upper layers, scored via the node's precomputed ADC table over the code-cache copy.
     * Upper-layer adjacency is in-memory, so this does no disk reads. Dead nodes (no new-ordinal
     * mapping, hence no cached code) are skipped during routing; a dead entry node returns 0 and
     * the caller falls back to the full search path.
     */
    private int descentSeeds(NodeContext ctx, int targetPartition, OnDiskGraphIndex.View targetView) {
        var entry = targetView.entryNode();
        FixedBitSet alive = liveNodes.get(targetPartition);
        OrdinalMapper mapper = remappers.get(targetPartition);
        if (entry == null || entry.node >= alive.length() || !alive.get(entry.node)) {
            return 0;
        }
        int cur = entry.node;
        int curNew = mapper.oldToNew(cur);
        float curScore = ctx.decoder.similarityTo(curNew);
        for (int level = entry.level; level >= 1; level--) {
            boolean improved = true;
            while (improved) {
                improved = false;
                var it = targetView.getNeighborsIterator(level, cur);
                while (it.hasNext()) {
                    int nb = it.nextInt();
                    if (!alive.get(nb)) {
                        continue;
                    }
                    float s = ctx.decoder.similarityTo(mapper.oldToNew(nb));
                    if (s > curScore) {
                        cur = nb;
                        curNew = mapper.oldToNew(nb);
                        curScore = s;
                        improved = true;
                        break; // restart neighbor iteration from the new position
                    }
                }
            }
        }

        int ranked = seedInsert(ctx, 0, curNew, curScore);
        if (entry.level >= 1) {
            // pad with the landing node's level-1 neighborhood for beam diversity
            var it = targetView.getNeighborsIterator(1, cur);
            while (it.hasNext()) {
                int nb = it.nextInt();
                if (nb == cur || !alive.get(nb)) {
                    continue;
                }
                int nbNew = mapper.oldToNew(nb);
                ranked = seedInsert(ctx, ranked, nbNew, ctx.decoder.similarityTo(nbNew));
            }
        }
        for (int j = 0; j < ranked; j++) {
            ctx.seedNodes[j] = oldOfNewOrd[ctx.seedNews[j]];
        }
        return ranked;
    }

    /** Offers a candidate into the (owner, partition) reverse slot, keeping the best-scored. */
    private void reverseOffer(int owner, int partition, int candNew, float score) {
        int slot = owner * numSources + partition;
        int base = slot * REVERSE_CAPACITY;
        // Unlocked pre-check: when the slot is full and the score can't beat its minimum, skip
        // the lock entirely. Stale reads can only cause a borderline skip or a redundant lock
        // acquisition, never corruption; hub nodes found by many concurrent searches make the
        // full-slot case the common one.
        if (revSizes[slot] == REVERSE_CAPACITY) {
            float min = Float.POSITIVE_INFINITY;
            for (int j = 0; j < REVERSE_CAPACITY; j++) {
                min = Math.min(min, revScores[base + j]);
            }
            if (score <= min) {
                return;
            }
        }
        ReentrantLock lock = stripes[owner & (LOCK_STRIPES - 1)];
        lock.lock();
        try {
            int size = revSizes[slot];
            if (size < REVERSE_CAPACITY) {
                revNodes[base + size] = candNew;
                revScores[base + size] = score;
                revSizes[slot] = (short) (size + 1);
                return;
            }
            int minIdx = 0;
            for (int j = 1; j < REVERSE_CAPACITY; j++) {
                if (revScores[base + j] < revScores[base + minIdx]) {
                    minIdx = j;
                }
            }
            if (score > revScores[base + minIdx]) {
                revNodes[base + minIdx] = candNew;
                revScores[base + minIdx] = score;
            }
        } finally {
            lock.unlock();
        }
    }

}
