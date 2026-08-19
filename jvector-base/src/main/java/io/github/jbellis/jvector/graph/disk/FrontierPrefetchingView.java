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

import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;

/**
 * An {@link OnDiskGraphIndex.View} that asynchronously warms the records a beam search is
 * likely to expand next. Best-first search keeps exactly one demand read in flight per thread
 * (read the popped node, score its neighbors, repeat), so an above-RAM search is bound by
 * single-read latency, not device throughput. This view mirrors the searcher's candidate
 * queue (it observes every scored neighbor and every expansion) and hints the queue's top
 * entries via {@link OnDiskGraphIndex#willNeedL0Record} — the search's actual next
 * expansions — putting several reads in flight per searching thread without spawning any.
 * <p>
 * Hints are speculative and best-effort: a wasted hint costs one record-sized readahead; a
 * successful one converts a blocking ~100us device read into a page-cache hit. Used by
 * {@link OnDiskGraphIndexCompactor} for cross-source searches; enabled by setting the system
 * property {@code jvector.compaction.frontierPrefetch} to the per-expansion hint width.
 */
final class FrontierPrefetchingView extends OnDiskGraphIndex.View {
    /** Number of top-scored shadow-queue entries hinted per expansion. 3 is the measured knee:
     * deeper ranks are displaced before expansion, so wider hinting buys waste, not overlap. */
    static final int WIDTH = 3;

    /** Shadow queue capacity. Must comfortably exceed WIDTH; 32 tracks the searcher's queue
     * top closely while keeping per-neighbor maintenance a few-element shift. */
    private static final int SHADOW_CAP = 32;

    private final OnDiskGraphIndex index;
    // Approximate dedup of recent hints: direct-mapped by low ordinal bits. A collision only
    // causes a redundant hint, and advising an already-cached page is nearly free.
    private final int[] recentlyHinted;
    // Shadow of the searcher's candidate queue, sorted by descending score. The view observes
    // every push (each neighbor it scores) and every pop (each node it is asked to expand), so
    // the head of this array is the search's actual next expansion — not a guess from the
    // current node's neighborhood. Expanding a node the shadow has never seen means a new
    // search started (entry point or seeds); the shadow resets and re-learns within one hop.
    private final int[] shadowNode = new int[SHADOW_CAP];
    private final float[] shadowScore = new float[SHADOW_CAP];
    private final int[] shadowStamp = new int[SHADOW_CAP];
    private int shadowSize;
    private int tick;
    /** Entries unexpanded for this many expansions are stale (old search's leftovers) — purge,
     * or they squat in the top slots, dedup-suppressed, and starve fresh hints. */
    private static final int SHADOW_MAX_AGE = 64;
    // Preallocated wrapper so the hot path does not allocate; views are single-threaded.
    private ImmutableGraphIndex.NeighborProcessor downstream;
    private final ImmutableGraphIndex.NeighborProcessor tracking = (friendOrd, similarity) -> {
        downstream.process(friendOrd, similarity);
        if (shadowSize == SHADOW_CAP && similarity <= shadowScore[SHADOW_CAP - 1]) {
            return;
        }
        int pos = shadowSize == SHADOW_CAP ? SHADOW_CAP - 1 : shadowSize++;
        while (pos > 0 && shadowScore[pos - 1] < similarity) {
            shadowNode[pos] = shadowNode[pos - 1];
            shadowScore[pos] = shadowScore[pos - 1];
            shadowStamp[pos] = shadowStamp[pos - 1];
            pos--;
        }
        shadowNode[pos] = friendOrd;
        shadowScore[pos] = similarity;
        shadowStamp[pos] = tick;
    };

    private FrontierPrefetchingView(OnDiskGraphIndex index) throws IOException {
        index.super(index.readerSupplier.get());
        this.index = index;
        this.recentlyHinted = new int[1024];
        Arrays.fill(recentlyHinted, -1);
    }

    static FrontierPrefetchingView wrap(OnDiskGraphIndex index) {
        try {
            return new FrontierPrefetchingView(index);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void processNeighbors(int level, int node, ScoreFunction scoreFunction,
                                 ImmutableGraphIndex.IntMarker visited,
                                 ImmutableGraphIndex.NeighborProcessor neighborProcessor) {
        if (level != 0) {
            super.processNeighbors(level, node, scoreFunction, visited, neighborProcessor);
            return;
        }
        tick++;
        purgeStale();
        expanded(node);
        downstream = neighborProcessor;
        super.processNeighbors(level, node, scoreFunction, visited, tracking);
        int hints = Math.min(WIDTH, shadowSize);
        for (int i = 0; i < hints; i++) {
            hint(shadowNode[i]);
        }
    }

    /**
     * Removes {@code node} from the shadow queue. Unknown nodes are tolerated, not treated as a
     * reset signal: resume() re-pushes evicted candidates this view never saw (124M resumes in a
     * 192M-node merge), and even across search boundaries the stale entries are near-neighbors
     * of a near-twin query under similarity-ordered processing — still worth hinting. Wrong
     * entries cost one wasted record read and are displaced by fresher, better-scored
     * candidates within a few expansions.
     */
    private void expanded(int node) {
        for (int i = 0; i < shadowSize; i++) {
            if (shadowNode[i] == node) {
                removeAt(i);
                return;
            }
        }
    }

    private void purgeStale() {
        for (int i = shadowSize - 1; i >= 0; i--) {
            if (tick - shadowStamp[i] > SHADOW_MAX_AGE) {
                removeAt(i);
            }
        }
    }

    private void removeAt(int i) {
        System.arraycopy(shadowNode, i + 1, shadowNode, i, shadowSize - i - 1);
        System.arraycopy(shadowScore, i + 1, shadowScore, i, shadowSize - i - 1);
        System.arraycopy(shadowStamp, i + 1, shadowStamp, i, shadowSize - i - 1);
        shadowSize--;
    }

    private void hint(int node) {
        int slot = node & (recentlyHinted.length - 1);
        if (recentlyHinted[slot] == node) {
            return;
        }
        recentlyHinted[slot] = node;
        index.willNeedL0Record(node);
    }
}
