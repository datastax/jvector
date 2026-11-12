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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * The probe step of one range of {@link OnDiskGraphIndexCompactor}'s cell walk, as handed to a
 * {@link CompactionAccelerator}: for each query, the cells to scan, best first.
 * <p>
 * The cells are the merged hierarchy's level-1 nodes, called centers here, at positions {@code 0} to
 * {@link #centerCount()}. The compactor ranks them with a beam search; an accelerator may rank them any way that orders
 * them by JVector's similarity to the query: EUCLIDEAN by squared distance, COSINE by cosine, DOT_PRODUCT by dot
 * product. It returns up to {@link #maxProbes()} center positions per query with {@link #setProbes}; the compactor
 * then takes cells from that list, best first, until each target source's scan budget is spent, as it does with the
 * beam's list.
 */
@Experimental
public final class CellProbe {

    private final VectorFloat<?>[] queries;
    private final int queryCount;
    private final HubMap centers;
    private final int dimension;
    private final VectorSimilarityFunction similarity;
    private final int maxProbes;
    final int[] probes;
    final int[] counts;

    CellProbe(VectorFloat<?>[] queries, int queryCount, HubMap centers, int dimension, VectorSimilarityFunction similarity,
              int maxProbes) {
        this.queries = queries;
        this.queryCount = queryCount;
        this.centers = centers;
        this.dimension = dimension;
        this.similarity = similarity;
        this.maxProbes = maxProbes;
        this.probes = new int[Math.multiplyExact(queryCount, maxProbes)];
        this.counts = new int[queryCount];
    }

    public int queryCount() {
        return queryCount;
    }

    public int dimension() {
        return dimension;
    }

    public VectorSimilarityFunction similarity() {
        return similarity;
    }

    /** Copies query {@code i}'s vector into {@code dst}. */
    public void query(int i, float[] dst) {
        VectorFloat<?> q = queries[i];
        for (int d = 0; d < dimension; d++) {
            dst[d] = q.get(d);
        }
    }

    /** Number of centers (level-1 positions). */
    public int centerCount() {
        return centers.level1Count();
    }

    /** Copies the vectors of center positions {@code [from, from + count)} into {@code dst} from {@code offset}, row-major. */
    public void copyCenters(int from, int count, float[] dst, int offset) {
        centers.copyLevel1Rows(from, count, dst, offset);
    }

    /** Most center positions kept per query. */
    public int maxProbes() {
        return maxProbes;
    }

    /**
     * Sets query {@code i}'s probes: {@code count} center positions from {@code positions[from]}, best first, at most
     * {@link #maxProbes()}. Calls for different queries may run concurrently.
     */
    public void setProbes(int i, int[] positions, int from, int count) {
        if (count > maxProbes) {
            throw new IllegalArgumentException(count + " probes for at most " + maxProbes);
        }
        System.arraycopy(positions, from, probes, i * maxProbes, count);
        counts[i] = count;
    }
}
