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

package io.github.jbellis.jvector.quantization;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.ParallelExecutor;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/// Tests that the quantization path accepts {@link ParallelExecutor} and works caller-runs.
///
/// The contract is deliberately split (see {@link ProductQuantization#compute} javadoc):
/// - **Encoding** is per-vector independent and deterministic given a fixed codebook, so
///   {@code encodeAll} produces <b>byte-identical</b> output under {@code callerRuns()} and a
///   {@code ForkJoinPool}.
/// - **Training** ({@code compute}/{@code refine}) draws k-means++ seeds from
///   {@code ThreadLocalRandom} and is therefore already non-deterministic run-to-run; it is
///   <b>not</b> byte-identical across executors, only recall/quality-equivalent.
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class QuantizationCallerRunsTest extends RandomizedTest {

    private static final int DIM = 64;
    private static final int SIZE = 2_000;
    private static final int M = 8;
    private static final int CLUSTERS = 256;

    private List<VectorFloat<?>> vectors;
    private ListRandomAccessVectorValues ravv;

    private void makeData() {
        vectors = TestUtil.createRandomVectors(SIZE, DIM);
        ravv = new ListRandomAccessVectorValues(vectors, DIM);
    }

    /// Encoding a fixed PQ codebook must be byte-identical whether run on a pool or caller-runs.
    @Test
    public void pqEncodeIsByteIdenticalAcrossExecutors() {
        makeData();
        ForkJoinPool pool = new ForkJoinPool(4);
        try {
            ProductQuantization pq = ProductQuantization.compute(ravv, M, CLUSTERS, true, UNWEIGHTED,
                    ParallelExecutor.forkJoin(pool), ParallelExecutor.forkJoin(pool));

            PQVectors onPool = pq.encodeAll(ravv, ParallelExecutor.forkJoin(pool));
            PQVectors callerRuns = pq.encodeAll(ravv, ParallelExecutor.callerRuns());

            assertEquals("PQ encoding must be byte-identical across executors", onPool, callerRuns);
        } finally {
            pool.shutdown();
        }
    }

    /// Binary quantization encoding must also be byte-identical across executors.
    @Test
    public void bqEncodeIsByteIdenticalAcrossExecutors() {
        makeData();
        ForkJoinPool pool = new ForkJoinPool(4);
        try {
            BinaryQuantization bq = new BinaryQuantization(DIM);
            CompressedVectors onPool = bq.encodeAll(ravv, ParallelExecutor.forkJoin(pool));
            CompressedVectors callerRuns = bq.encodeAll(ravv, ParallelExecutor.callerRuns());
            assertEquals("BQ encoding must be byte-identical across executors", onPool, callerRuns);
        } finally {
            pool.shutdown();
        }
    }

    /// Training on a pool vs caller-runs is NOT byte-identical (ThreadLocalRandom seeds), but must
    /// produce codebooks of equivalent quality — asserted via average reconstruction error.
    @Test
    public void pqTrainingIsQualityEquivalentAcrossExecutors() {
        makeData();
        ForkJoinPool pool = new ForkJoinPool(4);
        try {
            ProductQuantization onPool = ProductQuantization.compute(ravv, M, CLUSTERS, true, UNWEIGHTED,
                    ParallelExecutor.forkJoin(pool), ParallelExecutor.forkJoin(pool));
            ProductQuantization callerRuns = ProductQuantization.compute(ravv, M, CLUSTERS, true, UNWEIGHTED,
                    ParallelExecutor.callerRuns(), ParallelExecutor.callerRuns());

            double msePool = meanReconstructionError(onPool, ParallelExecutor.forkJoin(pool));
            double mseCaller = meanReconstructionError(callerRuns, ParallelExecutor.callerRuns());

            assertTrue("pool-trained codebook should reconstruct well: " + msePool, msePool < 0.05);
            assertTrue("caller-runs-trained codebook should reconstruct well: " + mseCaller, mseCaller < 0.05);
            assertTrue("training quality must be equivalent across executors (" + msePool + " vs " + mseCaller + ")",
                    Math.abs(msePool - mseCaller) < 0.01);
        } finally {
            pool.shutdown();
        }
    }

    /// refine() must also run caller-runs and produce a quality-equivalent codebook.
    @Test
    public void pqRefineRunsCallerRuns() {
        makeData();
        ForkJoinPool pool = new ForkJoinPool(4);
        try {
            ProductQuantization base = ProductQuantization.compute(ravv, M, CLUSTERS, true, UNWEIGHTED,
                    ParallelExecutor.forkJoin(pool), ParallelExecutor.forkJoin(pool));
            ProductQuantization refined = base.refine(ravv, 1, UNWEIGHTED,
                    ParallelExecutor.callerRuns(), ParallelExecutor.callerRuns());
            assertTrue("refined codebook should reconstruct well",
                    meanReconstructionError(refined, ParallelExecutor.callerRuns()) < 0.05);
        } finally {
            pool.shutdown();
        }
    }

    /// With callerRuns(), train + encode must execute only on the calling thread — no worker threads
    /// and the common pool untouched.
    @Test
    public void callerRunsStaysOnCallingThread() {
        makeData();
        Set<String> observed = ConcurrentHashMap.newKeySet();
        RecordingRAVV recording = new RecordingRAVV(ravv, observed);
        String mainName = Thread.currentThread().getName();

        ProductQuantization pq = ProductQuantization.compute(recording, M, CLUSTERS, true, UNWEIGHTED,
                ParallelExecutor.callerRuns(), ParallelExecutor.callerRuns());
        pq.encodeAll(recording, ParallelExecutor.callerRuns());

        assertFalse("some vector reads must have occurred", observed.isEmpty());
        for (String name : observed) {
            assertTrue("caller-runs must read only on the calling thread, but saw: " + name + " (all=" + observed + ")",
                    name.equals(mainName));
        }
    }

    private double meanReconstructionError(ProductQuantization pq, ParallelExecutor ex) {
        double[] errs = pq.reconstructionErrors(ravv, ex);
        double s = 0;
        for (double e : errs) {
            s += e;
        }
        return s / errs.length;
    }

    /// Records the thread of every getVector call, funnelling thread-local copies through one set.
    private static final class RecordingRAVV implements RandomAccessVectorValues {
        private final RandomAccessVectorValues delegate;
        private final Set<String> observed;

        RecordingRAVV(RandomAccessVectorValues delegate, Set<String> observed) {
            this.delegate = delegate;
            this.observed = observed;
        }

        @Override
        public int size() {
            return delegate.size();
        }

        @Override
        public int dimension() {
            return delegate.dimension();
        }

        @Override
        public VectorFloat<?> getVector(int nodeId) {
            observed.add(Thread.currentThread().getName());
            return delegate.getVector(nodeId);
        }

        @Override
        public boolean isValueShared() {
            return delegate.isValueShared();
        }

        @Override
        public RandomAccessVectorValues copy() {
            return new RecordingRAVV(delegate.copy(), observed);
        }
    }
}
