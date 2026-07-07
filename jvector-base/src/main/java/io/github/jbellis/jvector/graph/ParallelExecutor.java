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

package io.github.jbellis.jvector.graph;

import java.util.concurrent.ForkJoinPool;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Runs a {@link GraphIndexBuilder}'s internal build/finalize iterations to completion, blocking the
 * calling thread until every element has been processed. The implementation decides <em>how</em> the
 * iteration is distributed: {@link #forkJoin(ForkJoinPool)} hosts a parallel stream on a dedicated
 * pool (the historical behavior), while {@link #callerRuns()} runs everything sequentially on the
 * calling thread with no worker threads and no pool.
 * <p>
 * This is the seam that lets an embedder bound vector-graph construction to its own thread budget —
 * e.g. one thread per compaction — instead of a jvector-owned all-core pool. It is the build/finalize
 * counterpart to the caller-runs executor injection already available on the compaction merge path.
 */
public interface ParallelExecutor {
    /**
     * Runs {@code body} for each {@code i} in {@code [0, upperBound)}, blocking until all complete.
     *
     * @param upperBound the exclusive upper bound of the index range (may be {@code 0})
     * @param body       the action to apply to each index
     */
    void forEachInt(int upperBound, IntConsumer body);

    /**
     * Runs {@code body} for each element produced by {@code source}, blocking until all complete.
     * Callers pass a <em>sequential</em> stream; the implementation decides whether to parallelize it.
     *
     * @param source the (sequential) stream of primitive ints to iterate
     * @param body   the action to apply to each element
     */
    void forEach(IntStream source, IntConsumer body);

    /**
     * Runs {@code body} for each element produced by {@code source}, blocking until all complete.
     * Callers pass a <em>sequential</em> stream; the implementation decides whether to parallelize it.
     *
     * @param source the (sequential) stream to iterate
     * @param body   the action to apply to each element
     * @param <T>    the stream element type
     */
    <T> void forEach(Stream<T> source, Consumer<T> body);

    /**
     * Returns an executor backed by {@code pool}: each iteration is hosted as a parallel stream on
     * that pool and the calling thread blocks on the result. This reproduces the behavior of the
     * {@code ForkJoinPool}-based {@link GraphIndexBuilder} constructors.
     *
     * @param pool the pool that hosts the parallel iterations
     * @return a pool-backed {@code ParallelExecutor}
     */
    static ParallelExecutor forkJoin(ForkJoinPool pool) {
        return new ParallelExecutor() {
            @Override
            public void forEachInt(int upperBound, IntConsumer body) {
                pool.submit(() -> IntStream.range(0, upperBound).parallel().forEach(body)).join();
            }

            @Override
            public void forEach(IntStream source, IntConsumer body) {
                pool.submit(() -> source.parallel().forEach(body)).join();
            }

            @Override
            public <T> void forEach(Stream<T> source, Consumer<T> body) {
                pool.submit(() -> source.parallel().forEach(body)).join();
            }
        };
    }

    /**
     * Returns an executor that runs every iteration sequentially on the calling thread — no worker
     * threads, no pool, and the common pool is left untouched. Graph structure and recall are
     * equivalent to the {@link #forkJoin(ForkJoinPool)} path; only wall-clock and thread usage differ.
     *
     * @return a caller-runs {@code ParallelExecutor}
     */
    static ParallelExecutor callerRuns() {
        return new ParallelExecutor() {
            @Override
            public void forEachInt(int upperBound, IntConsumer body) {
                for (int i = 0; i < upperBound; i++) {
                    body.accept(i);
                }
            }

            @Override
            public void forEach(IntStream source, IntConsumer body) {
                source.forEach(body);
            }

            @Override
            public <T> void forEach(Stream<T> source, Consumer<T> body) {
                source.forEach(body);
            }
        };
    }
}
