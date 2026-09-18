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

import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Runs a {@link GraphIndexBuilder}'s internal build/finalize iterations to completion, blocking the
 * calling thread until every element has been processed. The implementation decides <em>how</em> the
 * iteration is distributed.
 * <p>
 * This is the seam that lets an embedder bound vector-graph construction to its own thread budget —
 * e.g. one thread per compaction — instead of a jvector-owned all-core pool. It is the build/finalize
 * counterpart to the caller-runs executor injection already available on the compaction merge path.
 *
 * <h2>Choosing a factory</h2>
 * The provided implementations trade off differently; the caveats below are relative to each other:
 * <ul>
 *   <li>{@link #forkJoin(ForkJoinPool)} — the historical behavior: each iteration is hosted as a
 *   parallel stream inside the pool, so the <em>whole</em> pipeline is decomposed — upstream stages
 *   of a stream source parallelize along with the body. Nested use from a body is safe (fork/join
 *   absorbs it). Caveats: requires a {@code ForkJoinPool} specifically; the blocking join does not
 *   respond to interruption; and parallel-stream semantics do not strictly guarantee that no body
 *   invocation is still running when a failure unwinds.</li>
 *   <li>{@link #callerRuns()} — zero threads, no pool: everything runs sequentially on the calling
 *   thread in encounter order, stopping at the first failure with nothing left running. Cannot
 *   deadlock and cannot leave work behind after unwind. Caveat: wall-clock scales with a single
 *   core.</li>
 *   <li>{@link #over(ExecutorService, int)} — for embedders holding a plain
 *   {@link ExecutorService}: elements are chunked and submitted while the calling thread
 *   orchestrates with a bounded in-flight window. Only the <em>body</em> is distributed — the
 *   source stream is traversed on the calling thread, so expensive upstream stages serialize
 *   (unlike {@code forkJoin}). Nested use from a body degrades to inline execution rather than
 *   deadlocking a bounded pool. On failure or interrupt it stops issuing chunks, waits out every
 *   started chunk — never unwinding beneath a still-running body — then rethrows the first failure
 *   and restores the interrupt flag. The executor's lifecycle stays with the caller; it is never
 *   shut down here.</li>
 * </ul>
 * Common to all three: the calling thread blocks until the iteration settles, and the first
 * observed body failure is rethrown to the caller; whether other elements still run after a
 * failure differs per the caveats above.
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
     * {@code ForkJoinPool}-based {@link GraphIndexBuilder} constructors. See the class javadoc for
     * how its caveats compare with the other factories.
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
     * See the class javadoc for how its caveats compare with the other factories.
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

    /**
     * Returns an executor backed by a caller-supplied {@code ExecutorService}: iterations are split
     * into chunks submitted to {@code executor} while the calling thread traverses the source and
     * bounds the in-flight window. Parallel streams cannot be hosted on a generic
     * {@code ExecutorService} (inside its workers they would silently run on the common pool), so
     * this adapter distributes only the <em>body</em>; see the class javadoc for the full caveat
     * comparison with {@link #forkJoin(ForkJoinPool)} and {@link #callerRuns()}.
     * <p>
     * {@code parallelism} must be stated explicitly because {@code ExecutorService} does not expose
     * its width: state the executor's actual thread count — a larger value merely queues, a smaller
     * one under-uses it. Passing a {@link ForkJoinPool} delegates to {@link #forkJoin(ForkJoinPool)}
     * (whole-pipeline stream decomposition is strictly better there) and {@code parallelism} is
     * ignored in that case.
     *
     * @param executor    runs the chunked iterations; its lifecycle remains the caller's (never shut down here)
     * @param parallelism the intended number of chunks in flight, typically the executor's thread
     *                    count; must be {@code >= 1}
     * @return a chunk-submitting {@code ParallelExecutor}
     * @throws IllegalArgumentException if {@code parallelism < 1}
     */
    static ParallelExecutor over(ExecutorService executor, int parallelism) {
        Objects.requireNonNull(executor, "executor");
        if (parallelism < 1) {
            throw new IllegalArgumentException("parallelism must be >= 1, got " + parallelism);
        }
        if (executor instanceof ForkJoinPool) {
            return forkJoin((ForkJoinPool) executor);
        }
        return new ChunkingParallelExecutor(executor, parallelism);
    }
}
