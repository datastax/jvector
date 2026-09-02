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
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * {@link ParallelExecutor} that hosts each iteration as a parallel stream on a {@link ForkJoinPool}.
 * Backs {@link ParallelExecutor#forkJoin(ForkJoinPool)}.
 *
 * <p><b>Why bodies never throw.</b> The obvious implementation lets a failing body throw straight
 * out of the stream, and that is what this class used to do. It unwinds too early: a parallel
 * stream's fork/join tree propagates the failure up through {@code join()} as soon as one branch
 * throws, so the submitted task can complete — and the caller resume — while sibling bodies are
 * still running on pool workers. For an embedder that owns the memory those bodies are reading,
 * that is a use-after-free waiting to happen: it releases the resource on the strength of the call
 * having returned, and a still-running body faults on an unmapped page (a JVM-level SIGSEGV, not a
 * catchable exception).
 *
 * <p>So the body is wrapped: a failure is <em>recorded</em> rather than thrown, elements after the
 * first failure become no-ops, the stream therefore always completes normally, and the recorded
 * failure is rethrown to the caller once {@code join()} has returned. Because the stream completed
 * normally, {@code join()} returning means every forked subtask finished — which is exactly the
 * guarantee "no body is still running when the call returns."
 *
 * <p>The wrapper is installed once per iteration, not per element.
 */
final class ForkJoinParallelExecutor implements ParallelExecutor {
    private final ForkJoinPool pool;

    ForkJoinParallelExecutor(ForkJoinPool pool) {
        this.pool = pool;
    }

    @Override
    public void forEachInt(int upperBound, IntConsumer body) {
        Guard guard = new Guard();
        IntConsumer guarded = guard.wrap(body);
        pool.submit(() -> guard.traverse(() -> IntStream.range(0, upperBound).parallel().forEach(guarded))).join();
        guard.rethrow();
    }

    @Override
    public void forEach(IntStream source, IntConsumer body) {
        Guard guard = new Guard();
        IntConsumer guarded = guard.wrap(body);
        pool.submit(() -> guard.traverse(() -> source.parallel().forEach(guarded))).join();
        guard.rethrow();
    }

    @Override
    public <T> void forEach(Stream<T> source, Consumer<T> body) {
        Guard guard = new Guard();
        Consumer<T> guarded = guard.wrap(body);
        pool.submit(() -> guard.traverse(() -> source.parallel().forEach(guarded))).join();
        guard.rethrow();
    }

    /** Collects the first failure of an iteration without letting it escape a body. */
    private static final class Guard {
        private final AtomicReference<Throwable> failure = new AtomicReference<>();

        private void record(Throwable t) {
            failure.compareAndSet(null, t);
        }

        IntConsumer wrap(IntConsumer body) {
            return i -> {
                if (failure.get() != null) {
                    return; // already failing: remaining elements are no-ops, not new work
                }
                try {
                    body.accept(i);
                } catch (Throwable t) {
                    record(t);
                }
            };
        }

        <T> Consumer<T> wrap(Consumer<T> body) {
            return t -> {
                if (failure.get() != null) {
                    return;
                }
                try {
                    body.accept(t);
                } catch (Throwable thrown) {
                    record(thrown);
                }
            };
        }

        /**
         * Runs the traversal, recording a failure raised by the stream <em>source</em> rather than
         * by a body (a throwing spliterator, say). Such a failure can still unwind the fork/join
         * tree early, so the drain guarantee covers body failures — the ones an embedder's own code
         * raises — and not a source that breaks mid-traversal.
         */
        void traverse(Runnable traversal) {
            try {
                traversal.run();
            } catch (Throwable t) {
                record(t);
            }
        }

        void rethrow() {
            Throwable t = failure.get();
            if (t instanceof RuntimeException) {
                throw (RuntimeException) t;
            }
            if (t instanceof Error) {
                throw (Error) t;
            }
            if (t != null) {
                throw new RuntimeException(t);
            }
        }
    }
}
