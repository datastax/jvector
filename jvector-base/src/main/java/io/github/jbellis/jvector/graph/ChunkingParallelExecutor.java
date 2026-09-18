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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.PrimitiveIterator;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * {@link ParallelExecutor} over a caller-supplied {@link ExecutorService}: the calling thread
 * traverses the source and submits fixed-size chunks to the executor, keeping a bounded number in
 * flight. Backs {@link ParallelExecutor#over(ExecutorService, int)}; see the {@link ParallelExecutor}
 * class javadoc for how this implementation's caveats compare with the other factories.
 */
final class ChunkingParallelExecutor implements ParallelExecutor {
    /** Elements per submitted chunk on the stream paths ({@code forEachInt} splits its range evenly instead). */
    private static final int BATCH_SIZE = 32;
    /** Chunks per worker on the {@code forEachInt} path: more than one smooths skewed per-element cost. */
    private static final int CHUNKS_PER_WORKER = 4;

    // Nested use from a body degrades to inline execution: a chunk that blocked on sub-chunks of
    // the same bounded executor could starve it into deadlock (every worker waiting on chunks that
    // can never be scheduled).
    private static final ThreadLocal<Boolean> IN_BODY = ThreadLocal.withInitial(() -> Boolean.FALSE);

    private final ExecutorService executor;
    private final int parallelism;

    ChunkingParallelExecutor(ExecutorService executor, int parallelism) {
        this.executor = executor;
        this.parallelism = parallelism;
    }

    @Override
    public void forEachInt(int upperBound, IntConsumer body) {
        if (upperBound <= 0) {
            return;
        }
        if (IN_BODY.get()) {
            for (int i = 0; i < upperBound; i++) {
                body.accept(i);
            }
            return;
        }
        int chunks = Math.min(upperBound, CHUNKS_PER_WORKER * parallelism);
        Drain drain = new Drain(Integer.MAX_VALUE); // chunk count is fixed and small: no window needed
        for (int c = 0; c < chunks && drain.healthy(); c++) {
            int start = (int) ((long) upperBound * c / chunks);
            int end = (int) ((long) upperBound * (c + 1) / chunks);
            drain.submit(() -> {
                for (int i = start; i < end; i++) {
                    body.accept(i);
                }
            });
        }
        drain.finish();
    }

    @Override
    public void forEach(IntStream source, IntConsumer body) {
        if (IN_BODY.get()) {
            source.forEach(body);
            return;
        }
        Drain drain = new Drain(2 * parallelism);
        try {
            PrimitiveIterator.OfInt it = source.iterator();
            int[] batch = new int[BATCH_SIZE];
            int n = 0;
            while (drain.healthy() && it.hasNext()) {
                batch[n++] = it.nextInt();
                if (n == BATCH_SIZE) {
                    int[] b = batch;
                    int len = n;
                    batch = new int[BATCH_SIZE];
                    n = 0;
                    drain.submit(() -> {
                        for (int i = 0; i < len; i++) {
                            body.accept(b[i]);
                        }
                    });
                }
            }
            if (drain.healthy() && n > 0) {
                int[] b = batch;
                int len = n;
                drain.submit(() -> {
                    for (int i = 0; i < len; i++) {
                        body.accept(b[i]);
                    }
                });
            }
        } catch (Throwable t) {
            drain.fail(t); // traversal failure: recorded, then drained below like any other
        }
        drain.finish();
    }

    @Override
    public <T> void forEach(Stream<T> source, Consumer<T> body) {
        if (IN_BODY.get()) {
            source.forEach(body);
            return;
        }
        Drain drain = new Drain(2 * parallelism);
        try {
            Iterator<T> it = source.iterator();
            List<T> batch = new ArrayList<>(BATCH_SIZE);
            while (drain.healthy() && it.hasNext()) {
                batch.add(it.next());
                if (batch.size() == BATCH_SIZE) {
                    List<T> b = batch;
                    batch = new ArrayList<>(BATCH_SIZE);
                    drain.submit(() -> b.forEach(body));
                }
            }
            if (drain.healthy() && !batch.isEmpty()) {
                List<T> b = batch;
                drain.submit(() -> b.forEach(body));
            }
        } catch (Throwable t) {
            drain.fail(t);
        }
        drain.finish();
    }

    /**
     * Chunk bookkeeping with the drain-before-unwind discipline used elsewhere in jvector: once any
     * chunk (or the traversal) fails, chunks that have not begun skip themselves via {@code aborted},
     * but every started chunk is waited out — the caller never unwinds (and so never releases
     * resources) beneath a still-running body. {@link Future#cancel} is deliberately never used:
     * {@code cancel(false)} succeeds on a <em>running</em> {@code FutureTask} (the flag only governs
     * interruption), making {@code get()} return while the body still executes — the exact unwind
     * this class exists to prevent. Interruption is noted, honored after the drain, and never used
     * to abort a running chunk.
     */
    private final class Drain {
        private final int window;
        private final ArrayDeque<Future<?>> inFlight = new ArrayDeque<>();
        // Written by the orchestrating thread, read by workers: a failing iteration stops issuing
        // chunks here and not-yet-started chunks become no-ops.
        private volatile boolean aborted;
        private Throwable failure;
        private boolean interrupted;

        Drain(int window) {
            this.window = window;
        }

        boolean healthy() {
            return failure == null && !interrupted;
        }

        void fail(Throwable t) {
            if (failure == null) {
                failure = t;
            }
            aborted = true;
        }

        void submit(Runnable chunk) {
            if (!healthy()) {
                return;
            }
            try {
                inFlight.add(executor.submit(() -> {
                    if (aborted) {
                        return; // iteration is failing: skip a chunk that has not begun
                    }
                    IN_BODY.set(Boolean.TRUE);
                    try {
                        chunk.run();
                    } finally {
                        IN_BODY.remove();
                    }
                }));
            } catch (RejectedExecutionException e) {
                fail(e);
                return;
            }
            if (inFlight.size() >= window) {
                settle(inFlight.poll());
            }
        }

        /** Waits {@code f} out (across interrupts), recording any failure. */
        private void settle(Future<?> f) {
            while (true) {
                try {
                    f.get();
                    return;
                } catch (ExecutionException e) {
                    fail(e.getCause());
                    return;
                } catch (InterruptedException e) {
                    interrupted = true; // note it and keep waiting: no unwind under running work
                    aborted = true;     // unstarted chunks need not run
                }
            }
        }

        /** Awaits every outstanding chunk, then rethrows the first failure and restores the interrupt flag. */
        void finish() {
            while (!inFlight.isEmpty()) {
                settle(inFlight.poll());
            }
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
            if (failure instanceof RuntimeException) {
                throw (RuntimeException) failure;
            }
            if (failure instanceof Error) {
                throw (Error) failure;
            }
            if (failure != null) {
                throw new RuntimeException(failure);
            }
            if (interrupted) {
                throw new RuntimeException(new InterruptedException("interrupted while awaiting parallel iteration"));
            }
        }
    }
}
