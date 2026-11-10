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
package io.github.jbellis.jvector.util;

import io.github.jbellis.jvector.vector.VectorizationProvider;

/**
 * Sets how many threads native BLAS and LAPACK may use for calls made on the current thread.
 *
 * <p>Native BLAS libraries such as MKL and OpenBLAS otherwise start one thread per CPU on their
 * first call, regardless of the host's own executors. JVector's BLAS calls therefore run inside a
 * scope: one thread by default, or the size of the executor a caller passes to the operation.
 * The native vectorization provider applies the limit through the library's per-thread setting
 * (MKL {@code MKL_Set_Num_Threads_Local}, OpenBLAS {@code openblas_set_num_threads_local}), or,
 * for OpenBLAS releases without it, the process-wide setting. Other providers leave BLAS threading
 * to the library's own configuration, such as {@code MKL_NUM_THREADS}.</p>
 *
 * <p>Scopes nest: closing one restores the count that was in effect when it opened.</p>
 */
public final class BlasThreads {
    private static final ThreadLocal<Integer> ACTIVE = new ThreadLocal<>();
    private static final Scope NONE = () -> { };

    private BlasThreads() {
    }

    /** Restores the previous BLAS thread count when closed. */
    public interface Scope extends AutoCloseable {
        @Override
        void close();
    }

    /**
     * Uses {@code threads} BLAS threads for native calls on this thread until the scope closes.
     *
     * @throws IllegalArgumentException if {@code threads} is not positive
     */
    public static Scope limit(int threads) {
        if (threads < 1) throw new IllegalArgumentException("BLAS thread count must be positive, got " + threads);
        var support = VectorizationProvider.getInstance().getVectorUtilSupport();
        Integer outer = ACTIVE.get();
        int previous = support.setBlasThreadsLocal(threads);
        ACTIVE.set(threads);
        return () -> {
            if (previous >= 0) support.setBlasThreadsLocal(previous);
            if (outer == null) ACTIVE.remove();
            else ACTIVE.set(outer);
        };
    }

    /** One BLAS thread, unless an enclosing {@link #limit} scope on this thread already sets the count. */
    public static Scope singleUnlessLimited() {
        return ACTIVE.get() != null ? NONE : limit(1);
    }

    /** The count set by the innermost open scope on this thread, or 0 if none is open. */
    public static int current() {
        Integer active = ACTIVE.get();
        return active == null ? 0 : active;
    }
}
