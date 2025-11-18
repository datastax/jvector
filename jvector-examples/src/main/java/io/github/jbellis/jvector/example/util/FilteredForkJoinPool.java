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

package io.github.jbellis.jvector.example.util;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinWorkerThread;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A custom ForkJoinPool implementation that creates worker threads with a specific naming pattern
 * to make them identifiable for thread leak detection.
 */
public class FilteredForkJoinPool extends ForkJoinPool {
    
    /**
     * Creates a ForkJoinPool with the same parallelism as {@link ForkJoinPool#commonPool()}
     * but with custom named threads for leak detection.
     *
     * @return a new ForkJoinPool instance with custom thread factory
     */
    public static ForkJoinPool createFilteredPool() {
        return createFilteredPool(Runtime.getRuntime().availableProcessors());
    }
    
    /**
     * Creates a ForkJoinPool with the specified parallelism and custom named threads for leak detection.
     *
     * @param parallelism the parallelism level
     * @return a new ForkJoinPool instance with custom thread factory
     */
    public static ForkJoinPool createFilteredPool(int parallelism) {
        return new ForkJoinPool(
                parallelism,
                new JVectorForkJoinWorkerThreadFactory(),
                null,
                false);
    }
    
    /**
     * Custom thread factory that creates worker threads with a specific naming pattern.
     */
    private static class JVectorForkJoinWorkerThreadFactory implements ForkJoinPool.ForkJoinWorkerThreadFactory {
        private static final AtomicInteger THREAD_COUNTER = new AtomicInteger(0);
        
        @Override
        public ForkJoinWorkerThread newThread(ForkJoinPool pool) {
            ForkJoinWorkerThread thread = new JVectorForkJoinWorkerThread(pool);
            thread.setName("jvector-fjp-worker-" + THREAD_COUNTER.getAndIncrement());
            thread.setDaemon(true);
            return thread;
        }
    }
    
    /**
     * Custom worker thread class that can be easily identified for thread leak detection.
     */
    private static class JVectorForkJoinWorkerThread extends ForkJoinWorkerThread {
        protected JVectorForkJoinWorkerThread(ForkJoinPool pool) {
            super(pool);
        }
    }
}
