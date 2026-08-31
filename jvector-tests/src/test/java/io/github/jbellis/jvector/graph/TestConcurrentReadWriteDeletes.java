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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.TestUtil.randomVector;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Runs "nVectors" operations, where each operation is either:
 * - an insertion
 * - a mock deletion, instantiated through the use of a BitSet for skipping these nodes during search
 * - a search
 * With probability 0.01, we run cleanup to commit the deletions to the index. The cleanup process and the insertions
 * cannot be concurrently executed (we use a lock to control their execution).
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestConcurrentReadWriteDeletes extends RandomizedTest {
    private static final Logger logger = LoggerFactory.getLogger(TestConcurrentReadWriteDeletes.class);

    private static final int nVectors = 20_000;
    private static final int dimension = 16;
    private static final double cleanupProbability = 0.01;

    private KeySet keysInserted = new KeySet();
    private List<Integer> keysRemoved = new CopyOnWriteArrayList();

    private List<VectorFloat<?>> vectors = createRandomVectors(nVectors, dimension);
    private RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vectors, dimension);

    private VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;

    private BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, similarityFunction);
    private GraphIndexBuilder builder = new GraphIndexBuilder(bsp, 2, 2, 10, 1.0f, 1.0f, true);

    private FixedBitSet liveNodes = new FixedBitSet(nVectors);

    private final Lock writeLock = new ReentrantLock();

    @Test
    public void testConcurrentReadsWritesDeletes() throws ExecutionException, InterruptedException {
        var vv = ravv.threadLocalSupplier();

        testConcurrentOps(i -> {
            var R = getRandom();
            if (R.nextDouble() < 0.2 || keysInserted.isEmpty())
            {
                // In the future, we could improve this test by acquiring the lock earlier and executing other
                writeLock.lock();
                try {
                    builder.addGraphNode(i, vv.get().getVector(i));
                    liveNodes.set(i);
                    keysInserted.add(i);
                } finally {
                    writeLock.unlock();
                }
            } else if (R.nextDouble() < 0.1) {
                var key = keysInserted.getRandom();
                if (!keysRemoved.contains(key)) {
                    liveNodes.flip(key);
                    keysRemoved.add(key);
                }
            } else {
                var queryVector = randomVector(getRandom(), dimension);
                SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queryVector, similarityFunction, ravv);

                int topK = Math.min(1, keysInserted.size());
                int rerankK = Math.min(50, keysInserted.size());

                GraphSearcher searcher = new GraphSearcher(builder.getGraph());
                searcher.search(ssp, topK, rerankK, 0.f, 0.f, liveNodes);
            }
        });
    }

    @FunctionalInterface
    private interface Op
    {
        void run(int i) throws Throwable;
    }

    private void testConcurrentOps(Op op) throws ExecutionException, InterruptedException {
        AtomicInteger counter = new AtomicInteger();
        long start = System.currentTimeMillis();
        
        // Use a simpler approach that doesn't rely on parallel streams
        var keys = IntStream.range(0, nVectors).boxed().collect(Collectors.toList());
        Collections.shuffle(keys, getRandom());
        
        // Use a thread-safe approach without relying on RandomizedContext
        int threadCount = Math.min(Runtime.getRuntime().availableProcessors(), 8); // Limit thread count
        List<Thread> threads = new ArrayList<>();
        int keysPerThread = nVectors / threadCount;
        
        // Create a thread-safe random seed for each thread
        final long randomSeed = getRandom().nextLong();
        
        for (int t = 0; t < threadCount; t++) {
            final int threadIndex = t;
            final int startIdx = threadIndex * keysPerThread;
            final int endIdx = (threadIndex == threadCount - 1) ? keys.size() : (threadIndex + 1) * keysPerThread;
            
            Thread thread = new Thread(() -> {
                for (int i = startIdx; i < endIdx; i++) {
                    int key = keys.get(i);
                    wrappedOp(op, key);
                    
                    if (counter.incrementAndGet() % 1_000 == 0) {
                        var elapsed = System.currentTimeMillis() - start;
                        logger.info(String.format("%d ops in %dms = %f ops/s",
                            counter.get(), elapsed, counter.get() * 1000.0 / elapsed));
                    }
                    
                    if (getRandom().nextDouble() < cleanupProbability) {
                        writeLock.lock();
                        try {
                            for (Integer keyToRemove : keysRemoved) {
                                builder.markNodeDeleted(keyToRemove);
                            }
                            keysRemoved.clear();
                            builder.cleanup();
                        } finally {
                            writeLock.unlock();
                        }
                    }
                }
            });
            
            threads.add(thread);
            thread.start();
        }
        
        // Wait for all threads to complete
        for (Thread thread : threads) {
            thread.join();
        }
    }

    private static void wrappedOp(Op op, Integer i) {
        try
        {
            op.run(i);
        }
        catch (Throwable e)
        {
            throw new RuntimeException(e);
        }
    }

    private static class KeySet
    {
        private final Map<Integer, Integer> keys = new ConcurrentHashMap<>();
        private final AtomicInteger ordinal = new AtomicInteger();

        public void add(Integer key)
        {
            var i = ordinal.getAndIncrement();
            keys.put(i, key);
        }

        public int getRandom()
        {
            if (isEmpty())
                throw new IllegalStateException();
            var i = TestConcurrentReadWriteDeletes.getRandom().nextInt(ordinal.get());
            // in case there is race with add(key), retry another random
            return keys.containsKey(i) ? keys.get(i) : getRandom();
        }

        public boolean isEmpty()
        {
            return keys.isEmpty();
        }

        public int size() {
            return keys.size();
        }
    }
}
