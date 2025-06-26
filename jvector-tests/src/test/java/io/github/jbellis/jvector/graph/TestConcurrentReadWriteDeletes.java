package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.TestUtil.randomVector;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestConcurrentReadWriteDeletes extends LuceneTestCase {
    private static final int nVectors = 200_000;
    private static final int dimension = 16;

    KeySet keysInserted = new KeySet();
    List<Integer> keysRemoved = new CopyOnWriteArrayList();

    List<VectorFloat<?>> vectors = createRandomVectors(nVectors, dimension);
    RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vectors, dimension);

    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;

    BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, similarityFunction);
    GraphIndexBuilder builder = new GraphIndexBuilder(bsp, 2, 2, 10, 1.0f, 1.0f, true);

    FixedBitSet liveNodes = new FixedBitSet(nVectors);

    @Test
    public void testConcurrentReadsWritesDeletes() {
        try {
            testConcurrentReadsWritesDeletes(false);
//            testConcurrentReadsWritesDeletes(true);
        } catch (ExecutionException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private void testConcurrentReadsWritesDeletes(boolean addHierarchy) throws ExecutionException, InterruptedException {
        var vv = ravv.threadLocalSupplier();

        testConcurrentOps(addHierarchy, i -> {
            var R = ThreadLocalRandom.current();
            if (R.nextDouble() < 0.2 || keysInserted.isEmpty())
            {
                builder.addGraphNode(i, vv.get().getVector(i));
                liveNodes.set(i);
                keysInserted.add(i);
            } else if (R.nextDouble() < 0.1) {
                var key = keysInserted.getRandom();
                liveNodes.flip(key);
                keysRemoved.add(key);
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

    private void testConcurrentOps(boolean addHierarchy, Op op) throws ExecutionException, InterruptedException {
        AtomicInteger counter = new AtomicInteger();
        long start = System.currentTimeMillis();
        var fjp = ForkJoinPool.commonPool();
        var keys = IntStream.range(0, nVectors).boxed().collect(Collectors.toList());
        Collections.shuffle(keys);
        var task = fjp.submit(() -> keys.stream().parallel().forEach(i ->
        {
            wrappedOp(op, i);
            if (counter.incrementAndGet() % 10_000 == 0)
            {
                var elapsed = System.currentTimeMillis() - start;
                System.out.println(String.format("%d ops in %dms = %f ops/s", counter.get(), elapsed, counter.get() * 1000.0 / elapsed));
            }
            if (ThreadLocalRandom.current().nextDouble() < 0.001) {
//                System.out.println("Cleanup");
                for (Integer key : keysRemoved) {
                    builder.markNodeDeleted(key);
                }
                keysRemoved.clear();
                builder.cleanup();
            }
        }));
        fjp.shutdown();
        task.get(); // re-throw
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
            var i = ThreadLocalRandom.current().nextInt(ordinal.get());
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
