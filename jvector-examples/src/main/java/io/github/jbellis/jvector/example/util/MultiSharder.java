package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.MultiSearchResult;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public interface MultiSharder {
    /**
     * Returns the RandomAccessVectorValues corresponding to the specified shard
     * @param shard the selected shard
     * @return the RandomAccessVectorValues
     */
    RandomAccessVectorValues getShard(int shard);

    /**
     * Returns the number of shards
     */
    int size();

    /**
     * Returns a function that converts a MultiSearchResult.NodeScore to a SearchResult.NodeScore
     */
    Function<MultiSearchResult.NodeScore, SearchResult.NodeScore> getConverter();

    class ContiguousSharder implements MultiSharder {
        private final List<RandomAccessVectorValues> ravvs;

        public ContiguousSharder(RandomAccessVectorValues ravv, int shards) {
            ravvs = new ArrayList<>();
            int batchSize = (int) Math.ceil(((double) ravv.size()) / shards);
            for (int ii = 0; ii < shards; ii++) {
                int maxSize = Math.min((ii + 1) * batchSize, ravv.size());
                ravvs.add(new ShardedRandomAccessVectorValues(ravv, ii * batchSize, maxSize));
            }
        }

        @Override
        public RandomAccessVectorValues getShard(int shard) {
            return ravvs.get(shard);
        }

        @Override
        public int size() {
            return ravvs.size();
        }

        @Override
        public Function<MultiSearchResult.NodeScore, SearchResult.NodeScore> getConverter() {
            return nodeScore -> {
                int globalId = 0;
                for (int ii = 0; ii < nodeScore.index; ii++) {
                    globalId += ravvs.get(ii).size();
                }
                return new SearchResult.NodeScore(globalId + nodeScore.node, nodeScore.score);
            };
        }

        private static class ShardedRandomAccessVectorValues implements RandomAccessVectorValues {
            private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
            private final VectorFloat<?> scratch;

            private final RandomAccessVectorValues ravv;
            private final int start;
            private final int end;

            ShardedRandomAccessVectorValues(RandomAccessVectorValues ravv, int start, int end) {
                this.ravv = ravv;
                this.start = start;
                this.end = end;

                this.scratch = vectorTypeSupport.createFloatVector(ravv.dimension());
            }

            @Override
            public int size() {
                return end - start;
            }

            @Override
            public int dimension() {
                return scratch.length();
            }

            @Override
            public VectorFloat<?> getVector(int nodeId) {
                var original = ravv.getVector(start + nodeId);
                // present a single vector reference to callers like the disk-backed RAVV implmentations,
                // to catch cases where they are not making a copy
                scratch.copyFrom(original, 0, 0, original.length());
                return scratch;
            }

            @Override
            public boolean isValueShared() {
                return true;
            }

            @Override
            public RandomAccessVectorValues copy() {
                return new ShardedRandomAccessVectorValues(ravv, start, end);
            }
        }
    }
}