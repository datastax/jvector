package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.example.dynamic.DynamicDataset;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Set;

public interface ConfiguredSystem extends AutoCloseable {
    SearchScoreProvider scoreProviderFor(VectorFloat<?> queryVector, GraphIndex.View view);

    GraphSearcher getSearcher();

    QueryBundle getQueryBundle();

    GraphIndex getIndex();

    class StaticConfiguredSystem implements ConfiguredSystem {
        private Dataset ds;
        private GraphIndex index;
        private CompressedVectors cv;
        private Set<FeatureId> features;

        private final ExplicitThreadLocal<GraphSearcher> searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(index));

        public StaticConfiguredSystem(Dataset ds, GraphIndex index, CompressedVectors cv, Set<FeatureId> features) {
            this.ds = ds;
            this.index = index;
            this.cv = cv;
            this.features = features;
        }

        public SearchScoreProvider scoreProviderFor(VectorFloat<?> queryVector, GraphIndex.View view) {
            // if we're not compressing then just use the exact score function
            if (cv == null) {
                return DefaultSearchScoreProvider.exact(queryVector, ds.getSimilarityFunction(), ds.getBaseRavv());
            }

            var scoringView = (GraphIndex.ScoringView) view;
            ScoreFunction.ApproximateScoreFunction asf;
            if (features.contains(FeatureId.FUSED_ADC)) {
                asf = scoringView.approximateScoreFunctionFor(queryVector, ds.getSimilarityFunction());
            } else {
                asf = cv.precomputedScoreFunctionFor(queryVector, ds.getSimilarityFunction());
            }
            var rr = scoringView.rerankerFor(queryVector, ds.getSimilarityFunction());
            return new DefaultSearchScoreProvider(asf, rr);
        }

        public GraphSearcher getSearcher() {
            return searchers.get();
        }

        public QueryBundle getQueryBundle() {
            return ds.getQueryBundle();
        }

        /**
         * Get the underlying graph index.
         *
         * @return the graph index.
         */
        public GraphIndex getIndex() {
            return index;
        }

        @Override
        public void close() throws Exception {
            searchers.close();
        }
    }

    class DynamicConfiguredSystem implements ConfiguredSystem {
        private DynamicDataset ds;
        private GraphIndex index;
        private CompressedVectors cv;
        private Set<FeatureId> features;
        private int epoch;

        private final ExplicitThreadLocal<GraphSearcher> searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(index));

        public DynamicConfiguredSystem(DynamicDataset ds, GraphIndex index, CompressedVectors cv, Set<FeatureId> features) {
            this.ds = ds;
            this.index = index;
            this.cv = cv;
            this.features = features;
            epoch = 0;
        }

        void setEpoch(int epoch) {
            this.epoch = epoch;
        }

        public SearchScoreProvider scoreProviderFor(VectorFloat<?> queryVector, GraphIndex.View view) {
            // if we're not compressing then just use the exact score function
            if (cv == null) {
                return DefaultSearchScoreProvider.exact(queryVector, ds.getSimilarityFunction(), ds.getBaseRavv());
            }

            var scoringView = (GraphIndex.ScoringView) view;
            ScoreFunction.ApproximateScoreFunction asf;
            if (features.contains(FeatureId.FUSED_ADC)) {
                asf = scoringView.approximateScoreFunctionFor(queryVector, ds.getSimilarityFunction());
            } else {
                asf = cv.precomputedScoreFunctionFor(queryVector, ds.getSimilarityFunction());
            }
            var rr = scoringView.rerankerFor(queryVector, ds.getSimilarityFunction());
            return new DefaultSearchScoreProvider(asf, rr);
        }

        public GraphSearcher getSearcher() {
            return searchers.get();
        }

        public QueryBundle getQueryBundle() {
            return ds.getQueryBundle(epoch);
        }

        /**
         * Get the underlying graph index.
         *
         * @return the graph index.
         */
        public GraphIndex getIndex() {
            return index;
        }

        @Override
        public void close() throws Exception {
            searchers.close();
        }
    }
}
