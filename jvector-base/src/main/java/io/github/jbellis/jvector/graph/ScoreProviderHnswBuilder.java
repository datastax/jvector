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

import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.management.CompressionType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An {@link HnswIndexBuilder} that builds with a caller-supplied {@link BuildScoreProvider}, e.g. one
 * that scores with compressed vectors. Returned by
 * {@code Indexes.hnswBuilder(BuildScoreProvider, int)}.
 * <p>
 * Because the score provider is fixed, {@link #withCompressionType} is not supported here. Having
 * no vectors of its own, it is populated with {@link #populateGraph(RandomAccessVectorValues)} or
 * {@link #addGraphNode}; {@link #buildAndPopulate()} throws.
 */
public class ScoreProviderHnswBuilder extends HnswIndexBuilder {
    private static final Logger logger = LoggerFactory.getLogger(ScoreProviderHnswBuilder.class);

    private final BuildScoreProvider scoreProvider;
    private final int dimension;

    /**
     * Creates a builder for a graph of vectors of the given {@code dimension}, scored with
     * {@code scoreProvider}.
     */
    public ScoreProviderHnswBuilder(BuildScoreProvider scoreProvider, int dimension) {
        this.scoreProvider = scoreProvider;
        this.dimension = dimension;
    }

    /**
     * Not supported: logs a warning and leaves the builder unchanged, since the score provider
     * passed to the constructor determines how the graph is scored.
     */
    @Override
    public HnswIndexBuilder withCompressionType(CompressionType compressionType) {
        logger.warn("Compression type is not supported when using a BuildScoreProvider. Ignoring the provided compression type: {}", compressionType);
        return this;
    }

    @Override
    protected GraphIndexBuilder createGraphBuilder() {
        return newGraphBuilder(scoreProvider, dimension);
    }

}
