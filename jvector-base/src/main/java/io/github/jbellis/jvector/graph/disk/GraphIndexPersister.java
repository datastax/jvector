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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.PersistableGraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * Fluent builder for persisting a {@link GraphIndex} to disk. Implements
 * {@link GraphIndex.WriteBuilder} and wraps the appropriate internal writer strategy
 * based on the graph type and output target.
 * <p>
 * Obtain instances via {@link GraphIndex#writer(Path)} or
 * {@link PersistableGraphIndex#writer(IndexWriter)}; do not instantiate directly.
 * <p>
 * All configuration methods must be called before the first
 * {@link #writeFeaturesInline} or {@link #write} call.
 */
public class GraphIndexPersister implements GraphIndex.WriteBuilder {
    private static final Logger logger = LoggerFactory.getLogger(GraphIndexPersister.class);

    private final PersistableGraphIndex graph;
    private final Path path;                // null for sequential mode
    private final IndexWriter sequentialOut; // null for path mode

    private final EnumMap<FeatureId, Feature> features = new EnumMap<>(FeatureId.class);
    private OrdinalMapper mapper;
    private int version = OnDiskGraphIndex.CURRENT_VERSION;
    private long startOffset = 0L;
    private int parallelWorkerThreads = 0;
    private boolean parallelDirectBuffers = false;

    /** Lazy-initialized once the first write call is made; null for sequential mode. */
    private RandomAccessOnDiskGraphIndexWriter lazyWriter;

    /** Path-based constructor: uses the parallel writer for on-heap graphs, random-access for on-disk. */
    public GraphIndexPersister(PersistableGraphIndex graph, Path path) throws FileNotFoundException {
        this.graph = graph;
        this.path = path;
        this.sequentialOut = null;
    }

    /** Sequential constructor: always uses the sequential writer regardless of graph type. */
    public GraphIndexPersister(PersistableGraphIndex graph, IndexWriter sequentialOut) {
        this.graph = graph;
        this.path = null;
        this.sequentialOut = sequentialOut;
    }

    @Override
    public GraphIndex.WriteBuilder with(Feature feature) {
        features.put(feature.id(), feature);
        return this;
    }

    @Override
    public GraphIndex.WriteBuilder withMapper(OrdinalMapper mapper) {
        this.mapper = mapper;
        return this;
    }

    @Override
    public GraphIndex.WriteBuilder withMap(Map<Integer, Integer> oldToNew) {
        return withMapper(new OrdinalMapper.MapMapper(oldToNew));
    }

    @Override
    public GraphIndex.WriteBuilder withVersion(int version) {
        this.version = version;
        return this;
    }

    @Override
    public GraphIndex.WriteBuilder withStartOffset(long offset) {
        this.startOffset = offset;
        return this;
    }

    @Override
    public GraphIndex.WriteBuilder withParallelWorkerThreads(int n) {
        if (!(graph instanceof OnHeapGraphIndex)) {
            logger.warn("withParallelWorkerThreads() has no effect for on-disk graph indexes");
        }
        this.parallelWorkerThreads = n;
        return this;
    }

    @Override
    public GraphIndex.WriteBuilder withParallelDirectBuffers(boolean useDirectBuffers) {
        if (!(graph instanceof OnHeapGraphIndex)) {
            logger.warn("withParallelDirectBuffers() has no effect for on-disk graph indexes");
        }
        this.parallelDirectBuffers = useDirectBuffers;
        return this;
    }

    @Override
    public GraphIndex.WriteBuilder writeFeaturesInline(int ordinal, Map<FeatureId, Feature.State> stateMap) throws IOException {
        getOrCreateRandomAccessWriter().writeFeaturesInline(ordinal, stateMap);
        return this;
    }

    @Override
    public void writeHeader(GraphIndex.View view) throws IOException {
        if (sequentialOut != null) {
            throw new UnsupportedOperationException("writeHeader is not supported for sequential writers");
        }
        getOrCreateRandomAccessWriter().writeHeader((PersistableGraphIndex.View) view);
    }

    @Override
    public long checksum() throws IOException {
        if (sequentialOut != null) {
            throw new UnsupportedOperationException("checksum is not supported for sequential writers");
        }
        return getOrCreateRandomAccessWriter().checksum();
    }

    @Override
    public void write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException {
        if (sequentialOut != null) {
            writeSequential(featureStateSuppliers);
        } else {
            getOrCreateRandomAccessWriter().write(featureStateSuppliers);
        }
    }

    @Override
    public void close() throws IOException {
        if (lazyWriter != null) {
            lazyWriter.close();
            lazyWriter = null;
        }
        // sequential writer is not owned by this persister; caller closes it
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    private RandomAccessOnDiskGraphIndexWriter getOrCreateRandomAccessWriter() throws IOException {
        if (lazyWriter == null) {
            lazyWriter = buildRandomAccessWriter();
        }
        return lazyWriter;
    }

    private RandomAccessOnDiskGraphIndexWriter buildRandomAccessWriter() throws IOException {
        if (graph instanceof OnHeapGraphIndex && parallelWorkerThreads != 0) {
            // parallelWorkerThreads < 0 means "use all available processors" (passes 0 to underlying writer)
            int threads = parallelWorkerThreads < 0 ? 0 : parallelWorkerThreads;
            var b = new OnDiskParallelGraphIndexWriter.Builder(graph, path)
                    .withStartOffset(startOffset)
                    .withParallelWorkerThreads(threads)
                    .withParallelDirectBuffers(parallelDirectBuffers);
            applyCommonConfig(b);
            return b.build();
        } else {
            var b = new OnDiskGraphIndexWriter.Builder(graph, path)
                    .withStartOffset(startOffset);
            applyCommonConfig(b);
            return b.build();
        }
    }

    private void writeSequential(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException {
        var b = new OnDiskSequentialGraphIndexWriter.Builder(graph, sequentialOut);
        applyCommonConfig(b);
        try (var writer = b.build()) {
            writer.write(featureStateSuppliers);
        }
    }

    private <K extends AbstractGraphIndexWriter<T>, T extends IndexWriter>
    void applyCommonConfig(AbstractGraphIndexWriter.Builder<K, T> b) {
        for (var feature : features.values()) {
            b.with(feature);
        }
        if (mapper != null) {
            b.withMapper(mapper);
        }
        b.withVersion(version);
    }
}
