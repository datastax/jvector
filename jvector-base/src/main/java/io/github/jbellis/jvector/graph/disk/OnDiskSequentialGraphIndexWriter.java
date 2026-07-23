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
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.*;

import java.io.IOException;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * Writes a graph index to disk in a format that can be loaded as an OnDiskGraphIndex.
 * <p>
 * Unlike {@link OnDiskGraphIndexWriter}, this class always writes in a sequential order and the header metadata is written as the footer.
 * <p>
 * Assumptions:
 * <ul>
 * <li> The graph already exists and is not modified as it is being written, and therefore can be written sequentially in a single pass.
 * <li> This assumption is valid for two common cases in Log Structure Merge Tree-based systems such as Cassandra and Lucene:
 *   <ol>
 *   <li> The graph is being written as part of compaction
 *   <li> The graph is being written for addition of a small immutable segment.
 *   </ol>
 * </ul>
 * <p>
 * Goals:
 * <ul>
 * <li> Immutability: Every byte written to the index file is immutable. This allows for running calculation of checksums without needing to re-read the file.
 * <li> Performance: We can take advantage of sequential writes for performance.
 * </ul>
 * <p>
 * The above goals are driven by the following motivations:
 * <ul>
 * <li> When we work with either cloud object storage where random writes are not supported on a single stream
 * <li> When we embed jVector in frameworks such as Lucene that rely on sequential writes for performance and correctness
 * </ul>
 */
public class OnDiskSequentialGraphIndexWriter extends AbstractGraphIndexWriter<IndexWriter> {

    OnDiskSequentialGraphIndexWriter(IndexWriter out,
                                             int version,
                                             ImmutableGraphIndex graph,
                                             OrdinalMapper oldToNewOrdinals,
                                             int dimension,
                                             EnumMap<FeatureId, Feature> features)
    {
        super(out, version, graph, oldToNewOrdinals, dimension, features);
    }

    @Override
    public synchronized void close() throws IOException {
        // Note: we don't close the output streams since we don't own them in this writer
    }

    /**
     * Note: There are several limitations you should be aware of when using:
     * <ul>
     * <li> This method doesn't persist (e.g. flush) the output streams.  The caller is responsible for doing so.
     * <li> This method does not support writing to "holes" in the ordinal space.  If your ordinal mapper
     *      maps a new ordinal to an old ordinal that does not exist in the graph, an exception will be thrown.
     * </ul>
     */
    @Override
    public synchronized void write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException
    {
        long startOffset = out.position();
        serializer.writeOnDiskSequential(createContext(startOffset), out, featureStateSuppliers);
    }


    /**
     * Builder for {@link OnDiskSequentialGraphIndexWriter}, with optional features.
     */
    public static class Builder extends AbstractGraphIndexWriter.Builder<OnDiskSequentialGraphIndexWriter, IndexWriter> {
        public Builder(ImmutableGraphIndex graphIndex, IndexWriter out) {
            super(graphIndex, out);
        }

        @Override
        protected OnDiskSequentialGraphIndexWriter reallyBuild(int dimension) {
            return new OnDiskSequentialGraphIndexWriter(out, version, graphIndex, ordinalMapper, dimension, features);

        }
    }
}