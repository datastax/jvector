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
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedFeature;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedNVQ;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedVectors;

import org.agrona.collections.Int2IntHashMap;

import java.io.IOException;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

/**
 * Abstract base class for writing graph indexes to disk.
 * @param <T> the type of the output writer
 */
public abstract class AbstractGraphIndexWriter<T extends IndexWriter> implements GraphIndexWriter {
    /** A magic number to indicate the file footer */
    public static final int FOOTER_MAGIC = 0x4a564244;
    /** The size of the offset in the footer. */
    public static final int FOOTER_OFFSET_SIZE = Long.BYTES;
    /** The size of the magic number in the footer. */
    public static final int FOOTER_MAGIC_SIZE = Integer.BYTES;
    /** The total size of the footer. */
    public static final int FOOTER_SIZE = FOOTER_MAGIC_SIZE + FOOTER_OFFSET_SIZE;
    final int version;
    final ImmutableGraphIndex graph;
    final OrdinalMapper ordinalMapper;
    final int dimension;
    final Map<FeatureId, Feature> featureMap;
    final T out; /* output for graph nodes and inline features */
    final long headerSize;

    volatile int maxOrdinalWritten = -1;
    final List<Feature> inlineFeatures;
    final GraphIndexFormat serializer;

    AbstractGraphIndexWriter(T out,
                             int version,
                             ImmutableGraphIndex graph,
                             OrdinalMapper oldToNewOrdinals,
                             int dimension,
                             EnumMap<FeatureId, Feature> features)
    {
        serializer = GraphIndexFormatFactory.forVersion(version);
        if (graph.isHierarchical() && !serializer.supportsMultiLayer()) {
            throw new IllegalArgumentException("Multilayer graphs must be written with version 4 or higher");
        }
        this.version = version;
        this.graph = graph;
        this.ordinalMapper = oldToNewOrdinals;
        this.dimension = dimension;

        if (version <= 5) {
            // Versions <= 5 use the old feature ordering, simply provided by the FeatureId
            this.featureMap = features;
            this.inlineFeatures = features.values().stream().filter(f -> !(f instanceof SeparatedFeature)).collect(Collectors.toList());
        } else {
            // Version 6 uses the new feature ordering to place fused features last in the list
            var sortedFeatures = features.values().stream().sorted().collect(Collectors.toList());
            this.featureMap = new LinkedHashMap<>();
            for (var feature : sortedFeatures) {
                this.featureMap.put(feature.id(), feature);
            }
            this.inlineFeatures = sortedFeatures.stream().filter(f -> !(f instanceof SeparatedFeature)).sorted().collect(Collectors.toList());
        }

        long fusedFeaturesCount = this.inlineFeatures.stream().filter(Feature::isFused).count();
        if (fusedFeaturesCount > 1) {
            throw new IllegalArgumentException("At most one fused feature is allowed");
        }
        if (fusedFeaturesCount == 1 && version < 6) {
            throw new IllegalArgumentException("Fused features require version 6 or higher");
        }
        this.out = out;
        // create a mock Header to determine the correct size
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var ch = new CommonHeader(version, dimension, 0, layerInfo, 0);
        var placeholderHeader = new Header(ch, featureMap);
        this.headerSize = placeholderHeader.size();
    }

    /**
     * Gets the maximum ordinal written so far.
     * @return the maximum ordinal written so far, or -1 if no ordinals have been written yet
     */
    public int getMaxOrdinal() {
        return maxOrdinalWritten;
    }

    /**
     * Gets the set of features.
     * @return the feature set
     */
    public Set<FeatureId> getFeatureSet() {
        return featureMap.keySet();
    }

    WriteContext createContext(long startOffset) {
        return new WriteContext(graph, ordinalMapper, featureMap, inlineFeatures, startOffset, headerSize, dimension);
    }

    long featureOffsetForOrdinal(long startOffset, int ordinal) {
        return serializer.featureOffsetForOrdinal(createContext(startOffset), ordinal);
    }

    /**
     * Computes sequential renumbering for graph ordinals.
     * @param graph the graph index to renumber
     * @return a Map of old to new graph ordinals where the new ordinals are sequential starting at 0,
     * while preserving the original relative ordering in `graph`.  That is, for all node ids i and j,
     * if i &lt; j in `graph` then map[i] &lt; map[j] in the returned map.  "Holes" left by
     * deleted nodes are filled in by shifting down the new ordinals.
     */
    public static Map<Integer, Integer> sequentialRenumbering(ImmutableGraphIndex graph) {
        try (var view = graph.getView()) {
            Int2IntHashMap oldToNewMap = new Int2IntHashMap(-1);
            int nextOrdinal = 0;
            for (int i = 0; i < view.getIdUpperBound(); i++) {
                if (graph.containsNode(i)) {
                    oldToNewMap.put(i, nextOrdinal++);
                }
            }
            return oldToNewMap;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Write the {@link Header} as a footer for the graph index.
     * <p>
     * To read the graph later, we will perform the following steps:
     * <ol>
     *     <li> Find the magic number at the end of the slice
     *     <li> Read the header offset from the end of the slice
     *     <li> Read the header
     *     <li> Read the neighbors offsets and graph metadata
     * </ol>
     * @param headerOffset the offset of the header in the slice
     * @throws IOException IOException
     */
    void writeFooter(ImmutableGraphIndex.View view, long headerOffset, long startOffset) throws IOException {
        serializer.writeFooter(createContext(startOffset), headerOffset, out);
    }

    /**
     * Writes the index header, including the graph size, so that OnDiskGraphIndex can open it.
     * The output IS flushed.
     * @param view the graph index view
     * @param startOffset the start offset
     * @throws IOException if an I/O error occurs
     */
    protected synchronized void writeHeader(ImmutableGraphIndex.View view, long startOffset) throws IOException {
        serializer.writeHeader(createContext(startOffset), out);
    }

    void writeSparseLevels(ImmutableGraphIndex.View view, Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers, long startOffset) throws IOException {
        serializer.writeSparseLevels(createContext(startOffset), out, featureStateSuppliers);
    }

    void writeSeparatedFeatures(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers, long startOffset) throws IOException {
        serializer.writeSeparatedFeatures(createContext(startOffset), out, featureStateSuppliers);
    }

    /**
     * Builder for {@link AbstractGraphIndexWriter}, with optional features.
     * <p>
     * Subclasses should implement `reallyBuild` to return the appropriate type.
     * <p>
     * K - the type of the writer to build
     * T - the type of the output stream
     * @param <K> the type of the writer to build
     * @param <T> the type of the output stream
     */
    public abstract static class Builder<K extends AbstractGraphIndexWriter<T>, T extends IndexWriter> {
        final ImmutableGraphIndex graphIndex;
        final EnumMap<FeatureId, Feature> features;
        final T out;
        OrdinalMapper ordinalMapper;
        int version;

        /**
         * Constructs a Builder.
         * @param graphIndex the graph index
         * @param out the output writer
         */
        public Builder(ImmutableGraphIndex graphIndex, T out) {
            this.graphIndex = graphIndex;
            this.out = out;
            this.features = new EnumMap<>(FeatureId.class);
            this.version = OnDiskGraphIndex.CURRENT_VERSION;
        }

        /**
         * Sets the version.
         * @param version the version
         * @return this builder
         */
        public Builder<K, T> withVersion(int version) {
            if (version > OnDiskGraphIndex.CURRENT_VERSION) {
                throw new IllegalArgumentException("Unsupported version: " + version);
            }

            this.version = version;
            return this;
        }

        /**
         * Adds a feature.
         * @param feature the feature
         * @return this builder
         */
        public Builder<K, T> with(Feature feature) {
            features.put(feature.id(), feature);
            return this;
        }

        /**
         * Sets the ordinal mapper.
         * @param ordinalMapper the ordinal mapper
         * @return this builder
         */
        public Builder<K, T> withMapper(OrdinalMapper ordinalMapper) {
            this.ordinalMapper = ordinalMapper;
            return this;
        }

        /**
         * Builds the writer.
         * @return the writer
         * @throws IOException if an I/O error occurs
         */
        public K build() throws IOException {
            if (version < 3 && (!features.containsKey(FeatureId.INLINE_VECTORS) || features.size() > 1)) {
                throw new IllegalArgumentException("Only INLINE_VECTORS is supported until version 3");
            }

            int dimension;
            if (features.containsKey(FeatureId.INLINE_VECTORS)) {
                dimension = ((InlineVectors) features.get(FeatureId.INLINE_VECTORS)).dimension();
            } else if (features.containsKey(FeatureId.NVQ_VECTORS)) {
                dimension = ((NVQ) features.get(FeatureId.NVQ_VECTORS)).dimension();
            } else if (features.containsKey(FeatureId.SEPARATED_VECTORS)) {
                dimension = ((SeparatedVectors) features.get(FeatureId.SEPARATED_VECTORS)).dimension();
            } else if (features.containsKey(FeatureId.SEPARATED_NVQ)) {
                dimension = ((SeparatedNVQ) features.get(FeatureId.SEPARATED_NVQ)).dimension();
            } else {
                throw new IllegalArgumentException("Inline or separated vector feature must be provided");
            }

            if (ordinalMapper == null) {
                ordinalMapper = new OrdinalMapper.MapMapper(sequentialRenumbering(graphIndex));
            }
            return reallyBuild(dimension);
        }

        /**
         * Actually builds the writer with the given dimension.
         * @param dimension the dimension
         * @return the writer
         * @throws IOException if an I/O error occurs
         */
        protected abstract K reallyBuild(int dimension) throws IOException;

        /**
         * Sets the ordinal mapping.
         * @param oldToNewOrdinals the old to new ordinals map
         * @return this builder
         */
        public Builder<K, T> withMap(Map<Integer, Integer> oldToNewOrdinals) {
            return withMapper(new OrdinalMapper.MapMapper(oldToNewOrdinals));
        }

        /**
         * Gets a feature by ID.
         * @param featureId the feature ID
         * @return the feature
         */
        public Feature getFeature(FeatureId featureId) {
            return features.get(featureId);
        }
    }
}
