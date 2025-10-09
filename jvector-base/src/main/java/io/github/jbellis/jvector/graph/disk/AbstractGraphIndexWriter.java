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
import io.github.jbellis.jvector.graph.disk.feature.*;
import org.agrona.collections.Int2IntHashMap;

import java.io.IOException;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

/**
 * Abstract base class for writing graph indexes to disk in various formats.
 * <p>
 * This writer handles the serialization of graph structure, including nodes, edges, and associated
 * features (such as vectors) to a persistent storage format. It supports both inline features
 * (written alongside graph nodes) and separated features (written in a dedicated section).
 * <p>
 * The on-disk format consists of:
 * <ul>
 *   <li>A header containing metadata and feature information</li>
 *   <li>Graph nodes with inline features and edge lists</li>
 *   <li>Sparse levels for hierarchical graphs (if applicable)</li>
 *   <li>Separated feature data (if any)</li>
 *   <li>A footer containing the header offset and magic number</li>
 * </ul>
 * <p>
 * Subclasses must implement the specific writing strategy (e.g., sequential or random access).
 * <p>
 * Thread safety: This class uses synchronized methods where necessary but is not designed
 * for concurrent writes from multiple threads. The {@code maxOrdinalWritten} field is volatile
 * to support visibility across threads.
 *
 * @param <T> the type of {@link IndexWriter} used for output operations
 */
public abstract class AbstractGraphIndexWriter<T extends IndexWriter> implements  GraphIndexWriter {
    /** Magic number written at the end of the index file to identify valid JVector graph files. */
    public static final int FOOTER_MAGIC = 0x4a564244; // "EOF magic"

    /** Size in bytes of the header offset field in the footer. */
    public static final int FOOTER_OFFSET_SIZE = Long.BYTES;

    /** Size in bytes of the magic number field in the footer. */
    public static final int FOOTER_MAGIC_SIZE = Integer.BYTES;

    /** Total size in bytes of the footer (magic number plus offset). */
    public static final int FOOTER_SIZE = FOOTER_MAGIC_SIZE + FOOTER_OFFSET_SIZE;

    /** The format version number for this graph index. */
    final int version;

    /** The immutable graph structure to be written to disk. */
    final ImmutableGraphIndex graph;

    /** Maps between original graph ordinals and the ordinals written to disk. */
    final OrdinalMapper ordinalMapper;

    /** The dimensionality of the vectors stored in this index. */
    final int dimension;

    /**
     * Map of features to be written with this index.
     * <p>
     * Uses {@code EnumMap} to ensure consistent ordering and avoid ordering bugs,
     * even though map-specific features are not utilized.
     */
    final EnumMap<FeatureId, Feature> featureMap;

    /** Output writer for graph nodes and inline features. */
    final T out;

    /** The size in bytes of the index header. */
    final int headerSize;

    /**
     * The maximum ordinal that has been written so far, or -1 if no ordinals have been written yet.
     * <p>
     * This field is volatile to ensure visibility across threads.
     */
    volatile int maxOrdinalWritten = -1;

    /** List of features that are written inline with graph nodes (not separated). */
    final List<Feature> inlineFeatures;

    /**
     * Constructs an abstract graph index writer with the specified configuration.
     *
     * @param out the output writer for graph nodes and inline features
     * @param version the format version number for this graph index
     * @param graph the immutable graph structure to be written to disk
     * @param oldToNewOrdinals maps original graph ordinals to new ordinals for writing
     * @param dimension the dimensionality of the vectors stored in this index
     * @param features map of features to be written with this index
     * @throws IllegalArgumentException if attempting to write a multilayer graph with version less than 4
     */
    AbstractGraphIndexWriter(T out,
                                     int version,
                                     ImmutableGraphIndex graph,
                                     OrdinalMapper oldToNewOrdinals,
                                     int dimension,
                                     EnumMap<FeatureId, Feature> features)
    {
        if (graph.getMaxLevel() > 0 && version < 4) {
            throw new IllegalArgumentException("Multilayer graphs must be written with version 4 or higher");
        }
        this.version = version;
        this.graph = graph;
        this.ordinalMapper = oldToNewOrdinals;
        this.dimension = dimension;
        this.featureMap = features;
        this.inlineFeatures = features.values().stream().filter(f -> !(f instanceof SeparatedFeature)).collect(Collectors.toList());
        this.out = out;

        // create a mock Header to determine the correct size
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var ch = new CommonHeader(version, dimension, 0, layerInfo, 0);
        var placeholderHeader = new Header(ch, featureMap);
        this.headerSize = placeholderHeader.size();
    }

    /**
     * Returns the maximum ordinal that has been written so far.
     *
     * @return the maximum ordinal written so far, or -1 if no ordinals have been written yet
     */
    public int getMaxOrdinal() {
        return maxOrdinalWritten;
    }

    /**
     * Returns the set of feature IDs that will be written with this index.
     *
     * @return an unmodifiable set of {@link FeatureId} values configured for this writer
     */
    public Set<FeatureId> getFeatureSet() {
        return featureMap.keySet();
    }

    /**
     * Calculates the byte offset where inline features for a given ordinal begin in the output stream.
     * <p>
     * This calculation accounts for the header size, all previous nodes' data (ordinals, inline features,
     * and edges), and the ordinal field of the target node.
     *
     * @param startOffset the starting offset in the output stream where the graph data begins
     * @param ordinal the node ordinal for which to calculate the feature offset
     * @return the absolute byte offset where the inline features for the specified ordinal are located
     */
    long featureOffsetForOrdinal(long startOffset, int ordinal) {
        int edgeSize = Integer.BYTES * (1 + graph.getDegree(0));
        long inlineBytes = ordinal * (long) (Integer.BYTES + inlineFeatures.stream().mapToInt(Feature::featureSize).sum() + edgeSize);
        return startOffset
                + headerSize
                + inlineBytes // previous nodes
                + Integer.BYTES; // the ordinal of the node whose features we're about to write
    }

    /**
     * Checks whether a feature should be written separately from the main graph data.
     *
     * @param feature the feature to check
     * @return {@code true} if the feature is a {@link SeparatedFeature}, {@code false} otherwise
     */
    boolean isSeparated(Feature feature) {
        return feature instanceof SeparatedFeature;
    }

    /**
     * Creates a sequential renumbering map that eliminates gaps in ordinal numbering.
     * <p>
     * Returns a map of old to new graph ordinals where the new ordinals are sequential starting at 0,
     * while preserving the original relative ordering in the graph. That is, for all node ids i and j,
     * if i &lt; j in the graph then map[i] &lt; map[j] in the returned map. "Holes" left by
     * deleted nodes are filled in by shifting down the new ordinals.
     * <p>
     * This is useful for creating compact on-disk representations where deleted nodes do not
     * leave unused space in the ordinal range.
     *
     * @param graph the immutable graph to renumber
     * @return a map from original ordinals to sequential new ordinals
     * @throws RuntimeException if an exception occurs while accessing the graph view
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
     * Writes the graph header as a footer at the end of the index file.
     * <p>
     * The footer format enables efficient index reading by storing the header at the end,
     * allowing readers to locate and parse metadata without scanning the entire file.
     * <p>
     * To read the graph later, the following steps are performed:
     * <ol>
     *     <li>Find the magic number at the end of the file</li>
     *     <li>Read the header offset from the end of the file</li>
     *     <li>Seek to the header offset and read the header</li>
     *     <li>Parse the graph metadata and feature information from the header</li>
     * </ol>
     *
     * @param view the graph view containing the entry node and other metadata
     * @param headerOffset the byte offset where the header begins in the output stream
     * @throws IOException if an I/O error occurs while writing the footer
     */
    void writeFooter(ImmutableGraphIndex.View view, long headerOffset) throws IOException {
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var commonHeader = new CommonHeader(version,
                dimension,
                ordinalMapper.oldToNew(view.entryNode().node),
                layerInfo,
                ordinalMapper.maxOrdinal() + 1);
        var header = new Header(commonHeader, featureMap);
        header.write(out); // write the header
        out.writeLong(headerOffset); // We write the offset of the header at the end of the file
        out.writeInt(FOOTER_MAGIC);
        final long expectedPosition = headerOffset + headerSize + FOOTER_SIZE;
        assert out.position() == expectedPosition : String.format("%d != %d", out.position(), expectedPosition);
    }

    /**
     * Writes the index header at the beginning of the output stream.
     * <p>
     * The header includes graph metadata such as version, dimension, entry node, layer information,
     * and feature configuration. After writing, the output is flushed to ensure the header is
     * persisted to disk.
     * <p>
     * This method is public to allow writing the header early in the process, enabling
     * {@code OnDiskGraphIndex} to open the index and read features before the edge data
     * is fully written. This is useful for incremental or staged writing scenarios.
     *
     * @param view the graph view containing the entry node and other metadata
     * @param startOffset the byte offset in the output stream where the header should begin
     * @throws IOException if an I/O error occurs while writing the header
     */
    public synchronized void writeHeader(ImmutableGraphIndex.View view, long startOffset) throws IOException {
        // graph-level properties
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var commonHeader = new CommonHeader(version,
                dimension,
                ordinalMapper.oldToNew(view.entryNode().node),
                layerInfo,
                ordinalMapper.maxOrdinal() + 1);
        var header = new Header(commonHeader, featureMap);
        header.write(out);
        assert out.position() == startOffset + headerSize : String.format("%d != %d", out.position(), startOffset + headerSize);
    }

    /**
     * Writes the sparse levels of a hierarchical graph to the output stream.
     * <p>
     * For graphs with multiple layers (levels &gt; 0), this method writes each sparse level
     * sequentially. Each level contains only a subset of nodes that participate in that level
     * of the hierarchy. For each node in a level, the method writes:
     * <ul>
     *   <li>The remapped node ordinal</li>
     *   <li>The number of neighbors at this level</li>
     *   <li>The remapped neighbor ordinals, padded to the level's degree with -1 values</li>
     * </ul>
     *
     * @param view the graph view providing access to nodes and neighbors at each level
     * @throws IOException if an I/O error occurs while writing the sparse levels
     * @throws IllegalStateException if the number of nodes written does not match the expected layer size
     */
    void writeSparseLevels(ImmutableGraphIndex.View view) throws IOException {
        // write sparse levels
        for (int level = 1; level <= graph.getMaxLevel(); level++) {
            int layerSize = graph.size(level);
            int layerDegree = graph.getDegree(level);
            int nodesWritten = 0;
            for (var it = graph.getNodes(level); it.hasNext(); ) {
                int originalOrdinal = it.nextInt();
                // node id
                final int newOrdinal = ordinalMapper.oldToNew(originalOrdinal);
                out.writeInt(newOrdinal);
                // neighbors
                var neighbors = view.getNeighborsIterator(level, originalOrdinal);
                out.writeInt(neighbors.size());
                int n = 0;
                for ( ; n < neighbors.size(); n++) {
                    out.writeInt(ordinalMapper.oldToNew(neighbors.nextInt()));
                }
                assert !neighbors.hasNext() : "Mismatch between neighbor's reported size and actual size";
                // pad out to degree
                for (; n < layerDegree; n++) {
                    out.writeInt(-1);
                }
                nodesWritten++;
            }
            if (nodesWritten != layerSize) {
                throw new IllegalStateException("Mismatch between layer size and nodes written");
            }
        }
    }

    /**
     * Writes separated features to dedicated sections in the output stream.
     * <p>
     * Separated features are stored apart from the main graph node data, which can improve
     * cache locality and enable more efficient access patterns for certain use cases.
     * This method iterates through all features marked as {@link SeparatedFeature} and writes
     * their data sequentially for each node ordinal.
     * <p>
     * For each separated feature:
     * <ul>
     *   <li>Records the current output position as the feature's offset</li>
     *   <li>Writes the feature data for each node in ordinal order</li>
     *   <li>Writes zero-padding for ordinals that have been omitted (deleted nodes)</li>
     * </ul>
     *
     * @param featureStateSuppliers a map from feature IDs to functions that provide feature state
     *                              for a given node ordinal; must contain suppliers for all separated features
     * @throws IOException if an I/O error occurs while writing feature data
     * @throws IllegalStateException if a supplier is missing for a separated feature
     */
    void writeSeparatedFeatures(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException {
        for (var featureEntry : featureMap.entrySet()) {
            if (isSeparated(featureEntry.getValue())) {
                var fid = featureEntry.getKey();
                var supplier = featureStateSuppliers.get(fid);
                if (supplier == null) {
                    throw new IllegalStateException("Supplier for feature " + fid + " not found");
                }

                // Set the offset for this feature
                var feature = (SeparatedFeature) featureEntry.getValue();
                feature.setOffset(out.position());

                // Write separated data for each node
                for (int newOrdinal = 0; newOrdinal <= ordinalMapper.maxOrdinal(); newOrdinal++) {
                    int originalOrdinal = ordinalMapper.newToOld(newOrdinal);
                    if (originalOrdinal != OrdinalMapper.OMITTED) {
                        feature.writeSeparately(out, supplier.apply(originalOrdinal));
                    } else {
                        // write zeros for missing data as padding
                        for (int i = 0; i < feature.featureSize(); i++) {
                            out.writeByte(0);
                        }
                    }
                }
            }
        }
    }

    /**
     * Builder for constructing {@link AbstractGraphIndexWriter} instances with configurable features.
     * <p>
     * This builder provides a fluent API for configuring graph index writers. It allows specifying:
     * <ul>
     *   <li>Format version</li>
     *   <li>Features to include (vectors, compression, etc.)</li>
     *   <li>Ordinal mapping strategy</li>
     * </ul>
     * <p>
     * The builder performs validation to ensure the requested configuration is compatible with
     * the specified format version. For example, version 3 and earlier only support inline vectors,
     * while version 4 and later are required for multilayer graphs.
     * <p>
     * Subclasses must implement {@link #reallyBuild(int)} to construct the appropriate writer type.
     * <p>
     * Example usage:
     * <pre>{@code
     * var writer = new MyGraphIndexWriter.Builder(graph, output)
     *     .withVersion(4)
     *     .with(new InlineVectors(dimension))
     *     .withMapper(ordinalMapper)
     *     .build();
     * }</pre>
     *
     * @param <K> the concrete type of {@link AbstractGraphIndexWriter} to build
     * @param <T> the type of {@link IndexWriter} used for output operations
     */
    public abstract static class Builder<K extends AbstractGraphIndexWriter<T>, T extends IndexWriter> {
        /** The immutable graph index to be written. */
        final ImmutableGraphIndex graphIndex;

        /** Map of features to include in the index, keyed by feature ID. */
        final EnumMap<FeatureId, Feature> features;

        /** The output writer for graph data. */
        final T out;

        /** Optional ordinal mapper for renumbering nodes; defaults to sequential renumbering if not set. */
        OrdinalMapper ordinalMapper;

        /** The format version to use; defaults to {@link OnDiskGraphIndex#CURRENT_VERSION}. */
        int version;

        /**
         * Constructs a new builder for the specified graph and output writer.
         *
         * @param graphIndex the immutable graph to write to disk
         * @param out the output writer for graph data
         */
        public Builder(ImmutableGraphIndex graphIndex, T out) {
            this.graphIndex = graphIndex;
            this.out = out;
            this.features = new EnumMap<>(FeatureId.class);
            this.version = OnDiskGraphIndex.CURRENT_VERSION;
        }

        /**
         * Sets the format version for the index.
         * <p>
         * Different versions support different features:
         * <ul>
         *   <li>Version 1-2: Basic graph structure with inline vectors only</li>
         *   <li>Version 3: Support for multiple feature types</li>
         *   <li>Version 4+: Required for multilayer graphs</li>
         * </ul>
         *
         * @param version the format version to use
         * @return this builder for method chaining
         * @throws IllegalArgumentException if the version is greater than {@link OnDiskGraphIndex#CURRENT_VERSION}
         */
        public Builder<K, T> withVersion(int version) {
            if (version > OnDiskGraphIndex.CURRENT_VERSION) {
                throw new IllegalArgumentException("Unsupported version: " + version);
            }

            this.version = version;
            return this;
        }

        /**
         * Adds a feature to be written with the index.
         * <p>
         * Features include vector storage (inline or separated), compression schemes,
         * and other node-associated data. Each feature is identified by its {@link FeatureId}.
         * If a feature with the same ID is already registered, it will be replaced.
         *
         * @param feature the feature to add
         * @return this builder for method chaining
         */
        public Builder<K, T> with(Feature feature) {
            features.put(feature.id(), feature);
            return this;
        }

        /**
         * Sets the ordinal mapper for renumbering nodes during writing.
         * <p>
         * The ordinal mapper controls how node ordinals in the source graph are mapped to
         * ordinals in the written index. This is useful for eliminating gaps from deleted nodes
         * or for mapping to external identifiers (e.g., database row IDs).
         * <p>
         * If no mapper is specified, {@link #build()} will use {@link #sequentialRenumbering(ImmutableGraphIndex)}
         * to create a mapper that eliminates gaps.
         *
         * @param ordinalMapper the ordinal mapper to use
         * @return this builder for method chaining
         */
        public Builder<K, T> withMapper(OrdinalMapper ordinalMapper) {
            this.ordinalMapper = ordinalMapper;
            return this;
        }

        /**
         * Builds and returns the configured graph index writer.
         * <p>
         * This method performs validation to ensure the configuration is valid:
         * <ul>
         *   <li>Version 3 and earlier must use only {@code INLINE_VECTORS}</li>
         *   <li>At least one vector feature must be present (inline, separated, or compressed)</li>
         *   <li>If no ordinal mapper is set, sequential renumbering is applied</li>
         * </ul>
         * <p>
         * The vector dimension is extracted from whichever vector feature is configured.
         *
         * @return the configured graph index writer
         * @throws IOException if an I/O error occurs during writer initialization
         * @throws IllegalArgumentException if the configuration is invalid for the specified version,
         *         or if no vector feature is provided
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
         * Constructs the concrete writer instance with the specified dimension.
         * <p>
         * This method is called by {@link #build()} after validation and dimension extraction
         * are complete. Subclasses must implement this to instantiate their specific writer type.
         *
         * @param dimension the vector dimensionality extracted from the configured features
         * @return the concrete graph index writer instance
         * @throws IOException if an I/O error occurs during writer construction
         */
        protected abstract K reallyBuild(int dimension) throws IOException;

        /**
         * Sets the ordinal mapper using a map from old to new ordinals.
         * <p>
         * This is a convenience method equivalent to calling
         * {@code withMapper(new OrdinalMapper.MapMapper(oldToNewOrdinals))}.
         *
         * @param oldToNewOrdinals a map from original graph ordinals to new ordinals for writing
         * @return this builder for method chaining
         */
        public Builder<K, T> withMap(Map<Integer, Integer> oldToNewOrdinals) {
            return withMapper(new OrdinalMapper.MapMapper(oldToNewOrdinals));
        }

        /**
         * Returns the feature associated with the specified feature ID.
         *
         * @param featureId the ID of the feature to retrieve
         * @return the feature with the specified ID, or {@code null} if no such feature is configured
         */
        public Feature getFeature(FeatureId featureId) {
            return features.get(featureId);
        }
    }
}
