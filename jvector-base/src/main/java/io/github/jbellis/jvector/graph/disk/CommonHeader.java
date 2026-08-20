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

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Base header for OnDiskGraphIndex functionality, containing essential metadata about the graph structure.
 * <p>
 * This class stores:
 * - Version information for format compatibility
 * - Vector dimension
 * - Entry node for graph traversal
 * - Layer information for multi-layer graphs (HNSW)
 * - ID upper bound (maximum node ID + 1)
 * <p>
 * The format evolves across versions:
 * - v2: Basic format with no magic number
 * - v3: Added magic number and feature set support
 * - v4: Added multi-layer support and ID upper bound
 * <p>
 * The on-disk layout for v4+ is:
 * - Magic number (to identify JVector files)
 * - Version
 * - Base layer size
 * - Vector dimension
 * - Entry node ID
 * - Base layer max degree
 * - ID upper bound
 * - Number of layers
 * - Layer info (size and degree for each layer)
 */
public class CommonHeader {
    private static final Logger logger = LoggerFactory.getLogger(CommonHeader.class);

    protected static final int V4_MAX_LAYERS = 32;

    public final int version;
    public final int dimension;
    public final int entryNode;
    public final List<LayerInfo> layerInfo;
    public final int idUpperBound;
    private final GraphIndexFormat graphIndexFormat;

    CommonHeader(int version, int dimension, int entryNode, List<LayerInfo> layerInfo, int idUpperBound) {
        this.version = version;
        this.dimension = dimension;
        this.entryNode = entryNode;
        this.layerInfo = layerInfo;
        this.idUpperBound = idUpperBound;
        this.graphIndexFormat = GraphIndexFormatFactory.forVersion(version);
    }

    void write(IndexWriter out) throws IOException {
        logger.debug("Writing common header at position {}", out.position());
        graphIndexFormat.writeCommonHeader(out, layerInfo, dimension, entryNode, idUpperBound);
        logger.debug("Common header finished writing at position {}", out.position());
    }

    static CommonHeader load(RandomAccessReader in) throws IOException {
        return GraphIndexFormat.loadCommonHeader(in);
    }

    int size() {
        return graphIndexFormat.commonHeaderSize();
    }

    GraphIndexFormat getGraphIndexFormat() {
        return graphIndexFormat;
    }

    @VisibleForTesting
    public static class LayerInfo {
        public final int size;
        public final int degree;

        public LayerInfo(int size, int degree) {
            this.size = size;
            this.degree = degree;
        }

        public static List<LayerInfo> fromGraph(ImmutableGraphIndex graph, OrdinalMapper mapper) {
            return IntStream.rangeClosed(0, graph.getMaxLevel())
                    .mapToObj(i -> new LayerInfo(graph.size(i), graph.getDegree(i)))
                    .collect(Collectors.toList());
        }

        @Override
        public String toString() {
            return "LayerInfo{" +
                    "size=" + size +
                    ", degree=" + degree +
                    '}';
        }

        @Override
        public int hashCode() {
            return Objects.hash(size, degree);
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            LayerInfo other = (LayerInfo) obj;
            return size == other.size && degree == other.degree;
        }
    }
}