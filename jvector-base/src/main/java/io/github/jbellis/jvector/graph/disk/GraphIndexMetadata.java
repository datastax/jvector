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

import java.util.List;
import java.util.Objects;

/**
 * Metadata about a graph index, containing all the information needed to
 * interpret the on-disk format. This replaces the scattered fields that were
 * previously in CommonHeader and provides a cleaner API.
 */
public class GraphIndexMetadata {
    private final int version;
    private final int dimension;
    private final int entryNode;
    private final List<CommonHeader.LayerInfo> layerInfo;
    private final int idUpperBound;

    public GraphIndexMetadata(int version, int dimension, int entryNode, 
                             List<CommonHeader.LayerInfo> layerInfo, int idUpperBound) {
        this.version = version;
        this.dimension = dimension;
        this.entryNode = entryNode;
        this.layerInfo = layerInfo;
        this.idUpperBound = idUpperBound;
    }

    public int getVersion() {
        return version;
    }

    public int getDimension() {
        return dimension;
    }

    public int getEntryNode() {
        return entryNode;
    }

    public List<CommonHeader.LayerInfo> getLayerInfo() {
        return layerInfo;
    }

    public int getIdUpperBound() {
        return idUpperBound;
    }

    /**
     * Gets the size of the base layer (layer 0).
     */
    public int getBaseLayerSize() {
        return layerInfo.get(0).size;
    }

    /**
     * Gets the max degree of the base layer (layer 0).
     */
    public int getBaseLayerDegree() {
        return layerInfo.get(0).degree;
    }

    /**
     * Gets the number of layers in the graph.
     */
    public int getNumLayers() {
        return layerInfo.size();
    }

    /**
     * Checks if this is a multi-layer (hierarchical) graph.
     */
    public boolean isMultiLayer() {
        return layerInfo.size() > 1;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GraphIndexMetadata that = (GraphIndexMetadata) o;
        return version == that.version &&
               dimension == that.dimension &&
               entryNode == that.entryNode &&
               idUpperBound == that.idUpperBound &&
               Objects.equals(layerInfo, that.layerInfo);
    }

    @Override
    public int hashCode() {
        return Objects.hash(version, dimension, entryNode, layerInfo, idUpperBound);
    }

    @Override
    public String toString() {
        return "GraphIndexMetadata{" +
               "version=" + version +
               ", dimension=" + dimension +
               ", entryNode=" + entryNode +
               ", layers=" + layerInfo.size() +
               ", idUpperBound=" + idUpperBound +
               '}';
    }
}

// Made with Bob
