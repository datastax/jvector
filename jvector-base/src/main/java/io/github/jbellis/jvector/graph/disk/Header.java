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
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.util.Map;

/**
 * Header information for an on-disk graph index, containing both common metadata and feature-specific headers.
 * <p>
 * This class encapsulates:
 * - Common header information (version, dimension, entry node, etc.)
 * - Feature set information (which features are included in the index)
 * - Feature-specific header data
 * <p>
 * The header can be written at the beginning of the index file or alternatively in a separate metadata file and is read when loading an index.
 * It provides all the metadata needed to correctly interpret the on-disk format of the graph.
 */
class Header {
    final CommonHeader common;

    // In V6, it is important that the features map is sorted according to the feature order defined in AbstractFeature
    // In V5 and older, these maps use the FeatureId as the sorting order
    final Map<FeatureId, ? extends Feature> features;

    Header(CommonHeader common, Map<FeatureId, ? extends Feature> features) {
        this.common = common;
        this.features = features;
    }

    void write(IndexWriter out) throws IOException {
        common.write(out);
        common.getGraphIndexFormat().writeHeaderFeatures(out, features);
    }

    public int size() {
        return common.getGraphIndexFormat().headerSize(features);
    }

    static Header load(RandomAccessReader reader, long offset) throws IOException {
        reader.seek(offset);
        Map<FeatureId, Feature> features;
        CommonHeader common = CommonHeader.load(reader);
        features = common.getGraphIndexFormat().loadHeaderFeatures(reader, common);
        return new Header(common, features);
    }
}