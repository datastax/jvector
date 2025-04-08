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

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.DataOutput;
import java.io.IOException;
import java.util.EnumMap;
import java.util.EnumSet;

/**
 * Header information for an on-disk graph index, reflecting the common header and feature-specific headers.
 */
class Header {
    final CommonHeader common;
    final EnumMap<FeatureId, ? extends Feature> features;

    Header(CommonHeader common, EnumMap<FeatureId, ? extends Feature> features) {
        this.common = common;
        this.features = features;
    }

    void write(RandomAccessWriter out) throws IOException {
        common.write(out);

        if (common.version >= 3) {
            out.writeInt(FeatureId.serialize(EnumSet.copyOf(features.keySet())));
        }

        // we restrict pre-version-3 writers to INLINE_VECTORS features, so we don't need additional version-handling here
        for (Feature writer : features.values()) {
            writer.writeHeader(out);
        }
    }

    public int size() {
        int size = common.size();

        if (common.version >= 3) {
            size += Integer.BYTES;
        }

        size += features.values().stream().mapToInt(Feature::headerSize).sum();

        return size;
    }

    static Header load(RandomAccessReader reader, long offset) throws IOException {
        reader.seek(offset);

        EnumSet<FeatureId> featureIds;
        EnumMap<FeatureId, Feature> features = new EnumMap<>(FeatureId.class);
        CommonHeader common = CommonHeader.load(reader);
        if (common.version >= 3) {
            featureIds = FeatureId.deserialize(reader.readInt());
        } else {
            featureIds = EnumSet.of(FeatureId.INLINE_VECTORS);
        }

        for (FeatureId featureId : featureIds) {
            features.put(featureId, featureId.load(common, reader));
        }

        return new Header(common, features);
    }
}