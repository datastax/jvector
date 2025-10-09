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

package io.github.jbellis.jvector.graph.disk.feature;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.disk.CommonHeader;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;
import java.util.function.BiFunction;

/**
 * An enum representing the features that can be stored in an on-disk graph index.
 * The order of this Enum SHOULD NOT be changed, as it affects serialization structure of graphs.
 * New features should be added to the end.
 * These are typically mapped to a Feature.
 */
public enum FeatureId {
    /** Vectors stored inline with the graph structure */
    INLINE_VECTORS(InlineVectors::load),

    /** Fused asymmetric distance computation for efficient similarity search */
    FUSED_ADC(FusedADC::load),

    /** Vectors compressed using Neighborhood Vector Quantization */
    NVQ_VECTORS(NVQ::load),

    /** Vectors stored separately from the graph structure */
    SEPARATED_VECTORS(SeparatedVectors::load),

    /** NVQ-compressed vectors stored separately from the graph structure */
    SEPARATED_NVQ(SeparatedNVQ::load);

    /** A set containing all available feature IDs */
    public static final Set<FeatureId> ALL = Collections.unmodifiableSet(EnumSet.allOf(FeatureId.class));

    private final BiFunction<CommonHeader, RandomAccessReader, Feature> loader;

    FeatureId(BiFunction<CommonHeader, RandomAccessReader, Feature> loader) {
        this.loader = loader;
    }

    /**
     * Loads the Feature implementation associated with this FeatureId from disk.
     *
     * @param header the common header containing graph metadata
     * @param reader the reader for accessing the on-disk data
     * @return the loaded Feature instance
     */
    public Feature load(CommonHeader header, RandomAccessReader reader) {
        return loader.apply(header, reader);
    }

    /**
     * Deserializes a set of FeatureIds from a bitfield representation.
     *
     * @param bitflags the bitfield where each bit represents the presence of a feature
     * @return an EnumSet containing the features indicated by the bitfield
     */
    public static EnumSet<FeatureId> deserialize(int bitflags) {
        EnumSet<FeatureId> set = EnumSet.noneOf(FeatureId.class);
        for (int n = 0; n < values().length; n++) {
            if ((bitflags & (1 << n)) != 0)
                set.add(values()[n]);
        }
        return set;
    }

    /**
     * Serializes a set of FeatureIds into a bitfield representation.
     *
     * @param flags the set of features to serialize
     * @return a bitfield where each bit represents the presence of a feature
     */
    public static int serialize(EnumSet<FeatureId> flags) {
        int i = 0;
        for (FeatureId flag : flags)
            i |= 1 << flag.ordinal();
        return i;
    }
}
