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

import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.util.EnumSet;
import java.util.Set;

/**
 * Abstract base class for graph index serializers providing common functionality.
 */
abstract class AbstractGraphIndexSerializer implements GraphIndexSerializer {
    private final int version;
    private final Set<FeatureId> supportedFeatures;
    private final boolean supportsMultiLayer;
    private final boolean usesFooter;
    private final FeatureOrdering featureOrdering;

    protected AbstractGraphIndexSerializer(int version, 
                                          Set<FeatureId> supportedFeatures,
                                          boolean supportsMultiLayer,
                                          boolean usesFooter,
                                          FeatureOrdering featureOrdering) {
        this.version = version;
        this.supportedFeatures = supportedFeatures;
        this.supportsMultiLayer = supportsMultiLayer;
        this.usesFooter = usesFooter;
        this.featureOrdering = featureOrdering;
    }

    @Override
    public int getVersion() {
        return version;
    }

    @Override
    public boolean supportsFeature(FeatureId feature) {
        return supportedFeatures.contains(feature);
    }

    @Override
    public boolean supportsMultiLayer() {
        return supportsMultiLayer;
    }

    @Override
    public boolean usesFooter() {
        return usesFooter;
    }

    @Override
    public FeatureOrdering getFeatureOrdering() {
        return featureOrdering;
    }

    /**
     * Helper to create a set of all features (for versions that support all features).
     */
    protected static Set<FeatureId> allFeatures() {
        return EnumSet.allOf(FeatureId.class);
    }

    /**
     * Helper to create a set with only inline vectors (for version 2).
     */
    protected static Set<FeatureId> inlineVectorsOnly() {
        return EnumSet.of(FeatureId.INLINE_VECTORS);
    }
}

// Made with Bob
