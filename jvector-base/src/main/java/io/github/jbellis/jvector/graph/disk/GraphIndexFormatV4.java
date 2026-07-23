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

import java.util.Set;

/**
 * Format for version 4 of the on-disk graph format.
 * Version 4 characteristics:
 * - Has magic number
 * - Supports multiple features
 * - Supports multi-layer (hierarchical) graphs
 * - Has idUpperBound field
 * - No footer
 */
class GraphIndexFormatV4 extends AbstractGraphIndexFormat {

    /** Creates the singleton format for version 4. */
    GraphIndexFormatV4() {
        super(4, nonFusedFeatures(), true, false, FeatureOrdering.BY_FEATURE_ID);
    }

    /**
     * Protected constructor allowing subclasses (V5, V6) to specify their own version,
     * feature set, footer flag, and feature ordering while sharing V4's wire format.
     */
    protected GraphIndexFormatV4(int version, Set<FeatureId> supportedFeatures, boolean usesFooter, FeatureOrdering featureOrdering) {
        super(version, supportedFeatures, true, usesFooter, featureOrdering);
    }
}
