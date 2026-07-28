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
 * Format for version 5 of the on-disk graph format.
 * Version 5 characteristics:
 * - Has magic number
 * - Supports multiple features
 * - Supports multi-layer (hierarchical) graphs
 * - Has idUpperBound field
 * - Uses footer for metadata (major change from V4)
 *
 * The wire format is identical to V4; the only behavioral difference is that
 * a footer is written after the graph data.
 */
class GraphIndexFormatV5 extends GraphIndexFormatV4 {

    /** Creates the singleton format for version 5. */
    GraphIndexFormatV5() {
        super(5, nonFusedFeatures(), true);
    }

    /**
     * Protected constructor for subclasses (V6) to specify their own version and feature set
     * while inheriting V5's footer-writing behavior.  Footer is always {@code true} for V5+.
     */
    protected GraphIndexFormatV5(int version, Set<FeatureId> supportedFeatures) {
        super(version, supportedFeatures, true);
    }
}
