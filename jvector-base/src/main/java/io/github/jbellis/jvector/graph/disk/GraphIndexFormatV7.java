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

/**
 * Format for version 7 of the on-disk graph format.
 * Version 7 characteristics:
 * - Has magic number
 * - Supports multiple features
 * - Supports multi-layer (hierarchical) graphs
 * - Has idUpperBound field
 * - Uses footer for metadata
 * - Places fused features last and writes feature count and ordinals explicitly (as in V6)
 * - Adds support for the {@link FeatureId#FUSED_ASH} feature
 *
 * The wire format is identical to V6; the only difference is the supported feature set.
 * FUSED_ASH uses the same fused-feature layout as FUSED_PQ: packed neighbor codes inline in
 * each L0 record, and the in-memory hierarchy's source vectors in a block after the sparse levels.
 */
class GraphIndexFormatV7 extends GraphIndexFormatV6 {

    /** Creates the singleton format for version 7. */
    GraphIndexFormatV7() {
        super(7, v7Features());
    }
}
