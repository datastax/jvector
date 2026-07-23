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

/**
 * Format for version 3 of the on-disk graph format.
 * Version 3 characteristics:
 * - Has magic number
 * - Supports multiple features (feature set serialization)
 * - Single layer only
 * - No footer
 */
class GraphIndexFormatV3 extends AbstractGraphIndexFormat {

    /** Creates the singleton format for version 3. */
    GraphIndexFormatV3() {
        super(3, nonFusedFeatures(), false, false, FeatureOrdering.BY_FEATURE_ID);
    }
}
