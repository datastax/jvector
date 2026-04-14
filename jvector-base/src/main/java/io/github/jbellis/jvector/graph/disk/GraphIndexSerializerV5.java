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
 * Serializer for version 5 of the on-disk graph format.
 * Version 5 characteristics:
 * - Has magic number
 * - Supports multiple features
 * - Supports multi-layer (hierarchical) graphs
 * - Has idUpperBound field
 * - Uses footer for metadata (major change from V4)
 * 
 * V5 is identical to V4 in terms of format, but uses a footer instead of
 * relying on the header at the beginning. The serialization logic is the same.
 */
class GraphIndexSerializerV5 extends GraphIndexSerializerV4 {

    GraphIndexSerializerV5() {
        // Call parent constructor but override version and footer flag
    }

    @Override
    public int getVersion() {
        return 5;
    }

    @Override
    public boolean usesFooter() {
        return true;
    }
}

// Made with Bob
