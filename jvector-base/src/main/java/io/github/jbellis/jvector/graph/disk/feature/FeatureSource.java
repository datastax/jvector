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

import java.io.Closeable;
import java.io.IOException;

/**
 * A source of features for graph nodes.
 */
public interface FeatureSource extends Closeable {
    /**
     * Get feature reader for node.
     * @param node the node
     * @param featureId the featureId
     * @return the return value
     * @throws IOException if an error occurs
     */
    RandomAccessReader featureReaderForNode(int node, FeatureId featureId) throws IOException;
}
