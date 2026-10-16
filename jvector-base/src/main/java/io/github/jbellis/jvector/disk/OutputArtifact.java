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

package io.github.jbellis.jvector.disk;

import io.github.jbellis.jvector.annotations.Experimental;

/**
 * The kinds of output one write operation can produce. Hosts map each to their own placement: a
 * component file, a codec file, or a section after the graph in the same file.
 */
@Experimental
public enum OutputArtifact {
    /** The graph index: header, records and footer, as read by {@code OnDiskGraphIndex.load}. */
    GRAPH,
    /**
     * A non-fused compressed-vectors sidecar (for example {@code PQVectors}), as read by
     * {@code CompressedVectors.load}.
     */
    COMPRESSED_VECTORS
}
