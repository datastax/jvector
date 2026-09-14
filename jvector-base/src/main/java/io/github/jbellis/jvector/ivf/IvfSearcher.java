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

package io.github.jbellis.jvector.ivf;

import io.github.jbellis.jvector.index.IndexSearcher;

/**
 * The IVF search entry point, mirroring the role {@code GraphSearcher} plays for graph/HNSW
 * indexes: returned by {@link IvfIndex#searcher()} with no cast required.
 * <p>
 * Left as an empty marker for now, kept as an interface rather than the concrete class its graph
 * counterpart is: IVF's search-time options (a per-query {@code nprobe} looks likely, mirroring
 * {@code efSearch} on the graph side, but is not yet decided) and its backing algorithm are both
 * still pending the IVF design. A concrete implementing class arrives alongside that algorithm.
 */
public interface IvfSearcher extends IndexSearcher {
}
