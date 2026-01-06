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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.vector.VectorRepresentation;

/**
 * This is an internal interface.
 * Graph indices should only be searched through a GraphSearcher. Thus, we do not want to expose the getView method
 * externally. This is a convenience interface so that, internally, graph indices, can provide their view to the
 * GraphSearcher. All graph indices should implement it.
 */
public interface Viewable<Primary extends VectorRepresentation, Secondary extends VectorRepresentation> {
    /**
     * Return a View with which to navigate the graph.  Views are not threadsafe -- that is,
     * only one search at a time should be run per View.
     * <p>
     * Additionally, the View represents a point of consistency in the graph, and in-use
     * Views prevent the removal of marked-deleted nodes from graphs that are being
     * concurrently modified.  Thus, it is good (and encouraged) to re-use Views for
     * on-disk, read-only graphs, but for in-memory graphs, it is better to create a new
     * View per search.
     */
    GraphIndexView<Primary, Secondary> getView();
}
