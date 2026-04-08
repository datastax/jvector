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

package io.github.jbellis.jvector.graph.representations;

import io.github.jbellis.jvector.vector.VectorRepresentation;


public interface VectorRepresentationCollection<Vec extends VectorRepresentation> {
    /**
     * Return the number of vector values.
     * <p>
     * All copies of a given RAVV should have the same size.  Typically this is achieved by either
     * (1) implementing a threadsafe, un-shared RAVV, where `copy` returns `this`, or
     * (2) implementing a fixed-size RAVV.
     */
    int size();

    /** Return the dimension of the returned vector values */
    int dimension();

    GlobalInformation<Vec> getGlobalInformation();

    PersistenceType getPersistenceType();
}