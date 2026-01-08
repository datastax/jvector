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

import io.github.jbellis.jvector.graph.representations.PersistenceType;
import io.github.jbellis.jvector.graph.representations.MutableRandomAccessVectorRepresentations;
import io.github.jbellis.jvector.graph.similarity.SearchScoreBundle;
import io.github.jbellis.jvector.vector.VectorRepresentation;

public abstract class AbstractMutableGraphIndex<Primary extends VectorRepresentation, Secondary extends VectorRepresentation> implements MutableGraphIndex, Viewable {
    private final MutableRandomAccessVectorRepresentations<Primary> primaries;
    private final MutableRandomAccessVectorRepresentations<Secondary> secondaries;
    private final SearchScoreBundle<Primary, Secondary> searchScoreBundle;

    private final PersistenceType primaryPersistenceType;
    private final PersistenceType secondaryPersistenceType;

    AbstractMutableGraphIndex(MutableRandomAccessVectorRepresentations<Primary> primaries,
                              MutableRandomAccessVectorRepresentations<Secondary> secondaries,
                              PersistenceType primaryPersistenceType,
                              PersistenceType secondaryPersistenceType,
                              SearchScoreBundle<Primary, Secondary> searchScoreBundle) {
        if (primaries.dimension() != secondaries.dimension()) {
            throw new IllegalArgumentException("The dimensions of the primary and secondary representations must be the same");
        }
        this.primaries = primaries;
        this.secondaries = secondaries;
        this.searchScoreBundle = searchScoreBundle;
        this.primaryPersistenceType = primaryPersistenceType;
        this.secondaryPersistenceType = secondaryPersistenceType;
    }

    public MutableRandomAccessVectorRepresentations<Primary> getPrimaryRepresentations() {
        return primaries;
    }

    public MutableRandomAccessVectorRepresentations<Secondary> getSecondaryRepresentations() {
        return secondaries;
    }

    public SearchScoreBundle<Primary, Secondary> getSearchScoreBundle() {
        return searchScoreBundle;
    }

    public PersistenceType getPrimaryPersistenceType() {
        return primaryPersistenceType;
    }

    public PersistenceType getSecondaryPersistenceType() {
        return secondaryPersistenceType;
    }
}
