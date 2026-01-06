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

import io.github.jbellis.jvector.graph.representations.MutableRandomAccessVectorRepresentations;
import io.github.jbellis.jvector.vector.VectorRepresentation;

public abstract class AbstractMutableGraphIndex<Primary extends VectorRepresentation, Secondary extends VectorRepresentation> implements MutableGraphIndex, Viewable{
    final private MutableRandomAccessVectorRepresentations<Primary> primaries;
    final private MutableRandomAccessVectorRepresentations<Secondary> secondaries;

    AbstractMutableGraphIndex(MutableRandomAccessVectorRepresentations<Primary> primaries, MutableRandomAccessVectorRepresentations<Secondary> secondaries) {
        if (primaries.dimension() != secondaries.dimension()) {
            throw new IllegalArgumentException("The dimensions of the primary and secondary representations must be the same");
        }
        this.primaries = primaries;
        this.secondaries = secondaries;
    }

    public MutableRandomAccessVectorRepresentations<Primary> getPrimaryRepresentations() {
        return primaries;
    }

    public MutableRandomAccessVectorRepresentations<Secondary> getSecondaryRepresentations() {
        return secondaries;
    }
}
