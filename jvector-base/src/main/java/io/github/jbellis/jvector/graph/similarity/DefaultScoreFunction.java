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

package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.vector.VectorRepresentation;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public class DefaultScoreFunction implements SimilarityFunction<VectorFloat<?>> {
    private VectorFloat<?> query;
    private VectorSimilarityFunction vsf;

    public DefaultScoreFunction(VectorSimilarityFunction vsf) {
        query = null;
        this.vsf = vsf;
    }

    @Override
    public boolean isExact() {
        return false;
    }

    @Override
    public void fixQuery(VectorFloat<?> query) {
        this.query = query;
    }

    @Override
    public float similarityTo(VectorFloat<?> other) {
        return vsf.compare(query, other);
    }

    @Override
    public float similarity(VectorFloat<?> vec1, VectorFloat<?> vec2) {
        return vsf.compare(vec1, vec2);
    }

    @Override
    public VectorSimilarityFunction getSimilarityFunction() {
        return vsf;
    }

    @Override
    public <Vec2 extends VectorRepresentation> boolean compatible(SimilarityFunction<Vec2> other) {
        return vsf == other.getSimilarityFunction();
    }

    @Override
    public SimilarityFunction<VectorFloat<?>> copy() {
        return new DefaultScoreFunction(vsf);
    }
}
