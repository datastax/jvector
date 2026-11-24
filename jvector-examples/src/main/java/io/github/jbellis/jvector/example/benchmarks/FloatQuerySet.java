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

package io.github.jbellis.jvector.example.benchmarks;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.concurrent.ThreadLocalRandom;

/** The float queries of a float dataset, scored by the configured system's float scorers. */
public final class FloatQuerySet implements QuerySet<VectorFloat<?>> {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ConfiguredSystem system;
    private final DataSet ds;

    public FloatQuerySet(ConfiguredSystem system, DataSet ds) {
        this.system = system;
        this.ds = ds;
    }

    @Override
    public int size() {
        return ds.getQueryVectors().size();
    }

    @Override
    public VectorFloat<?> get(int i) {
        return ds.getQueryVectors().get(i);
    }

    @Override
    public int baseCount() {
        return ds.getBaseVectors().size();
    }

    @Override
    public VectorFloat<?> baseQuery(int ordinal) {
        return ds.getBaseVectors().get(ordinal).copy();
    }

    @Override
    public VectorFloat<?> newQuery() {
        return vts.createFloatVector(ds.getDimension());
    }

    @Override
    public void randomize(VectorFloat<?> query) {
        randomizeVector(query);
    }

    /** Fills {@code query} with uniform values in [0, 1), at least one nonzero, and normalizes it. */
    static void randomizeVector(VectorFloat<?> query) {
        var random = ThreadLocalRandom.current();
        boolean nonZero = false;
        for (int d = 0; d < query.length(); d++) {
            float value = random.nextFloat();
            query.set(d, value);
            nonZero |= value != 0;
        }
        if (!nonZero) query.set(0, 1);
        VectorUtil.l2normalize(query);
    }

    @Override
    public VectorFloat<?>[] newArray(int size) {
        return new VectorFloat<?>[size];
    }

    @Override
    public SearchScoreProvider scoreProviderFor(VectorFloat<?> query, GraphIndex.View view) {
        return system.scoreProviderFor(query, view);
    }

    @Override
    public VectorFloat<?> asFloat(VectorFloat<?> query) {
        return query;
    }
}
