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

import io.github.jbellis.jvector.example.benchmarks.datasets.ByteDataSet;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.concurrent.ThreadLocalRandom;

/**
 * The int8 queries of an int8 dataset, scored exactly against the int8 vectors stored inline in an on-disk index.
 * Int8 indexes have no compression or reranking, so searches use the standard stopping rule.
 */
public final class Int8QuerySet implements QuerySet<ByteSequence<?>> {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ByteDataSet ds;

    public Int8QuerySet(ByteDataSet ds) {
        this.ds = ds;
    }

    @Override
    public int size() {
        return ds.getQueryVectors().size();
    }

    @Override
    public ByteSequence<?> get(int i) {
        return ds.getQueryVectors().get(i);
    }

    @Override
    public int baseCount() {
        return ds.getBaseVectors().size();
    }

    @Override
    public ByteSequence<?> baseQuery(int ordinal) {
        return ds.getBaseVectors().get(ordinal).copy();
    }

    @Override
    public ByteSequence<?> newQuery() {
        return vts.createByteSequence(ds.getDimension());
    }

    /** Uniform signed bytes, with at least one nonzero component so cosine is defined. */
    @Override
    public void randomize(ByteSequence<?> query) {
        var random = ThreadLocalRandom.current();
        boolean nonZero = false;
        for (int d = 0; d < query.length(); d++) {
            byte value = (byte) random.nextInt(-128, 128);
            query.set(d, value);
            nonZero |= value != 0;
        }
        if (!nonZero) query.set(0, (byte) 1);
    }

    @Override
    public ByteSequence<?>[] newArray(int size) {
        return new ByteSequence<?>[size];
    }

    @Override
    public SearchScoreProvider scoreProviderFor(ByteSequence<?> query, GraphIndex.View view) {
        if (!(view instanceof OnDiskGraphIndex.View)) {
            throw new IllegalStateException("Int8 queries are scored against an on-disk index with inline int8 vectors");
        }
        var scorer = ((OnDiskGraphIndex.View) view).byteVectorRerankerFor(query, ds.getSimilarityFunction());
        return new DefaultSearchScoreProvider(scorer);
    }
}
