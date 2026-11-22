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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineByteVectors;

/** Signed int8 vectors stored inline with each node, scored exactly when searching int8 datasets. */
final class InlineByteVectorsProvider implements FeatureProvider {
    @Override
    public FeatureId id() {
        return FeatureId.INLINE_BYTE_VECTORS;
    }

    @Override
    public void configure(FeatureWriteContext ctx) {
        var byteVectors = ctx.byteVectors();
        // A parallel writer reads from several threads; each reads through its own view.
        var reader = byteVectors.threadLocalSupplier();
        ctx.add(new InlineByteVectors(byteVectors.dimension()));
        ctx.supplyByOrdinal(FeatureId.INLINE_BYTE_VECTORS,
                            ordinal -> new InlineByteVectors.State(reader.get().getVector(ordinal)));
    }

    @Override
    public boolean providesRerankVectors() {
        return true;
    }
}
