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
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.quantization.NVQuantization;

/** NVQ-quantized vectors stored alongside each node, used for reranking. */
final class NVQProvider implements FeatureProvider {
    @Override
    public FeatureId id() {
        return FeatureId.NVQ_VECTORS;
    }

    @Override
    public void configure(FeatureWriteContext ctx) {
        var vectors = ctx.vectors();
        int nSubVectors = vectors.dimension() == 2 ? 1 : 2;
        var nvq = ctx.timeIndexCompute("NVQ", () -> NVQuantization.compute(vectors, nSubVectors));
        ctx.add(new NVQ(nvq));
        ctx.supplyFromVector(FeatureId.NVQ_VECTORS, vector -> new NVQ.State(nvq.encode(vector)));
    }

    @Override
    public boolean providesRerankVectors() {
        return true;
    }
}
