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
import io.github.jbellis.jvector.graph.disk.feature.FusedASH;
import io.github.jbellis.jvector.quantization.ASHVectors;

/**
 * ASH codes of each node's neighbors stored with the node, so searches score approximately from the
 * index without separate compressed vectors. The codes come from the build-time ASH.
 */
final class FusedASHProvider implements FeatureProvider {
    @Override
    public FeatureId id() {
        return FeatureId.FUSED_ASH;
    }

    @Override
    public void configure(FeatureWriteContext ctx) {
        var ash = ctx.buildASH();
        if (ash == null) {
            System.out.println("Skipping Fused ASH feature due to null ASHVectors");
            return;
        }
        ctx.add(new FusedASH(ctx.graph().maxDegree(), ash.getCompressor()));
        // Neighbor codes need final adjacency, so there is no per-vector supplier for inline writes.
        ctx.supplyAfterBuild(FeatureId.FUSED_ASH, view -> ordinal -> new FusedASH.State(view, ash, ordinal));
    }

    @Override
    public boolean supportsInMemoryBuild() {
        return false;
    }

    @Override
    public boolean providesApproximateScoring() {
        return true;
    }

    @Override
    public void checkCompressedBuild(io.github.jbellis.jvector.quantization.CompressedVectors encoded) {
        if (!(encoded instanceof ASHVectors)) {
            throw new IllegalArgumentException("FusedASH requires ASH build compression");
        }
    }
}
