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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

/**
 * Trains a fresh {@link VectorCompressor} on the merged source vectors during a compaction run.
 * <p>
 * One implementation per quantization scheme (today: {@link PQRetrainer} wrapped behind a lambda;
 * future: {@code ASHRetrainer}, etc.). Both fused and sidecar generic strategies receive a
 * retrainer via their constructor and invoke it at {@code retrain(vsf)} time — the strategies
 * stay quantization-agnostic and the retrainer encapsulates the scheme-specific training math.
 */
@FunctionalInterface
public interface VectorCompressorRetrainer {
    VectorCompressor<?> retrain(VectorSimilarityFunction vsf);
}
