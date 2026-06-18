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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.disk.CompactionContext;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.QuantizationCompactionStrategy;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;

public interface CompressedVectors extends Accountable {
    /**
     * Write the compressed vectors to the given IndexWriter
     * @param out the IndexWriter to write to
     * @param version the serialization version.  versions 2 and 3 are supported
     */
    void write(IndexWriter out, int version) throws IOException;

    /**
     * Write the compressed vectors to the given IndexWriter at the current serialization version
     */
    default void write(IndexWriter out) throws IOException {
        write(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    /** @return the original size of each vector, in bytes, before compression */
    int getOriginalSize();

    /** @return the compressed size of each vector, in bytes */
    int getCompressedSize();

    /** @return the compressor used by this instance */
    VectorCompressor<?> getCompressor();

    /** precomputes partial scores for the given query with every centroid; suitable for most searches */
    ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction);

    /** no precomputation; suitable when just a handful of score computations are performed */
    ScoreFunction.ApproximateScoreFunction diversityFunctionFor(int nodeId, VectorSimilarityFunction similarityFunction);

    /** no precomputation; suitable when just a handful of score computations are performed */
    ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction);


    @Deprecated
    default ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        return precomputedScoreFunctionFor(q, similarityFunction);
    }

    /** the number of vectors */
    int count();

    /**
     * For compaction use: returns the {@link QuantizationCompactionStrategy} the compactor should
     * run when merging graphs whose non-fused compressed sidecars are this kind of
     * {@code CompressedVectors}. One strategy instance per compaction; it retrains the compressor
     * on the merged source vectors and streams the merged sidecar to disk.
     * <p>
     * Called by {@code OnDiskGraphIndexCompactor.detectSidecarStrategy()}. Named to mirror
     * {@code FusedFeature.createCompactionStrategy} — same verb, receiver type disambiguates
     * whether the returned strategy drives the inline-fused or sidecar workflow. Default throws —
     * implementations supporting compaction must override.
     */
    default QuantizationCompactionStrategy createCompactionStrategy(CompactionContext ctx) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " does not support sidecar compaction");
    }

    // ---- For compaction use: sidecar-streaming-write hooks. Called by the generic
    // SidecarCompactionStrategy to produce a merged-format-compatible sidecar without that
    // strategy knowing the format. ----

    /**
     * For compaction use: writes the format-specific sidecar header (compressor params + vector
     * count + any extras the reader expects between count and the chunk stream). Called once at
     * the start of a streaming sidecar write by {@code SidecarCompactionStrategy.writeSidecar},
     * after which the strategy emits chunks of {@code sidecarVectorsPerChunk()} codes each.
     * Default throws — implementations supporting sidecar compaction must override.
     */
    default void writeSidecarHeader(IndexWriter out, VectorCompressor<?> mergedCompressor, int count) throws IOException {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " does not support sidecar compaction");
    }

    /**
     * For compaction use: vectors per chunk for streaming sidecar writes. The chunk size must
     * match the format the reader expects (e.g. {@code PQVectors} uses 1024 to align with
     * {@code MutablePQVectors}'s on-disk layout). Read by
     * {@code SidecarCompactionStrategy.writeSidecar} to size each emitted chunk.
     */
    default int sidecarVectorsPerChunk() {
        return 1024;
    }
}
