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

import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;

import java.io.IOException;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Optional;

import java.util.concurrent.ForkJoinPool;

@SuppressWarnings("OptionalUsedAsFieldOrParameterType")
public final class CompactOptions {

    /** Features to write into the OUTPUT index. */
    public final EnumSet<FeatureId> writeFeatures;

    /** How compaction evaluates candidate neighbors. */
    public final CompactionPrecision precision;

    /** Executor controls (optional). If null, compactor may create its own. */
    public final ForkJoinPool executor;

    /** Window size for in-flight tasks / scratch pool size. 0 => use executor parallelism or a reasonable default. */
    public final int taskWindowSize;

    /** Compression configuration (may be NONE). */
    public final CompressionConfig compressionConfig;

    private CompactOptions(Builder b) {
        this.writeFeatures = b.writeFeatures.clone();
        this.precision = b.precision;
        this.executor = b.executor;
        this.taskWindowSize = b.taskWindowSize;
        this.compressionConfig = b.compressionConfig;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Convenience: inline vectors */
    public static CompactOptions withInlineVectors() {
        return builder()
                .writeFeatures(EnumSet.of(FeatureId.INLINE_VECTORS))
                .precision(CompactionPrecision.EXACT)
                .build();
    }

    /** Convenience: inline vectors, PQ provided, write PQ vectors to a separate file. */
    public static CompactOptions withPQVectorsSeparate(ProductQuantization pq, Path pqVecPath) {
        return builder()
                .writeFeatures(EnumSet.of(FeatureId.INLINE_VECTORS))
                .precision(CompactionPrecision.EXACT)
                .compressionConfig(CompressionConfig.withPQCodebook(pq, pqVecPath))
                .build();
    }

    /** Convenience: inline vectors + FusedPQ, automatically chose the PQ codebook from one of the sources */
    public static CompactOptions withFusedPQ() {
        return builder()
                .writeFeatures(EnumSet.of(FeatureId.INLINE_VECTORS, FeatureId.FUSED_PQ))
                .precision(CompactionPrecision.COMPRESSED)
                .compressionConfig(CompressionConfig.withSourcePQ(CompressionConfig.PQSourcePolicy.AUTO))
                .build();
    }

    /** Convenience: inline vectors + FusedPQ, PQ provided */
    public static CompactOptions withFusedPQ(ProductQuantization pq) {
        return builder()
                .writeFeatures(EnumSet.of(FeatureId.INLINE_VECTORS, FeatureId.FUSED_PQ))
                .precision(CompactionPrecision.COMPRESSED)
                .compressionConfig(CompressionConfig.withPQCodebook(pq))
                .build();
    }

    // -------------------------------------------------------------------------
    // Validation / Derivation helpers used by the compactor
    // -------------------------------------------------------------------------

    /**
     * Validate options independent of sources.
     * Call this early in compactor.compact().
     */
    public void validateStatic() {
        if (writeFeatures == null || writeFeatures.isEmpty()) {
            throw new IllegalArgumentException("writeFeatures must not be empty");
        }
        if (!writeFeatures.contains(FeatureId.INLINE_VECTORS)) {
            throw new IllegalArgumentException("writeFeatures must include INLINE_VECTORS (unless you explicitly support vectorless indexes)");
        }
        if (taskWindowSize < 0) {
            throw new IllegalArgumentException("taskWindowSize must be >= 0");
        }
        if (precision == null) {
            throw new IllegalArgumentException("precision must not be null");
        }
        compressionConfig.validateAgainstRequestedFeatures(writeFeatures, precision);
    }

    /**
     * Validate options against runtime facts (dimension, fused requirements, etc.).
     */
    public void validateWithRuntime(int indexDimension) {
        ProductQuantization pq = compressionConfig.pqCodebook().orElse(null);
            if (pq != null && pq.getOriginalDimension() != indexDimension) {
                throw new IllegalArgumentException(
                "PQ dimension mismatch: pq=" + pq.getOriginalDimension() + " index=" + indexDimension);
        }
    }

    /** Derive effective task window size. */
    public int effectiveTaskWindowSize() {
        if (taskWindowSize > 0) return taskWindowSize;
        if (executor != null) return Math.max(1, executor.getParallelism());
        return Math.max(1, Runtime.getRuntime().availableProcessors());
    }

    // -------------------------------------------------------------------------
    // Builder
    // -------------------------------------------------------------------------

    public static final class Builder {
        private EnumSet<FeatureId> writeFeatures = EnumSet.of(FeatureId.INLINE_VECTORS);
        private java.util.concurrent.ForkJoinPool executor = null;
        private CompactionPrecision precision = CompactionPrecision.EXACT;
        private int taskWindowSize = 0;
        private CompressionConfig compressionConfig = CompressionConfig.none();

        private Builder() {}

        public Builder writeFeatures(EnumSet<FeatureId> features) {
            this.writeFeatures = Objects.requireNonNull(features, "features");
            return this;
        }

        public Builder addFeature(FeatureId id) {
            this.writeFeatures.add(Objects.requireNonNull(id, "id"));
            return this;
        }

        public Builder executor(ForkJoinPool executor) {
            this.executor = executor;
            return this;
        }

        public Builder precision(CompactionPrecision precision) {
            this.precision = Objects.requireNonNull(precision, "precision");
            return this;
        }

        public Builder taskWindowSize(int taskWindowSize) {
            this.taskWindowSize = taskWindowSize;
            return this;
        }

        public Builder compressionConfig(CompressionConfig compressionConfig) {
            this.compressionConfig = Objects.requireNonNull(compressionConfig, "compressionConfig");
            return this;
        }

        public CompactOptions build() {
            CompactOptions opts = new CompactOptions(this);
            opts.validateStatic();
            return opts;
        }
    }

    public enum CompactionPrecision {
        EXACT,
        COMPRESSED
    }

    // -------------------------------------------------------------------------
    // CompressionConfig
    // -------------------------------------------------------------------------

    public static final class CompressionConfig {


    public enum Kind {
        NONE,
        PQ_VECTORS,
        PQ_CODEBOOK,
        SOURCE_PQ
    }

    public enum PQSourcePolicy {
        AUTO,
        LARGEST_LIVE,
        FIRST
    }

    public final Kind kind;
    public final Optional<PQVectors> pqVectors;
    public final Optional<ProductQuantization> pqCodebook;
    public final PQSourcePolicy sourcePolicy;
    public final Path pqVectorsOutputPath;

    private CompressionConfig(Kind kind,
                              PQVectors pqVectors,
                              ProductQuantization pqCodebook,
                              PQSourcePolicy sourcePolicy,
                              Path pqVectorsOutputPath) {
        this.kind = Objects.requireNonNull(kind, "kind");
        this.pqVectors = Optional.ofNullable(pqVectors);
        this.pqCodebook = Optional.ofNullable(pqCodebook);
        this.sourcePolicy = sourcePolicy;
        this.pqVectorsOutputPath = pqVectorsOutputPath;

        int configured = (pqVectors != null ? 1 : 0) +
                         (pqCodebook != null ? 1 : 0) +
                         (sourcePolicy != null ? 1 : 0);
        if (configured > 1) {
            throw new IllegalArgumentException("Only one compression source may be configured");
        }

        if (kind == Kind.PQ_VECTORS && pqVectorsOutputPath != null) {
            throw new IllegalArgumentException("withPQVectors(...) may not be combined with pqVectorsOutputPath");
        }
    }

    // --------------------------
    // Factories
    // --------------------------

    public static CompressionConfig none() {
        return new CompressionConfig(Kind.NONE, null, null, null, null);
    }

    public static CompressionConfig withPQVectors(PQVectors pqVectors) {
        Objects.requireNonNull(pqVectors, "pqVectors");
        return new CompressionConfig(Kind.PQ_VECTORS, pqVectors, null, null, null);
    }

    public static CompressionConfig withPQCodebook(ProductQuantization pq) {
        Objects.requireNonNull(pq, "pq");
        return new CompressionConfig(Kind.PQ_CODEBOOK, null, pq, null, null);
    }

    public static CompressionConfig withPQCodebook(ProductQuantization pq, Path pqVectorsOutputPath) {
        Objects.requireNonNull(pq, "pq");
        Objects.requireNonNull(pqVectorsOutputPath, "pqVectorsOutputPath");
        return new CompressionConfig(Kind.PQ_CODEBOOK, null, pq, null, pqVectorsOutputPath);
    }

    public static CompressionConfig withSourcePQ(PQSourcePolicy policy) {
        Objects.requireNonNull(policy, "policy");
        return new CompressionConfig(Kind.SOURCE_PQ, null, null, policy, null);
    }

    public static CompressionConfig withSourcePQ(PQSourcePolicy policy, Path pqVectorsOutputPath) {
        Objects.requireNonNull(policy, "policy");
        Objects.requireNonNull(pqVectorsOutputPath, "pqVectorsOutputPath");
        return new CompressionConfig(Kind.SOURCE_PQ, null, null, policy, pqVectorsOutputPath);
    }

    // --------------------------
    // Validation (called by CompactOptions.validateStatic)
    // --------------------------

    public void validateAgainstRequestedFeatures(EnumSet<FeatureId> requested, CompactionPrecision precision) {
        boolean fusedEnabled = requested.contains(FeatureId.FUSED_PQ);
        boolean pqVectorsOutputEnabled = (pqVectorsOutputPath != null);
        boolean compressedEnabled = (precision == CompactionPrecision.COMPRESSED);

        boolean pqNeeded = fusedEnabled || pqVectorsOutputEnabled || compressedEnabled;

        if (!pqNeeded) {
            return;
        }

        if (kind == Kind.NONE) {
            throw new IllegalArgumentException(
                    "Compression is required (FUSED_PQ, pqVectorsOutput, or COMPRESSED precision) but compressionConfig is NONE");
        }


        if (compressedEnabled && kind == Kind.PQ_CODEBOOK) {
            throw new IllegalArgumentException(
                    "CompactionPrecision.COMPRESSED requires PQ vectors or source-side PQ; PQ codebook alone is insufficient");
        }

    }

    public Optional<ProductQuantization> pqCodebook() {
        if (pqVectors.isPresent()) {
            return Optional.of(pqVectors.get().getCompressor());
        }
        return pqCodebook;
    }
}
}
