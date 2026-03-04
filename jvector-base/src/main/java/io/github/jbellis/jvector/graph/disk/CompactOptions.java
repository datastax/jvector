import java.io.IOException;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Optional;

import java.annotation.Nullable;

@SuppressWarnings("OptionalUsedAsFieldOrParameterType")
public final class CompactOptions {

    /** Features to write into the OUTPUT index. */
    public final EnumSet<FeatureId> writeFeatures;

    /** Executor controls (optional). If null, compactor may create its own. */
    @Nullable public final ForkJoinPool executor;

    /** Window size for in-flight tasks / scratch pool size. 0 => use executor parallelism or a reasonable default. */
    public final int taskWindowSize;

    /**
     * If non-null, compactor writes PQ vectors (one per node) to this file.
     */
    @Nullable
    public final Path pqOutputPath;

    /** PQ configuration (may be NONE). */
    public final PQConfig pqConfig;

    private CompactOptions(Builder b) {
        this.writeFeatures = b.writeFeatures.clone();
        this.executor = b.executor;
        this.taskWindowSize = b.taskWindowSize;
        this.pqConfig = b.pqConfig;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static CompactOptions basicInlineVectors() {
        return builder()
                .writeFeatures(EnumSet.of(FeatureId.INLINE_VECTORS))
                .build();
    }

    /** Convenience: inline vectors, PQ provided, write PQ to sidecar. */
    public static CompactOptions withPQVectorsSidecar(ProductQuantization pq, Path sidecarPath) {
        return builder()
                .writeFeatures(EnumSet.of(FeatureId.INLINE_VECTORS))
                .pq(PQConfig.provided(pq).sidecar(sidecarPath))
                .build();
    }

    /** Convenience: inline vectors + FusedPQ, get PQ from the source with the largest number of live nodes */
    public static CompactOptions withFusedPQLargestLive(ProductQuantization pq) {
        return builder()
                .writeFeatures(EnumSet.of(FeatureId.INLINE_VECTORS, FeatureId.FUSED_PQ))
                .pq(PQConfig.fromSources(PQConfig.PQSourcePolicy.LARGEST_LIVE))
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
        pqConfig.validateAgainstRequestedFeatures(writeFeatures);
    }

    /**
     * Validate options against runtime facts (dimension, fused requirements, etc.).
     */
    public void validateWithRuntime(int indexDimension) {
        // PQ-related runtime validation
        if (pqConfig.mode != PQConfig.Mode.NONE) {
            ProductQuantization pq = pqConfig.pq.orElse(null);
            if (pqConfig.mode == PQConfig.Mode.PROVIDED) {
                if (pq == null) throw new IllegalStateException("PQ mode PROVIDED but pq is empty");
            }
            if (pq != null) {
                if (pq.getOriginalDimension != indexDimension) {
                    throw new IllegalArgumentException(
                            "PQ dimension mismatch: pq=" + pq.getOriginalDimension + " index=" + indexDimension);
                }
            }
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
        private int taskWindowSize = 0;
        private PQConfig pqConfig = PQConfig.none();

        private Builder() {}

        public Builder writeFeatures(EnumSet<FeatureId> features) {
            this.writeFeatures = Objects.requireNonNull(features, "features");
            return this;
        }

        public Builder addFeature(FeatureId id) {
            this.writeFeatures.add(Objects.requireNonNull(id, "id"));
            return this;
        }

        public Builder executor(@Nullable java.util.concurrent.ForkJoinPool executor) {
            this.executor = executor;
            return this;
        }

        public Builder taskWindowSize(int taskWindowSize) {
            this.taskWindowSize = taskWindowSize;
            return this;
        }

        public Builder pqConfig(PQConfig pqConfig) {
            this.pqConfig = Objects.requireNonNull(pqConfig, "pq");
            return this;
        }

        public CompactOptions build() {
            CompactOptions opts = new CompactOptions(this);
            opts.validateStatic();
            return opts;
        }
    }

    // -------------------------------------------------------------------------
    // PQConfig
    // -------------------------------------------------------------------------

    public static final class PQConfig {

    public enum Mode {
        /** No PQ used. */
        NONE,
        /** PQ is provided by the caller. */
        PROVIDED,
        /** PQ should be chosen from sources according to sourcePolicy. */
        FROM_SOURCES
        // TRAIN can be added later without changing CompactOptions signature
    }

    /** Keep this for future expansion but only NONE for now. */
    public enum PQStorage {
        /** No PQ written anywhere. */
        NONE
        // SIDECAR / EMBEDDED can be added later if desired
    }

    public enum PQSourcePolicy {
        AUTO,
        LARGEST_LIVE,
        FIRST
    }

    /** Controls writing per-node PQ vectors (codes) as a sidecar artifact. */
    public enum PQVectorsOutput {
        NONE,
        SIDECAR_FILE
    }

    public final Mode mode;
    public final PQStorage storage;

    /** Present when mode==PROVIDED. (When FROM_SOURCES, resolved later by compactor.) */
    public final Optional<ProductQuantization> pq;

    /** Used when mode==FROM_SOURCES. */
    public final PQSourcePolicy sourcePolicy;

    /** Whether to write PQ vectors out, and where. */
    public final PQVectorsOutput pqVectorsOutput;

    /** Required when pqVectorsOutput==SIDECAR_FILE. */
    @Nullable public final Path pqVectorsOutputPath;

    private PQConfig(Mode mode,
                     PQStorage storage,
                     @Nullable ProductQuantization pq,
                     PQSourcePolicy sourcePolicy,
                     PQVectorsOutput pqVectorsOutput,
                     @Nullable Path pqVectorsOutputPath) {
        this.mode = Objects.requireNonNull(mode, "mode");
        this.storage = Objects.requireNonNull(storage, "storage");
        this.pq = Optional.ofNullable(pq);
        this.sourcePolicy = Objects.requireNonNull(sourcePolicy, "sourcePolicy");
        this.pqVectorsOutput = Objects.requireNonNull(pqVectorsOutput, "pqVectorsOutput");
        this.pqVectorsOutputPath = pqVectorsOutputPath;

        // internal consistency
        if (this.pqVectorsOutput == PQVectorsOutput.SIDECAR_FILE && this.pqVectorsOutputPath == null) {
            throw new IllegalArgumentException("pqVectorsOutput==SIDECAR_FILE requires pqVectorsOutputPath");
        }
        if (this.pqVectorsOutput != PQVectorsOutput.SIDECAR_FILE && this.pqVectorsOutputPath != null) {
            throw new IllegalArgumentException("pqVectorsOutputPath is set but pqVectorsOutput is not SIDECAR_FILE");
        }
    }

    // --------------------------
    // Factories
    // --------------------------

    public static PQConfig none() {
        return new PQConfig(
                Mode.NONE,
                PQStorage.NONE,
                null,
                PQSourcePolicy.AUTO,
                PQVectorsOutput.NONE,
                null
        );
    }

    public static PQConfig provided(ProductQuantization pq) {
        Objects.requireNonNull(pq, "pq");
        return new PQConfig(
                Mode.PROVIDED,
                PQStorage.NONE,
                pq,
                PQSourcePolicy.AUTO,
                PQVectorsOutput.NONE,
                null
        );
    }

    public static PQConfig fromSources(PQSourcePolicy policy) {
        Objects.requireNonNull(policy, "policy");
        return new PQConfig(
                Mode.FROM_SOURCES,
                PQStorage.NONE,
                null,
                policy,
                PQVectorsOutput.NONE,
                null
        );
    }

    /** Enable writing PQ vectors to a sidecar file. */
    public PQConfig withPQVectorsSidecar(Path pqVectorsOutputPath) {
        Objects.requireNonNull(pqVectorsOutputPath, "pqVectorsOutputPath");
        return new PQConfig(
                this.mode,
                this.storage,
                this.pq.orElse(null),
                this.sourcePolicy,
                PQVectorsOutput.SIDECAR_FILE,
                pqVectorsOutputPath
        );
    }

    /** Disable writing PQ vectors. */
    public PQConfig withoutPQVectors() {
        return new PQConfig(
                this.mode,
                this.storage,
                this.pq.orElse(null),
                this.sourcePolicy,
                PQVectorsOutput.NONE,
                null
        );
    }

    // --------------------------
    // Validation (called by CompactOptions.validateStatic)
    // --------------------------

    public void validateAgainstRequestedFeatures(EnumSet<FeatureId> requested) {
        boolean fusedEnabled = requested.contains(FeatureId.FUSED_PQ);
        boolean pqVectorsEnabled = (pqVectorsOutput == PQVectorsOutput.SIDECAR_FILE);

        boolean pqNeeded = fusedEnabled || pqVectorsEnabled;

        if (!pqNeeded) {
            // Nothing PQ-related requested -> PQConfig must be NONE-ish
            if (mode != Mode.NONE || pq.isPresent()) {
                throw new IllegalArgumentException(
                        "PQ configured (mode=" + mode + ") but neither FUSED_PQ nor pqVectorsOutput is enabled");
            }
            return;
        }

        // PQ needed
        if (mode == Mode.NONE) {
            throw new IllegalArgumentException("PQ is required (FUSED_PQ or pqVectorsOutput enabled) but pqConfig.mode==NONE");
        }

        if (mode == Mode.PROVIDED && pq.isEmpty()) {
            throw new IllegalArgumentException("pqConfig.mode==PROVIDED requires pqConfig.pq");
        }

        // If pq vectors sidecar requested, ensure path exists (already enforced in ctor)
        if (pqVectorsEnabled && pqVectorsOutputPath == null) {
            throw new IllegalArgumentException("pqVectorsOutput==SIDECAR_FILE requires pqVectorsOutputPath");
        }

        if (mode == Mode.FROM_SOURCES && pq.isPresent()) {
            throw new IllegalArgumentException("pqConfig.mode==FROM_SOURCES must not set pqConfig.pq (it will be resolved from sources)");
        }
    }
}
}
