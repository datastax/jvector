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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.util.work.ProgressTracker;
import io.github.jbellis.jvector.util.work.WorkStage;

/**
 * The phases {@link OnDiskGraphIndexCompactor} reports to a
 * {@link io.github.jbellis.jvector.util.work.ProgressLimiter}. {@link WorkStage} deliberately
 * leaves the stage set to the consumer; this is jvector's, so an embedder can render compaction
 * progress without knowing the compactor's internals.
 *
 * <p>Not every stage runs in every compaction: the quantization stages are skipped for sources
 * with no inline codes, {@link #SIDECAR} only for the sidecar entry point, and {@link #REFINE}
 * only when refinement is enabled. Progress within a stage is counted in nodes, and a phase whose
 * total is not known up front reports {@code -1} per
 * {@link ProgressTracker.PhaseScope#onProgress}.
 */
@Experimental
public enum CompactionStage implements WorkStage {
    /** Streaming a window of each source's base-layer records into the page cache, when enabled. */
    SOURCE_PRETOUCH,
    /** Assigning similarity-clustered output ordinals, when enabled. */
    SIMILARITY_ORDINALS,
    /** Retraining the quantization codebook on a balanced sample of the merged sources. */
    PQ_RETRAIN,
    /** Encoding every live node against the retrained codebook into the pre-encode cache. */
    CODE_PRE_ENCODE,
    /** Merging and writing the base layer: the bulk of both the reads and the output. */
    BASE_LAYER,
    /** Merging and writing the upper layers. */
    UPPER_LAYERS,
    /** Writing the trailing feature records and the footer after every layer is written. */
    FINALIZE,
    /** Second-pass neighbor refinement over the written graph. */
    REFINE,
    /** Writing the merged non-fused compressed sidecar. */
    SIDECAR
}
