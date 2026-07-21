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

package io.github.jbellis.jvector.management;

/**
 * JMX Standard MBean interface for {@link GraphIndexBuilderConfig}.
 *
 * <p>Exposes {@link io.github.jbellis.jvector.graph.GraphIndexBuilder} construction
 * parameters as JMX-managed attributes so they can be inspected and updated at runtime
 * via any JMX client (JConsole, jvisualvm, jmxterm, etc.) without restarting the
 * application.
 *
 * <p>Changes to these attributes take effect the next time a
 * {@link io.github.jbellis.jvector.graph.GraphIndexBuilder} reads the value from
 * {@link GraphIndexBuilderConfig#getInstance()}.  They do not affect indexes that are
 * already being built or have already been built.
 *
 * <p>The interface follows the Standard MBean naming convention: the implementation
 * class ({@link GraphIndexBuilderConfig}) has the same simple name as this interface
 * without the {@code MBean} suffix.
 */
public interface GraphIndexBuilderConfigMBean {

    // ── Graph topology ────────────────────────────────────────────────────────

    /**
     * Returns whether HNSW-style hierarchy layers are added on top of the base Vamana
     * graph during index construction.
     *
     * <p>When {@code true}, the graph has multiple levels (like HNSW), which improves
     * search speed on large datasets by reducing the number of distance computations
     * needed to reach the entry point region.  When {@code false}, only the flat
     * level-0 graph is built (equivalent to a plain Vamana index), which uses less
     * memory and may build faster on small datasets.
     */
    boolean isAddHierarchy();

    /**
     * Enables or disables HNSW-style hierarchy layers for subsequent index builds.
     *
     * @param addHierarchy {@code true} to enable hierarchy (default), {@code false} to disable
     */
    void setAddHierarchy(boolean addHierarchy);

    /**
     * Returns whether a second refinement pass is run over each node's edges after
     * the initial graph build completes.
     *
     * <p>Refinement improves recall at the cost of additional build time.
     */
    boolean isRefineFinalGraph();

    /**
     * Enables or disables the final graph refinement pass.
     *
     * @param refineFinalGraph {@code true} to enable refinement (default), {@code false} to skip
     */
    void setRefineFinalGraph(boolean refineFinalGraph);

    // ── Write path ────────────────────────────────────────────────────────────

    /**
     * Returns whether graph index writes use the parallel writer
     * ({@link io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter}) or the
     * sequential writer ({@link io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter}).
     *
     * <p>The parallel writer serialises level-0 records concurrently via an
     * {@code AsynchronousFileChannel}, which substantially reduces wall-clock write time for
     * large indexes.  Both writers produce an identical on-disk format; switching this flag
     * does not require re-reading or re-indexing existing data.
     */
    boolean isParallelBuild();

    /**
     * Enables or disables the parallel graph index writer for subsequent builds.
     *
     * @param parallelBuild {@code true} to use the parallel writer, {@code false} for sequential (default)
     */
    void setParallelBuild(boolean parallelBuild);

    /**
     * Returns the compression type used during graph construction scoring.
     * Valid values are the names of {@link CompressionType} constants: {@code "NONE"}, {@code "PQ"}, {@code "BQ"}.
     */
    String getBuildCompressionType();

    /**
     * Sets the compression type used during graph construction scoring.
     *
     * @param compressionType one of {@code "NONE"}, {@code "PQ"}, {@code "BQ"}
     * @throws IllegalArgumentException if the value is not a valid {@link CompressionType} name
     */
    void setBuildCompressionType(String compressionType);

    // ── PQ build compression parameters ──────────────────────────────────────
    // These are only consulted when BuildCompressionType is "PQ".

    /**
     * Returns the PQ subspace divisor. The number of PQ subspaces {@code m} is computed as
     * {@code dimension / mFactor}. Ignored when {@code BuildCompressionType} is not {@code "PQ"}.
     */
    int getPqMFactor();

    /**
     * Sets the PQ subspace divisor.
     *
     * @param mFactor must be a positive integer that evenly divides the vector dimension
     */
    void setPqMFactor(int mFactor);

    /**
     * Returns the number of centroids per PQ subspace (default 256).
     * Ignored when {@code BuildCompressionType} is not {@code "PQ"}.
     */
    int getPqK();

    /**
     * Sets the number of centroids per PQ subspace.
     *
     * @param k must be a positive power of two; typical value is 256
     */
    void setPqK(int k);

    /**
     * Returns whether PQ training globally centers the data before clustering.
     * Recommended {@code true} for Euclidean similarity, {@code false} otherwise.
     * Ignored when {@code BuildCompressionType} is not {@code "PQ"}.
     */
    boolean isPqCenterData();

    /**
     * Enables or disables global centering during PQ training.
     *
     * @param centerData {@code true} to center, {@code false} to skip (default)
     */
    void setPqCenterData(boolean centerData);

    /**
     * Returns the anisotropic loss threshold used during PQ encoding.
     * {@code -1.0} disables anisotropic weighting (default).
     * Ignored when {@code BuildCompressionType} is not {@code "PQ"}.
     */
    float getPqAnisotropicThreshold();

    /**
     * Sets the anisotropic loss threshold.
     *
     * @param threshold use {@code -1.0} to disable anisotropic weighting (default)
     */
    void setPqAnisotropicThreshold(float threshold);
}
