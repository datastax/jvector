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

import io.github.jbellis.jvector.annotations.Experimental;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Locale;

/**
 * Singleton that holds runtime-tunable default values for
 * {@link io.github.jbellis.jvector.graph.GraphIndexBuilder} construction parameters.
 *
 * <h2>Management</h2>
 *
 * <p>This class is a plain domain object: it implements {@link GraphIndexBuilderSettings} and
 * registers itself with {@link ManagementRegistry}, which in turn exposes it through whichever
 * {@link io.github.jbellis.jvector.management.spi.ManagementBackend} is active in this JVM. By
 * default that backend is JMX (see
 * {@code io.github.jbellis.jvector.management.jmx.JmxManagementBackend}), which makes this
 * class's attributes inspectable and updatable via any JMX client (JConsole, jvisualvm,
 * jmxterm, etc.) without restarting the application, under the object name
 * {@code io.github.jbellis.jvector:type=GraphIndexBuilderConfig}. See
 * {@link ManagementBackendProvider} for how to select a different backend, or disable external
 * exposure entirely, via the {@code jvector.management.backend} system property.
 *
 * <p>For example, using JConsole with the default JMX backend:
 * <pre>
 *   MBeans → io.github.jbellis.jvector → GraphIndexBuilderConfig → Attributes
 *       AddHierarchy : true   ← current value
 *                    [edit to false and press Enter to apply]
 * </pre>
 *
 * Or programmatically via {@code jmxterm}:
 * <pre>
 *   open &lt;pid&gt;
 *   bean io.github.jbellis.jvector:type=GraphIndexBuilderConfig
 *   get AddHierarchy
 *   set AddHierarchy false
 * </pre>
 *
 * <h2>Usage</h2>
 *
 * <p>Code that creates a {@code GraphIndexBuilder} and wants to respect the managed
 * value reads from the singleton before construction:
 * <pre>{@code
 * boolean addHierarchy = GraphIndexBuilderConfig.getInstance().isAddHierarchy();
 * var builder = new GraphIndexBuilder(scoreProvider, dimension, M, beamWidth,
 *                                     neighborOverflow, alpha, addHierarchy);
 * }</pre>
 *
 * <h2>Thread Safety</h2>
 *
 * <p>All managed attributes are stored as {@code volatile} fields so that writes from a
 * management-backend thread (e.g. a JMX client thread) are immediately visible to application
 * threads without additional synchronization.
 *
 * <h2>Failure Policy</h2>
 *
 * <p>Registration with {@link ManagementRegistry} happens in the constructor. The registry and
 * every {@link io.github.jbellis.jvector.management.spi.ManagementBackend} implementation treat
 * registration failure (e.g., no platform MBeanServer, or a name collision) as non-fatal: it
 * logs a warning and is otherwise silently ignored — the singleton is still usable with its
 * default values, so management-backend availability is never on the critical path.
 */
@Experimental
public class GraphIndexBuilderConfig implements GraphIndexBuilderSettings, ManagedResource {

    private static final Logger logger = LoggerFactory.getLogger(GraphIndexBuilderConfig.class);

    // ── Singleton ────────────────────────────────────────────────────────────
    // Initialized at class-load time; the JVM guarantees exactly-once, thread-safe
    // initialization of static fields.
    private static final GraphIndexBuilderConfig INSTANCE = new GraphIndexBuilderConfig();

    public static GraphIndexBuilderConfig getInstance() {
        return INSTANCE;
    }

    // ── Managed attributes ───────────────────────────────────────────────────
    // volatile ensures writes by a JMX client thread are immediately visible
    // to any thread that subsequently reads the field.

    private volatile boolean addHierarchy = true;
    private volatile boolean refineFinalGraph = true;
    private volatile boolean parallelBuild = false;
    private volatile String buildCompressionType = CompressionType.NONE.name();

    // PQ build compression parameters — only used when buildCompressionType == "PQ"
    private volatile int pqMFactor = 8;
    private volatile int pqK = 256;
    private volatile boolean pqCenterData = false;
    private volatile float pqAnisotropicThreshold = -1.0f;

    // ── Constructor ──────────────────────────────────────────────────────────

    private GraphIndexBuilderConfig() {
        ManagementRegistry.getInstance().register(this, GraphIndexBuilderSettings.class);
    }

    // ── GraphIndexBuilderSettings ─────────────────────────────────────────────

    @Override
    public boolean isAddHierarchy() {
        return addHierarchy;
    }

    @Override
    public void setAddHierarchy(boolean addHierarchy) {
        boolean previous = this.addHierarchy;
        this.addHierarchy = addHierarchy;
        if (previous != addHierarchy) {
            logger.info("JMX: addHierarchy changed {} → {}", previous, addHierarchy);
        }
    }

    @Override
    public boolean isRefineFinalGraph() {
        return refineFinalGraph;
    }

    @Override
    public void setRefineFinalGraph(boolean refineFinalGraph) {
        boolean previous = this.refineFinalGraph;
        this.refineFinalGraph = refineFinalGraph;
        if (previous != refineFinalGraph) {
            logger.info("JMX: refineFinalGraph changed {} → {}", previous, refineFinalGraph);
        }
    }

    @Override
    public boolean isParallelBuild() {
        return parallelBuild;
    }

    @Override
    public void setParallelBuild(boolean parallelBuild) {
        boolean previous = this.parallelBuild;
        this.parallelBuild = parallelBuild;
        if (previous != parallelBuild) {
            logger.info("JMX: parallelBuild changed {} → {}", previous, parallelBuild);
        }
    }

    @Override
    public String getBuildCompressionType() {
        return this.buildCompressionType;
    }

    @Override
    public void setBuildCompressionType(String compressionType) {
        // Validate eagerly so JMX clients get an error immediately rather than at build time.
        // Matching is case-insensitive; the canonical enum name is stored for consistency.
        CompressionType ct;
        try {
            ct = CompressionType.valueOf(compressionType.toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid build compression type: '" + compressionType + "'. Valid values: NONE, PQ, BQ", e);
        }
        String canonical = ct.name();
        String previous = this.buildCompressionType;
        this.buildCompressionType = canonical;
        if (!previous.equals(canonical)) {
            logger.info("JMX: buildCompressionType changed {} → {}", previous, canonical);
        }
    }

    // ── PQ build compression parameters ─────────────────────────────────────

    @Override
    public int getPqMFactor() {
        return pqMFactor;
    }

    @Override
    public void setPqMFactor(int mFactor) {
        if (mFactor <= 0) throw new IllegalArgumentException("pqMFactor must be positive");
        int previous = this.pqMFactor;
        this.pqMFactor = mFactor;
        if (previous != mFactor) {
            logger.info("JMX: pqMFactor changed {} → {}", previous, mFactor);
        }
    }

    @Override
    public int getPqK() {
        return pqK;
    }

    @Override
    public void setPqK(int k) {
        if (k <= 0) throw new IllegalArgumentException("pqK must be positive");
        int previous = this.pqK;
        this.pqK = k;
        if (previous != k) {
            logger.info("JMX: pqK changed {} → {}", previous, k);
        }
    }

    @Override
    public boolean isPqCenterData() {
        return pqCenterData;
    }

    @Override
    public void setPqCenterData(boolean centerData) {
        boolean previous = this.pqCenterData;
        this.pqCenterData = centerData;
        if (previous != centerData) {
            logger.info("JMX: pqCenterData changed {} → {}", previous, centerData);
        }
    }

    @Override
    public float getPqAnisotropicThreshold() {
        return pqAnisotropicThreshold;
    }

    @Override
    public void setPqAnisotropicThreshold(float threshold) {
        float previous = this.pqAnisotropicThreshold;
        this.pqAnisotropicThreshold = threshold;
        if (Float.compare(previous, threshold) != 0) {
            logger.info("JMX: pqAnisotropicThreshold changed {} → {}", previous, threshold);
        }
    }
}
