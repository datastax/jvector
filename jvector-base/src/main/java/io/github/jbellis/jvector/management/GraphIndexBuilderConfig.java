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

import javax.management.MBeanServer;
import javax.management.ObjectName;
import java.lang.management.ManagementFactory;

/**
 * Singleton that holds JMX-managed default values for
 * {@link io.github.jbellis.jvector.graph.GraphIndexBuilder} construction parameters.
 *
 * <h2>JMX Pattern — Standard MBean</h2>
 *
 * <p>This class uses Java's <em>Standard MBean</em> pattern, the simplest form of JMX
 * management.  The rules are:
 * <ol>
 *   <li>Define an interface whose name ends in {@code MBean}
 *       ({@link GraphIndexBuilderConfigMBean}).</li>
 *   <li>Implement that interface in a class with the same name minus the {@code MBean}
 *       suffix (this class).</li>
 *   <li>Register an instance with the platform {@link MBeanServer} under a unique
 *       {@link ObjectName}.</li>
 * </ol>
 *
 * <p>Once registered, any JMX client can inspect and modify the exposed attributes.
 * For example, using JConsole:
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
 * <p>Code that creates a {@code GraphIndexBuilder} and wants to respect the JMX-managed
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
 * JMX thread are immediately visible to application threads without additional
 * synchronization.
 *
 * <h2>Failure Policy</h2>
 *
 * <p>MBean registration is performed in the constructor and wrapped in a try/catch.
 * Registration failure (e.g., because the JVM has no platform MBeanServer or the name
 * is already taken) logs a warning and is otherwise silently ignored — the singleton is
 * still usable with its default values, so JMX availability is never on the critical
 * path.
 */
@Experimental
public class GraphIndexBuilderConfig implements GraphIndexBuilderConfigMBean {

    private static final Logger logger = LoggerFactory.getLogger(GraphIndexBuilderConfig.class);

    /**
     * JMX ObjectName under which this MBean is registered.
     * Domain: project base package.  Type: simple class name.
     */
    public static final String OBJECT_NAME = "io.github.jbellis.jvector:type=GraphIndexBuilderConfig";

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

    // ── Constructor ──────────────────────────────────────────────────────────

    private GraphIndexBuilderConfig() {
        try {
            MBeanServer server = ManagementFactory.getPlatformMBeanServer();
            ObjectName name = new ObjectName(OBJECT_NAME);
            server.registerMBean(this, name);
            logger.info("Registered JMX MBean: {}", OBJECT_NAME);
        } catch (Exception e) {
            // JMX registration is best-effort; do not disrupt normal operation.
            logger.warn("Failed to register JMX MBean '{}': {}", OBJECT_NAME, e.getMessage());
        }
    }

    // ── GraphIndexBuilderConfigMBean ─────────────────────────────────────────

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
}
