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

import io.github.jbellis.jvector.management.spi.ManagementBackend;
import io.github.jbellis.jvector.management.spi.NoopManagementBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Selects the single {@link ManagementBackend} active for this JVM, based on the
 * {@code jvector.management.backend} system property:
 * <ul>
 *   <li>unset, blank, or {@code "jmx"} (default) &mdash;
 *       {@code io.github.jbellis.jvector.management.jmx.JmxManagementBackend}, preserving
 *       JVector's zero-configuration JMX behavior.</li>
 *   <li>{@code "none"} &mdash; {@link NoopManagementBackend}; managed resources remain usable
 *       programmatically but are not exposed through any external transport.</li>
 *   <li>any other value &mdash; treated as the fully-qualified class name of a custom
 *       {@link ManagementBackend} implementation with a public no-arg constructor.</li>
 * </ul>
 *
 * <p>Backend selection happens once per JVM (via the initialization-on-demand holder idiom),
 * mirroring {@link io.github.jbellis.jvector.vector.VectorizationProvider}'s lookup pattern.
 * Failure to construct the requested backend falls back to {@link NoopManagementBackend} with a
 * warning: management-backend selection must never be able to prevent the application from
 * starting.
 */
public final class ManagementBackendProvider {

    private static final Logger logger = LoggerFactory.getLogger(ManagementBackendProvider.class);

    /** System property used to select the active {@link ManagementBackend}. */
    public static final String BACKEND_PROPERTY = "jvector.management.backend";

    private static final String JMX_BACKEND_CLASS = "io.github.jbellis.jvector.management.jmx.JmxManagementBackend";

    private ManagementBackendProvider() {
    }

    /** Returns the {@link ManagementBackend} selected for this JVM. */
    public static ManagementBackend getInstance() {
        return Holder.INSTANCE;
    }

    // visible for testing
    static ManagementBackend lookup() {
        String configured = System.getProperty(BACKEND_PROPERTY);
        if (configured != null) {
            configured = configured.trim();
        }

        if (configured == null || configured.isEmpty() || "jmx".equalsIgnoreCase(configured)) {
            return load(JMX_BACKEND_CLASS);
        }
        if ("none".equalsIgnoreCase(configured)) {
            return new NoopManagementBackend();
        }
        return load(configured);
    }

    private static ManagementBackend load(String className) {
        try {
            Class<?> clazz = Class.forName(className);
            ManagementBackend backend = (ManagementBackend) clazz.getConstructor().newInstance();
            logger.info("Using management backend: {}", className);
            return backend;
        } catch (Exception e) {
            logger.warn("Failed to load management backend '{}' ({}); falling back to no-op. " +
                            "Managed resources remain usable programmatically but will not be exposed externally.",
                    className, e.toString());
            return new NoopManagementBackend();
        }
    }

    /** Initialization-on-demand holder; prevents classloading deadlock, as in {@code VectorizationProvider}. */
    private static final class Holder {
        static final ManagementBackend INSTANCE = lookup();
    }
}
