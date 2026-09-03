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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * In-process directory of {@link ManagedResource}s.
 *
 * <p>Domain objects (such as {@link GraphIndexBuilderConfig}) register themselves here; the
 * registry hands each registration to the single {@link ManagementBackend} active in this JVM
 * (selected by {@link ManagementBackendProvider}) so it can be exposed externally. This class
 * has no dependency on JMX or any other specific transport &mdash; see
 * {@link ManagementBackendProvider} for backend selection and
 * {@code io.github.jbellis.jvector.management.jmx.JmxManagementBackend} for the default JMX
 * implementation.
 *
 * <p>Thread-safe: registrations may occur concurrently from any thread.
 */
public final class ManagementRegistry {

    private static final Logger logger = LoggerFactory.getLogger(ManagementRegistry.class);

    private static final class Holder {
        static final ManagementRegistry INSTANCE = new ManagementRegistry();
    }

    public static ManagementRegistry getInstance() {
        return Holder.INSTANCE;
    }

    private final ConcurrentMap<String, ManagementEntry> entries = new ConcurrentHashMap<>();
    private final ManagementBackend backend;

    private ManagementRegistry() {
        this(ManagementBackendProvider.getInstance());
    }

    // visible for testing — lets tests exercise register/unregister logic against a fake
    // backend without going through the single JVM-wide backend selected by
    // ManagementBackendProvider.
    ManagementRegistry(ManagementBackend backend) {
        this.backend = backend;
    }

    /**
     * Registers {@code resource} as implementing {@code serviceInterface} and binds it into the
     * active {@link ManagementBackend}.
     *
     * <p>{@code resource.managementName()} must be unique within the JVM; re-registering under
     * the same name unbinds the previous entry first.
     *
     * @return the {@link ManagementEntry} created for this registration
     */
    public ManagementEntry register(ManagedResource resource, Class<?> serviceInterface) {
        ManagementEntry entry = new ManagementEntry(
                resource.managementName(), resource.managementDescription(), resource, serviceInterface);
        ManagementEntry previous = entries.put(entry.name(), entry);
        if (previous != null) {
            logger.warn("Replacing existing management registration for '{}'", entry.name());
            safeUnbind(previous);
        }
        safeBind(entry);
        return entry;
    }

    /** Removes a previously-registered resource by name, unbinding it from the active backend. */
    public void unregister(String name) {
        ManagementEntry entry = entries.remove(name);
        if (entry != null) {
            safeUnbind(entry);
        }
    }

    private void safeBind(ManagementEntry entry) {
        try {
            backend.bind(entry);
        } catch (Exception e) {
            logger.warn("Management backend failed to bind '{}': {}", entry.name(), e.getMessage());
        }
    }

    private void safeUnbind(ManagementEntry entry) {
        try {
            backend.unbind(entry);
        } catch (Exception e) {
            logger.warn("Management backend failed to unbind '{}': {}", entry.name(), e.getMessage());
        }
    }
}
