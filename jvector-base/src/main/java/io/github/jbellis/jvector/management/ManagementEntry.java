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

import java.util.Objects;

/**
 * Immutable description of a {@link ManagedResource} bound into a {@link ManagementRegistry},
 * handed to the active {@link io.github.jbellis.jvector.management.spi.ManagementBackend} on
 * {@code bind()}/{@code unbind()}.
 */
public final class ManagementEntry {

    private final String name;
    private final String description;
    private final ManagedResource resource;
    private final Class<?> serviceInterface;

    public ManagementEntry(String name, String description, ManagedResource resource, Class<?> serviceInterface) {
        this.name = Objects.requireNonNull(name, "name");
        this.description = Objects.requireNonNull(description, "description");
        this.resource = Objects.requireNonNull(resource, "resource");
        this.serviceInterface = Objects.requireNonNull(serviceInterface, "serviceInterface");
        if (!serviceInterface.isInstance(resource)) {
            throw new IllegalArgumentException(
                    resource.getClass().getName() + " does not implement " + serviceInterface.getName());
        }
    }

    /** Short identifier for this resource, unique within the JVM. Used, e.g., to derive a JMX {@code ObjectName}. */
    public String name() {
        return name;
    }

    /** Human-readable description of what this resource manages. May be empty. */
    public String description() {
        return description;
    }

    /** The registered resource itself. */
    public ManagedResource resource() {
        return resource;
    }

    /** The interface under which {@link #resource()} is exposed to backends. */
    public Class<?> serviceInterface() {
        return serviceInterface;
    }
}
