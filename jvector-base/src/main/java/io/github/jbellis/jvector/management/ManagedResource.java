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
 * A resource that can be exposed through a {@link ManagementRegistry} to whichever
 * {@link io.github.jbellis.jvector.management.spi.ManagementBackend} is active in this JVM
 * (JMX by default — see {@code io.github.jbellis.jvector.management.jmx.JmxManagementBackend}).
 *
 * <p>Implementations are plain domain objects; they carry no dependency on any particular
 * management transport. A backend consults {@link #managementName()} and
 * {@link #managementDescription()} purely as presentation metadata — for example, the JMX
 * backend derives an {@code ObjectName} from {@link #managementName()}.
 */
public interface ManagedResource {

    /**
     * A short identifier for this resource, unique within the JVM. Defaults to the implementing
     * class's simple name, which is sufficient for singleton-style resources such as
     * {@link GraphIndexBuilderConfig}.
     */
    default String managementName() {
        return getClass().getSimpleName();
    }

    /**
     * A human-readable description of what this resource manages. Purely informational;
     * backends may surface it (for example, as an MBean description) or ignore it.
     */
    default String managementDescription() {
        return "";
    }
}
