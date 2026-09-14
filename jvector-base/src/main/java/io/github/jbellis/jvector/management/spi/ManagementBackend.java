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

package io.github.jbellis.jvector.management.spi;

import io.github.jbellis.jvector.management.ManagementEntry;
import io.github.jbellis.jvector.management.ManagementRegistry;

/**
 * A pluggable transport that exposes {@link ManagementEntry} resources for external inspection
 * and control.
 *
 * <p>Exactly one backend is active per JVM, selected by
 * {@code io.github.jbellis.jvector.management.ManagementBackendProvider} via the
 * {@code jvector.management.backend} system property. The default,
 * {@code io.github.jbellis.jvector.management.jmx.JmxManagementBackend}, exposes resources as
 * JMX MBeans. A deployment that wants no external exposure can select
 * {@link NoopManagementBackend}; a deployment that wants a different transport entirely can
 * supply its own implementation.
 *
 * <p>Implementations must treat registration/deregistration failures as non-fatal — a
 * management backend is a convenience for operators, never on the critical path of the
 * application it's embedded in. {@link ManagementRegistry} additionally guards every call
 * against unexpected exceptions, but a well-behaved backend should not rely on that.
 */
public interface ManagementBackend {

    /** Exposes {@code entry} through this backend. Must not throw. */
    void bind(ManagementEntry entry);

    /** Withdraws a previously-{@link #bind}ed entry. Must not throw. */
    void unbind(ManagementEntry entry);
}
