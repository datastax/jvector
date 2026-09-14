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

/**
 * A {@link ManagementBackend} that does nothing. Selected via
 * {@code -Djvector.management.backend=none} for deployments that want managed resources (such
 * as {@code io.github.jbellis.jvector.management.GraphIndexBuilderConfig}) to remain usable
 * programmatically without exposing them through any external transport — for example, when
 * JMX is disallowed in the target environment.
 *
 * <p>This is also the fallback used automatically when the configured backend fails to load,
 * so that a misconfigured or unavailable management backend never prevents the application
 * from starting.
 */
public final class NoopManagementBackend implements ManagementBackend {

    @Override
    public void bind(ManagementEntry entry) {
        // intentionally no-op
    }

    @Override
    public void unbind(ManagementEntry entry) {
        // intentionally no-op
    }
}
