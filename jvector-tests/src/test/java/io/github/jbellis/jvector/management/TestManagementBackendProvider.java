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

import io.github.jbellis.jvector.management.jmx.JmxManagementBackend;
import io.github.jbellis.jvector.management.spi.ManagementBackend;
import io.github.jbellis.jvector.management.spi.NoopManagementBackend;
import org.junit.After;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Exercises {@link ManagementBackendProvider#lookup()} directly (bypassing the memoized,
 * once-per-JVM {@code getInstance()} holder) to verify backend selection for every value of the
 * {@code jvector.management.backend} system property, including the fallback behavior when a
 * requested backend cannot be loaded.
 */
public class TestManagementBackendProvider {

    @After
    public void clearProperty() {
        System.clearProperty(ManagementBackendProvider.BACKEND_PROPERTY);
    }

    @Test
    public void defaultsToJmxWhenUnset() {
        System.clearProperty(ManagementBackendProvider.BACKEND_PROPERTY);
        assertTrue(ManagementBackendProvider.lookup() instanceof JmxManagementBackend);
    }

    @Test
    public void selectsJmxExplicitly() {
        System.setProperty(ManagementBackendProvider.BACKEND_PROPERTY, "jmx");
        assertTrue(ManagementBackendProvider.lookup() instanceof JmxManagementBackend);
    }

    @Test
    public void selectsJmxCaseInsensitively() {
        System.setProperty(ManagementBackendProvider.BACKEND_PROPERTY, "JMX");
        assertTrue(ManagementBackendProvider.lookup() instanceof JmxManagementBackend);
    }

    @Test
    public void selectsNoopBackend() {
        System.setProperty(ManagementBackendProvider.BACKEND_PROPERTY, "none");
        assertTrue(ManagementBackendProvider.lookup() instanceof NoopManagementBackend);
    }

    @Test
    public void loadsCustomBackendByClassName() {
        System.setProperty(ManagementBackendProvider.BACKEND_PROPERTY, StubBackend.class.getName());
        assertTrue(ManagementBackendProvider.lookup() instanceof StubBackend);
    }

    @Test
    public void fallsBackToNoopOnUnknownClassName() {
        System.setProperty(ManagementBackendProvider.BACKEND_PROPERTY, "not.a.real.ClassName");
        // must not throw
        assertTrue(ManagementBackendProvider.lookup() instanceof NoopManagementBackend);
    }

    /** Public, no-arg-constructible backend used to test the custom-class-name path. */
    public static final class StubBackend implements ManagementBackend {
        @Override
        public void bind(ManagementEntry entry) {
        }

        @Override
        public void unbind(ManagementEntry entry) {
        }
    }
}
