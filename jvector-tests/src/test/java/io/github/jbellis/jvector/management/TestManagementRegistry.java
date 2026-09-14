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
import org.junit.Test;

import java.util.ArrayDeque;
import java.util.Deque;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * Verifies {@link ManagementRegistry}'s register/unregister wiring against a fake
 * {@link ManagementBackend}, independent of JMX or any other real transport &mdash; this is
 * what actually proves the management abstraction is swappable: nothing here touches
 * {@code javax.management}.
 */
public class TestManagementRegistry {

    private interface Greeter {
        String greeting();
    }

    private static class FakeResource implements ManagedResource, Greeter {
        private final String name;

        FakeResource(String name) {
            this.name = name;
        }

        @Override
        public String managementName() {
            return name;
        }

        @Override
        public String managementDescription() {
            return "a fake resource for testing";
        }

        @Override
        public String greeting() {
            return "hello from " + name;
        }
    }

    private enum Event { BIND, UNBIND }

    private static class RecordingBackend implements ManagementBackend {
        final Deque<Event> events = new ArrayDeque<>();
        final Deque<ManagementEntry> entries = new ArrayDeque<>();
        boolean throwOnBind = false;
        boolean throwOnUnbind = false;

        @Override
        public void bind(ManagementEntry entry) {
            events.add(Event.BIND);
            entries.add(entry);
            if (throwOnBind) {
                throw new RuntimeException("boom (bind)");
            }
        }

        @Override
        public void unbind(ManagementEntry entry) {
            events.add(Event.UNBIND);
            entries.add(entry);
            if (throwOnUnbind) {
                throw new RuntimeException("boom (unbind)");
            }
        }
    }

    @Test
    public void registerBindsIntoBackend() {
        var backend = new RecordingBackend();
        var registry = new ManagementRegistry(backend);
        var resource = new FakeResource("Greeter1");

        ManagementEntry entry = registry.register(resource, Greeter.class);

        assertEquals("Greeter1", entry.name());
        assertEquals("a fake resource for testing", entry.description());
        assertSame(resource, entry.resource());
        assertEquals(Greeter.class, entry.serviceInterface());

        assertEquals(1, backend.events.size());
        assertEquals(Event.BIND, backend.events.peek());
        assertSame(entry, backend.entries.peek());
    }

    @Test
    public void unregisterUnbindsFromBackend() {
        var backend = new RecordingBackend();
        var registry = new ManagementRegistry(backend);
        var resource = new FakeResource("Greeter2");

        registry.register(resource, Greeter.class);
        backend.events.clear();
        backend.entries.clear();

        registry.unregister("Greeter2");

        assertEquals(1, backend.events.size());
        assertEquals(Event.UNBIND, backend.events.peek());
    }

    @Test
    public void unregisterUnknownNameIsNoop() {
        var backend = new RecordingBackend();
        var registry = new ManagementRegistry(backend);

        registry.unregister("does-not-exist");

        assertTrue(backend.events.isEmpty());
    }

    @Test
    public void reregisteringSameNameUnbindsPreviousThenBindsNew() {
        var backend = new RecordingBackend();
        var registry = new ManagementRegistry(backend);
        var first = new FakeResource("Shared");
        var second = new FakeResource("Shared");

        registry.register(first, Greeter.class);
        backend.events.clear();
        backend.entries.clear();

        registry.register(second, Greeter.class);

        assertEquals(2, backend.events.size());
        var iterator = backend.events.iterator();
        assertEquals(Event.UNBIND, iterator.next());
        assertEquals(Event.BIND, iterator.next());

        var entryIterator = backend.entries.iterator();
        assertSame(first, entryIterator.next().resource());
        assertSame(second, entryIterator.next().resource());
    }

    @Test
    public void backendExceptionsOnBindDoNotPropagate() {
        var backend = new RecordingBackend();
        backend.throwOnBind = true;
        var registry = new ManagementRegistry(backend);

        // must not throw
        ManagementEntry entry = registry.register(new FakeResource("Flaky"), Greeter.class);

        assertEquals("Flaky", entry.name());
    }

    @Test
    public void backendExceptionsOnUnbindDoNotPropagate() {
        var backend = new RecordingBackend();
        var registry = new ManagementRegistry(backend);
        registry.register(new FakeResource("Flaky2"), Greeter.class);
        backend.throwOnUnbind = true;

        // must not throw
        registry.unregister("Flaky2");
    }

    @Test(expected = IllegalArgumentException.class)
    public void registeringUnderWrongInterfaceThrows() {
        var backend = new RecordingBackend();
        var registry = new ManagementRegistry(backend);

        // FakeResource does not implement Runnable
        registry.register(new FakeResource("Mismatched"), Runnable.class);
    }
}
