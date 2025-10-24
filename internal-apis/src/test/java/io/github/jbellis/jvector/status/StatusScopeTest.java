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

package io.github.jbellis.jvector.status;

import io.github.jbellis.jvector.status.eventing.RunState;
import io.github.jbellis.jvector.status.eventing.StatusSink;
import io.github.jbellis.jvector.status.eventing.StatusSource;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;
import io.github.jbellis.jvector.status.sinks.NoopStatusSink;
import io.github.jbellis.jvector.status.StatusScope;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies that {@link StatusContext} manages track hierarchies and monitoring resources.
 */
public class StatusScopeTest {

    private static final class InstrumentedTask implements StatusSource<InstrumentedTask> {
        private final String name;
        private volatile double progress;
        private volatile RunState state = RunState.PENDING;

        private InstrumentedTask(String name) {
            this.name = name;
        }

        void start() {
            state = RunState.RUNNING;
            progress = 0.0;
        }

        void advance(double increment) {
            progress = Math.min(1.0, progress + increment);
            if (progress >= 1.0) {
                state = RunState.SUCCESS;
            }
        }

        @Override
        public StatusUpdate<InstrumentedTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state);
        }

        @Override
        public String toString() {
            return name;
        }
    }

    @Test
    public void tracksRootTaskInSingleContext() {
        try (StatusContext context = new StatusContext("root");
             StatusScope scope = context.createScope("root-scope")) {
            InstrumentedTask task = new InstrumentedTask("root-task");
            try (StatusTracker<InstrumentedTask> tracker = scope.trackTask(task)) {
                assertEquals("root", context.getName());
                assertEquals(Duration.ofMillis(100), context.getDefaultPollInterval());
                assertEquals(1, context.getActiveTrackerCount());
                assertEquals(scope, tracker.getParentScope());
            }
            assertEquals(0, context.getActiveTrackerCount());
        }
    }

    @Test
    public void scopedTasksShareContextAndHierarchy() {
        try (StatusContext context = new StatusContext("root")) {
            InstrumentedTask task1 = new InstrumentedTask("task1");
            InstrumentedTask task2 = new InstrumentedTask("task2");

            try (StatusScope scope = context.createScope("TestScope")) {
                try (StatusTracker<InstrumentedTask> tracker1 = scope.trackTask(task1);
                     StatusTracker<InstrumentedTask> tracker2 = scope.trackTask(task2)) {
                    assertSame(scope, tracker1.getParentScope());
                    assertSame(scope, tracker2.getParentScope());
                    assertEquals(2, scope.getChildTasks().size());
                    assertEquals(2, context.getActiveTrackerCount());
                }
                assertTrue(scope.getChildTasks().isEmpty());
            }
        }
    }

    @Test
    public void contextOwnsMonitorLifecycle() {
        try (StatusContext context = new StatusContext("root");
             StatusScope scope = context.createScope("test")) {
            InstrumentedTask task = new InstrumentedTask("monitored");
            try (StatusTracker<InstrumentedTask> tracker = scope.trackTask(task)) {
                task.start();
                task.advance(0.5);
                assertEquals(RunState.RUNNING, tracker.getStatus().runstate);
            }
            assertEquals(0, context.getActiveTrackerCount());
        }
    }

    @Test
    public void additionalSinksApplyToNewTrackers() {
        RecordingSink sink = new RecordingSink();
        try (StatusContext context = new StatusContext("root")) {
            context.addSink(sink);
            try (var scope = context.createScope("test-scope");
                 StatusTracker<InstrumentedTask> tracker = scope.trackTask(new InstrumentedTask("task"))) {
                // Sinks are managed by context, verify notifications reach the sink
                tracker.getStatus();
            }
            assertTrue(sink.events.contains(RecordingSink.Event.START));

            sink.events.clear();
            context.removeSink(sink);

            try (var scope = context.createScope("test-scope2");
                 StatusTracker<InstrumentedTask> tracker = scope.trackTask(new InstrumentedTask("task2"))) {
                tracker.getStatus();
            }
            // After removal, sink should not receive events
            assertFalse(sink.events.contains(RecordingSink.Event.START));
        }
    }

    @Test
    public void closingContextStopsAllTrackersAndScopes() {
        StatusContext context = new StatusContext("root");
        InstrumentedTask task1 = new InstrumentedTask("task1");
        InstrumentedTask task2 = new InstrumentedTask("task2");

        StatusScope scope = context.createScope("TestScope");
        StatusTracker<InstrumentedTask> tracker1 = scope.trackTask(task1);
        StatusTracker<InstrumentedTask> tracker2 = scope.trackTask(task2);

        context.close();

        assertTrue(context.isClosed());
        assertEquals(0, context.getActiveTrackerCount());
        assertTrue(scope.getChildTasks().isEmpty());
        assertNotNull(tracker1.getStatus());
        assertNotNull(tracker2.getStatus());
    }

    private static final class RecordingSink implements StatusSink {
        enum Event { START, UPDATE, FINISH }

        final List<Event> events = new ArrayList<>();

        @Override
        public void taskStarted(StatusTracker<?> task) {
            events.add(Event.START);
        }

        @Override
        public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
            events.add(Event.UPDATE);
        }

        @Override
        public void taskFinished(StatusTracker<?> task) {
            events.add(Event.FINISH);
        }
    }
}
