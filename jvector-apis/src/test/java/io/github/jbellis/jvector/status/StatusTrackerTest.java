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
import org.junit.jupiter.api.Test;
import io.github.jbellis.jvector.status.TrackerScope;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class StatusTrackerTest {

    private static final class InstrumentedTask implements StatusSource<InstrumentedTask> {
        private final String name;
        private volatile double progress;
        private volatile RunState state = RunState.PENDING;

        private InstrumentedTask(String name) {
            this.name = name;
        }

        void setProgress(double value) {
            progress = value;
            if (value >= 1.0) {
                state = RunState.SUCCESS;
            } else if (value > 0) {
                state = RunState.RUNNING;
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
    public void instrumentedTaskProgressIsReported() {
        try (StatusContext context = new StatusContext("tracker-test")) {
            InstrumentedTask task = new InstrumentedTask("task");
            try (StatusTracker<InstrumentedTask> tracker = context.track(task)) {
                task.setProgress(0.5);
                StatusUpdate<InstrumentedTask> update = tracker.getStatus();
                assertEquals(0.5, update.progress, 1e-6);
                assertEquals(RunState.RUNNING, update.runstate);
            }
        }
    }

    @Test
    public void functorBasedTrackingUsesCustomFunction() {
        try (StatusContext context = new StatusContext("functor")) {
            List<Double> samples = new ArrayList<>();
            FunctionTask task = new FunctionTask("functor");
            try (StatusTracker<FunctionTask> tracker = context.track(task, t -> {
                samples.add(t.progress);
                return new StatusUpdate<>(t.progress, t.state);
            }, Duration.ofMillis(25))) {
                task.advance(0.25);
                tracker.getStatus();
                task.advance(0.5);
                tracker.getStatus();
            }
            assertTrue(samples.stream().anyMatch(v -> v >= 0.5));
        }
    }

    private static final class FunctionTask {
        private final String name;
        private double progress;
        private RunState state = RunState.PENDING;

        private FunctionTask(String name) {
            this.name = name;
        }

        void advance(double value) {
            progress = Math.min(1.0, progress + value);
            state = progress >= 1.0 ? RunState.SUCCESS : RunState.RUNNING;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    @Test
    public void scopedTasksInheritContext() {
        RecordingSink sink = new RecordingSink();
        try (StatusContext context = new StatusContext("hierarchy", List.of(sink))) {
            InstrumentedTask task1 = new InstrumentedTask("task1");
            InstrumentedTask task2 = new InstrumentedTask("task2");

            try (TrackerScope scope = context.createScope("TestScope")) {
                try (StatusTracker<InstrumentedTask> tracker1 = scope.trackTask(task1);
                     StatusTracker<InstrumentedTask> tracker2 = scope.trackTask(task2)) {
                    // Verify scope relationship and context
                    assertEquals(scope, tracker1.getParentScope());
                    assertEquals(scope, tracker2.getParentScope());
                    assertEquals(context, tracker1.getContext());
                    assertEquals(context, tracker2.getContext());
                    task2.setProgress(1.0);
                    tracker2.getStatus();
                }
            }

            assertEquals(RecordingSink.Event.FINISH, sink.events.get(sink.events.size() - 1));
        }
    }

    @Test
    public void addAndRemoveSinksDynamically() {
        RecordingSink sink = new RecordingSink();
        try (StatusContext context = new StatusContext("dynamics")) {
            InstrumentedTask task = new InstrumentedTask("task");
            try (StatusTracker<InstrumentedTask> tracker = context.track(task)) {
                // Sinks are managed at context level
                context.addSink(sink);
                task.setProgress(1.0);
                tracker.getStatus();
                context.removeSink(sink);
            }
            assertTrue(sink.events.contains(RecordingSink.Event.START));
            assertTrue(sink.events.contains(RecordingSink.Event.FINISH));
        }
    }

    @Test
    public void closingTrackerIdempotent() {
        StatusContext context = new StatusContext("idempotent");
        InstrumentedTask task = new InstrumentedTask("task");
        StatusTracker<InstrumentedTask> tracker = context.track(task);
        tracker.close();
        tracker.close();
        assertEquals(0, context.getActiveTrackerCount());
        context.close();
    }

    @Test
    public void contextClosesAllScopesAndTasks() {
        StatusContext context = new StatusContext("parent");
        TrackerScope scope = context.createScope("TestScope");
        InstrumentedTask task1 = new InstrumentedTask("task1");
        InstrumentedTask task2 = new InstrumentedTask("task2");
        StatusTracker<InstrumentedTask> tracker1 = scope.trackTask(task1);
        StatusTracker<InstrumentedTask> tracker2 = scope.trackTask(task2);

        context.close();

        assertTrue(scope.getChildTasks().isEmpty());
        assertTrue(context.getActiveTrackers().isEmpty());
        assertNotNull(tracker1.getStatus());
        assertNotNull(tracker2.getStatus());
    }

    @Test
    public void scopeMustShareSameContext() {
        try (StatusContext contextA = new StatusContext("A");
             StatusContext contextB = new StatusContext("B")) {
            TrackerScope scopeA = contextA.createScope("ScopeA");
            TrackerScope scopeB = contextB.createScope("ScopeB");

            // Attempting to track a task in scopeB using contextA should fail
            IllegalArgumentException ex = assertThrows(
                    IllegalArgumentException.class,
                    () -> contextA.trackInScope(scopeB, new InstrumentedTask("task"), StatusSource::getTaskStatus, null));

            assertTrue(ex.getMessage().contains("different StatusContext"));

            scopeA.close();
            scopeB.close();
        }
    }

    @Test
    public void elapsedRunningTimeTracksExecution() throws InterruptedException {
        try (StatusContext context = new StatusContext("timing")) {
            InstrumentedTask task = new InstrumentedTask("task");
            try (StatusTracker<InstrumentedTask> tracker = context.track(task)) {
                assertNull(tracker.getRunningStartTime());
                assertEquals(0, tracker.getElapsedRunningTime());

                task.setProgress(0.1);
                tracker.refreshAndGetStatus();
                assertNotNull(tracker.getRunningStartTime());

                Thread.sleep(5);
                long runningElapsed = tracker.getElapsedRunningTime();
                assertTrue(runningElapsed >= 5);

                task.setProgress(1.0);
                tracker.refreshAndGetStatus();

                long finalElapsed = tracker.getElapsedRunningTime();
                assertTrue(finalElapsed >= runningElapsed);
                assertNotNull(tracker.getRunningStartTime());
            }
        }
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
