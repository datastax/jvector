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

package io.github.jbellis.jvector.status.sinks;

import io.github.jbellis.jvector.status.StatusContext;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.eventing.RunState;
import io.github.jbellis.jvector.status.eventing.StatusSink;
import io.github.jbellis.jvector.status.eventing.StatusSource;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class StatusSinkTest {

    private static final class TestTask implements StatusSource<TestTask> {
        private final String name;
        private volatile double progress;
        private volatile RunState state = RunState.PENDING;

        TestTask(String name) {
            this.name = name;
        }

        @Override
        public StatusUpdate<TestTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state, this);
        }

        void advance(double delta) {
            progress = Math.min(1.0, progress + delta);
            state = progress >= 1.0 ? RunState.SUCCESS : RunState.RUNNING;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    @Test
    void noopSinkAcceptsLifecycle() throws InterruptedException {
        StatusSink sink = NoopStatusSink.getInstance();
        try (StatusContext context = new StatusContext("noop", List.of(sink))) {
            TestTask task = new TestTask("noop-task");
            try (StatusTracker<TestTask> tracker = context.track(task)) {
                task.advance(1.0);
                Thread.sleep(30);
                assertEquals(RunState.SUCCESS, tracker.getStatus().runstate);
            }
        }
    }

    @Test
    void consoleLoggerFormatsOutput() throws InterruptedException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        ConsoleLoggerSink sink = new ConsoleLoggerSink(new PrintStream(buffer), false, true);

        try (StatusContext context = new StatusContext("console", List.of(sink))) {
            TestTask task = new TestTask("console-task");
            try (StatusTracker<TestTask> tracker = context.track(task, Duration.ofMillis(15))) {
                task.advance(0.5);
                Thread.sleep(40);
                task.advance(0.5);
                Thread.sleep(40);
                assertTrue(buffer.toString().contains("console-task"));
            }
        }
    }

    @Test
    void metricsSinkAggregatesStats() throws InterruptedException {
        MetricsStatusSink sink = new MetricsStatusSink();
        try (StatusContext context = new StatusContext("metrics", List.of(sink))) {
            TestTask task = new TestTask("metrics-task");
            try (StatusTracker<TestTask> tracker = context.track(task, Duration.ofMillis(10))) {
                for (int i = 0; i < 3; i++) {
                    task.advance(0.4);
                    Thread.sleep(25);
                }
            }
        }
        assertTrue(sink.getTotalUpdates() >= 3);
        assertEquals(1, sink.getTotalTasksFinished());
    }

    @Test
    void multipleSinksReceiveUpdates() throws InterruptedException {
        List<RunState> observedStates = new ArrayList<>();
        StatusSink collector = new StatusSink() {
            @Override
            public void taskStarted(StatusTracker<?> task) {
                observedStates.add(RunState.PENDING);
            }

            @Override
            public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
                observedStates.add(status.runstate);
            }

            @Override
            public void taskFinished(StatusTracker<?> task) {
                observedStates.add(RunState.SUCCESS);
            }
        };

        MetricsStatusSink metrics = new MetricsStatusSink();
        try (StatusContext context = new StatusContext("multi", List.of(collector, metrics))) {
            TestTask task = new TestTask("multi-task");
            try (StatusTracker<TestTask> tracker = context.track(task, Duration.ofMillis(10))) {
                task.advance(1.0);
                Thread.sleep(30);
                assertEquals(RunState.SUCCESS, tracker.getStatus().runstate);
            }
        }

        assertTrue(observedStates.contains(RunState.RUNNING));
        assertTrue(metrics.getTotalTasksFinished() >= 1);
    }
}
