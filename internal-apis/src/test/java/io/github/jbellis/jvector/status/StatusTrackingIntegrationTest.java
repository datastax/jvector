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
import io.github.jbellis.jvector.status.sinks.MetricsStatusSink;
import io.github.jbellis.jvector.status.TrackerScope;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class StatusTrackingIntegrationTest {

    private static final class RecordingSink implements StatusSink {
        final List<RunState> states = new ArrayList<>();

        @Override
        public void taskStarted(StatusTracker<?> task) {
            states.add(RunState.PENDING);
        }

        @Override
        public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
            states.add(status.runstate);
        }

        @Override
        public void taskFinished(StatusTracker<?> task) {
            states.add(RunState.SUCCESS);
        }
    }

    private static final class WorkTask implements StatusSource<WorkTask> {
        private volatile double progress;
        private volatile RunState state = RunState.PENDING;

        @Override
        public StatusUpdate<WorkTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state, this);
        }

        void advance(double delta) {
            progress = Math.min(1.0, progress + delta);
            state = progress >= 1.0 ? RunState.SUCCESS : RunState.RUNNING;
        }
    }

    @Test
    void trackersShareContextSinks() throws InterruptedException {
        RecordingSink sink = new RecordingSink();
        MetricsStatusSink metrics = new MetricsStatusSink();

        try (StatusContext context = new StatusContext("integration", Duration.ofMillis(15), List.of(sink, metrics))) {
            try (TrackerScope scope = context.createScope("TestWorkload")) {
                WorkTask task1 = new WorkTask();
                WorkTask task2 = new WorkTask();
                try (StatusTracker<WorkTask> tracker1 = scope.trackTask(task1);
                     StatusTracker<WorkTask> tracker2 = scope.trackTask(task2)) {
                    task1.advance(0.5);
                    task2.advance(1.0);
                    Thread.sleep(50);
                    assertEquals(RunState.RUNNING, tracker1.getStatus().runstate);
                    assertEquals(RunState.SUCCESS, tracker2.getStatus().runstate);
                }
            }
        }

        assertTrue(metrics.getTotalTasksStarted() >= 2);
        assertTrue(metrics.getTotalTasksFinished() >= 2);
        assertTrue(sink.states.contains(RunState.RUNNING));
    }
}
