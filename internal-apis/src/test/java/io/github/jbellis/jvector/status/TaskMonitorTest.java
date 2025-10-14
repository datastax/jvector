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
import io.github.jbellis.jvector.status.eventing.StatusSource;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;
import org.junit.jupiter.api.Test;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.*;

class TaskMonitorTest {

    private static final class SampleTask implements StatusSource<SampleTask> {
        private volatile RunState state = RunState.PENDING;
        private volatile double progress = 0.0;

        @Override
        public StatusUpdate<SampleTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state, this);
        }

        void start(double progress) {
            this.state = RunState.RUNNING;
            this.progress = progress;
        }

        void finish() {
            this.progress = 1.0;
            this.state = RunState.SUCCESS;
        }
    }

    @Test
    void contextPollsTrackedTasks() throws InterruptedException {
        SampleTask task = new SampleTask();
        try (StatusContext context = new StatusContext("polling", Duration.ofMillis(20));
             TrackerScope scope = context.createScope("test-scope");
             StatusTracker<SampleTask> tracker = scope.trackTask(task)) {

            assertEquals(RunState.PENDING, tracker.getStatus().runstate);

            task.start(0.25);
            Thread.sleep(50);
            assertEquals(RunState.RUNNING, tracker.getStatus().runstate);
            assertEquals(0.25, tracker.getStatus().progress, 1e-6);

            task.finish();
            Thread.sleep(50);
            assertEquals(RunState.SUCCESS, tracker.getStatus().runstate);
            assertEquals(1.0, tracker.getStatus().progress, 1e-6);
        }
    }

    @Test
    void closingContextStopsPolling() throws InterruptedException {
        SampleTask task = new SampleTask();
        StatusContext context = new StatusContext("closable", Duration.ofMillis(10));
        TrackerScope scope = context.createScope("test-scope");
        StatusTracker<SampleTask> tracker = scope.trackTask(task);

        task.start(0.1);
        Thread.sleep(30);
        assertEquals(RunState.RUNNING, tracker.getStatus().runstate);

        context.close();

        task.finish();
        Thread.sleep(30);
        // Status should remain RUNNING because polling thread is stopped.
        assertEquals(RunState.RUNNING, tracker.getStatus().runstate);
        tracker.close();
    }
}
