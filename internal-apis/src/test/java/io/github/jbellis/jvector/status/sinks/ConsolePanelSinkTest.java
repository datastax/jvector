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
import io.github.jbellis.jvector.status.eventing.StatusSource;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class ConsolePanelSinkTest {

    private static final class DemoTask implements StatusSource<DemoTask> {
        private final String name;
        private volatile double progress;
        private volatile RunState state = RunState.PENDING;

        DemoTask(String name) {
            this.name = name;
        }

        @Override
        public StatusUpdate<DemoTask> getTaskStatus() {
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
    void panelReceivesLifecycleEvents() throws InterruptedException {
        ConsolePanelSink sink = ConsolePanelSink.builder()
                .withRefreshRateMs(50)
                .withCompletedTaskRetention(1, TimeUnit.SECONDS)
                .withCaptureSystemStreams(false)
                .withColorOutput(false)
                .build();

        try (StatusContext context = new StatusContext("panel", Duration.ofMillis(20), List.of(sink))) {
            DemoTask task = new DemoTask("panel-demo");
            try (var scope = context.createScope("test-scope");
                 StatusTracker<DemoTask> tracker = scope.trackTask(task)) {
                task.advance(0.5);
                Thread.sleep(60);
                assertEquals(RunState.RUNNING, tracker.getStatus().runstate);
                task.advance(0.5);
                Thread.sleep(60);
                assertEquals(RunState.SUCCESS, tracker.getStatus().runstate);
            }
        } finally {
            sink.close();
        }
    }
}
