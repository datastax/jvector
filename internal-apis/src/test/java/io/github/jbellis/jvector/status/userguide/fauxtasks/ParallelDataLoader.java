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

package io.github.jbellis.jvector.status.userguide.fauxtasks;

import io.github.jbellis.jvector.status.eventing.RunState;
import io.github.jbellis.jvector.status.eventing.StatusSource;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;

import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

/**
 * Example parallel data loader using AtomicLong for thread-safe progress tracking.
 */
public class ParallelDataLoader implements StatusSource<ParallelDataLoader> {
    private final long totalRecords = 1000;
    private final AtomicLong recordsLoaded = new AtomicLong(0);
    private volatile RunState state = RunState.PENDING;

    @Override
    public StatusUpdate<ParallelDataLoader> getTaskStatus() {
        double progress = (double) recordsLoaded.get() / totalRecords;
        return new StatusUpdate<>(progress, state, this);
    }

    public void load() {
        state = RunState.RUNNING;
        // Multiple threads can safely call recordsLoaded.incrementAndGet()
        IntStream.range(0, (int) totalRecords).parallel().forEach(i -> {
            loadRecord(i);
            recordsLoaded.incrementAndGet(); // Atomic - thread-safe
        });
        state = RunState.SUCCESS;
    }

    private void loadRecord(int i) {
        try {
            Thread.sleep(1);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public String getName() {
        return "ParallelDataLoader";
    }
}
