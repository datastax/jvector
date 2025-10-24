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

/**
 * Example processing task used in user guide examples.
 */
public class DataProcessor implements StatusSource<DataProcessor> {
    private final long totalRecords = 1000;
    private volatile long recordsProcessed = 0;
    private volatile RunState state = RunState.PENDING;

    @Override
    public StatusUpdate<DataProcessor> getTaskStatus() {
        return new StatusUpdate<>((double) recordsProcessed / totalRecords, state, this);
    }

    public void process() {
        state = RunState.RUNNING;
        for (int i = 0; i < totalRecords; i++) {
            processRecord(i);
            recordsProcessed++;
        }
        state = RunState.SUCCESS;
    }

    private void processRecord(int i) {
        try {
            Thread.sleep(1);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public String getName() {
        return "DataProcessor";
    }
}
