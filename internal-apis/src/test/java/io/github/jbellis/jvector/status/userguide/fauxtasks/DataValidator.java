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
 * Example validation task used in user guide examples.
 */
public class DataValidator implements StatusSource<DataValidator> {
    private final long totalRecords = 1000;
    private volatile long recordsValidated = 0;
    private volatile RunState state = RunState.PENDING;

    @Override
    public StatusUpdate<DataValidator> getTaskStatus() {
        return new StatusUpdate<>((double) recordsValidated / totalRecords, state, this);
    }

    public void validate() {
        state = RunState.RUNNING;
        for (int i = 0; i < totalRecords; i++) {
            validateRecord(i);
            recordsValidated++;
        }
        state = RunState.SUCCESS;
    }

    private void validateRecord(int i) {
        try {
            Thread.sleep(1);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public String getName() {
        return "DataValidator";
    }
}
