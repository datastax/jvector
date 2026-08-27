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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.util.work.ProgressLimiter;
import io.github.jbellis.jvector.util.work.ProgressTracker;
import io.github.jbellis.jvector.util.work.WorkLimiter;
import io.github.jbellis.jvector.util.work.WorkStage;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The instrumentation wraps the embedder's limiter at the one point every stage passes through.
 * It must be transparent to the delegate — same calls, same order, same grant — and must produce
 * the stage log lines and the timing summary from those calls alone.
 */
public class TestStageInstrumentation {

    /** Records every call it receives, in order, and hands out a distinguishable grant. */
    private static final class RecordingLimiter implements ProgressLimiter {
        final List<String> calls = new ArrayList<>();
        final WorkLimiter.Grant grant = () -> calls.add("release");

        @Override
        public PhaseScope startPhase(WorkStage stage) {
            calls.add("start:" + stage.name());
            return new PhaseScope() {
                @Override
                public void onProgress(long completed, long total) {
                    calls.add("progress:" + stage.name() + ":" + completed + "/" + total);
                }

                @Override
                public void close() {
                    calls.add("close:" + stage.name());
                }
            };
        }

        @Override
        public Grant acquire(long amount) {
            calls.add("acquire:" + amount);
            return grant;
        }
    }

    @Test
    public void testDelegateSeesExactlyTheOriginalCalls() throws Exception {
        RecordingLimiter delegate = new RecordingLimiter();
        List<String> lines = new ArrayList<>();
        StageInstrumentation inst = new StageInstrumentation(delegate, lines::add);

        try (ProgressTracker.PhaseScope scope = inst.startPhase(CompactionStage.BASE_LAYER)) {
            scope.onProgress(0, 10);
            scope.onProgress(5, 10);
            scope.onProgress(10, 10);
            try (WorkLimiter.Grant g = inst.acquire(4096)) {
                assertSame("the delegate's grant is returned unchanged", delegate.grant, g);
            }
        }
        assertEquals(List.of("start:BASE_LAYER", "progress:BASE_LAYER:0/10", "progress:BASE_LAYER:5/10",
                             "progress:BASE_LAYER:10/10", "acquire:4096", "release", "close:BASE_LAYER"),
                     delegate.calls);
    }

    @Test
    public void testStageLinesAndSummary() throws Exception {
        List<String> lines = new ArrayList<>();
        StageInstrumentation inst = new StageInstrumentation(ProgressLimiter.UNLIMITED, lines::add);
        inst.beginRun();

        try (ProgressTracker.PhaseScope scope = inst.startPhase(CompactionStage.PQ_RETRAIN)) {
            scope.onProgress(0, 100);
            scope.onProgress(50, 100);
            scope.onProgress(51, 100);   // same 50% bucket: no second line
            scope.onProgress(100, 100);  // 100% is the completion line, not a progress line
        }
        // A stage may open several phases (upper layers, one per level): times accumulate.
        try (ProgressTracker.PhaseScope scope = inst.startPhase(CompactionStage.UPPER_LAYERS)) {
            scope.onProgress(3, 3);
        }
        try (ProgressTracker.PhaseScope scope = inst.startPhase(CompactionStage.UPPER_LAYERS)) {
            scope.onProgress(2, 2);
        }

        assertEquals("Stage PQ_RETRAIN started", lines.get(0));
        assertEquals("Stage PQ_RETRAIN: 100 units", lines.get(1));
        assertEquals("Stage PQ_RETRAIN progress: 50/100 (50%)", lines.get(2));
        assertTrue("completion line names the units and elapsed ms: " + lines.get(3),
                lines.get(3).startsWith("Stage PQ_RETRAIN completed: 100/100 units in ") && lines.get(3).endsWith(" ms"));
        assertEquals("one progress line per 10% bucket, none for 100%",
                4, lines.stream().filter(l -> l.startsWith("Stage PQ_RETRAIN")).count());

        Map<String, Long> ms = inst.stageMillis();
        assertEquals(List.of("PQ_RETRAIN", "UPPER_LAYERS"), new ArrayList<>(ms.keySet()));
        String summary = inst.summary();
        assertTrue(summary, summary.startsWith("Compaction stage times: PQ_RETRAIN="));
        assertTrue("repeated phases are counted: " + summary, summary.contains("UPPER_LAYERS=") && summary.contains("(x2)"));
        assertTrue(summary, summary.contains("| wall ") && summary.contains("throttle wait 0 ms (0 blocked admissions)"));

        inst.beginRun();
        assertTrue("beginRun clears the previous run", inst.stageMillis().isEmpty());
    }

    @Test
    public void testCloseForwardedOnce() throws Exception {
        RecordingLimiter delegate = new RecordingLimiter();
        StageInstrumentation inst = new StageInstrumentation(delegate, s -> { });
        ProgressTracker.PhaseScope scope = inst.startPhase(CompactionStage.FINALIZE);
        scope.close();
        scope.close();
        assertEquals(List.of("start:FINALIZE", "close:FINALIZE"), delegate.calls);
        assertEquals("a double close does not double-count the stage", 1, inst.stageMillis().size());
    }

    @Test
    public void testThrottleWaitIsAccounted() throws Exception {
        ProgressLimiter slow = new ProgressLimiter() {
            @Override
            public PhaseScope startPhase(WorkStage stage) {
                return PhaseScope.NOOP;
            }

            @Override
            public Grant acquire(long amount) throws InterruptedException {
                Thread.sleep(5);
                return Grant.NOOP;
            }
        };
        StageInstrumentation inst = new StageInstrumentation(slow, s -> { });
        inst.acquire(1);
        inst.acquire(1);
        assertTrue("blocked time is measured: " + inst.throttleMillis(), inst.throttleMillis() >= 8);
        assertTrue(inst.summary(), inst.summary().contains("(2 blocked admissions)"));
    }
}
