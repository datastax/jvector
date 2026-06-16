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

package io.github.jbellis.jvector.graph;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests adaptive early termination based on per-query result-set discovery.
 *
 * <p>The tracker models the "Patience in Proximity" idea: terminate only after
 * the tracked nearest-neighbor set stops changing for a sustained period. The
 * adaptive variant uses per-query discovery-rate statistics to decide saturation
 * and patience.</p>
 */
class AdaptiveTerminationTest {
    private static final int K = 10;

    @Test
    void resetRejectsZeroK() {
        AdaptiveTermination tracker = new AdaptiveTermination();

        assertThrows(IllegalArgumentException.class, () -> tracker.reset(0));
    }

    @Test
    void resetRejectsNegativeK() {
        AdaptiveTermination tracker = new AdaptiveTermination();

        assertThrows(IllegalArgumentException.class, () -> tracker.reset(-1));
    }

    @Test
    void doesNotTerminateWithoutObservations() {
        AdaptiveTermination tracker = new AdaptiveTermination();
        tracker.reset(K);

        assertFalse(tracker.shouldTerminate());
    }

    @Test
    void oneObservationIsInsufficient() {
        AdaptiveTermination tracker = new AdaptiveTermination();
        tracker.reset(K);

        tracker.observe(0, 1, true);

        assertFalse(tracker.shouldTerminate());
    }

    @Test
    void steadyHealthyDiscoveryDoesNotTerminate() {
        AdaptiveTermination tracker = new AdaptiveTermination();
        tracker.reset(K);

        for (int i = 0; i < 200; i++) {
            tracker.observe(K, 1, true);
            assertFalse(tracker.shouldTerminate());
        }
    }

    @Test
    void largeKSteadyHealthyDiscoveryDoesNotTerminate() {
        int k = 1_000;
        AdaptiveTermination tracker = new AdaptiveTermination();
        tracker.reset(k);

        for (int i = 0; i < 200; i++) {
            tracker.observe(k, 1, true);
            assertFalse(tracker.shouldTerminate());
        }
    }

    @Test
    void terminationRequiresResultSetFull() {
        AdaptiveTermination tracker = new AdaptiveTermination();
        tracker.reset(K);

        seedHealthyDiscovery(tracker, K);

        boolean terminated = observeUntilTerminated(
                tracker,
                0,
                1,
                false,
                200
        );

        assertFalse(terminated);
    }

    @Test
    void terminatesWhenDiscoveryDriesUpAfterResultSetIsFull() {
        AdaptiveTermination tracker = new AdaptiveTermination();
        tracker.reset(K);

        seedHealthyDiscovery(tracker, K);

        boolean terminated = observeUntilTerminated(
                tracker,
                0,
                1,
                true,
                200
        );

        assertTrue(terminated);
    }

    @Test
    void terminatesWhenDryDiscoveryIsReportedAcrossMultipleSteps() {
        AdaptiveTermination tracker = new AdaptiveTermination();
        tracker.reset(K);

        seedHealthyDiscovery(tracker, K);

        boolean terminated = observeUntilTerminated(
                tracker,
                0,
                5,
                true,
                200
        );

        assertTrue(terminated);
    }

    @Test
    void resetClearsPriorTerminationState() {
        AdaptiveTermination tracker = new AdaptiveTermination();
        tracker.reset(K);

        seedHealthyDiscovery(tracker, K);
        assertTrue(observeUntilTerminated(tracker, 0, 1, true, 200));

        tracker.reset(K);

        assertFalse(tracker.shouldTerminate());
        tracker.observe(0, 1, true);
        assertFalse(tracker.shouldTerminate());
    }

    @Test
    void terminationIsRepeatableAfterReset() {
        AdaptiveTermination tracker = new AdaptiveTermination();
        tracker.reset(K);

        seedHealthyDiscovery(tracker, K);
        assertTrue(observeUntilTerminated(tracker, 0, 1, true, 200));

        tracker.reset(K);

        seedHealthyDiscovery(tracker, K);
        assertTrue(observeUntilTerminated(tracker, 0, 1, true, 200));
    }

    @Test
    void observeGuardsInvalidInputs() {
        AdaptiveTermination tracker = new AdaptiveTermination();
        tracker.reset(K);

        assertDoesNotThrow(() -> {
            for (int i = 0; i < 100; i++) {
                tracker.observe(-10, 0, true);
            }
        });
    }



    @Test
    void lowVarianceRequiresMorePatienceThanHighVariance() {
        AdaptiveTermination stableTracker = new AdaptiveTermination();
        stableTracker.reset(K);
        for (int i = 0; i < 100; i++) {
            stableTracker.observe(5, 1, true);
        }

        AdaptiveTermination chaoticTracker = new AdaptiveTermination();
        chaoticTracker.reset(K);
        for (int i = 0; i < 100; i++) {
            chaoticTracker.observe(i % 2 == 0 ? K : 0, 1, true);
        }

        // Force a non-saturated visit to synchronize state
        stableTracker.observe(K, 1, true);
        chaoticTracker.observe(K, 1, true);

        int stableSteps = 0;
        while (!stableTracker.shouldTerminate() && stableSteps < 50) {
            stableTracker.observe(0, 1, true);
            stableSteps++;
        }

        int chaoticSteps = 0;
        while (!chaoticTracker.shouldTerminate() && chaoticSteps < 50) {
            chaoticTracker.observe(0, 1, true);
            chaoticSteps++;
        }

        assertTrue(stableTracker.shouldTerminate());
        assertTrue(chaoticTracker.shouldTerminate());
        assertTrue(chaoticSteps < stableSteps,
                "High variance should reduce patience, terminating faster than a stable history");
    }

    private static void seedHealthyDiscovery(AdaptiveTermination tracker, int k) {
        for (int i = 0; i < 20; i++) {
            tracker.observe(k, 1, true);
            assertFalse(tracker.shouldTerminate());
        }
    }


    private static boolean observeUntilTerminated(
            AdaptiveTermination tracker,
            int introduced,
            int steps,
            boolean resultSetFull,
            int maxObservations
    ) {
        for (int i = 0; i < maxObservations; i++) {
            tracker.observe(introduced, steps, resultSetFull);
            if (tracker.shouldTerminate()) {
                return true;
            }
        }
        return false;
    }
}
