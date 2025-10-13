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

import io.github.jbellis.jvector.status.eventing.StatusUpdate;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Background polling engine used by {@link StatusContext} to periodically observe tracked objects.
 * Each context owns exactly one monitor, which runs a single daemon thread that continuously
 * polls registered {@link StatusTracker}s at their configured intervals.
 *
 * <p>Key Responsibilities:
 * <ul>
 *   <li><strong>Polling Loop:</strong> Maintains a single daemon thread that polls all registered trackers</li>
 *   <li><strong>Interval Management:</strong> Respects per-tracker poll intervals to balance responsiveness and overhead</li>
 *   <li><strong>Status Observation:</strong> Calls {@link StatusTracker#refreshAndGetStatus()} on trackers
 *       to observe their tracked objects</li>
 *   <li><strong>Event Routing:</strong> Forwards observed status to {@link StatusContext#pushStatus} for
 *       distribution to sinks</li>
 *   <li><strong>Cleanup:</strong> Automatically removes closed trackers from the polling loop</li>
 * </ul>
 *
 * <p>Architectural Flow:
 * <ol>
 *   <li>Monitor thread wakes up and checks all registered trackers</li>
 *   <li>For each tracker whose poll interval has elapsed:
 *     <ul>
 *       <li>Monitor calls {@link StatusTracker#refreshAndGetStatus()}</li>
 *       <li>Tracker observes its object and caches the status</li>
 *       <li>Monitor receives the observed status</li>
 *       <li>Monitor forwards status to {@link StatusContext#pushStatus}</li>
 *       <li>Context routes status to all registered sinks</li>
 *     </ul>
 *   </li>
 *   <li>Monitor calculates next wake time based on shortest remaining interval</li>
 *   <li>Monitor sleeps until next wake time</li>
 * </ol>
 *
 * <p>This class is package-private and should only be instantiated by {@link StatusContext}.
 *
 * @see StatusContext
 * @see StatusTracker
 * @see StatusTracker#refreshAndGetStatus()
 * @since 4.0.0
 */
final class StatusMonitor implements AutoCloseable {

    private static final long MIN_SLEEP_MILLIS = 10;

    private final StatusContext context;
    private final ConcurrentHashMap<StatusTracker<?>, MonitoredEntry<?>> entries = new ConcurrentHashMap<>();
    private final AtomicBoolean running = new AtomicBoolean(true);
    private final Thread monitorThread;

    StatusMonitor(StatusContext context) {
        this.context = context;
        this.monitorThread = new Thread(this::runLoop, "StatusMonitor");
        this.monitorThread.setDaemon(true);
        this.monitorThread.start();
    }

    /**
     * Registers a tracker for periodic polling. The initial status is immediately
     * pushed to the context to notify sinks of the new tracker.
     *
     * @param tracker the tracker to monitor
     * @param pollInterval the interval between status observations
     * @param initialStatus the initial status to push to sinks
     * @param <T> the type of object being tracked
     */
    <T> void register(StatusTracker<T> tracker,
                      Duration pollInterval,
                      StatusUpdate<T> initialStatus) {
        entries.put(tracker, new MonitoredEntry<>(tracker, pollInterval));
        context.pushStatus(tracker, initialStatus);
    }

    /**
     * Unregisters a tracker from polling. After this call, the monitor will no
     * longer observe the tracker or forward its status updates.
     *
     * @param tracker the tracker to unregister
     */
    void unregister(StatusTracker<?> tracker) {
        entries.remove(tracker);
    }

    /**
     * Main polling loop that runs continuously until the monitor is closed.
     * For each registered tracker, checks if its poll interval has elapsed and
     * if so, calls {@link #pollTracker} to observe and forward status.
     * <p>
     * The loop automatically removes closed trackers and calculates optimal
     * sleep times based on the shortest remaining poll interval.
     */
    private void runLoop() {
        while (running.get()) {
            long now = System.currentTimeMillis();
            long nextWake = now + MIN_SLEEP_MILLIS;

            for (Map.Entry<StatusTracker<?>, MonitoredEntry<?>> mapEntry : entries.entrySet()) {
                StatusTracker<?> tracker = mapEntry.getKey();
                MonitoredEntry<?> entry = mapEntry.getValue();

                if (tracker.isClosed()) {
                    entries.remove(tracker);
                    continue;
                }

                if (now >= entry.nextPollMillis) {
                    pollTracker(entry);
                    entry.nextPollMillis = now + entry.intervalMillis;
                }

                nextWake = Math.min(nextWake, entry.nextPollMillis);
            }

            long sleepMillis = Math.max(MIN_SLEEP_MILLIS, nextWake - System.currentTimeMillis());
            try {
                TimeUnit.MILLISECONDS.sleep(sleepMillis);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    /**
     * Polls a single tracker by calling its {@link StatusTracker#refreshAndGetStatus()} method
     * and forwarding the result to the context. Errors during polling are caught and logged
     * to prevent one failing tracker from affecting others.
     *
     * @param entry the monitored entry containing the tracker to poll
     * @param <T> the type of object being tracked
     */
    @SuppressWarnings("unchecked")
    private <T> void pollTracker(MonitoredEntry<T> entry) {
        StatusTracker<T> tracker = entry.tracker;
        try {
            // Tracker observes its own tracked object and caches the result
            StatusUpdate<T> status = tracker.refreshAndGetStatus();
            // Context routes the observed status to sinks
            context.pushStatus(tracker, status);
        } catch (Throwable t) {
            System.err.println("Error polling status for tracker: " + t.getMessage());
        }
    }

    /**
     * Closes the monitor, stopping the polling thread and clearing all registered trackers.
     * This method is idempotent and safe to call multiple times.
     */
    @Override
    public void close() {
        if (running.compareAndSet(true, false)) {
            monitorThread.interrupt();
            try {
                monitorThread.join(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            entries.clear();
        }
    }

    private static final class MonitoredEntry<T> {
        final StatusTracker<T> tracker;
        final long intervalMillis;
        volatile long nextPollMillis;

        MonitoredEntry(StatusTracker<T> tracker, Duration pollInterval) {
            this.tracker = tracker;
            this.intervalMillis = Math.max(pollInterval.toMillis(), MIN_SLEEP_MILLIS);
            this.nextPollMillis = System.currentTimeMillis();
        }
    }
}
