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

import java.util.concurrent.atomic.AtomicLong;

/**
 * A simulated clock that can run faster or slower than real time, useful for
 * demonstrations, testing, and controlled time-dependent behavior. The clock
 * maintains virtual time that advances based on a configurable speed multiplier.
 *
 * <p>The clock runs in a dedicated background thread that continuously advances
 * virtual time based on elapsed real time and the current speed multiplier.
 * This allows tasks to sleep in "virtual time" while actual wall-clock time
 * passes more quickly or slowly.
 *
 * <h2>Speed Control</h2>
 * <p>The clock supports preset speed multipliers ranging from 0.0001x (extremely slow)
 * to 50x (very fast). Common presets include:</p>
 * <ul>
 *   <li>0.0001x, 0.001x, 0.01x, 0.1x - Slower than real time</li>
 *   <li>1.0x - Real-time speed (default)</li>
 *   <li>2.0x, 5.0x, 10.0x, 20.0x, 50.0x - Faster than real time</li>
 * </ul>
 *
 * <p>Speed can be adjusted dynamically using {@link #speedUp()} and {@link #slowDown()},
 * which cycle through the preset values. Current speed can be queried via
 * {@link #getSpeedMultiplier()}, {@link #getSpeedDescription()}, or
 * {@link #getSpeedIndicator()}.</p>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * try (SimulatedClock clock = new SimulatedClock()) {
 *     // Start at normal speed (1.0x)
 *     long startTime = clock.currentTimeMillis();
 *
 *     // Speed up simulation
 *     clock.speedUp();  // Now at 2.0x
 *     clock.speedUp();  // Now at 5.0x
 *
 *     // Sleep for 1000ms virtual time (takes ~200ms real time at 5.0x)
 *     clock.sleep(1000);
 *
 *     long elapsed = clock.currentTimeMillis() - startTime;
 *     System.out.println("Virtual time elapsed: " + elapsed + "ms");
 * }
 * }</pre>
 *
 * <h2>Integration with ConsolePanelSink</h2>
 * <p>This clock is commonly used with {@link io.github.jbellis.jvector.status.sinks.ConsolePanelSink}
 * custom keyboard handlers to provide interactive time control in demonstrations:</p>
 * <pre>{@code
 * SimulatedClock clock = new SimulatedClock();
 * ConsolePanelSink sink = ConsolePanelSink.builder()
 *     .withKeyHandler("shift-right", () -> {
 *         clock.speedUp();
 *         sink.addLogMessage("Speed: " + clock.getSpeedDescription());
 *     })
 *     .withKeyHandler("shift-left", () -> {
 *         clock.slowDown();
 *         sink.addLogMessage("Speed: " + clock.getSpeedDescription());
 *     })
 *     .build();
 * }</pre>
 *
 * <h2>Thread Safety</h2>
 * <p>This class is thread-safe. The {@link #sleep(long)} method can be called
 * from multiple threads concurrently, and speed adjustments are safely visible
 * across all threads. The background clock thread is marked as a daemon and will
 * not prevent JVM shutdown.</p>
 *
 * @see AutoCloseable
 * @since 4.0.0
 */
public class SimulatedClock implements AutoCloseable {
    private static final double[] SPEED_PRESETS = {0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0};
    private static final int DEFAULT_SPEED_INDEX = 6; // 1.0x

    private final AtomicLong virtualTimeMillis;
    private final long realStartTime;
    private volatile double speedMultiplier;
    private volatile int speedIndex;
    private volatile boolean running;
    private final Thread clockThread;

    /**
     * Creates a new simulated clock starting at normal speed (1.0x).
     * The clock immediately begins advancing virtual time in a background thread.
     * Virtual time starts at 0ms when the clock is created.
     */
    public SimulatedClock() {
        this.realStartTime = System.currentTimeMillis();
        this.virtualTimeMillis = new AtomicLong(0);
        this.speedIndex = DEFAULT_SPEED_INDEX;
        this.speedMultiplier = SPEED_PRESETS[speedIndex];
        this.running = true;

        this.clockThread = new Thread(this::runClock, "SimulatedClock");
        this.clockThread.setDaemon(true);
        this.clockThread.start();
    }

    private void runClock() {
        long lastUpdateTime = System.currentTimeMillis();

        while (running) {
            try {
                Thread.sleep(10); // Update every 10ms

                long now = System.currentTimeMillis();
                long realElapsed = now - lastUpdateTime;
                long virtualElapsed = (long) (realElapsed * speedMultiplier);

                virtualTimeMillis.addAndGet(virtualElapsed);
                lastUpdateTime = now;

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    /**
     * Returns the current virtual time in milliseconds since clock creation.
     * Virtual time starts at 0 and advances based on the current speed multiplier.
     *
     * @return the current virtual time in milliseconds
     */
    public long currentTimeMillis() {
        return virtualTimeMillis.get();
    }

    /**
     * Sleeps for the specified virtual duration. The actual real-world sleep time
     * is adjusted based on the current speed multiplier. This method is responsive
     * to speed changes during the sleep by checking the speed multiplier periodically.
     *
     * <p>For example, if the speed is 5.0x, sleeping for 1000ms virtual time will
     * take approximately 200ms of real time. If the speed changes during the sleep,
     * the remaining sleep duration will be adjusted accordingly.</p>
     *
     * <p>If {@code virtualMillis} is 0 or negative, this method returns immediately
     * without sleeping.</p>
     *
     * @param virtualMillis the virtual time to sleep in milliseconds
     * @throws InterruptedException if the current thread is interrupted while sleeping
     */
    public void sleep(long virtualMillis) throws InterruptedException {
        if (virtualMillis <= 0) {
            return;
        }

        long targetVirtualTime = virtualTimeMillis.get() + virtualMillis;

        // Sleep in small chunks to be responsive to speed changes
        // Use 100ms real-time chunks for responsiveness
        while (virtualTimeMillis.get() < targetVirtualTime) {
            long remaining = targetVirtualTime - virtualTimeMillis.get();
            long realSleepMillis = (long) Math.min(100, remaining / speedMultiplier);

            if (realSleepMillis > 0) {
                Thread.sleep(realSleepMillis);
            } else {
                // Speed is very fast, just yield to let clock thread advance
                Thread.yield();
                // Add a tiny sleep to prevent busy-waiting
                Thread.sleep(1);
            }
        }
    }

    /**
     * Increases the clock speed to the next preset value.
     * If already at the maximum speed (50.0x), this method has no effect.
     * Speed changes take effect immediately and apply to all ongoing sleep operations.
     */
    public void speedUp() {
        if (speedIndex < SPEED_PRESETS.length - 1) {
            speedIndex++;
            speedMultiplier = SPEED_PRESETS[speedIndex];
        }
    }

    /**
     * Decreases the clock speed to the previous preset value.
     * If already at the minimum speed (0.0001x), this method has no effect.
     * Speed changes take effect immediately and apply to all ongoing sleep operations.
     */
    public void slowDown() {
        if (speedIndex > 0) {
            speedIndex--;
            speedMultiplier = SPEED_PRESETS[speedIndex];
        }
    }

    /**
     * Returns the current speed multiplier as a numeric value.
     * For example, 1.0 means real-time, 2.0 means twice as fast, 0.5 means half speed.
     *
     * @return the current speed multiplier
     */
    public double getSpeedMultiplier() {
        return speedMultiplier;
    }

    /**
     * Returns a human-readable description of the current speed.
     * Examples: "x1 (realtime)", "x10 (faster)", "x1/100 (slower)".
     *
     * @return a formatted string describing the current speed setting
     */
    public String getSpeedDescription() {
        if (speedMultiplier == 1.0) {
            return "x1 (realtime)";
        } else if (speedMultiplier >= 1.0) {
            // Faster: x2, x5, x10, x20, x50
            int speedInt = (int) speedMultiplier;
            if (speedInt == speedMultiplier) {
                return "x" + speedInt + " (faster)";
            } else {
                return String.format("x%.1f (faster)", speedMultiplier);
            }
        } else {
            // Slower: x1/10, x1/100, x1/1000, x1/10000
            double inverse = 1.0 / speedMultiplier;
            int inverseInt = (int) Math.round(inverse);
            return "x1/" + inverseInt + " (slower)";
        }
    }

    /**
     * Returns a compact speed indicator suitable for inline status display.
     * Examples: "x1", "x10", "x1/10", "x1/100", "x1/10000".
     * This is a shorter version of {@link #getSpeedDescription()} without the descriptive suffix.
     *
     * @return a compact string indicating the current speed
     */
    public String getSpeedIndicator() {
        if (speedMultiplier == 1.0) {
            return "x1";
        } else if (speedMultiplier >= 1.0) {
            // Faster: x2, x5, x10, x20, x50
            int speedInt = (int) speedMultiplier;
            if (speedInt == speedMultiplier) {
                return "x" + speedInt;
            } else {
                return String.format("x%.1f", speedMultiplier);
            }
        } else {
            // Slower: x1/10, x1/100, x1/1000, x1/10000
            double inverse = 1.0 / speedMultiplier;
            int inverseInt = (int) Math.round(inverse);
            return "x1/" + inverseInt;
        }
    }

    /**
     * Stops the clock's background thread and releases resources.
     * This method blocks for up to 1 second waiting for the background thread
     * to terminate. After calling close, the clock should not be used further.
     */
    @Override
    public void close() {
        running = false;
        clockThread.interrupt();
        try {
            clockThread.join(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
