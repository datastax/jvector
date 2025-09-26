package io.github.jbellis.jvector.status;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.Callable;

public class SimulatedTask implements Callable<String> {

    public final String name;
    public final int count;
    public final Duration stepTime;
    private final StringBuilder buffer;

    public SimulatedTask(String name, StringBuilder buffer, int count, Duration stepTime) {
        this.name = name;
        this.count = count;
        this.stepTime = stepTime;
        this.buffer = buffer;
    }

    @Override
    public String call() throws Exception {
        int divisor = count / 100;
        var startAt = Instant.now();

        for (int micro = 0; micro < count; micro+=divisor) {
            int end = Math.min(micro + divisor, count);
            buffer.append(name).append(" micro:").append(micro).append(" count:").append(count).append("\n");
            for (int i = micro; i < end; i++) {
                Thread.sleep(stepTime.toMillis());
            }
        }
        var endAt = Instant.now();
        String summary= name + " completed in " + Duration.between(startAt, endAt).toString() + " \n";
        return buffer.toString();

    }
}
