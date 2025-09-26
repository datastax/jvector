package io.github.jbellis.jvector.status;

import org.junit.Test;
import static org.junit.Assert.*;

public class SimpleStatusUpdateTest {

    @Test
    public void testTaskStatusCreation() {
        StatusUpdate<String> status = new StatusUpdate<>(0.5, StatusUpdate.RunState.RUNNING);
        assertEquals(0.5, status.progress, 0.001);
        assertEquals(StatusUpdate.RunState.RUNNING, status.runstate);
        assertTrue(status.timestamp > 0);
    }
}