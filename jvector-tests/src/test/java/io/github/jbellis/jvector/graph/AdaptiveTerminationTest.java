package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.util.BoundedLongHeap;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class AdaptiveTerminationTest {
    private static NodeQueue results(float... scores) {
        NodeQueue q = new NodeQueue(new BoundedLongHeap(scores.length), NodeQueue.Order.MIN_HEAP);
        for (int i = 0; i < scores.length; i++) q.push(i, scores[i]);
        return q;
    }

    @Test void requiresKResults() {
        var termination = new AdaptiveTermination();
        termination.reset(2, 0.1f);
        assertEquals(AdaptiveTermination.Decision.CONTINUE,
                termination.shouldTerminate(0.1f, results(0.9f)));
    }

    @Test void usesDotProductDistanceMargin() {
        var termination = new AdaptiveTermination();
        termination.reset(1, 0.1f);
        NodeQueue q = results(0.9f);
        assertEquals(AdaptiveTermination.Decision.CONTINUE, termination.shouldTerminate(0.89f, q));
        assertEquals(AdaptiveTermination.Decision.TERMINATE, termination.shouldTerminate(0.88f, q));
    }

    @Test void invalidApproximateScoresFallBack() {
        var termination = new AdaptiveTermination();
        termination.reset(1);
        assertEquals(AdaptiveTermination.Decision.FALL_BACK_TO_STANDARD,
                termination.shouldTerminate(1.1f, results(0.9f)));
        assertEquals(AdaptiveTermination.Decision.FALL_BACK_TO_STANDARD,
                termination.shouldTerminate(0.1f, results(0.9f)));
    }

    @Test void resetClearsFallbackState() {
        var termination = new AdaptiveTermination();
        termination.reset(1);
        termination.shouldTerminate(Float.NaN, results(0.9f));
        termination.reset(1, 0.1f);
        assertEquals(AdaptiveTermination.Decision.TERMINATE,
                termination.shouldTerminate(0.88f, results(0.9f)));
    }
    @Test void rawDotProductBuildScoresUseTheSameDistanceRule() {
        var t = new AdaptiveTermination();
        t.reset(1, 0.1f, true);
        assertEquals(AdaptiveTermination.Decision.CONTINUE, t.shouldTerminate(0.78f, results(0.8f)));
        assertEquals(AdaptiveTermination.Decision.TERMINATE, t.shouldTerminate(0.76f, results(0.8f)));
        assertEquals(AdaptiveTermination.Decision.FALL_BACK_TO_STANDARD, t.shouldTerminate(-1.1f, results(0.8f)));
    }

}
