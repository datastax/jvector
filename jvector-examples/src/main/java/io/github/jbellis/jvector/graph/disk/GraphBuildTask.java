package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.status.eventing.RunState;
import io.github.jbellis.jvector.status.eventing.StatusSource;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;

/**
 * Wrapper for GraphIndexBuilder that provides progress monitoring.
 */
public class GraphBuildTask implements StatusSource<GraphBuildTask> {
    private final GraphIndexBuilder builder;
    private final RandomAccessVectorValues vectors;
    private final String name;
    private volatile RunState state = RunState.PENDING;
    private volatile ImmutableGraphIndex result = null;
    private final long totalNodes;

    public GraphBuildTask(String name, GraphIndexBuilder builder, RandomAccessVectorValues vectors) {
        this.name = name;
        this.builder = builder;
        this.vectors = vectors;
        this.totalNodes = vectors.size();
    }

    public ImmutableGraphIndex execute() {
        try {
            state = RunState.RUNNING;
            result = builder.build(vectors);
            state = RunState.SUCCESS;
            return result;
        } catch (Exception e) {
            state = RunState.FAILED;
            throw e;
        }
    }

    @Override
    public StatusUpdate<GraphBuildTask> getTaskStatus() {
        // Estimate progress based on graph size
        long currentSize = (result != null) ? result.size(0) : builder.getGraph().size(0);

        double progress = totalNodes > 0 ? (double) currentSize / totalNodes : 0.0;

        return new StatusUpdate<>(
                progress,
                state,
                this
        );
    }

    public String getName() {
        return name;
    }
}
