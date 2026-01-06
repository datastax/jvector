package io.github.jbellis.jvector.graph;

public abstract class GraphIndexPrinter {
    public static <G extends ImmutableGraphIndex & Viewable> String  prettyPrint(G graph) {
        StringBuilder sb = new StringBuilder();
        sb.append(graph);
        sb.append("\n");

        try (var view = graph.getView()) {
            for (int level = 0; level <= graph.getMaxLevel(); level++) {
                sb.append(String.format("# Level %d\n", level));
                NodesIterator it = graph.getNodes(level);
                while (it.hasNext()) {
                    int node = it.nextInt();
                    sb.append("  ").append(node).append(" -> ");
                    for (var neighbors = view.getNeighborsIterator(level, node); neighbors.hasNext(); ) {
                        sb.append(" ").append(neighbors.nextInt());
                    }
                    sb.append("\n");
                }
                sb.append("\n");
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return sb.toString();
    }
}
