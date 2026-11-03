package io.github.jbellis.jvector.fastervector;

import java.util.List;

/**
 * Result of a nearest-neighbour search.
 *
 * @param results one list per query (in query order); each list holds that
 *                query's {@code topk} neighbours in rank order
 */
public record SearchResult(List<List<NodeScore>> results) {}
