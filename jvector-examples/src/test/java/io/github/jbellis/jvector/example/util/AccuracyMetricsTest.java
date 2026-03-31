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
package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.SearchResult;

import java.util.List;

/**
 * Simple test class to verify the AccuracyMetrics functionality.
 */
public class AccuracyMetricsTest {
    public static void main(String[] args) {
        System.out.println("Running AccuracyMetrics tests...");

        testRecallPerfect();
        testRecallPartial();
        testRecallAveragesAcrossQueries();

        testAveragePrecisionPerfect();
        testAveragePrecisionPartial();
        testAveragePrecisionNoHits();
        testAveragePrecisionRespectsRankingOrder();

        testMeanAveragePrecision();

        testRecallThrowsOnMismatchedQueryCounts();
        testMeanAveragePrecisionThrowsOnMismatchedQueryCounts();

        testRecallThrowsWhenKgtExceedsKRetrieved();
        testRecallThrowsWhenKgtExceedsGtSize();
        testRecallThrowsWhenKRetrievedExceedsRetrievedSize();

        testAveragePrecisionThrowsWhenKExceedsGtSize();
        testAveragePrecisionThrowsWhenKExceedsRetrievedSize();

        testRecallThrowsOnDuplicateGroundTruthOrdinal();
        testRecallThrowsOnDuplicateRetrievedOrdinal();
        testAveragePrecisionThrowsOnDuplicateGroundTruthOrdinal();
        testAveragePrecisionThrowsOnDuplicateRetrievedOrdinal();

        System.out.println("All AccuracyMetrics tests completed successfully!");
    }

    private static void testRecallPerfect() {
        System.out.println("\nTest: Recall perfect");

        double recall = AccuracyMetrics.recallFromSearchResults(
                List.of(List.of(1, 2, 3)),
                List.of(result(1, 2, 3)),
                3,
                3
        );

        assertEquals("Recall", 1.0, recall, 0.0000001);
    }

    private static void testRecallPartial() {
        System.out.println("\nTest: Recall partial");

        double recall = AccuracyMetrics.recallFromSearchResults(
                List.of(List.of(1, 2, 3)),
                List.of(result(1, 4, 5)),
                3,
                3
        );

        assertEquals("Recall", 1.0 / 3.0, recall, 0.0000001);
    }

    private static void testRecallAveragesAcrossQueries() {
        System.out.println("\nTest: Recall averages across queries");

        double recall = AccuracyMetrics.recallFromSearchResults(
                List.of(
                        List.of(1, 2, 3),
                        List.of(10, 20, 30)
                ),
                List.of(
                        result(1, 2, 3),
                        result(10, 99, 98)
                ),
                3,
                3
        );

        assertEquals("Recall", 2.0 / 3.0, recall, 0.0000001);
    }

    private static void testAveragePrecisionPerfect() {
        System.out.println("\nTest: Average precision perfect");

        double ap = AccuracyMetrics.averagePrecisionAtK(
                List.of(1, 2, 3),
                result(1, 2, 3),
                3
        );

        assertEquals("Average precision", 1.0, ap, 0.0000001);
    }

    private static void testAveragePrecisionPartial() {
        System.out.println("\nTest: Average precision partial");

        double ap = AccuracyMetrics.averagePrecisionAtK(
                List.of(1, 2, 3),
                result(1, 4, 2),
                3
        );

        // Relevant hits at ranks 1 and 3:
        // P@1 = 1/1
        // P@3 = 2/3
        // AP@3 = (1 + 2/3) / 3 = 5/9
        assertEquals("Average precision", 5.0 / 9.0, ap, 0.0000001);
    }

    private static void testAveragePrecisionNoHits() {
        System.out.println("\nTest: Average precision no hits");

        double ap = AccuracyMetrics.averagePrecisionAtK(
                List.of(1, 2, 3),
                result(4, 5, 6),
                3
        );

        assertEquals("Average precision", 0.0, ap, 0.0000001);
    }

    private static void testAveragePrecisionRespectsRankingOrder() {
        System.out.println("\nTest: Average precision respects ranking order");

        double better = AccuracyMetrics.averagePrecisionAtK(
                List.of(1, 2, 3),
                result(1, 2, 9),
                3
        );

        double worse = AccuracyMetrics.averagePrecisionAtK(
                List.of(1, 2, 3),
                result(9, 1, 2),
                3
        );

        assertEquals("Better-ranked AP", 2.0 / 3.0, better, 0.0000001);
        assertEquals("Worse-ranked AP", (1.0 / 2.0 + 2.0 / 3.0) / 3.0, worse, 0.0000001);
    }

    private static void testMeanAveragePrecision() {
        System.out.println("\nTest: Mean average precision");

        double map = AccuracyMetrics.meanAveragePrecisionAtK(
                List.of(
                        List.of(1, 2, 3),
                        List.of(10, 20, 30)
                ),
                List.of(
                        result(1, 2, 3),
                        result(99, 98, 97)
                ),
                3
        );

        assertEquals("Mean average precision", 0.5, map, 0.0000001);
    }

    private static void testRecallThrowsOnMismatchedQueryCounts() {
        System.out.println("\nTest: Recall throws on mismatched query counts");

        assertThrows(
                "Insufficient ground truth for the number of retrieved elements",
                () -> AccuracyMetrics.recallFromSearchResults(
                        List.of(List.of(1, 2, 3)),
                        List.of(result(1, 2, 3), result(4, 5, 6)),
                        3,
                        3
                )
        );
    }

    private static void testMeanAveragePrecisionThrowsOnMismatchedQueryCounts() {
        System.out.println("\nTest: MAP throws on mismatched query counts");

        assertThrows(
                "Insufficient ground truth for the number of retrieved elements",
                () -> AccuracyMetrics.meanAveragePrecisionAtK(
                        List.of(List.of(1, 2, 3)),
                        List.of(result(1, 2, 3), result(4, 5, 6)),
                        3
                )
        );
    }

    private static void testRecallThrowsWhenKgtExceedsKRetrieved() {
        System.out.println("\nTest: Recall throws when kGT exceeds kRetrieved");

        assertThrows(
                "kGT: 3 > kRetrieved: 2",
                () -> AccuracyMetrics.recallFromSearchResults(
                        List.of(List.of(1, 2, 3)),
                        List.of(result(1, 2)),
                        3,
                        2
                )
        );
    }

    private static void testRecallThrowsWhenKgtExceedsGtSize() {
        System.out.println("\nTest: Recall throws when kGT exceeds GT size");

        assertThrows(
                "kGT: 3 > Gt size: 2",
                () -> AccuracyMetrics.recallFromSearchResults(
                        List.of(List.of(1, 2)),
                        List.of(result(1, 2, 3)),
                        3,
                        3
                )
        );
    }

    private static void testRecallThrowsWhenKRetrievedExceedsRetrievedSize() {
        System.out.println("\nTest: Recall throws when kRetrieved exceeds retrieved size");

        assertThrows(
                "kRetrieved: 3 > retrieved size: 2",
                () -> AccuracyMetrics.recallFromSearchResults(
                        List.of(List.of(1, 2, 3)),
                        List.of(result(1, 2)),
                        2,
                        3
                )
        );
    }

    private static void testAveragePrecisionThrowsWhenKExceedsGtSize() {
        System.out.println("\nTest: AP throws when k exceeds GT size");

        assertThrows(
                "k: 3 > Gt size: 2",
                () -> AccuracyMetrics.averagePrecisionAtK(
                        List.of(1, 2),
                        result(1, 2, 3),
                        3
                )
        );
    }

    private static void testAveragePrecisionThrowsWhenKExceedsRetrievedSize() {
        System.out.println("\nTest: AP throws when k exceeds retrieved size");

        assertThrows(
                "k: 3 > retrieved size: 2",
                () -> AccuracyMetrics.averagePrecisionAtK(
                        List.of(1, 2, 3),
                        result(1, 2),
                        3
                )
        );
    }

    private static void testRecallThrowsOnDuplicateGroundTruthOrdinal() {
        System.out.println("\nTest: Recall throws on duplicate ground truth ordinal");

        assertThrows(
                "Duplicate ground truth ordinal in top-3: 1",
                () -> AccuracyMetrics.recallFromSearchResults(
                        List.of(List.of(1, 1, 2)),
                        List.of(result(1, 2, 3)),
                        3,
                        3
                )
        );
    }

    private static void testRecallThrowsOnDuplicateRetrievedOrdinal() {
        System.out.println("\nTest: Recall throws on duplicate retrieved ordinal");

        assertThrows(
                "Duplicate retrieved ordinal in top-3: 1",
                () -> AccuracyMetrics.recallFromSearchResults(
                        List.of(List.of(1, 2, 3)),
                        List.of(result(1, 1, 2)),
                        3,
                        3
                )
        );
    }

    private static void testAveragePrecisionThrowsOnDuplicateGroundTruthOrdinal() {
        System.out.println("\nTest: AP throws on duplicate ground truth ordinal");

        assertThrows(
                "Duplicate ground truth ordinal in top-3: 1",
                () -> AccuracyMetrics.averagePrecisionAtK(
                        List.of(1, 1, 2),
                        result(1, 2, 3),
                        3
                )
        );
    }

    private static void testAveragePrecisionThrowsOnDuplicateRetrievedOrdinal() {
        System.out.println("\nTest: AP throws on duplicate retrieved ordinal");

        assertThrows(
                "Duplicate retrieved ordinal in top-3: 1",
                () -> AccuracyMetrics.averagePrecisionAtK(
                        List.of(1, 2, 3),
                        result(1, 1, 2),
                        3
                )
        );
    }

    private static SearchResult result(int... nodes) {
        SearchResult.NodeScore[] nodeScores = new SearchResult.NodeScore[nodes.length];
        for (int i = 0; i < nodes.length; i++) {
            nodeScores[i] = new SearchResult.NodeScore(nodes[i], 0.0f);
        }
        return new SearchResult(nodeScores, 0, 0, 0, 0, Float.POSITIVE_INFINITY);
    }

    private static void assertEquals(String message, double expected, double actual, double delta) {
        if (Math.abs(expected - actual) > delta) {
            throw new AssertionError(message + " - Expected: " + expected + ", Actual: " + actual);
        }
        System.out.println("✓ " + message + " - Value: " + actual);
    }

    private static void assertThrows(String expectedMessage, Runnable runnable) {
        try {
            runnable.run();
            throw new AssertionError("Expected exception with message: " + expectedMessage);
        } catch (IllegalArgumentException e) {
            if (!expectedMessage.equals(e.getMessage())) {
                throw new AssertionError(
                        "Expected exception message: " + expectedMessage + ", Actual: " + e.getMessage()
                );
            }
            System.out.println("✓ Threw expected exception - Message: " + e.getMessage());
        }
    }
}
