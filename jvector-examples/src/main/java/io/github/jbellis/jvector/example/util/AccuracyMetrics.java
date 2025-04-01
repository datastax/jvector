package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.SearchResult;

import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class AccuracyMetrics {
    /**
     * Compute kGT-recall@kRetrieved, which is the fraction of
     * the kGT ground-truth nearest neighbors that are in the kRetrieved
     * first search results (with kGT ≤ kRetrieved)
     * @param gt the ground truth
     * @param retrieved the retrieved elements
     * @param kGT the number of considered ground truth elements
     * @param kRetrieved the number of retrieved elements
     * @return the recall
     */
    public static double recallFromSearchResults(List<List<Integer>> gt, List<SearchResult> retrieved, int kGT, int kRetrieved) {
        if (gt.size() != retrieved.size()) {
            throw new IllegalArgumentException("We should have ground truth for each result");
        }
        Long correctCount = IntStream.range(0, gt.size())
                .mapToObj(i -> topKCorrect(gt.get(i), retrieved.get(i), kGT, kRetrieved))
                .reduce(0L, Long::sum);
        return (double) correctCount / gt.size();
    }

    /**
     * Compute kGT-recall@kRetrieved, which is the fraction of
     * the kGT ground-truth nearest neighbors that are in the kRetrieved
     * first search results (with kGT ≤ kRetrieved)
     * @param gt the ground truth
     * @param retrieved the retrieved elements
     * @param kGT the number of considered ground truth elements
     * @param kRetrieved the number of retrieved elements
     * @return the recall
     */
    public static double recall(List<List<Integer>> gt, List<List<Integer>> retrieved, int kGT, int kRetrieved) {
        if (gt.size() != retrieved.size()) {
            throw new IllegalArgumentException("We should have ground truth for each result");
        }
        Long correctCount = IntStream.range(0, gt.size())
                .mapToObj(i -> topKCorrect(gt.get(i), retrieved.get(i), kGT, kRetrieved))
                .reduce(0L, Long::sum);
        return (double) correctCount / (kGT * gt.size());
    }

    private static long topKCorrect(List<Integer> gt, List<Integer> retrieved, int kGT, int kRetrieved) {
        if (kGT > kRetrieved) {
            throw new IllegalArgumentException("kGT: " + kGT + " > kRetrieved: " + kRetrieved);
        }
        var gtView = crop(gt, kGT);
        var retrievedView = crop(retrieved, kRetrieved);

        if (gtView.size() > retrieved.size()) {
            return gtView.stream().filter(retrievedView::contains).count();
        } else {
            return retrievedView.stream().filter(gtView::contains).count();
        }
    }

    public static long topKCorrect(List<Integer> gt, SearchResult retrieved, int kGT, int kRetrieved) {
        var temp = Arrays.stream(retrieved.getNodes()).mapToInt(nodeScore -> nodeScore.node)
                .boxed()
                .collect(Collectors.toList());
        return topKCorrect(gt, temp, kGT, kRetrieved);
    }

    private static List<Integer> crop(List<Integer> list, int k) {
        int count = Math.min(list.size(), k);
        return list.subList(0, count);
    }
}
