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

import java.util.Arrays;
import java.util.List;
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
    public static double recallFromSearchResults(List<? extends List<Integer>> gt, List<SearchResult> retrieved, int kGT, int kRetrieved) {
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
        if (kGT > gt.size()) {
            throw new IllegalArgumentException("kGT: " + kGT + " > Gt size: " + gt.size());
        }
        if (kRetrieved > retrieved.size()) {
            throw new IllegalArgumentException("kRetrieved: " + kRetrieved + " > retrieved size: " + retrieved.size());
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

    /**
     * Compute the average precision at k.
     * See the definition <a href="https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision">here</a>.
     * @param gt the ground truth
     * @param retrieved the retrieved elements
     * @param k the number of retrieved elements
     * @return the average precision
     */
    public static double averagePrecisionAtK(List<Integer> gt, SearchResult retrieved, int k) {
        var retrievedTemp = Arrays.stream(retrieved.getNodes()).mapToInt(nodeScore -> nodeScore.node)
                .boxed()
                .collect(Collectors.toList());

        if (k > gt.size()) {
            throw new IllegalArgumentException("k: " + k + " > Gt size: " + gt.size());
        }
        if (k > retrievedTemp.size()) {
            throw new IllegalArgumentException("k: " + k + " > retrieved size: " + retrievedTemp.size());
        }

        var gtView = crop(gt, k);
        var retrievedView = crop(retrievedTemp, k);

        double score = 0.;
        int num_hits = 0;
        int i = 0;

        for (var p : retrievedView) {
            if (gtView.contains(p) && !retrievedView.subList(0, i).contains(p)) {
                num_hits += 1;
                score += num_hits / (i + 1.0);
            }
            i++;
        }

        return score / gtView.size();
    }

    public static double meanAveragePrecisionAtK(List<? extends List<Integer>> gt, List<SearchResult> retrieved, int k) {
        if (gt.size() != retrieved.size()) {
            throw new IllegalArgumentException("We should have ground truth for each result");
        }
        Double apk = IntStream.range(0, gt.size())
                .mapToObj(i -> averagePrecisionAtK(gt.get(i), retrieved.get(i), k))
                .reduce(0., Double::sum);
        return apk / gt.size();
    }

}
