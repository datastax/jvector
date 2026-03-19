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
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Computes accuracy metrics, such as recall and mean average precision.
 */
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
            throw new IllegalArgumentException("Insufficient ground truth for the number of retrieved elements");
        }

        long correctCount = 0;
        for (int i = 0; i < gt.size(); i++) {
            correctCount += topKCorrect(gt.get(i), retrieved.get(i), kGT, kRetrieved);
        }

        return (double) correctCount / (kGT * gt.size());
    }

    private static long topKCorrect(List<Integer> gt, SearchResult retrieved, int kGT, int kRetrieved) {
        // Exception validation
        var nodes = retrieved.getNodes();
        if (kGT > kRetrieved) {
            throw new IllegalArgumentException("kGT: " + kGT + " > kRetrieved: " + kRetrieved);
        }
        if (kGT > gt.size()) {
            throw new IllegalArgumentException("kGT: " + kGT + " > Gt size: " + gt.size());
        }
        if (kRetrieved > nodes.length) {
            throw new IllegalArgumentException("kRetrieved: " + kRetrieved + " > retrieved size: " + nodes.length);
        }

        // Build HashSet with explicit capacity to avoid rehashing.
        // Load factor is 0.75, so sized to kGT / 0.75.
        Set<Integer> gtSet = new HashSet<>((int) (kGT / 0.75f) + 1);
        for (int i = 0; i < kGT; i++) {
            gtSet.add(gt.get(i));
        }

        // Manual primitive loop for speed (no Stream setup).
        int hits = 0;
        for (int i = 0; i < kRetrieved; i++) {
            if (gtSet.contains(nodes[i].node)) {
                hits++;
            }
        }

        return hits;
    }

    /**
     * Computes the average precision at k.
     * See the definition <a href="https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision">here</a>.
     * @param gt the ground truth
     * @param retrieved the retrieved elements
     * @param k the number of retrieved elements
     * @return the average precision
     */
    public static double averagePrecisionAtK(List<Integer> gt, SearchResult retrieved, int k) {
        var nodes = retrieved.getNodes();
        if (k > gt.size()) {
            throw new IllegalArgumentException("k: " + k + " > Gt size: " + gt.size());
        }
        if (k > nodes.length) {
            throw new IllegalArgumentException("k: " + k + " > retrieved size: " + nodes.length);
        }

        // Sized hashset used for performance.
        Set<Integer> gtSet = new HashSet<>((int) (k / 0.75f) + 1);
        for (int i = 0; i < k; i++) {
            gtSet.add(gt.get(i));
        }

        // Handles potential duplicates in O(1).
        Set<Integer> seen = new HashSet<>((int) (k / 0.75f) + 1);

        double score = 0.;
        int hits = 0;
        for (int i = 0; i < k; i++) {
            int p = nodes[i].node;
            if (gtSet.contains(p) && seen.add(p)) {
                hits++;
                score += (double) hits / (i + 1);
            }
        }

        return score / k;
    }

    /**
     * Computes the mean average precision at k.
     * See the definition <a href="https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision">here</a>.
     * @param gt the ground truth
     * @param retrieved the retrieved elements
     * @param k the number of retrieved elements
     * @return the mean average precision
     */
    public static double meanAveragePrecisionAtK(List<? extends List<Integer>> gt, List<SearchResult> retrieved, int k) {
        if (gt.size() != retrieved.size()) {
            throw new IllegalArgumentException("Insufficient ground truth for the number of retrieved elements");
        }

        double totalAp = 0;
        for (int i = 0; i < gt.size(); i++) {
            totalAp += averagePrecisionAtK(gt.get(i), retrieved.get(i), k);
        }

        return totalAp / gt.size();
    }
}
