/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.util.FixedBitSet;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashSet;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestNodeArray extends RandomizedTest {
  static void validateSortedByNode(NodeArray na) {
    for (int i = 0; i < na.size() - 1; i++) {
      assertTrue(na.getNode(i) < na.getNode(i + 1));
    }
  }

  @Test
  public void testScoresDescOrder() {
    NodeArray neighbors = new NodeArray(10);
    neighbors.addInOrder(0, 1);
    neighbors.addInOrder(1, 0.8f);

    AssertionError ex = assertThrows(AssertionError.class, () -> neighbors.addInOrder(-1, 0.9f));
    assert ex.getMessage().startsWith("Nodes are added in the incorrect order!") : ex.getMessage();

    neighbors.insertSorted(3, 0.9f);
    assertScoresEqual(new float[] {1, 0.9f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {0, 1, 3}, neighbors);

    neighbors.insertSorted(4, 1f);
    assertScoresEqual(new float[] {1, 1, 0.9f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {0, 1, 3, 4}, neighbors);

    neighbors.insertSorted(5, 1.1f);
    assertScoresEqual(new float[] {1.1f, 1, 1, 0.9f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {0, 1, 3, 4, 5}, neighbors);

    neighbors.insertSorted(6, 0.8f);
    assertScoresEqual(new float[] {1.1f, 1, 1, 0.9f, 0.8f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {0, 1, 3, 4, 5, 6}, neighbors);

    neighbors.insertSorted(7, 0.8f);
    assertScoresEqual(new float[] {1.1f, 1, 1, 0.9f, 0.8f, 0.8f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {0, 1, 3, 4, 5, 6, 7}, neighbors);

    neighbors.removeIndex(2);
    assertScoresEqual(new float[] {1.1f, 1, 1, 0.8f, 0.8f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {0, 1, 4, 5, 6, 7}, neighbors);

    neighbors.removeIndex(0);
    NodesIterator it = neighbors.getIteratorSortedByScores();
    assertScoresEqual(new float[] {1.1f, 1, 0.8f, 0.8f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {1, 4, 5, 6, 7}, neighbors);

    neighbors.removeIndex(4);
    assertScoresEqual(new float[] {1.1f, 1, 0.8f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {1, 4, 5, 6}, neighbors);

    neighbors.removeLast();
    assertScoresEqual(new float[] {1.1f, 1, 0.8f}, neighbors);
    assertNodesEqual(new int[] {1, 4, 5}, neighbors);

    neighbors.insertSorted(8, 0.9f);
    assertScoresEqual(new float[] {1.1f, 1, 0.9f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {1, 4, 5, 8}, neighbors);
  }

  private void assertScoresEqual(float[] scores, NodeArray neighbors) {
    NodesIterator it = neighbors.getIteratorSortedByScores();
    int i = 0;
    while (it.hasNext()) {
      assertEquals(scores[i++], neighbors.getScore(it.nextInt()), 0.01f);
    }
  }

  private void assertNodesEqual(int[] nodes, NodeArray neighbors) {
    for (int i = 0; i < nodes.length; i++) {
      assertEquals(nodes[i], neighbors.getNode(i));
    }
  }

  @Test
  public void testRetainNoneSelected() {
    var array = new NodeArray(10);
    for (int i = 1; i <= 10; i++) {
      array.addInOrder(i, 11 - i);
    }
    var selected = new FixedBitSet(10); // All bits are false by default
    array.retain(selected);
    Assert.assertEquals(0, array.size());
  }

  @Test
  public void testRetainAllSelected() {
    var array = new NodeArray(10);
    for (int i = 1; i <= 10; i++) {
      array.addInOrder(i, 11 - i);
    }
    var selected = new FixedBitSet(10);
    selected.set(0, 10); // Set all bits to true
    array.retain(selected);
    Assert.assertEquals(10, array.size());
  }

  @Test
  public void testRetainSomeSelectedNotFront() {
    var array = new NodeArray(10);
    for (int i = 1; i <= 10; i++) {
      array.addInOrder(i, 11 - i);
    }
    var selected = new FixedBitSet(10);
    selected.set(5, 10); // Select last 5 elements
    array.retain(selected);
    Assert.assertEquals(5, array.size());
    for (int i = 0; i < array.size(); i++) {
      assertTrue(selected.get(i + 5));
    }
  }

  @Test
  public void testRetainSomeSelectedAtFront() {
    var array = new NodeArray(10);
    for (int i = 1; i <= 10; i++) {
      array.addInOrder(i, 11 - i);
    }
    var selected = new FixedBitSet(10);
    selected.set(0, 5); // Select first 5 elements
    array.retain(selected);
    Assert.assertEquals(5, array.size());
    for (int i = 0; i < array.size(); i++) {
      assertTrue(selected.get(i));
    }
  }

  @Test
  public void testNoDuplicatesDescOrder() {
    NodeArray cna = new NodeArray(5);
    cna.insertSorted(1, 10.0f);
    cna.insertSorted(2, 9.0f);
    cna.insertSorted(3, 8.0f);
    cna.insertSorted(1, 10.0f); // This is a duplicate and should be ignored
    cna.insertSorted(3, 8.0f); // This is also a duplicate
    Assert.assertArrayEquals(new int[] {1, 2, 3}, cna.copyDenseNodes());
    assertArrayEquals(new float[] {10.0f, 9.0f, 8.0f}, cna.copyDenseScores(), 0.01f);
    validateSortedByNode(cna);
  }

  @Test
  public void testNoDuplicatesSameScores() {
    NodeArray cna = new NodeArray(5);
    cna.insertSorted(1, 10.0f);
    cna.insertSorted(2, 10.0f);
    cna.insertSorted(3, 10.0f);
    cna.insertSorted(1, 10.0f); // This is a duplicate and should be ignored
    cna.insertSorted(3, 10.0f); // This is also a duplicate
    assertArrayEquals(new int[] {1, 2, 3}, cna.copyDenseNodes());
    assertArrayEquals(new float[] {10.0f, 10.0f, 10.0f}, cna.copyDenseScores(), 0.01f);
    validateSortedByNode(cna);
  }

  @Test
  public void testMergeCandidatesSimple() {
    var arr1 = new NodeArray(1);
    arr1.addInOrder(1, 1.0f);

    var arr2 = new NodeArray(1);
    arr2.addInOrder(0, 2.0f);

    var merged = NodeArray.merge(arr1, arr2);
    // Expected result: [0, 1]
    assertArrayEquals(new int[] {0, 1}, merged.copyDenseNodes());

    arr1 = new NodeArray(3);
    arr1.addInOrder(1, 1.0f);
    arr1.addInOrder(2, 2.0f);
    arr1.addInOrder(3, 3.0f);

    arr2 = new NodeArray(3);
    arr2.addInOrder(1, 1.05f);
    arr2.addInOrder(2, 2.0f);
    arr2.addInOrder(4, 4.0f);

    merged = NodeArray.merge(arr1, arr2);
    assertArrayEquals(new int[] {1, 2, 3, 4}, merged.copyDenseNodes());
    assertScoresEqual(new float[] {4.0f, 3.0f, 2.0f, 1.0f}, merged);

    // Testing boundary conditions
    arr1 = new NodeArray(2);
    arr1.addInOrder(2, 2.0f);
    arr1.addInOrder(3, 3.0f);

    arr2 = new NodeArray(1);
    arr2.addInOrder(2, 2.0f);

    merged = NodeArray.merge(arr1, arr2);
    // Expected result: [3, 2]
    assertArrayEquals(new int[] {2, 3}, merged.copyDenseNodes());
    assertScoresEqual(new float[] {3.0f, 2.0f}, merged);
    validateSortedByNode(merged);
  }

  private void testMergeCandidatesOnce() {
    // test merge emphasizing dealing with tied scores
    int maxSize = 1 + getRandom().nextInt(5);

    // fill arr1 with nodes from 0..size, with random scores assigned (so random order of nodes)
    NodeArray arr1 = new NodeArray(maxSize);
    int a1Size = getRandom().nextBoolean() ? maxSize : 1 + getRandom().nextInt(maxSize);
    for (int i = 0; i < a1Size; i++) {
      arr1.insertSorted(i, getRandom().nextFloat());
    }

    // arr2 contains either
    // -- an exact duplicates of the corresponding node in arr1, or
    // -- a random score chosen from arr1
    // this is designed to maximize the need for correct handling of corner cases in the merge
    NodeArray arr2 = new NodeArray(maxSize);
    int a2Size = getRandom().nextBoolean() ? maxSize : 1 + getRandom().nextInt(maxSize);
    for (int i = 0; i < a2Size; i++) {
      if (i < a1Size && getRandom().nextBoolean()) {
        // duplicate entry
        int j = getRandom().nextInt(a1Size);
        if (!arr2.contains(arr1.getNode(j))) {
          arr2.insertSorted(arr1.getNode(j), arr1.getScore(j));
        }
      } else {
        // duplicate just score
        float score;
        if (getRandom().nextBoolean()) {
          score = getRandom().nextFloat();
        } else {
          score = arr1.getScore(getRandom().nextInt(a1Size));
        }
        arr2.insertSorted(i + arr1.size(), score);
      }
    }

    // merge!
    var merged = NodeArray.merge(arr1, arr2);

    // sanity check
    assert merged.size() <= arr1.size() + arr2.size();
    assert merged.size() >= Math.max(arr1.size(), arr2.size());
    var uniqueNodes = new HashSet<>();

    // results should be sorted by score, and not contain duplicates
    for (int i = 0; i < merged.size() - 1; i++) {
      assertTrue(merged.getNode(i) < merged.getNode(i + 1));
      assertTrue(uniqueNodes.add(merged.getNode(i)));
    }
    assertTrue(uniqueNodes.add(merged.getNode(merged.size() - 1)));

    // results should contain all the nodes that were in the source arrays
    for (int i = 0; i < arr1.size(); i++) {
      assertTrue(String.format("%s missing%na1: %s%na2: %s%nmerged: %s%n",
                               arr1.getNode(i), arr1, arr2, merged),
                 uniqueNodes.contains(arr1.getNode(i)));
    }
    for (int i = 0; i < arr2.size(); i++) {
      assertTrue(String.format("%s missing%na1: %s%na2: %s%nmerged: %s%n",
                               arr2.getNode(i), arr1, arr2, merged),
                 uniqueNodes.contains(arr2.getNode(i)));
    }
  }

  @Test
  public void testMergeCandidatesRandom() {
    for (int i = 0; i < 10000; i++) {
      testMergeCandidatesOnce();
    }
  }
}

