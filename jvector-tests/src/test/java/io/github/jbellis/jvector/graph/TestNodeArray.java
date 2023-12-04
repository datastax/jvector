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

import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestNodeArray extends RandomizedTest {
  @Test
  public void testScoresDescOrder() {
    NodeArray neighbors = new NodeArray(10);
    neighbors.addInOrder(0, 1);
    neighbors.addInOrder(1, 0.8f);

    AssertionError ex = assertThrows(AssertionError.class, () -> neighbors.addInOrder(2, 0.9f));
    assert ex.getMessage().startsWith("Nodes are added in the incorrect order!") : ex.getMessage();

    neighbors.insertSorted(3, 0.9f);
    assertScoresEqual(new float[] {1, 0.9f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {0, 3, 1}, neighbors);

    neighbors.insertSorted(4, 1f);
    assertScoresEqual(new float[] {1, 1, 0.9f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {0, 4, 3, 1}, neighbors);

    neighbors.insertSorted(5, 1.1f);
    assertScoresEqual(new float[] {1.1f, 1, 1, 0.9f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {5, 0, 4, 3, 1}, neighbors);

    neighbors.insertSorted(6, 0.8f);
    assertScoresEqual(new float[] {1.1f, 1, 1, 0.9f, 0.8f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {5, 0, 4, 3, 1, 6}, neighbors);

    neighbors.insertSorted(7, 0.8f);
    assertScoresEqual(new float[] {1.1f, 1, 1, 0.9f, 0.8f, 0.8f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {5, 0, 4, 3, 1, 6, 7}, neighbors);

    neighbors.removeIndex(2);
    assertScoresEqual(new float[] {1.1f, 1, 0.9f, 0.8f, 0.8f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {5, 0, 3, 1, 6, 7}, neighbors);

    neighbors.removeIndex(0);
    assertScoresEqual(new float[] {1, 0.9f, 0.8f, 0.8f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {0, 3, 1, 6, 7}, neighbors);

    neighbors.removeIndex(4);
    assertScoresEqual(new float[] {1, 0.9f, 0.8f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {0, 3, 1, 6}, neighbors);

    neighbors.removeLast();
    assertScoresEqual(new float[] {1, 0.9f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {0, 3, 1}, neighbors);

    neighbors.insertSorted(8, 0.9f);
    assertScoresEqual(new float[] {1, 0.9f, 0.9f, 0.8f}, neighbors);
    assertNodesEqual(new int[] {0, 3, 8, 1}, neighbors);
  }

  private void assertScoresEqual(float[] scores, NodeArray neighbors) {
    for (int i = 0; i < scores.length; i++) {
      assertEquals(scores[i], neighbors.score[i], 0.01f);
    }
  }

  private void assertNodesEqual(int[] nodes, NodeArray neighbors) {
    for (int i = 0; i < nodes.length; i++) {
      assertEquals(nodes[i], neighbors.node[i]);
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
}
