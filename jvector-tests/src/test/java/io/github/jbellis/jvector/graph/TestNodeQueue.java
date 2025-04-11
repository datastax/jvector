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
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.GrowableLongHeap;
import org.junit.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TestNodeQueue extends RandomizedTest {
  @Test
  public void testNeighborsProduct() {
    // make sure we have the sign correct
    NodeQueue nn = new NodeQueue(new BoundedLongHeap(2), NodeQueue.Order.MIN_HEAP);
    assertTrue(nn.push(2, 0.5f));
    assertTrue(nn.push(1, 0.2f));
    assertTrue(nn.push(3, 1f));
    assertEquals(0.5f, nn.topScore(), 0);
    nn.pop();
    assertEquals(1f, nn.topScore(), 0);
    nn.pop();
  }

  @Test
  public void testNeighborsMaxHeap() {
    NodeQueue nn = new NodeQueue(new BoundedLongHeap(2), NodeQueue.Order.MAX_HEAP);
    assertTrue(nn.push(2, 2));
    assertTrue(nn.push(1, 1));
    assertFalse(nn.push(3, 3));
    assertEquals(2f, nn.topScore(), 0);
    nn.pop();
    assertEquals(1f, nn.topScore(), 0);
  }

  @Test
  public void testTopMaxHeap() {
    NodeQueue nn = new NodeQueue(new GrowableLongHeap(2), NodeQueue.Order.MAX_HEAP);
    nn.push(1, 2);
    nn.push(2, 1);
    // lower scores are better; highest score on top
    assertEquals(2, nn.topScore(), 0);
    assertEquals(1, nn.topNode());
  }

  @Test
  public void testTopMinHeap() {
    NodeQueue nn = new NodeQueue(new GrowableLongHeap(2), NodeQueue.Order.MIN_HEAP);
    nn.push(1, 0.5f);
    nn.push(2, -0.5f);
    // higher scores are better; lowest score on top
    assertEquals(-0.5f, nn.topScore(), 0);
    assertEquals(2, nn.topNode());
  }

  @Test
  public void testClear() {
    NodeQueue nn = new NodeQueue(new GrowableLongHeap(2), NodeQueue.Order.MIN_HEAP);
    nn.push(1, 1.1f);
    nn.push(2, -2.2f);
    nn.clear();

    assertEquals(0, nn.size());
  }

  @Test
  public void testMaxSizeQueue() {
    NodeQueue nn = new NodeQueue(new BoundedLongHeap(2), NodeQueue.Order.MIN_HEAP);
    nn.push(1, 1);
    nn.push(2, 2);
    assertEquals(2, nn.size());
    assertEquals(1, nn.topNode());

    // BoundedLongHeap does not extend the queue
    nn.push(3, 3);
    assertEquals(2, nn.size());
    assertEquals(2, nn.topNode());
  }

  @Test
  public void testUnboundedQueue() {
    NodeQueue nn = new NodeQueue(new GrowableLongHeap(1), NodeQueue.Order.MAX_HEAP);
    float maxScore = -2;
    int maxNode = -1;
    for (int i = 0; i < 256; i++) {
      // initial size is 32
      float score = getRandom().nextFloat();
      if (score > maxScore) {
        maxScore = score;
        maxNode = i;
      }
      nn.push(i, score);
    }
    assertEquals(maxScore, nn.topScore(), 0);
    assertEquals(maxNode, nn.topNode());
  }

  @Test
  public void testPushManyMinHeap() {
    // Build a NodeQueue with a GrowableLongHeap, using MIN_HEAP order
    NodeQueue queue = new NodeQueue(new GrowableLongHeap(2), NodeQueue.Order.MIN_HEAP);

    // Let's prepare some node, score pairs
    int[] nodes = { 5, 1, 3, 2, 8 };
    float[] scores = { 2.2f, -1.0f, 0.5f, 2.1f, -0.9f };

    // We'll create a TestNodeScoreIterator with these arrays
    TestNodeScoreIterator it = new TestNodeScoreIterator(nodes, scores);

    // Bulk-add all pairs in one go
    queue.pushMany(it, nodes.length);

    // The queue should now contain 5 elements
    assertEquals(5, queue.size());

    // Because it's a MIN_HEAP, the top (root) should be the "smallest" score
    // We have scores: [2.2, -1.0, 0.5, 2.1, -0.9]
    // The minimum is -1.0. Let's see which node that corresponds to: node=1
    assertEquals(-1.0f, queue.topScore(), 0.000001);
    assertEquals(1, queue.topNode());
  }

  @Test
  public void testPushManyMinHeapEdgeCase() {
    // Build a NodeQueue with a GrowableLongHeap, using MIN_HEAP order
    NodeQueue queue = new NodeQueue(new GrowableLongHeap(2), NodeQueue.Order.MIN_HEAP);

    // Let's prepare some node, score pairs
    // We select 3 elements in this case because it was a missed edge case in the original code
    int[] nodes = { 5, 1, 3};
    float[] scores = { 2.2f, -1.0f, 0.5f};

    // We'll create a TestNodeScoreIterator with these arrays
    TestNodeScoreIterator it = new TestNodeScoreIterator(nodes, scores);

    // Bulk-add all pairs in one go
    queue.pushMany(it, nodes.length);

    // The queue should now contain 3 elements
    assertEquals(3, queue.size());
  }

  @Test
  public void testPushManyMaxHeap() {
    // Build a NodeQueue with a GrowableLongHeap, using MAX_HEAP order
    NodeQueue queue = new NodeQueue(new GrowableLongHeap(2), NodeQueue.Order.MAX_HEAP);

    // Let's prepare some node, score pairs
    int[] nodes = { 10, 20, 30, 40, 50 };
    float[] scores = { -2.5f, 1.0f, 0.0f, 1.5f, 3.0f };

    // We'll create a TestNodeScoreIterator with these arrays
    TestNodeScoreIterator it = new TestNodeScoreIterator(nodes, scores);

    // Bulk-add all pairs in one go
    queue.pushMany(it, nodes.length);

    // The queue should now contain 5 elements
    assertEquals(5, queue.size());

    // Because it's a MAX_HEAP, the top (root) should be the "largest" score
    // The largest among [-2.5, 1.0, 0.0, 1.5, 3.0] is 3.0 => node=50
    assertEquals(3.0f, queue.topScore(), 0.000001);
    assertEquals(50, queue.topNode());
  }

  @Test
  public void testPushManyBoundedHeapAtCapacity() {
    NodeQueue queue = new NodeQueue(new BoundedLongHeap(2), NodeQueue.Order.MAX_HEAP);
    queue.pushMany(new TestNodeScoreIterator(new int[] { 1, 2 }, new float[] { 1, 2 }), 2);
    assertEquals(2, queue.size());
    assertEquals(2, queue.topNode());
    assertEquals(2, queue.topScore(), 0.000001);
  }

  @Test
  public void testPushManyBoundedHeapExceedsCapacity() {
    assertThrows(IllegalArgumentException.class, () -> {
      NodeQueue queue = new NodeQueue(new BoundedLongHeap(2), NodeQueue.Order.MAX_HEAP);
      queue.pushMany(new TestNodeScoreIterator(new int[] { 1, 2, 3 }, new float[] { 1, 2, 3 }), 3);
    });
    NodeQueue queue = new NodeQueue(new BoundedLongHeap(2), NodeQueue.Order.MAX_HEAP);
    queue.push(1, 1);
    assertThrows(IllegalArgumentException.class, () -> {
      queue.pushMany(new TestNodeScoreIterator(new int[] { 1, 2 }, new float[] { 1, 2 }), 2);
    });
  }

  @Test
  public void testInvalidArguments() {
    assertThrows(IllegalArgumentException.class, () -> new NodeQueue(new GrowableLongHeap(0), NodeQueue.Order.MIN_HEAP));
  }

  @Test
  public void testToString() {
    assertEquals("Nodes[0]", new NodeQueue(new GrowableLongHeap(2), NodeQueue.Order.MIN_HEAP).toString());
  }

  @Test
  public void testPushManyPartialIterator() {
    // Test with MIN_HEAP
    NodeQueue minQueue = new NodeQueue(new GrowableLongHeap(2), NodeQueue.Order.MIN_HEAP);
    int[] nodes = { 1, 2, 3, 4, 5 };
    float[] scores = { 2.0f, 1.0f, 3.0f, 0.5f, 4.0f };
    TestNodeScoreIterator it = new TestNodeScoreIterator(nodes, scores);

    // Only add first 3 elements from a 5-element iterator
    minQueue.pushMany(it, 3);
    assertEquals(3, minQueue.size());
    assertEquals(1.0f, minQueue.topScore(), 0.000001); // Smallest among 2.0, 1.0, 3.0
    assertEquals(2, minQueue.topNode());
    assertTrue(it.hasNext()); // Iterator should still have more elements

    // Test with MAX_HEAP
    NodeQueue maxQueue = new NodeQueue(new GrowableLongHeap(2), NodeQueue.Order.MAX_HEAP);
    nodes = new int[]{ 10, 20, 30, 40, 50 };
    scores = new float[]{ 1.0f, 3.0f, 2.0f, 4.0f, 5.0f };
    it = new TestNodeScoreIterator(nodes, scores);

    // Only add first 2 elements from a 5-element iterator
    maxQueue.pushMany(it, 2);
    assertEquals(2, maxQueue.size());
    assertEquals(3.0f, maxQueue.topScore(), 0.000001); // Largest among 1.0, 3.0
    assertEquals(20, maxQueue.topNode());
    assertTrue(it.hasNext()); // Iterator should still have more elements
  }

  @Test
  public void testPushManyBoundedHeapPartial() {
    NodeQueue queue = new NodeQueue(new BoundedLongHeap(3), NodeQueue.Order.MAX_HEAP);
    int[] nodes = { 1, 2, 3, 4, 5 };
    float[] scores = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    TestNodeScoreIterator it = new TestNodeScoreIterator(nodes, scores);

    // Add 2 elements to a heap with capacity 3
    queue.pushMany(it, 2);
    assertEquals(2, queue.size());
    assertEquals(2.0f, queue.topScore(), 0.000001);
    assertEquals(2, queue.topNode());

    // Add 1 more element
    queue.push(3, 3.0f);
    assertEquals(3, queue.size());
    assertEquals(3.0f, queue.topScore(), 0.000001);
    assertEquals(3, queue.topNode());
  }

  /**
   * Simple iterator that yields a fixed array of (node, score) pairs
   * for testing the pushAll method.
   */
  static class TestNodeScoreIterator implements NodeQueue.NodeScoreIterator {
    private final int[] nodes;
    private final float[] scores;
    private int index = 0;

    TestNodeScoreIterator(int[] nodes, float[] scores) {
      assert nodes.length == scores.length;
      this.nodes = nodes;
      this.scores = scores;
    }

    @Override
    public boolean hasNext() {
      return index < nodes.length;
    }

    @Override
    public int pop() {
      return nodes[index++];
    }

    @Override
    public float topScore() {
      return scores[index];
    }
  }

}
