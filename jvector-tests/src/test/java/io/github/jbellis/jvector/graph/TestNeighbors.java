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

package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Test;

import java.util.stream.IntStream;

import static io.github.jbellis.jvector.graph.TestNodeArray.validateSortedByScore;
import static org.junit.Assert.assertEquals;

public class TestNeighbors extends RandomizedTest {

  @Test
  public void testInsertDiverse() {
    // set up BSP
    var similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
    var vectors = new TestVectorGraph.CircularFloatVectorValues(10);
    var candidates = new NodeArray(10);
    var bsp = BuildScoreProvider.randomAccessScoreProvider(vectors, similarityFunction);
    // fill candidates with all the nodes except 7
    IntStream.range(0, 10)
        .filter(i -> i != 7)
        .forEach(i -> candidates.insertSorted(i, scoreBetween(bsp, 7, i)));
    assert candidates.size() == 9;

    // only nodes 6 and 8 are diverse wrt 7
    var cnm = new ConcurrentNeighborMap(new VamanaDiversityProvider(bsp, 1.0f), 10, 10);
    cnm.addNode(7);
    var neighbors = cnm.insertDiverse(7, candidates);
    assertEquals(2, neighbors.size());
    assert neighbors.contains(8);
    assert neighbors.contains(6);
    validateSortedByScore(neighbors);
  }

  private static float scoreBetween(BuildScoreProvider bsp, int i, int j) {
    return bsp.searchProviderFor(i).exactScoreFunction().similarityTo(j);
  }

  @Test
  public void testInsertDiverseConcurrent() {
    // set up BSP
    var sf = VectorSimilarityFunction.DOT_PRODUCT;
    var vectors = new TestVectorGraph.CircularFloatVectorValues(10);
    var natural = new NodeArray(10);
    var concurrent = new NodeArray(10);
    var bsp = BuildScoreProvider.randomAccessScoreProvider(vectors, sf);
    // "natural" candidates are [0..7), "concurrent" are [8..10)
    IntStream.range(0, 7)
        .forEach(i -> natural.insertSorted(i, scoreBetween(bsp, 7, i)));
    IntStream.range(8, 10)
        .forEach(
            i -> concurrent.insertSorted(i, scoreBetween(bsp, 7, i)));

    // only nodes 6 and 8 are diverse wrt 7
    var cnm = new ConcurrentNeighborMap(new VamanaDiversityProvider(bsp, 1.0f), 10, 10);
    cnm.addNode(7);
    var neighbors = cnm.insertDiverse(7, NodeArray.merge(natural, concurrent));
    assertEquals(2, neighbors.size());
    assert neighbors.contains(8);
    assert neighbors.contains(6);
    validateSortedByScore(neighbors);
  }

  @Test
  public void testInsertDiverseRetainsNatural() {
    // set up BSP
    var vectors = new TestVectorGraph.CircularFloatVectorValues(10);
    var similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
    var bsp = BuildScoreProvider.randomAccessScoreProvider(vectors, similarityFunction);

    // check that the new neighbor doesn't replace the existing one (since both are diverse, and the max degree accommodates both)
    var cna = new NodeArray(1);
    cna.addInOrder(6, scoreBetween(bsp, 7, 6));

    var cna2 = new NodeArray(1);
    cna2.addInOrder(8, scoreBetween(bsp, 7, 8));

    var cnm = new ConcurrentNeighborMap(new VamanaDiversityProvider(bsp, 1.0f), 10, 10);
    cnm.addNode(7, cna);
    var neighbors = cnm.insertDiverse(7, cna2);
    assertEquals(2, neighbors.size());
  }


  @Test
  public void testConcurrentDuplicateOffersWithDifferentScores() throws Exception {
    var bsp=BuildScoreProvider.randomAccessScoreProvider(new TestVectorGraph.CircularFloatVectorValues(32),VectorSimilarityFunction.DOT_PRODUCT);
    var map=new ConcurrentNeighborMap(new VamanaDiversityProvider(bsp,1.2f),64,80);
    map.addNode(0);
    var pool=new java.util.concurrent.ForkJoinPool(8);
    try {
      pool.submit(() -> IntStream.range(0,2000).parallel().forEach(i -> map.insertEdge(0,1+i%20,i,1.2f))).join();
    } finally {
      pool.shutdown();
      org.junit.Assert.assertTrue(pool.awaitTermination(10,java.util.concurrent.TimeUnit.SECONDS));
    }
    assertEquals(20,map.get(0).size());
    var ids=new java.util.HashSet<Integer>();
    for(int i=0;i<map.get(0).size();i++) org.junit.Assert.assertTrue(ids.add(map.get(0).getNode(i)));
    validateSortedByScore(map.get(0));
  }

  @Test
  public void testInitialCandidateBatchIsUnique() {
    var bsp=BuildScoreProvider.randomAccessScoreProvider(new TestVectorGraph.CircularFloatVectorValues(32),VectorSimilarityFunction.DOT_PRODUCT);
    var map=new ConcurrentNeighborMap(new VamanaDiversityProvider(bsp,1.2f),64,80);
    map.addNode(0);
    var candidates=new NodeArray(4);
    candidates.addInOrder(1,10f); candidates.addInOrder(2,9f);
    candidates.addInOrder(1,8f); candidates.addInOrder(2,7f);
    var result=map.insertDiverse(0,candidates);
    assertEquals(2,result.size());
    org.junit.Assert.assertNotEquals(result.getNode(0),result.getNode(1));
  }

  @Test
  public void testDuplicateOffersDoNotTriggerPruning() {
    for (int degree : new int[] {2, 3, 32, 64, 128}) {
      int hardMax = degree + degree / 2;
      var prunes = new java.util.concurrent.atomic.AtomicInteger();
      io.github.jbellis.jvector.graph.diversity.DiversityProvider diversity =
          (neighbors, maxDegree, diverseBefore, selected) -> {
            prunes.incrementAndGet();
            for (int i = 0; i < Math.min(maxDegree, neighbors.size()); i++) selected.set(i);
            return 1.0;
          };
      var map = new ConcurrentNeighborMap(diversity, degree, hardMax);
      map.addNode(0);
      for (int count = 1; count <= hardMax; count++) {
        map.insertEdge(0, count, -count, 1.5f);
        var before = map.get(0);
        assertEquals(count, before.size());
        for (int id = 1; id <= count; id++) {
          for (float score : new float[] {-id, 1000f, -1000f}) {
            // Also reject without pruning when the requested limit is below current size.
            for (float overflow : new float[] {1.0f, 1.5f}) {
              map.insertEdge(0, id, score, overflow);
              org.junit.Assert.assertSame(before, map.get(0));
              assertEquals(0, prunes.get());
            }
          }
        }
        for (int i = 0; i < count; i++) {
          assertEquals(i + 1, before.getNode(i));
          assertEquals(-(i + 1), before.getScore(i), 0f);
        }
      }

      // A distinct insertion beyond the limit must still prune normally.
      map.insertEdge(0, hardMax + 1, -(hardMax + 1), 1.5f);
      assertEquals(1, prunes.get());
      assertEquals(degree, map.get(0).size());
      validateSortedByScore(map.get(0));

      // Refill to the boundary: rejected duplicates must not prevent explicit cleanup.
      for (int id = degree + 1; id <= hardMax; id++) {
        map.insertEdge(0, id, -id, 1.5f);
      }
      var before = map.get(0);
      map.insertEdge(0, 1, 1000f, 1.5f);
      org.junit.Assert.assertSame(before, map.get(0));
      assertEquals(1, prunes.get());
      map.enforceDegree(0);
      assertEquals(2, prunes.get());
      assertEquals(degree, map.get(0).size());
      for (int i = 0; i < degree; i++) {
        assertEquals(i + 1, map.get(0).getNode(i));
        assertEquals(-(i + 1), map.get(0).getScore(i), 0f);
      }
    }
  }
}
