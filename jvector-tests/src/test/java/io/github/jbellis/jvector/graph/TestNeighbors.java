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

}
