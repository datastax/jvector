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

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.util.ArrayUtil;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.IntHashSet;

import java.util.Arrays;

import static java.lang.Math.min;

/**
 * NodeArray encodes nodeids and their scores relative to some other element 
 * (a query vector, or another graph node) as a pair of growable arrays. 
 * Nodes are arranged in the sorted order of their scores in descending order,
 * i.e. the most-similar nodes are first.
 */
public class NodeScoreArray {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private VectorFloat<?> scores;
    private int[] nodes;

    public NodeScoreArray(int size) {
        this.nodes = new int[size];
        this.scores = vectorTypeSupport.createFloatVector(size);
    }

    /**
     * Add a new node to the NodeArray into the specified insertion point.
     */
    void insertAt(int insertionPoint, int newNode, float newScore) {
        this.nodes[insertionPoint] = newNode;
        this.scores.set(insertionPoint, newScore);
    }

    public int size() {
        return nodes.length;
    }

    public void clear() {
        this.scores.zero();
    }

    public float getScore(int i) {
        return scores.get(i);
    }

    public int getNode(int i) {
        return nodes[i];
    }

    public void setScore(int i, float score) {
        scores.set(i, score);
    }

    public void setNode(int i, int node) {
        nodes[i] = node;
    }

    public VectorFloat<?> getScores() {
        return scores;
    }
}
