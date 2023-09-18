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

/**
 * Provides random access to vectors by dense ordinal. This interface is used by graph-based
 * implementations of KNN search.
 */
public interface RandomAccessVectorValues<T> {

  /** Return the number of vector values */
  int size();

  /** Return the dimension of the returned vector values */
  int dimension();

  /**
   * Return the vector value indexed at the given ordinal.
   *
   * <p>For performance, implementations are free to re-use the same object across invocations.
   * That is, you will get back the same float[]
   * reference (for instance) for every requested ordinal. If you want to use those values across
   * calls, you should make a copy.
   *
   * @param targetOrd a valid ordinal, &ge; 0 and &lt; {@link #size()}.
   */
  T vectorValue(int targetOrd);

  /**
   * @return true iff the vector returned is shared.  A shared vector will
   * only be valid until the next call to vectorValue overwrites it.
   */
  boolean isValueShared();

  /**
   * Creates a new copy of this {@link RandomAccessVectorValues}. This is helpful when you need to
   * access different values at once, to avoid overwriting the underlying float vector returned by
   * a shared {@link RandomAccessVectorValues#vectorValue}.
   */
  RandomAccessVectorValues<T> copy();
}
