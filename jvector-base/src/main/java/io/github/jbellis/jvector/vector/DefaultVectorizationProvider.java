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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/** Default provider returning scalar implementations. */
final public class DefaultVectorizationProvider extends VectorizationProvider {

  private final VectorUtilSupport vectorUtilSupport;
  private final VectorTypeSupport vectorTypes;

  /**
   * Constructor.
   */
  public DefaultVectorizationProvider() {
    vectorUtilSupport = new DefaultVectorUtilSupport();
    vectorTypes = new ArrayVectorProvider();
  }

  @Override
  public VectorUtilSupport getVectorUtilSupport() {
    return vectorUtilSupport;
  }

  @Override
  public VectorTypeSupport getVectorTypeSupport() {
    return vectorTypes;
  }
}
