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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PanamaVectorUtilSupportTest {

  public static final String REQUIRE_SPECIFIC_VECTORIZATION_PROVIDER =
      "Test_RequireSpecificVectorizationProvider";

  /// The only time this test should be run is when it is expected to be within a JVM which
  /// supports native access, whether it is on hardware that natively supports the Java vector
  /// API or not.
  ///
  /// If it is being run without
  /// ```
  /// -ea
  /// --add-modules jdk.incubator.vector
  /// --enable-native-access=ALL-UNNAMED
  /// -Djvector.experimental.enable_native_vectorization=true
  ///```
  /// Then the vector API will not be detected anyway.
  ///
  /// The purpose of this test is to provide clear status for diagnosing test coverage, thus it only
  /// emits impl selection details without actually testing anything. The output is formatted to
  /// be easily findable in a build log.
  ///
  /// If `-DTest_RequireSpecificVectorizationProvider=<simple name>` is provided, then the test
  /// will fail if the detected implementation doesn't match the {@link Class#getSimpleName()}
  ///  value.
  @Test
  void testVectorSupportTypeisPanema() {
    VectorizationProvider provider = VectorizationProvider.getInstance();
    System.out.println("PROVIDER: using " + provider.getClass().getSimpleName());

    boolean readable = VectorizationProvider.vectorModulePresentAndReadable();
    System.out.println("VECTOR MODULE READABLE: " + readable);

    String requiredProvider = System.getProperty(REQUIRE_SPECIFIC_VECTORIZATION_PROVIDER);
    if (requiredProvider != null) {
      System.out.println("REQUIRED PROVIDER: " + requiredProvider);
      assertEquals(
          requiredProvider,
          provider.getClass().getSimpleName(),
          "Provider mismatch, " + "required " + requiredProvider + ", detected "
          + provider.getClass().getSimpleName()
      );
    }
  }


}