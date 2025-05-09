package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NativeVectorUtilSupportTest {

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
  /// ```
  /// Then the vector API will not be detected anyway.
  ///
  /// The purpose of this test is to provide clear status for diagnosing test coverage, thus it only
  /// emits impl selection details without actually testing anything. The output is formatted to
  /// be easily findable in a build log.
  @Test
  void testVectorTypeSupportIsNative() {
    VectorizationProvider provider = VectorizationProvider.getInstance();
    if (provider instanceof NativeVectorizationProvider) {
      System.out.println("PROVIDER: NativeVectorizationProvider detected");
    } else {
      System.out.println("PROVIDER: NativeVectorizationProvider not detected: using " + provider.getClass().getSimpleName());
    }

    boolean readable = VectorizationProvider.vectorModulePresentAndReadable();
    System.out.println("VECTOR MODULE READABLE: " + readable);
  }


}