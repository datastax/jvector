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

import java.lang.Runtime.Version;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Locale;
import java.util.Objects;
import java.util.logging.Logger;


/**
 * A provider of vectorization implementations. Depending on the Java version and availability of
 * vectorization modules in the Java runtime this class provides optimized implementations (using
 * SIMD) of several algorithms used throughout JVector.
 */
public abstract class VectorizationProvider {

  /**
   * Returns the default instance of the provider matching vectorization possibilities of actual
   * runtime.
   */
  public static VectorizationProvider getInstance() {
    return Objects.requireNonNull(
        Holder.INSTANCE, "call to getInstance() from subclass of VectorizationProvider");
  }

  protected VectorizationProvider() {

  }

  /**
   * Returns a singleton (stateless) {@link VectorUtilSupport} to support SIMD usage in {@link
   * VectorUtil}.
   */
  public abstract VectorUtilSupport getVectorUtilSupport();

  /**
   * Returns a singleton (stateless) {@link VectorTypeSupport} which works with the corresponding {@link VectorUtilSupport}
   * implementation
   */
  public abstract VectorTypeSupport getVectorTypeSupport();

  // *** Lookup mechanism: ***

  protected static final Logger LOG = Logger.getLogger(VectorizationProvider.class.getName());

  /** The minimal version of Java that has the bugfix for JDK-8301190. */
  private static final Version VERSION_JDK8301190_FIXED = Version.parse("20.0.2");

  // visible for tests
  static VectorizationProvider lookup(boolean testMode) {
    final int runtimeVersion = Runtime.version().feature();
    if (runtimeVersion >= 20) {
      // is locale sane (only buggy in Java 20)
      if (isAffectedByJDK8301190()) {
        LOG.warning(
            "Java runtime is using a buggy default locale; Java vector incubator API can't be enabled: "
                + Locale.getDefault());
        return new DefaultVectorizationProvider();
      }
      // is the incubator module present and readable (JVM providers may to exclude them or it is
      // build with jlink)
      if (!vectorModulePresentAndReadable()) {
        LOG.warning(
            "Java vector incubator module is not readable. For optimal vector performance, pass '--add-modules jdk.incubator.vector' to enable Vector API.");
        return new DefaultVectorizationProvider();
      }
      if (!testMode && isClientVM()) {
        LOG.warning("C2 compiler is disabled; Java vector incubator API can't be enabled");
        return new DefaultVectorizationProvider();
      }

      // check for jvector.experimental.enable_native_vectorization
      // this is an experimental feature subject to change
      if (Boolean.getBoolean("jvector.experimental.enable_native_vectorization")) {
        try {
          Class<?> clazz = Class.forName("io.github.jbellis.jvector.vector.NativeVectorizationProvider");
          Constructor<?> ctor = clazz.getConstructor();
          Object instance = ctor.newInstance();
          var provider = (VectorizationProvider) instance;
          LOG.info("Native Vector API enabled. Using NativeVectorizationProvider.");
          return provider;
        } catch (UnsupportedOperationException uoe) {
          LOG.warning("Native vector API was not enabled. " + uoe.getMessage());
        } catch (ClassNotFoundException e) {
          LOG.warning("Java version does not support native vector API");
        } catch (InvocationTargetException e) {
          // unwrap, check for UnsupportedOperationException
          if (e.getCause() instanceof UnsupportedOperationException) {
            LOG.warning("Native vector API was not enabled. " + e.getCause().getMessage());
          } else {
            throw new AssertionError(e);
          }
        } catch (Throwable th) {
          throw new AssertionError(th);
        }
      }

      try {
        var provider = (VectorizationProvider) Class.forName("io.github.jbellis.jvector.vector.PanamaVectorizationProvider").getConstructor().newInstance();
        LOG.info("Java incubating Vector API enabled. Using PanamaVectorizationProvider.");
        return provider;
      } catch (UnsupportedOperationException uoe) {
        // not supported because preferred vector size too small or similar
        LOG.warning("Java vector API was not enabled. " + uoe.getMessage());
        return new DefaultVectorizationProvider();
      } catch (ClassNotFoundException e) {
        LOG.warning("Java version does not support vector API");
        return new DefaultVectorizationProvider();
      } catch (RuntimeException | Error e) {
        throw e;
      } catch (Throwable th) {
        throw new AssertionError(th);
      }
    } else {
      LOG.warning("You are running with Java 19 or earlier, which do not support the required incubating Vector API. Falling back to slower defaults.");
    }
    return new DefaultVectorizationProvider();
  }

  static boolean vectorModulePresentAndReadable() {
    var opt =
        ModuleLayer.boot().modules().stream()
            .filter(m -> m.getName().equals("jdk.incubator.vector"))
            .findFirst();
    if (opt.isPresent()) {
      VectorizationProvider.class.getModule().addReads(opt.get());
      return true;
    }
    return false;
  }

  /**
   * Check if runtime is affected by JDK-8301190 (avoids assertion when default language is say
   * "tr").
   */
  private static boolean isAffectedByJDK8301190() {
    return VERSION_JDK8301190_FIXED.compareToIgnoreOptional(Runtime.version()) > 0
        && !Objects.equals("I", "i".toUpperCase(Locale.getDefault()));
  }

  @SuppressWarnings("removal")
  private static boolean isClientVM() {
    try {
      final PrivilegedAction<Boolean> action =
          () -> System.getProperty("java.vm.info", "").contains("emulated-client");
      return AccessController.doPrivileged(action);
    } catch (
        @SuppressWarnings("unused")
        SecurityException e) {
      LOG.warning(
          "SecurityManager denies permission to 'java.vm.info' system property, so state of C2 compiler can't be detected. "
              + "In case of performance issues allow access to this property.");
      return false;
    }
  }

  /** This static holder class prevents classloading deadlock. */
  private static final class Holder {
    private Holder() {}

    static final VectorizationProvider INSTANCE = lookup(false);
  }
}
