package io.github.jbellis.jvector.api.startup;

public enum JVProfile {
  /**
   When this profile is selected, then any of the profiles will suffice, in the order they appear
   as enum values. This is the default.
   */
  auto,
  /**
   When base_jvm is selected, then the core JVM implementation of the JVector engine is loaded.
   */
  base_jvm,
  /**
   When panama_jvm is selected, then the JDK Panama project version of the JVector engine is
   loaded. This uses a direct vectorization API which effectively allows intrinsic calls to native
   CPU instructions.
   */
  panama_jvm,
  /**
   When native_ffi is selected, then the FFI implementation of the JVector engine is loaded. This
   uses the most recent FFI interface to call linked code for vectorization which is written in
   C with specialized SIMD instructions.
   */
  native_ffi
}
