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

package io.github.jbellis.jvector.util;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.security.AccessControlException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.text.DecimalFormat;
import java.util.Collection;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Estimates the size (memory representation) of Java objects.
 *
 * <p>This class uses assumptions that were discovered for the Hotspot virtual machine. If you use a
 * non-OpenJDK/Oracle-based JVM, the measurements may be slightly wrong.
 *
 * <p>For predictive, configuration-driven sizing of JVector indexes (as opposed to inspecting
 * already instantiated objects) see {@code devdocs/memory_estimator.md} and
 * {@link io.github.jbellis.jvector.MemoryCostEstimator}.
 *
 * @see #shallowSizeOf(Object)
 * @see #shallowSizeOfInstance(Class)
 */
public final class RamUsageEstimator {

  /** One kilobyte bytes. */
  public static final long ONE_KB = 1024;

  /** One megabyte bytes. */
  public static final long ONE_MB = ONE_KB * ONE_KB;

  /** One gigabyte bytes. */
  public static final long ONE_GB = ONE_KB * ONE_MB;

  /** No instantiation. */
  private RamUsageEstimator() {}

  /** True, iff compressed references (oops) are enabled by this JVM */
  public static final boolean COMPRESSED_REFS_ENABLED;

  /** Number of bytes this JVM uses to represent an object reference. */
  public static final int NUM_BYTES_OBJECT_REF;

  /** Number of bytes to represent an object header (no fields, no alignments). */
  public static final int NUM_BYTES_OBJECT_HEADER;

  /** Number of bytes to represent an array header (no content, but with alignments). */
  public static final int NUM_BYTES_ARRAY_HEADER;

  /**
   * A constant specifying the object alignment boundary inside the JVM. Objects will always take a
   * full multiple of this constant, possibly wasting some space.
   */
  public static final int NUM_BYTES_OBJECT_ALIGNMENT;

  /**
   * Approximate memory usage that we assign to all unknown objects - this maps roughly to a few
   * primitive fields and a couple short String-s.
   */
  public static final int UNKNOWN_DEFAULT_RAM_BYTES_USED = 256;

  /** Sizes of primitive classes. */
  public static final Map<Class<?>, Integer> primitiveSizes;

  static {
    Map<Class<?>, Integer> primitiveSizesMap = new IdentityHashMap<>();
    primitiveSizesMap.put(boolean.class, 1);
    primitiveSizesMap.put(byte.class, 1);
    primitiveSizesMap.put(char.class, Integer.valueOf(Character.BYTES));
    primitiveSizesMap.put(short.class, Integer.valueOf(Short.BYTES));
    primitiveSizesMap.put(int.class, Integer.valueOf(Integer.BYTES));
    primitiveSizesMap.put(float.class, Integer.valueOf(Float.BYTES));
    primitiveSizesMap.put(double.class, Integer.valueOf(Double.BYTES));
    primitiveSizesMap.put(long.class, Integer.valueOf(Long.BYTES));

    primitiveSizes = Collections.unmodifiableMap(primitiveSizesMap);
  }

  static final int INTEGER_SIZE, LONG_SIZE, STRING_SIZE;

  static final String MANAGEMENT_FACTORY_CLASS = "java.lang.management.ManagementFactory";
  static final String HOTSPOT_BEAN_CLASS = "com.sun.management.HotSpotDiagnosticMXBean";

  /** Initialize constants and try to collect information about the JVM internals. */
  static {
    if (Constants.JRE_IS_64BIT) {
      // Try to get compressed oops and object alignment (the default seems to be 8 on Hotspot);
      // (this only works on 64 bit, on 32 bits the alignment and reference size is fixed):
      boolean compressedOops = false;
      int objectAlignment = 8;
      boolean isHotspot = false;
      try {
        final Class<?> beanClazz = Class.forName(HOTSPOT_BEAN_CLASS);
        // we use reflection for this, because the management factory is not part
        // of Java 8's compact profile:
        final Object hotSpotBean =
            Class.forName(MANAGEMENT_FACTORY_CLASS)
                .getMethod("getPlatformMXBean", Class.class)
                .invoke(null, beanClazz);
        if (hotSpotBean != null) {
          isHotspot = true;
          final Method getVMOptionMethod = beanClazz.getMethod("getVMOption", String.class);
          try {
            final Object vmOption = getVMOptionMethod.invoke(hotSpotBean, "UseCompressedOops");
            compressedOops =
                Boolean.parseBoolean(
                    vmOption.getClass().getMethod("getValue").invoke(vmOption).toString());
          } catch (@SuppressWarnings("unused") ReflectiveOperationException | RuntimeException e) {
            isHotspot = false;
          }
          try {
            final Object vmOption = getVMOptionMethod.invoke(hotSpotBean, "ObjectAlignmentInBytes");
            objectAlignment =
                Integer.parseInt(
                    vmOption.getClass().getMethod("getValue").invoke(vmOption).toString());
          } catch (@SuppressWarnings("unused") ReflectiveOperationException | RuntimeException e) {
            isHotspot = false;
          }
        }
      } catch (@SuppressWarnings("unused") ReflectiveOperationException | RuntimeException e) {
        isHotspot = false;
        final Logger log = Logger.getLogger(RamUsageEstimator.class.getName());
        final Module module = RamUsageEstimator.class.getModule();
        final ModuleLayer layer = module.getLayer();
        // classpath / unnamed module has no layer, so we need to check:
        if (layer != null
            && layer.findModule("jdk.management").map(module::canRead).orElse(false) == false) {
          log.warning(
              "JVector cannot correctly calculate object sizes on 64bit JVMs, unless the 'jdk.management' Java module "
                  + "is readable [please add 'jdk.management' to modular application either by command line or its module descriptor]");
        } else {
          log.warning(
              "JVector cannot correctly calculate object sizes on 64bit JVMs that are not based on Hotspot or a compatible implementation.");
        }
      }
      COMPRESSED_REFS_ENABLED = compressedOops;
      NUM_BYTES_OBJECT_ALIGNMENT = objectAlignment;
      // reference size is 4, if we have compressed oops:
      NUM_BYTES_OBJECT_REF = COMPRESSED_REFS_ENABLED ? 4 : 8;
      // "best guess" based on reference size:
      NUM_BYTES_OBJECT_HEADER = 8 + NUM_BYTES_OBJECT_REF;
      // array header is NUM_BYTES_OBJECT_HEADER + NUM_BYTES_INT, but aligned (object alignment):
      NUM_BYTES_ARRAY_HEADER = (int) alignObjectSize(NUM_BYTES_OBJECT_HEADER + Integer.BYTES);
    } else {
      COMPRESSED_REFS_ENABLED = false;
      NUM_BYTES_OBJECT_ALIGNMENT = 8;
      NUM_BYTES_OBJECT_REF = 4;
      NUM_BYTES_OBJECT_HEADER = 8;
      // For 32 bit JVMs, no extra alignment of array header:
      NUM_BYTES_ARRAY_HEADER = NUM_BYTES_OBJECT_HEADER + Integer.BYTES;
    }

    INTEGER_SIZE = (int) shallowSizeOfInstance(Integer.class);
    LONG_SIZE = (int) shallowSizeOfInstance(Long.class);
    STRING_SIZE = (int) shallowSizeOfInstance(String.class);
  }

  /** Approximate memory usage that we assign to a Hashtable / HashMap entry. */
  public static final long HASHTABLE_RAM_BYTES_PER_ENTRY =
      // key + value *
      // hash tables need to be oversized to avoid collisions, assume 2x capacity
      (2L * NUM_BYTES_OBJECT_REF) * 2;

  /** Aligns an object size to be the next multiple of {@link #NUM_BYTES_OBJECT_ALIGNMENT}. */
  public static long alignObjectSize(long size) {
    size += (long) NUM_BYTES_OBJECT_ALIGNMENT - 1L;
    return size - (size % NUM_BYTES_OBJECT_ALIGNMENT);
  }

  /**
   * Return the shallow size of the provided {@link Integer} object. Ignores the possibility that
   * this object is part of the VM IntegerCache
   */
  public static long sizeOf(Integer ignored) {
    return INTEGER_SIZE;
  }

  /**
   * Return the shallow size of the provided {@link Long} object. Ignores the possibility that this
   * object is part of the VM LongCache
   */
  public static long sizeOf(Long ignored) {
    return LONG_SIZE;
  }

  /** Returns the size in bytes of the byte[] object. */
  public static long sizeOf(byte[] arr) {
    return alignObjectSize((long) NUM_BYTES_ARRAY_HEADER + arr.length);
  }

  /** Returns the size in bytes of the boolean[] object. */
  public static long sizeOf(boolean[] arr) {
    return alignObjectSize((long) NUM_BYTES_ARRAY_HEADER + arr.length);
  }

  /** Returns the size in bytes of the char[] object. */
  public static long sizeOf(char[] arr) {
    return alignObjectSize((long) NUM_BYTES_ARRAY_HEADER + (long) Character.BYTES * arr.length);
  }

  /** Returns the size in bytes of the short[] object. */
  public static long sizeOf(short[] arr) {
    return alignObjectSize((long) NUM_BYTES_ARRAY_HEADER + (long) Short.BYTES * arr.length);
  }

  /** Returns the size in bytes of the int[] object. */
  public static long sizeOf(int[] arr) {
    return alignObjectSize((long) NUM_BYTES_ARRAY_HEADER + (long) Integer.BYTES * arr.length);
  }

  /** Returns the size in bytes of the float[] object. */
  public static long sizeOf(float[] arr) {
    return alignObjectSize((long) NUM_BYTES_ARRAY_HEADER + (long) Float.BYTES * arr.length);
  }

  /** Returns the size in bytes of the long[] object. */
  public static long sizeOf(long[] arr) {
    return alignObjectSize((long) NUM_BYTES_ARRAY_HEADER + (long) Long.BYTES * arr.length);
  }

  /** Returns the size in bytes of the double[] object. */
  public static long sizeOf(double[] arr) {
    return alignObjectSize((long) NUM_BYTES_ARRAY_HEADER + (long) Double.BYTES * arr.length);
  }

  /** Returns the size in bytes of the String[] object. */
  public static long sizeOf(String[] arr) {
    long size = shallowSizeOf(arr);
    for (String s : arr) {
      if (s == null) {
        continue;
      }
      size += sizeOf(s);
    }
    return size;
  }

  public static long sizeOf(Accountable a) {
    return a.ramBytesUsed();
  }

  /** Recurse only into immediate descendants. */
  public static final int MAX_DEPTH = 1;

  private static long sizeOfMap(Map<?, ?> map, int depth, long defSize) {
    if (map == null) {
      return 0;
    }
    long size = shallowSizeOf(map);
    if (depth > MAX_DEPTH) {
      return size;
    }
    long sizeOfEntry = -1;
    for (Map.Entry<?, ?> entry : map.entrySet()) {
      if (sizeOfEntry == -1) {
        sizeOfEntry = shallowSizeOf(entry);
      }
      size += sizeOfEntry;
      size += sizeOfObject(entry.getKey(), depth, defSize);
      size += sizeOfObject(entry.getValue(), depth, defSize);
    }
    return alignObjectSize(size);
  }

  private static long sizeOfCollection(Collection<?> collection, int depth, long defSize) {
    if (collection == null) {
      return 0;
    }
    long size = shallowSizeOf(collection);
    if (depth > MAX_DEPTH) {
      return size;
    }
    // assume array-backed collection and add per-object references
    size += NUM_BYTES_ARRAY_HEADER + collection.size() * NUM_BYTES_OBJECT_REF;
    for (Object o : collection) {
      size += sizeOfObject(o, depth, defSize);
    }
    return alignObjectSize(size);
  }

  private static long sizeOfObject(Object o, int depth, long defSize) {
    if (o == null) {
      return 0;
    }
    long size;
    if (o instanceof Accountable) {
      size = ((Accountable) o).ramBytesUsed();
    } else if (o instanceof String) {
      size = sizeOf((String) o);
    } else if (o instanceof boolean[]) {
      size = sizeOf((boolean[]) o);
    } else if (o instanceof byte[]) {
      size = sizeOf((byte[]) o);
    } else if (o instanceof char[]) {
      size = sizeOf((char[]) o);
    } else if (o instanceof double[]) {
      size = sizeOf((double[]) o);
    } else if (o instanceof float[]) {
      size = sizeOf((float[]) o);
    } else if (o instanceof int[]) {
      size = sizeOf((int[]) o);
    } else if (o instanceof Integer) {
      size = sizeOf((Integer) o);
    } else if (o instanceof Long) {
      size = sizeOf((Long) o);
    } else if (o instanceof long[]) {
      size = sizeOf((long[]) o);
    } else if (o instanceof short[]) {
      size = sizeOf((short[]) o);
    } else if (o instanceof String[]) {
      size = sizeOf((String[]) o);
    } else if (o instanceof Map) {
      size = sizeOfMap((Map<?, ?>) o, ++depth, defSize);
    } else if (o instanceof Collection) {
      size = sizeOfCollection((Collection<?>) o, ++depth, defSize);
    } else {
      if (defSize > 0) {
        size = defSize;
      } else {
        size = shallowSizeOf(o);
      }
    }
    return size;
  }

  /** Returns the size in bytes of the String object. */
  public static long sizeOf(String s) {
    if (s == null) {
      return 0;
    }
    // may not be true in Java 9+ and CompactStrings - but we have no way to determine this

    // char[] + hashCode
    long size = STRING_SIZE + (long) NUM_BYTES_ARRAY_HEADER + (long) Character.BYTES * s.length();
    return alignObjectSize(size);
  }

  /** Returns the shallow size in bytes of the Object[] object. */
  // Use this method instead of #shallowSizeOf(Object) to avoid costly reflection
  public static long shallowSizeOf(Object[] arr) {
    return alignObjectSize(
        (long) NUM_BYTES_ARRAY_HEADER + (long) NUM_BYTES_OBJECT_REF * arr.length);
  }

  /**
   * Estimates a "shallow" memory usage of the given object. For arrays, this will be the memory
   * taken by array storage (no subreferences will be followed). For objects, this will be the
   * memory taken by the fields.
   *
   * <p>JVM object alignments are also applied.
   */
  public static long shallowSizeOf(Object obj) {
    if (obj == null) return 0;
    final Class<?> clz = obj.getClass();
    if (clz.isArray()) {
      return shallowSizeOfArray(obj);
    } else {
      return shallowSizeOfInstance(clz);
    }
  }

  /**
   * Returns the shallow instance size in bytes an instance of the given class would occupy. This
   * works with all conventional classes and primitive types, but not with arrays (the size then
   * depends on the number of elements and varies from object to object).
   *
   * @see #shallowSizeOf(Object)
   * @throws IllegalArgumentException if {@code clazz} is an array class.
   */
  public static long shallowSizeOfInstance(Class<?> clazz) {
    if (clazz.isArray())
      throw new IllegalArgumentException("This method does not work with array classes.");
    if (clazz.isPrimitive()) return primitiveSizes.get(clazz);

    long size = NUM_BYTES_OBJECT_HEADER;

    // Walk type hierarchy
    for (; clazz != null; clazz = clazz.getSuperclass()) {
      final Class<?> target = clazz;
      final Field[] fields;
      try {
        fields = doPrivileged((PrivilegedAction<Field[]>) target::getDeclaredFields);
      } catch (
          @SuppressWarnings("removal")
          AccessControlException e) {
        throw new RuntimeException("Can't access fields of class: " + target, e);
      }

      for (Field f : fields) {
        if (!Modifier.isStatic(f.getModifiers())) {
          size = adjustForField(size, f);
        }
      }
    }
    return alignObjectSize(size);
  }

  @SuppressWarnings("removal")
  private static <T> T doPrivileged(PrivilegedAction<T> action) {
    return AccessController.doPrivileged(action);
  }

  /** Return shallow size of any <code>array</code>. */
  private static long shallowSizeOfArray(Object array) {
    long size = NUM_BYTES_ARRAY_HEADER;
    final int len = Array.getLength(array);
    if (len > 0) {
      Class<?> arrayElementClazz = array.getClass().getComponentType();
      if (arrayElementClazz.isPrimitive()) {
        size += (long) len * primitiveSizes.get(arrayElementClazz);
      } else {
        size += (long) NUM_BYTES_OBJECT_REF * len;
      }
    }
    return alignObjectSize(size);
  }

  /**
   * This method returns the maximum representation size of an object. <code>sizeSoFar</code> is the
   * object's size measured so far. <code>f</code> is the field being probed.
   *
   * <p>The returned offset will be the maximum of whatever was measured so far and <code>f</code>
   * field's offset and representation size (unaligned).
   */
  public static long adjustForField(long sizeSoFar, final Field f) {
    final Class<?> type = f.getType();
    final int fsize = type.isPrimitive() ? primitiveSizes.get(type) : NUM_BYTES_OBJECT_REF;
    return sizeSoFar + fsize;
  }

  /** Returns <code>size</code> in human-readable units (GB, MB, KB or bytes). */
  public static String humanReadableUnits(long bytes, DecimalFormat df) {
    if (bytes / ONE_GB > 0) {
      return df.format((float) bytes / ONE_GB) + " GB";
    } else if (bytes / ONE_MB > 0) {
      return df.format((float) bytes / ONE_MB) + " MB";
    } else if (bytes / ONE_KB > 0) {
      return df.format((float) bytes / ONE_KB) + " KB";
    } else {
      return bytes + " bytes";
    }
  }
}
