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

/**
 * Helper APIs to encode numeric values as sortable bytes and vice-versa.
 *
 * <p>To also index floating point numbers, this class supplies a method to convert them to
 * integer values by changing their bit layout: {@link
 * #floatToSortableInt}. You will have no precision loss by converting floating point numbers to
 * integers and back (only that the integer form is not usable). Other data types like dates can
 * easily converted to longs or ints (e.g. date to long: {@link java.util.Date#getTime}).
 */
public final class NumericUtils {

  private NumericUtils() {} // no instance!

    /**
   * Converts a <code>float</code> value to a sortable signed <code>int</code>. The value is
   * converted by getting their IEEE 754 floating-point &quot;float format&quot; bit layout and then
   * some bits are swapped, to be able to compare the result as int. By this the precision is not
   * reduced, but the value can easily used as an int. The sort order (including {@link Float#NaN})
   * is defined by {@link Float#compareTo}; {@code NaN} is greater than positive infinity.
   *
   * @param value the float value to convert to sortable integer representation
   * @return the sortable integer encoding of the float value
   * @see #sortableIntToFloat
   */
  public static int floatToSortableInt(float value) {
    return sortableFloatBits(Float.floatToIntBits(value));
  }

  /**
   * Converts a sortable <code>int</code> back to a <code>float</code>.
   *
   * @param encoded the sortable integer representation to decode
   * @return the original float value
   * @see #floatToSortableInt
   */
  public static float sortableIntToFloat(int encoded) {
    return Float.intBitsToFloat(sortableFloatBits(encoded));
  }

  /** Converts IEEE 754 representation of a float to sortable order (or back to the original)
   * @param bits the IEEE 754 bit representation to transform
   * @return the transformed bit pattern suitable for integer comparison
   */
  public static int sortableFloatBits(int bits) {
    return bits ^ (bits >> 31) & 0x7fffffff;
  }
}
