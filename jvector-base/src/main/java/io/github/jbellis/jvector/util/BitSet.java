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
 * Base implementation for a bit set.
 */
public abstract class BitSet implements Bits, Accountable {
  /** Creates a BitSet instance. */
  public BitSet() {}
  /**
   * Clear all the bits of the set.
   *
   * <p>Depending on the implementation, this may be significantly faster than clear(0, length).
   */
  public void clear() {
    // default implementation for compatibility
    clear(0, length());
  }

  /**
   * Returns the number of bits in the set.
   * @return the number of bits in the set
   */
  public abstract int length();

  /**
   * Sets the bit at the specified index.
   * @param i the index of the bit to set
   */
  public abstract void set(int i);

  /**
   * Sets the bit at the specified index and returns its previous value.
   * @param i the index of the bit to set
   * @return {@code true} if the bit was previously set, {@code false} otherwise
   */
  public abstract boolean getAndSet(int i);

  /**
   * Clears the bit at the specified index.
   * @param i the index of the bit to clear
   */
  public abstract void clear(int i);

  /**
   * Clears a range of bits.
   *
   * @param startIndex lower index
   * @param endIndex one-past the last bit to clear
   */
  public abstract void clear(int startIndex, int endIndex);

  /**
   * Returns the number of bits that are set.
   * <p>
   * NOTE: this method is likely to run in linear time.
   * @return the number of bits that are set
   */
  public abstract int cardinality();

  /**
   * Returns an approximation of the cardinality of this set. Some implementations may trade accuracy
   * for speed if they have the ability to estimate the cardinality of the set without iterating
   * over all the data. The default implementation returns {@link #cardinality()}.
   * @return an approximation of the number of bits that are set
   */
  public abstract int approximateCardinality();

  /**
   * Returns the index of the last set bit before or on the index specified.
   * @param index the index to start searching backwards from (inclusive)
   * @return the index of the previous set bit, or -1 if there are no more set bits
   */
  public abstract int prevSetBit(int index);

  /**
   * Returns the index of the first set bit starting at the index specified.
   * @param index the index to start searching from (inclusive)
   * @return the index of the next set bit, or {@link DocIdSetIterator#NO_MORE_DOCS} if there are no more set bits
   */
  public abstract int nextSetBit(int index);

  @Override
  public String toString() {
    return getClass().getSimpleName() + "(length=" + length() + ", cardinality=~" + approximateCardinality() + ")";
  }
}
