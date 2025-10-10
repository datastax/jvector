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
  /** Creates a BitSet. */
  protected BitSet() {}

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
   * The number of bits in the set.
   * @return the number of bits
   */
  public abstract int length();

  /**
   * Set the bit at <code>i</code>.
   * @param i the bit index
   */
  public abstract void set(int i);

  /**
   * Set the bit at <code>i</code>, returning <code>true</code> if it was previously set.
   * @param i the bit index
   * @return true if the bit was previously set
   */
  public abstract boolean getAndSet(int i);

  /**
   * Clear the bit at <code>i</code>.
   * @param i the bit index
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
   * Return the number of bits that are set. NOTE: this method is likely to run in linear time
   * @return the number of set bits
   */
  public abstract int cardinality();

  /**
   * Return an approximation of the cardinality of this set. Some implementations may trade accuracy
   * for speed if they have the ability to estimate the cardinality of the set without iterating
   * over all the data. The default implementation returns {@link #cardinality()}.
   * @return the approximate cardinality
   */
  public abstract int approximateCardinality();

  /**
   * Returns the index of the last set bit before or on the index specified. -1 is returned if there
   * are no more set bits.
   * @param index the index
   * @return the index of the previous set bit
   */
  public abstract int prevSetBit(int index);

  /**
   * Returns the index of the first set bit starting at the index specified. {@link
   * DocIdSetIterator#NO_MORE_DOCS} is returned if there are no more set bits.
   * @param index the index
   * @return the index of the next set bit
   */
  public abstract int nextSetBit(int index);

  @Override
  public String toString() {
    return getClass().getSimpleName() + "(length=" + length() + ", cardinality=~" + approximateCardinality() + ")";
  }
}
