/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.util;

import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * A thread-safe {@link BitSet} implementation that grows as needed to accommodate set(index) calls. When it
 * does so, it will grow its internal storage multiplicatively, assuming that more growth will be
 * needed in the future. This is the important difference from FixedBitSet + FBS.ensureCapacity,
 * which grows the minimum necessary each time.
 *
 * @see GrowableBitSet
 */
public class ThreadSafeGrowableBitSet extends BitSet {

  /** The underlying JDK BitSet that stores the actual bit data */
  private final java.util.BitSet bitSet;
  /** Lock for coordinating concurrent read and write access */
  private final ReadWriteLock lock = new ReentrantReadWriteLock();

  /**
   * Creates a thread-safe wrapper around an existing BitSet.
   * @param bitSet the JDK BitSet to wrap with thread-safe operations
   */
  public ThreadSafeGrowableBitSet(java.util.BitSet bitSet) {
    this.bitSet = bitSet;
  }

  /**
   * Creates a new thread-safe growable bit set with the specified initial capacity.
   * @param initialBits the initial number of bits to allocate
   */
  public ThreadSafeGrowableBitSet(int initialBits) {
    this.bitSet = new java.util.BitSet(initialBits);
  }

  @Override
  public void clear(int index) {
    lock.writeLock().lock();
    try {
      bitSet.clear(index);
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public void clear() {
    lock.writeLock().lock();
    try {
      bitSet.clear();
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public boolean get(int index) {
    lock.readLock().lock();
    try {
      return bitSet.get(index);
    } finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public boolean getAndSet(int index) {
    lock.writeLock().lock();
    try {
      boolean v = get(index);
      set(index);
      return v;
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public int length() {
    lock.readLock().lock();
    try {
      return bitSet.length();
    } finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public void set(int i) {
    lock.writeLock().lock();
    try {
      bitSet.set(i);
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public void clear(int startIndex, int endIndex) {
    lock.writeLock().lock();
    try {
      if (startIndex == 0 && endIndex == bitSet.length()) {
        bitSet.clear();
        return;
      } else if (startIndex >= endIndex) {
        return;
      }
      bitSet.clear(startIndex, endIndex);
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public int cardinality() {
    lock.readLock().lock();
    try {
      return bitSet.cardinality();
    } finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public int approximateCardinality() {
    lock.readLock().lock();
    try {
      return bitSet.cardinality();
    } finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public int prevSetBit(int index) {
    lock.readLock().lock();
    try {
      return bitSet.previousSetBit(index);
    } finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public int nextSetBit(int i) {
    lock.readLock().lock();
    try {
      int next = bitSet.nextSetBit(i);
      if (next == -1) {
        next = DocIdSetIterator.NO_MORE_DOCS;
      }
      return next;
    } finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public long ramBytesUsed() {
    throw new UnsupportedOperationException();
  }

  /**
   * Creates a copy of this bit set including all currently set bits.
   * @return a new independent ThreadSafeGrowableBitSet with the same bit values
   */
  public ThreadSafeGrowableBitSet copy() {
    lock.readLock().lock();
    try {
      return new ThreadSafeGrowableBitSet((java.util.BitSet) bitSet.clone());
    } finally {
      lock.readLock().unlock();
    }
  }
}
