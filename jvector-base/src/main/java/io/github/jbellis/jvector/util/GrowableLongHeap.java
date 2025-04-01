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
 * An AbstractLongHeap that can grow in size (unbounded, except for memory and array size limits).
 */
public class GrowableLongHeap extends AbstractLongHeap {
    /**
     * Create an empty heap with the configured initial size.
     *
     * @param initialSize the initial size of the heap
     */
    public GrowableLongHeap(int initialSize) {
        super(initialSize);
    }

    /**
     * Adds a value to an LongHeap in log(size) time.
     *
     * @return true always
     */
    @Override
    public boolean push(long element) {
        add(element);
        return true;
    }
}
