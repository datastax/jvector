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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.util.RamUsageEstimator;

import java.lang.foreign.MemorySegment;

/**
 * Heap held by a native segment's Java handle: the segment object and its scope. The arena's internal bookkeeping
 * behind the scope is not reachable through the public API, so this is an estimate of a fixed few dozen bytes per
 * segment; the native bytes themselves are reported separately.
 */
final class NativeSegmentHeap {
    private NativeSegmentHeap() {}

    static long of(MemorySegment segment) {
        return segment == null ? 0 : RamUsageEstimator.shallowSizeOf(segment) + RamUsageEstimator.shallowSizeOf(segment.scope());
    }
}
