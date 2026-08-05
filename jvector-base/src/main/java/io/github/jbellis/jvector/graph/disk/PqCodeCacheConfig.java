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

package io.github.jbellis.jvector.graph.disk;

/// Immutable configuration for the fused-PQ pre-encode code cache ([PqCodeCache]), composed into a
/// compaction run via {@code OnDiskGraphIndexCompactor.setPqCodeCacheConfig(...)}.
///
/// Two independent controls:
/// - **compose in or leave off** — [#enabled]. When off, no cache is built and every consumer
///   encodes per neighbor write (the pre-fix behavior); the strategy hands consumers
///   [PqCodeCache#NONE] so the object graph stays uniform.
/// - **chunk sizing** — [#maxChunkBytes], the soft cap on bytes per mmap chunk (clamped to
///   [PqCodeCache#DEFAULT_MAX_CHUNK_BYTES]). Smaller values force more chunks; the default keeps a
///   graph in as few ≤ 1 GiB mappings as possible.
///
/// Distinct from the cache's *internal* [PqCodeCache#isActive() dynamic bypass]: this config is the
/// build-time, compose-in/leave-off decision, whereas the bypass can flip a built cache off at
/// runtime.
public final class PqCodeCacheConfig {
    /// Cache composed in, chunks sized to the 1 GiB default. The compactor default.
    public static final PqCodeCacheConfig DEFAULT = new PqCodeCacheConfig(true, PqCodeCache.DEFAULT_MAX_CHUNK_BYTES);

    /// Cache left off: consumers encode per neighbor write (no pre-encode cache built).
    public static final PqCodeCacheConfig DISABLED = new PqCodeCacheConfig(false, PqCodeCache.DEFAULT_MAX_CHUNK_BYTES);

    private final boolean enabled;
    private final long maxChunkBytes;

    public PqCodeCacheConfig(boolean enabled, long maxChunkBytes) {
        if (maxChunkBytes <= 0) {
            throw new IllegalArgumentException("maxChunkBytes must be > 0, got " + maxChunkBytes);
        }
        this.enabled = enabled;
        this.maxChunkBytes = maxChunkBytes;
    }

    /// Whether to build the pre-encode cache at all.
    public boolean enabled() {
        return enabled;
    }

    /// Soft cap on bytes per mmap chunk (clamped to 1 GiB by [PqCodeCache#codesPerChunkFor]).
    public long maxChunkBytes() {
        return maxChunkBytes;
    }

    /// Returns a copy with {@link #enabled()} set as given.
    public PqCodeCacheConfig withEnabled(boolean enabled) {
        return new PqCodeCacheConfig(enabled, maxChunkBytes);
    }

    /// Returns a copy with {@link #maxChunkBytes()} set as given.
    public PqCodeCacheConfig withMaxChunkBytes(long maxChunkBytes) {
        return new PqCodeCacheConfig(enabled, maxChunkBytes);
    }

    @Override
    public String toString() {
        return "PqCodeCacheConfig{enabled=" + enabled + ", maxChunkBytes=" + maxChunkBytes + '}';
    }
}
