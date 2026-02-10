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

/**
 * Provides utilities for characterizing the underlying storage hardware and layout.
 * <p>
 * This package contains logic to detect and classify storage tiers (e.g., Local SSD,
 * Persistent Disk, Network Filesystem) across different environments including
 * AWS, GCP, and local development machines.
 * <p>
 * The primary entry point is {@link io.github.jbellis.jvector.bench.storage.CloudStorageLayoutUtil},
 * which provides a unified view of the system's mount points and their corresponding
 * {@link io.github.jbellis.jvector.bench.storage.CloudStorageLayoutUtil.StorageClass}.
 */
package io.github.jbellis.jvector.bench.storage;
