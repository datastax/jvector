<!--
  ~ Copyright DataStax, Inc.
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~ http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
  -->

# Compactor Benchmark

The `CompactorBenchmark` measures the performance and accuracy of merging multiple `OnDiskGraphIndex` segments into a single unified index using the `OnDiskGraphIndexCompactor`.

## Overview

The benchmark follows these steps:
1.  **Setup**: Loads a specified dataset and splits it into `numSources` independent on-disk graph segments.
2.  **Benchmark**: Merges these segments into a single "compacted" index.
3.  **Verification**: Measures the search recall of the compacted index against the dataset's ground truth to ensure no quality loss occurred during merging.

## Parameters

You can tune the benchmark using the following JMH parameters:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `datasetName` | `siftsmall` | The symbolic name of the dataset to load (must be supported by `DataSets` loader). |
| `numSources` | `4` | The number of segments to create before compacting. Tests the impact of fragmentation. |
| `graphDegree` | `32` | The `M` parameter (max neighbors) used when building the source segments. |
| `beamWidth` | `100` | The search depth used when building the source segments. |
| `storageDirectories` | *(empty)* | Comma-separated list of directories to use for storing segments. If provided, overrides `tempDir`. Used in round-robin fashion. |
| `storageClasses` | *(empty)* | Comma-separated list of expected `io.github.jbellis.jvector.bench.storage.CloudStorageLayoutUtil.StorageClass` values (e.g., `LOCAL_SSD`, `NETWORK_FILESYSTEM`). Must match the count of `storageDirectories`. The benchmark will validate that the storage path actually resides on the specified storage class. |

## Execution Examples

You can execute the benchmark directly from the command line using `mvn exec:java`.

### Run with Defaults
```bash
./mvnw -pl benchmarks-jmh exec:java@compactor
```

### Testing a Parameter Matrix
JMH allows you to specify multiple values for any parameter by separating them with commas. This is useful for exploring how compaction performance scales across different datasets and configurations.

Example: Testing two datasets across different fragmentation levels:
```bash
./mvnw -pl benchmarks-jmh exec:java@compactor \
  -Dexec.args="-p datasetName=siftsmall,gist -p numSources=2,4,8 -p graphDegree=32,64"
```

### Run with Storage Validation
To benchmark compaction across specific storage tiers (e.g., merging segments stored on a local SSD vs a network mount), provide comma-separated directories and their expected classes:

```bash
./mvnw -pl benchmarks-jmh exec:java@compactor \
  -Dexec.args="-p storageDirectories=/mnt/fast_ssd,/mnt/slow_network \
               -p storageClasses=LOCAL_SSD,NETWORK_FILESYSTEM \
               -p numSources=2"
```
*Note: If `storageClasses` is provided, it must have the same number of entries as `storageDirectories`, and the system must be able to verify the storage class of the provided paths.*

### Full JMH Control
Since the benchmark uses the JMH `Runner`, you can pass any standard JMH arguments (like `-wi` for warmups or `-i` for iterations):
```bash
./mvnw -pl benchmarks-jmh exec:java@compactor \
  -Dexec.args="-wi 2 -i 5 -p numSources=4"
```

## Interpreting Results

- **Score (ms/op)**: Represents the average time taken to perform the compaction (merging) of the segments.
- **Compacted Graph Recall**: The benchmark logs the recall of the resulting index. High recall indicates the diversity heuristic (`CompactVamanaDiversityProvider`) correctly maintained the graph's navigability.