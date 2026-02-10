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

__TLDR__: `mvn compile exec:exec@compactor -pl benchmarks-jmh -am`

## Overview

The benchmark follows these steps:
1.  **Setup**: Loads a specified dataset and splits it into `numSources` independent on-disk graph segments.
2.  **Benchmark**: Merges these segments into a single "compacted" index.
3.  **Verification**: Measures the search recall of the compacted index against the dataset's ground truth to ensure search quality is maintained after merging.

## Parameters

You can tune the benchmark using the following JMH parameters via `-Dargs="-p <param>=<value>"`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `datasetNames` | `glove-100-angular` | The symbolic name of the dataset to load. Supports qualified names like `ibm_datapile:1m`. |
| `numSources` | `1, 4` | The number of segments to create before compacting. A value of `1` acts as a pass-through baseline (no compaction). |
| `graphDegree` | `32` | The `M` parameter (max neighbors) used when building the source segments. |
| `beamWidth` | `100` | The search depth used when building the source segments. |
| `storageDirectories` | *(empty)* | Comma-separated list of directories to use for storing segments. Used in round-robin fashion. |
| `storageClasses` | *(empty)* | Comma-separated list of expected `StorageClass` values. Must match the count of `storageDirectories`. |

## Parameter Validation

The benchmark performs strict validation before execution:
- `numSources`, `graphDegree`, and `beamWidth` must all be positive integers.
- If `numSources` is `1`, the benchmark establishes a baseline by skipping the compaction call and searching the source segment directly.
- If `storageDirectories` are provided, they must be writable.
- If `storageClasses` are provided, the benchmark verifies that the actual mount points match the asserted classes using `CloudStorageLayoutUtil`.

Invalid JMH flags or parameter types passed via `-Dargs` will be detected by the JMH parser and result in a build failure.

## Execution Examples

The benchmark uses the `exec:exec` goal to ensure that forked JVM processes correctly inherit the full project environment, including the classpath and specialized JVector JVM arguments.

### Run with Defaults
```bash
./mvnw -pl benchmarks-jmh exec:exec@compactor
```

### Debugging (Disable Forking)
To attach a debugger or inspect logs in the same process, disable JMH forking:
```bash
./mvnw -pl benchmarks-jmh exec:exec@compactor -Dargs="-f 0"
```

### Run with a Specific Dataset
```bash
./mvnw -pl benchmarks-jmh exec:exec@compactor -Dargs="-p datasetNames=ibm_datapile:1m"
```

### Run with Storage Validation
To benchmark compaction across specific storage tiers (e.g., merging segments stored on a local SSD vs a network mount), provide comma-separated directories and their expected classes:

```bash
./mvnw -pl benchmarks-jmh exec:exec@compactor \
  -Dargs="-p storageDirectories=/mnt/fast_ssd,/mnt/slow_network \
          -p storageClasses=LOCAL_SSD,NETWORK_FILESYSTEM \
          -p numSources=2"
```

### Testing a Parameter Matrix
Example: Testing across different fragmentation levels and degrees:
```bash
./mvnw -pl benchmarks-jmh exec:exec@compactor \
  -Dargs="-p datasetNames=glove-100-angular -p numSources=1,2,4,8 -p graphDegree=32,64"
```

### Full JMH Control
You can pass any standard JMH arguments (like `-wi` for warmups or `-i` for iterations):
```bash
./mvnw -pl benchmarks-jmh exec:exec@compactor \
  -Dargs="-wi 2 -i 5 -p numSources=1,4"
```

## Interpreting Results

- **Score (ms/op)**: The average time taken to perform the compaction. For `numSources=1`, this represents the baseline search time (compaction is bypassed).
- **recall (AuxCounter)**: The search recall of the index. Compare the `recall` of `numSources > 1` against the `numSources=1` baseline to verify that the merge process maintained graph quality.