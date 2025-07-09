# JMH Benchmarks
Micro benchmarks for jVector. While {@link Bench.java} is about recall, the JMH benchmarks
are mostly targeting scalability and latency aspects.

## Building and running the benchmark

1. You can build and then run
```shell
# Get version from pom.xml
VERSION=$(mvn help:evaluate -Dexpression=revision -q -DforceStdout)
mvn clean install -DskipTests=true
java --enable-native-access=ALL-UNNAMED \
  --add-modules=jdk.incubator.vector \
  -XX:+HeapDumpOnOutOfMemoryError \
  -Xmx14G -Djvector.experimental.enable_native_vectorization=true \
  -jar benchmarks-jmh/target/benchmarks-jmh-${VERSION}.jar 
```

You can add additional optional JMH arguments dynamically from command line. For example, to run the benchmarks with 4 forks, 5 warmup iterations, 5 measurement iterations, 2 threads, and 10 seconds warmup time per iteration, use the following command:
```shell
# Get version from pom.xml
VERSION=$(mvn help:evaluate -Dexpression=revision -q -DforceStdout)
java --enable-native-access=ALL-UNNAMED \
  --add-modules=jdk.incubator.vector \
  -XX:+HeapDumpOnOutOfMemoryError \
  -Xmx14G -Djvector.experimental.enable_native_vectorization=true \
  -jar benchmarks-jmh/target/benchmarks-jmh-${VERSION}.jar \
  -f 4 -wi 5 -i 5 -t 2 -w 10s
```

Common JMH command line options you can use in the configuration or command line:
- `-f <num>` - Number of forks
- `-wi <num>` - Number of warmup iterations
- `-i <num>` - Number of measurement iterations
- `-w <time>` - Warmup time per iteration
- `-r <time>` - Measurement time per iteration
- `-t <num>` - Number of threads
- `-p <param>=<value>` - Benchmark parameters
- `-prof <profiler>` - Add profiler


2. Focus on specific benchmarks

For example in the below command lines we are going to run only `IndexConstructionWithRandomSetBenchmark`
```shell
# Get version from pom.xml
VERSION=$(mvn help:evaluate -Dexpression=revision -q -DforceStdout)
BENCHMARK_NAME="IndexConstructionWithRandomSetBenchmark"
mvn clean install -DskipTests=true
java --enable-native-access=ALL-UNNAMED \
  --add-modules=jdk.incubator.vector \
  -XX:+HeapDumpOnOutOfMemoryError \
  -Xmx20G -Djvector.experimental.enable_native_vectorization=true \
  -jar benchmarks-jmh/target/benchmarks-jmh-${VERSION}.jar $BENCHMARK_NAME
```

Same example for PQ training benchmark
```shell
# Get version from pom.xml
VERSION=$(mvn help:evaluate -Dexpression=revision -q -DforceStdout)
BENCHMARK_NAME="PQTrainingWithRandomVectorsBenchmark"
mvn clean install -DskipTests=true
java --enable-native-access=ALL-UNNAMED \
  --add-modules=jdk.incubator.vector \
  -XX:+HeapDumpOnOutOfMemoryError \
  -Xmx20G -Djvector.experimental.enable_native_vectorization=true \
  -jar benchmarks-jmh/target/benchmarks-jmh-${VERSION}.jar $BENCHMARK_NAME
```

If you want to rerun a specific benchmark without testing the entire grid of scenarios defined in the benchmark.
You can just do the following to set M and beamWidth:
```shell
# Get version from pom.xml
VERSION=$(mvn help:evaluate -Dexpression=revision -q -DforceStdout)
java -jar benchmarks-jmh/target/benchmarks-jmh-${VERSION}.jar IndexConstructionWithStaticSetBenchmark -p M=32 -p beamWidth=100 
```
### Running benchmarks with auxiliary counters

For benchmarks that include auxiliary counters (like `RecallWithRandomVectorsBenchmark`), run with CSV output to capture all metrics:

```shell
# Get version from pom.xml
VERSION=$(mvn help:evaluate -Dexpression=revision -q -DforceStdout)
BENCHMARK_NAME="RecallWithRandomVectorsBenchmark"
mvn clean install -DskipTests=true
java --enable-native-access=ALL-UNNAMED \
  --add-modules=jdk.incubator.vector \
  -XX:+HeapDumpOnOutOfMemoryError \
  -Xmx20G -Djvector.experimental.enable_native_vectorization=true \
  -jar benchmarks-jmh/target/benchmarks-jmh-${VERSION}.jar $BENCHMARK_NAME -rf csv -rff results.csv
```

## Formatting benchmark results

For benchmarks that output auxiliary counters (like recall metrics, visited counts, etc.), you can use the provided Python formatter to create a clean tabular view of the results.

### Setting up the Python environment

First, create a virtual environment and install the required dependencies:

```shell
# Create virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install pandas dependency
pip install pandas
```

### Using the results formatter

After running a benchmark with CSV output (using `-rf csv -rff results.csv`), you can format the results:

```shell
# Make sure your virtual environment is activated
source .venv/bin/activate

# Run the formatter script (assumes results.csv is in the current directory)
python benchmarks-jmh/scripts/jmh_results_formatter.py
```

The formatter will output a clean table showing:
- **k**: Number of nearest neighbors requested
- **PQ_Subspaces**: Number of Product Quantization subspaces
- **Time_ms**: Execution time in milliseconds
- **Recall**: Average recall score
- **ReRanked_Count**: Average number of vectors re-ranked
- **Visited_Count**: Average number of nodes visited during search
- **Expanded_Count_BaseLayer**: Average number of nodes expanded in base layer

Example output:
```
 k  PQ_Subspaces   Time_ms  Recall  ReRanked_Count  Visited_Count  Expanded_Count_BaseLayer
50             0  19.283   1.000             0.0        3290.8                     253.7
50            16   4.137   0.700           250.0        2849.6                     252.1
50            32   4.531   0.500           250.0        2881.9                     254.2
```



