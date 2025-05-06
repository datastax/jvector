# JMH Benchmarks
Micro benchmarks for jVector. While {@link Bench.java} is about recall, the JMH benchmarks
are mostly targeting scalability and latency aspects.

## Building and running the benchmark

1. You can build and then run
```shell
mvn clean install -DskipTests=true
java --enable-native-access=ALL-UNNAMED \
  --add-modules=jdk.incubator.vector \
  -XX:+HeapDumpOnOutOfMemoryError \
  -Xmx14G -Djvector.experimental.enable_native_vectorization=true \
  -jar benchmarks-jmh/target/benchmarks-jmh-4.0.0-beta.2-SNAPSHOT.jar 
```

You can add additional optional JMH arguments dynamically from command line. For example, to run the benchmarks with 4 forks, 5 warmup iterations, 5 measurement iterations, 2 threads, and 10 seconds warmup time per iteration, use the following command:
```shell
java --enable-native-access=ALL-UNNAMED \
  --add-modules=jdk.incubator.vector \
  -XX:+HeapDumpOnOutOfMemoryError \
  -Xmx14G -Djvector.experimental.enable_native_vectorization=true \
  -jar benchmarks-jmh/target/benchmarks-jmh-4.0.0-beta.2-SNAPSHOT.jar \
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
mvn clean install -DskipTests=true
BENCHMARK_NAME="IndexConstructionWithRandomSetBenchmark"
java --enable-native-access=ALL-UNNAMED \
  --add-modules=jdk.incubator.vector \
  -XX:+HeapDumpOnOutOfMemoryError \
  -Xmx20G -Djvector.experimental.enable_native_vectorization=true \
  -jar benchmarks-jmh/target/benchmarks-jmh-4.0.0-beta.3-SNAPSHOT.jar $BENCHMARK_NAME
```

If you want to rerun a specific benchmark without testing the entire grid of scenarios defined in the benchmark.
You can just do the following to set M and beamWidth:
```shell
java -jar benchmarks-jmh/target/benchmarks-jmh-4.0.0-beta.3-SNAPSHOT.jar IndexConstructionWithStaticSetBenchmark -p M=32 -p beamWidth=100 
```



