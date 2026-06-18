### Parallel Graph Index Construction

**Description**  
OnDiskParallelGraphIndexWriter significantly accelerates graph index construction by addressing the disk I/O bottleneck that limits the serial OnDiskGraphIndexWriter. This implementation uses asynchronous file I/O with multiple worker threads to write graph records in parallel, with parallelism automatically determined by available system resources (or configurable via builder options). By parallelizing both record building and disk writes while maintaining correct ordering, this approach dramatically reduces the time required to persist large graph indexes to disk.

**Purpose / Impact**
- Eliminates i/o bottleneck in on disk graph construction
- Maintains backwards compatibility for existing clients of the JVector library

**How to Enable**  
To enable parallel graph index writes, simply use `OnDiskParallelGraphIndexWriter.Builder` instead of `OnDiskGraphIndexWriter.Builder`:

**Basic usage (uses default parallelism based on available processors):**
```java
try (var writer = new OnDiskParallelGraphIndexWriter.Builder(graph, outputPath)
        .with(features...)
        .build()) {
    writer.write(featureSuppliers);
}
```

**Advanced configuration:**
```java
try (var writer = new OnDiskParallelGraphIndexWriter.Builder(graph, outputPath)
        .with(features...)
        .withParallelWorkerThreads(8)           // Optional: specify thread count (0 = auto)
        .withParallelDirectBuffers(true)        // Optional: use direct ByteBuffers for better performance
        .build()) {
    writer.write(featureSuppliers);
}
```

The parallel writer is a drop-in replacement for the standard writer with the same API, automatically leveraging multiple threads and asynchronous I/O to accelerate the write process.

**Notes**
- Currently still marked as @experimental
- Includes deprecation of method
```java 
public synchronized void writeInline(int ordinal, Map<FeatureId, Feature.State> stateMap)
``` 
in favor of the more descriptive method
```java
public synchronized void writeFeaturesInline(int ordinal, Map<FeatureId, Feature.State> stateMap)
```

**Related Issues**
- [579](https://github.com/datastax/jvector/issues/579)

