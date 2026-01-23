# Parallel Writer Direct-Write Architecture Design

## Overview
This document describes a new architecture for `OnDiskParallelGraphIndexWriter` that eliminates the `featuresPreWritten` flag by having tasks write directly to the async file channel, similar to how the sequential writer operates.

## Current Architecture Problems

### Current Flow
1. `NodeRecordTask` builds complete records in memory (ByteBuffer)
2. Returns `List<Result>` containing ByteBuffers and file offsets
3. `ParallelGraphWriter.writeRecordsAsync()` writes these buffers to disk
4. Requires `featuresPreWritten` flag to handle pre-written features differently

### Issues
- **Complexity**: Separate code paths for pre-written vs. normal features
- **State Management**: Flag must be maintained and passed through multiple layers
- **Inconsistency**: Different logic than sequential writer for same scenario
- **Memory**: Builds entire records in memory before writing

## Proposed Architecture

### New Flow
1. `NodeRecordTask` writes directly to `AsynchronousFileChannel`
2. For each ordinal, calculates file offsets and writes:
   - Ordinal at offset
   - Each feature (if supplier != null) at its offset
   - Neighbors at their offset
3. Returns `List<Future<Integer>>` (write completion futures)
4. `ParallelGraphWriter` waits for all write futures to complete

### Key Benefits
- **Eliminates flag**: Just check if supplier is null (like sequential writer)
- **Consistency**: Same logic as `OnDiskGraphIndexWriter.writeL0Records()`
- **Simplicity**: Single code path for all cases
- **Memory efficiency**: No need to buffer entire records

## Detailed Design

### 1. NodeRecordTask Changes

#### Current Signature
```java
class NodeRecordTask implements Callable<List<NodeRecordTask.Result>> {
    private final ByteBuffer buffer;
    private final boolean featuresPreWritten;
    
    NodeRecordTask(..., ByteBuffer buffer, boolean featuresPreWritten) { }
    
    public List<Result> call() {
        // Build records in buffer
        // Return List<Result> with buffers
    }
}
```

#### New Signature
```java
class NodeRecordTask implements Callable<List<Future<Integer>>> {
    private final AsynchronousFileChannel fileChannel;
    // Remove: buffer, featuresPreWritten
    
    NodeRecordTask(..., AsynchronousFileChannel fileChannel) { }
    
    public List<Future<Integer>> call() {
        // Write directly to fileChannel
        // Return list of write futures
    }
}
```

#### New call() Implementation Logic
```java
public List<Future<Integer>> call() throws Exception {
    List<Future<Integer>> writeFutures = new ArrayList<>();
    
    for (int newOrdinal = startOrdinal; newOrdinal < endOrdinal; newOrdinal++) {
        var originalOrdinal = ordinalMapper.newToOld(newOrdinal);
        long recordOffset = baseOffset + (long) newOrdinal * recordSize;
        long currentOffset = recordOffset;
        
        // Write ordinal
        ByteBuffer ordinalBuf = ByteBuffer.allocate(Integer.BYTES);
        ordinalBuf.order(ByteOrder.BIG_ENDIAN);
        ordinalBuf.putInt(newOrdinal);
        ordinalBuf.flip();
        writeFutures.add(fileChannel.write(ordinalBuf, currentOffset));
        currentOffset += Integer.BYTES;
        
        // Handle OMITTED nodes
        if (originalOrdinal == OrdinalMapper.OMITTED) {
            // Write zeros for features and empty neighbor list
            // (similar to current implementation)
        } else {
            // Write inline features
            for (var feature : inlineFeatures) {
                var supplier = featureStateSuppliers.get(feature.id());
                if (supplier == null) {
                    // Feature pre-written via writeInline() - skip it
                    currentOffset += feature.featureSize();
                } else {
                    // Write feature data
                    ByteBuffer featureBuf = ByteBuffer.allocate(feature.featureSize());
                    featureBuf.order(ByteOrder.BIG_ENDIAN);
                    var writer = new ByteBufferIndexWriter(featureBuf);
                    feature.writeInline(writer, supplier.apply(originalOrdinal));
                    featureBuf.flip();
                    writeFutures.add(fileChannel.write(featureBuf, currentOffset));
                    currentOffset += feature.featureSize();
                }
            }
            
            // Write neighbors
            var neighbors = view.getNeighborsIterator(0, originalOrdinal);
            int neighborDataSize = Integer.BYTES * (1 + graph.getDegree(0));
            ByteBuffer neighborBuf = ByteBuffer.allocate(neighborDataSize);
            neighborBuf.order(ByteOrder.BIG_ENDIAN);
            
            neighborBuf.putInt(neighbors.size());
            int n = 0;
            for (; n < neighbors.size(); n++) {
                neighborBuf.putInt(ordinalMapper.oldToNew(neighbors.nextInt()));
            }
            for (; n < graph.getDegree(0); n++) {
                neighborBuf.putInt(-1);
            }
            
            neighborBuf.flip();
            writeFutures.add(fileChannel.write(neighborBuf, currentOffset));
        }
    }
    
    return writeFutures;
}
```

### 2. ParallelGraphWriter Changes

#### Current writeL0Records
```java
public void writeL0Records(..., boolean featuresPreWritten) {
    // Create tasks with ByteBuffer
    // Collect List<Result> from tasks
    // Write results to async channel
}
```

#### New writeL0Records
```java
public void writeL0Records(...) {  // No featuresPreWritten parameter
    // Open AsynchronousFileChannel
    AsynchronousFileChannel afc = AsynchronousFileChannel.open(
        filePath, 
        EnumSet.of(StandardOpenOption.WRITE, StandardOpenOption.READ),
        executor
    );
    
    // Create tasks with channel (no ByteBuffer)
    List<Future<List<Future<Integer>>>> taskFutures = new ArrayList<>();
    for (int i = 0; i < numTasks; i++) {
        // ... calculate range ...
        Future<List<Future<Integer>>> future = executor.submit(() -> {
            var view = viewPerThread.get();
            var task = new NodeRecordTask(
                start, end, ordinalMapper, graph, view,
                inlineFeatures, featureStateSuppliers,
                recordSize, baseOffset,
                afc  // Pass channel instead of buffer
            );
            return task.call();
        });
        taskFutures.add(future);
    }
    
    // Collect all write futures from all tasks
    List<Future<Integer>> allWriteFutures = new ArrayList<>();
    for (var taskFuture : taskFutures) {
        allWriteFutures.addAll(taskFuture.get());
    }
    
    // Wait for all writes to complete
    for (var writeFuture : allWriteFutures) {
        writeFuture.get();  // Blocks until write completes
    }
    
    // Close channel
    afc.close();
}
```

### 3. OnDiskParallelGraphIndexWriter Changes

#### Remove
- `private volatile boolean featuresPreWritten` field
- `writeFeaturesInline()` override method
- `featuresPreWritten` parameter from `writeL0Records()` call

#### Keep
- Everything else remains the same
- `writeFeaturesInline()` inherited from parent still works correctly

### 4. Result Class Changes

#### Current
```java
static class Result {
    final int newOrdinal;
    final long fileOffset;
    final ByteBuffer data;
}
```

#### New
```java
// Result class no longer needed - tasks return write futures directly
// Can be removed entirely
```

## Implementation Steps

1. **Phase 1**: Update NodeRecordTask
   - Change constructor to accept `AsynchronousFileChannel`
   - Remove `ByteBuffer` and `featuresPreWritten` parameters
   - Rewrite `call()` method to write directly to channel
   - Change return type to `List<Future<Integer>>`

2. **Phase 2**: Update ParallelGraphWriter
   - Remove `featuresPreWritten` parameter from `writeL0Records()`
   - Open `AsynchronousFileChannel` at start
   - Pass channel to tasks instead of buffer
   - Collect and wait for all write futures
   - Remove `writeRecordsAsync()` method (no longer needed)

3. **Phase 3**: Update OnDiskParallelGraphIndexWriter
   - Remove `featuresPreWritten` field
   - Remove `writeFeaturesInline()` override
   - Remove `featuresPreWritten` from `writeL0Records()` call

4. **Phase 4**: Cleanup
   - Remove `NodeRecordTask.Result` class
   - Remove buffer pool from `ParallelGraphWriter`
   - Update documentation

## Performance Considerations

### Potential Benefits
- **Less memory**: No buffering of complete records
- **Better parallelism**: Writes can start immediately, don't wait for all records to build
- **Simpler code**: Easier to maintain and understand

### Potential Concerns
- **More write operations**: Each feature/neighbor section is a separate write
- **Overhead**: More Future objects to manage
- **Testing needed**: Must benchmark to ensure performance is acceptable

### Mitigation Strategies
- **Batch small writes**: Combine ordinal + features into single write when possible
- **Tune executor pool**: Adjust thread pool size for optimal performance
- **Monitor**: Add metrics to track write performance

## Testing Strategy

1. **Unit Tests**: Verify correct file layout with and without pre-written features
2. **Integration Tests**: Test with real graph data
3. **Performance Tests**: Benchmark against current implementation
4. **Correctness Tests**: Verify recall rates match current implementation

## Rollback Plan

If performance is not acceptable:
- Keep both implementations
- Add configuration flag to choose between them
- Default to current implementation
- Allow opt-in to new implementation

## Success Criteria

1. ✅ `featuresPreWritten` flag completely removed
2. ✅ Code simpler and more maintainable
3. ✅ Recall rates identical to current implementation
4. ✅ Performance within 10% of current implementation (or better)
5. ✅ All existing tests pass

## Open Questions

1. **Buffer allocation**: Should we pool small buffers for ordinal/feature writes?
2. **Error handling**: How to handle partial write failures?
3. **Executor sharing**: Should file channel use same executor as task pool?
4. **Write ordering**: Do we need to ensure writes complete in order?

## Conclusion

This design eliminates the `featuresPreWritten` flag by adopting a direct-write architecture that mirrors the sequential writer's approach. The key insight is that `AsynchronousFileChannel.write(buffer, position)` provides the same "seek" capability as `RandomAccessWriter.seek()`, allowing us to skip pre-written features by simply not writing to those positions.

The implementation is straightforward but requires careful attention to:
- File offset calculations
- Buffer management
- Future collection and waiting
- Error handling

With proper implementation and testing, this should result in simpler, more maintainable code with comparable or better performance.