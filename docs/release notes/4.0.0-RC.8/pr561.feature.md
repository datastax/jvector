### Fused Product Quantization (Fused PQ)

**Description**  
Fused Product Quantization (Fused PQ) is a performance optimization that embeds compressed Product Quantization (PQ) codes directly into the graph index structure alongside each node's neighbor lists. This eliminates the need for separate lookups to retrieve compressed vectors during graph traversal, significantly improving query performance by reducing memory access overhead. The feature stores PQ-encoded neighbor vectors inline with the graph edges, enabling faster approximate similarity scoring during search operations. By embedding the compressed neighbor vectors into the graph index itself, Fused PQ eliminates the need to maintain a separate in-memory data structure for PQ-encoded vectors, reducing heap memory usage while maintaining fast approximate similarity search performance.​

**Purpose / Impact**
- Reduces memory usage for large-scale vector datasets
- Improves cache locality during graph traversal
- Enables higher writer scalability for large / high-dimensional vector workloads

**How to Enable**  
To enable Fused PQ when writing an on-disk graph index:

1. **Create the FusedPQ feature** by passing your graph's max degree and a ProductQuantization compressor to the constructor:
   ```java
   var fusedPQFeature = new FusedPQ(graph.maxDegree(), pq);
   ```

2. **Add it to your OnDiskGraphIndexWriter builder**:
   ```java
   var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
       .with(fusedPQFeature)
       .build();
   ```

3. **Provide a state supplier during the write phase** that includes the graph view and PQ vectors:
   ```java
   Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
   writeSuppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(view, pqVectors, ordinal));
   writer.write(writeSuppliers);
   ```

**Notes**
- Fused PQ requires a 256-cluster ProductQuantization compressor. The feature automatically embeds compressed neighbor vectors inline with the graph structure during the write operation.
- To enable the FUSED_PQ feature, we introduced the new version 6 file format for our graph indices.

