# CompactOptions Usage Guide

`CompactOptions` controls how `OnDiskGraphIndexCompactor` writes the output index during compaction. It determines:

- What **features** are written into the index (e.g., inline vectors, FusedPQ).
- Whether **PQ vectors** are written to a separate file.
- Where the **PQ codebook** comes from.
- Parallelism settings (task window size and executor).

Most users should rely on the provided **convenience factory methods**, which cover the most common configurations.

---

# Basic Workflow

Typical compaction workflow:

```java
OnDiskGraphIndexCompactor compactor =
        new OnDiskGraphIndexCompactor(List.of(sourceIndex1, sourceIndex2));

CompactOptions opts = CompactOptions.withInlineVectors();

compactor.compact(
        outputPath,
        VectorSimilarityFunction.DOT_PRODUCT,
        opts
);
```

---

# Convenience Configurations

## 1. Basic Inline Vectors

Stores full vectors inline in the graph index.

```java
CompactOptions opts = CompactOptions.withInlineVectors();
```

Usage:

```java
compactor.compact(
        outputPath,
        VectorSimilarityFunction.DOT_PRODUCT,
        CompactOptions.withInlineVectors()
);
```

Result:

```
index/
 └── graph.bin
```

Vectors are stored directly in the graph records.

---

## 2. PQ Vectors Stored Separately

Stores inline vectors in the graph but writes **PQ-compressed vectors** into a separate file.

```java
ProductQuantization pq = ...;
Path pqVectorsFile = Path.of("index.pq");

CompactOptions opts =
        CompactOptions.withPQVectorsSeparate(pq, pqVectorsFile);
```

Usage:

```java
compactor.compact(
        outputPath,
        VectorSimilarityFunction.DOT_PRODUCT,
        opts
);
```

Result:

```
index/
 ├── graph.bin
 └── index.pq
```

The `.pq` file follows the **PQVectors format** and can be loaded with:

```java
PQVectors.load(reader);
```

---

## 3. FusedPQ

Stores PQ codes **directly inside graph neighbor lists** for faster approximate search.

```java
CompactOptions opts = CompactOptions.withFusedPQ();
```

Usage:

```java
compactor.compact(
        outputPath,
        VectorSimilarityFunction.DOT_PRODUCT,
        opts
);
```

Result:

```
index/
 └── graph.bin
```

Each node contains **FusedPQ neighbor encoding**.

---

# Advanced Configuration

If more control is needed, use the builder API.

Example:

```java
CompactOptions opts = CompactOptions.builder()
        .writeFeatures(EnumSet.of(
                FeatureId.INLINE_VECTORS,
                FeatureId.FUSED_PQ
        ))
        .pqConfig(PQConfig.fromSources(
                PQConfig.PQSourcePolicy.AUTO
        ))
        .taskWindowSize(32)
        .build();
```

Usage:

```java
compactor.compact(outputPath, similarityFunction, opts);
```

---

# Parallelism Configuration

`CompactOptions` can control compaction parallelism.

Example:

```java
CompactOptions opts = CompactOptions.builder()
        .writeFeatures(EnumSet.of(FeatureId.INLINE_VECTORS))
        .taskWindowSize(32)
        .executor(new ForkJoinPool(16))
        .build();
```

Behavior:

- If `executor` is provided → compactor uses it and does not shutdown.
- Otherwise → a `ForkJoinPool` is created automatically.
- Thread count is limited by available processors.

---

# PQ Codebook Sources

When PQ is used, the codebook must be provided or selected.

Available options:

| Mode | Description |
|-----|-------------|
| `PROVIDED` | PQ codebook supplied by caller |
| `FROM_SOURCES` | Select codebook from existing indices |
| `NONE` | PQ disabled |

Example:

```java
PQConfig.fromSources(PQConfig.PQSourcePolicy.AUTO)
```

Policies:

| Policy | Behavior |
|------|---------|
| `FIRST` | Use first source index |
| `LARGEST_LIVE` | Choose source with most live nodes |
| `AUTO` | Let compactor decide |

---

# Example: Full Compaction Pipeline

```java
OnDiskGraphIndex index1 = OnDiskGraphIndex.load(path1);
OnDiskGraphIndex index2 = OnDiskGraphIndex.load(path2);

OnDiskGraphIndexCompactor compactor =
        new OnDiskGraphIndexCompactor(List.of(index1, index2));

ProductQuantization pq =
        ProductQuantization.load(new RandomAccessReader(pqFile));

CompactOptions opts =
        CompactOptions.withPQVectorsSeparate(pq, Path.of("merged.pq"));

compactor.compact(
        Path.of("merged-index"),
        VectorSimilarityFunction.DOT_PRODUCT,
        opts
);
```

Result:

```
merged-index/
 ├── graph.bin
 └── merged.pq
```

# Summary

`CompactOptions` allows flexible control over compaction output while keeping common configurations simple.

Most users should start with one of the convenience methods:

```java
CompactOptions.withInlineVectors()
CompactOptions.withPQVectorsSeparate(...)
CompactOptions.withFusedPQ()
```

and only use the builder API for advanced tuning.
