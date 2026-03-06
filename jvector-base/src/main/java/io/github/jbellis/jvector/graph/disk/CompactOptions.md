# CompactOptions

`CompactOptions` configures how multiple `OnDiskGraphIndex` sources are compacted into a single output index.

The design separates configuration into **three independent concepts**:

1. **Output Format** – what features the output index stores  
2. **Compaction Precision** – how compaction evaluates candidate neighbors internally  
3. **PQ Configuration** – where PQ resources come from and what PQ artifacts are produced  

---

# 1. Output Format

Output format determines **what the resulting index stores**.

```java
EnumSet<FeatureId> writeFeatures;
```

Typical values:

| Feature | Meaning |
|---|---|
| `INLINE_VECTORS` | Store full vectors inline in the index |
| `FUSED_PQ` | Store fused PQ codes alongside neighbors |

Example:

```java
EnumSet.of(FeatureId.INLINE_VECTORS)
EnumSet.of(FeatureId.INLINE_VECTORS, FeatureId.FUSED_PQ)
```

Important:

- `writeFeatures` controls **output storage only**
- It **does not control how compaction selects neighbors**

For example, compaction may use compressed vectors internally even if the output stores only full vectors.

---

# 2. Compaction Precision

Compaction precision determines **how candidate neighbors are evaluated during compaction**.

```java
enum CompactionPrecision {
    EXACT,
    COMPRESSED
}
```

## EXACT

Compaction evaluates neighbors using **full vectors**.

Properties:

- highest quality
- highest memory usage
- does **not require PQ**

---

## COMPRESSED

Compaction evaluates neighbors using **PQ-encoded vectors**.

Properties:

- lower memory footprint
- faster candidate scoring
- requires PQ resources

---

# 3. PQ Configuration

`PQConfig` defines **how PQ resources are obtained and what PQ-related artifacts should be produced**.

PQ configuration describes **PQ availability**, not compaction policy.

```java
CompactOptions.PQConfig
```

PQ resources consist of two components:

1. **PQ codebook** – the `ProductQuantization` model
2. **PQ vectors** – the encoded vectors (`PQVectors`)

---

# 3.1 PQ Codebook Source

The PQ codebook can be obtained in two ways.

## Provide an Existing Codebook

If a global PQ model already exists, it can be provided directly:

```java
PQConfig.withPQCodebook(ProductQuantization pq)
```

Example:

```java
PQConfig.withPQCodebook(pq)
```

This is typically used when:

- the system already maintains a global PQ model
- PQ vectors were precomputed offline
- compaction should reuse an existing quantization model

---

## Resolve Codebook From Source Indexes

The codebook can also be selected from the source indexes:

```java
PQConfig.withPQCodebook(PQSourcePolicy policy)
```

Available policies:

```java
enum PQSourcePolicy {
    LARGEST_LIVE,
    FIRST
}
```

| Policy | Behavior |
|---|---|
| LARGEST_LIVE | use the source with the most live nodes |
| FIRST | use the first source index |

Example:

```java
PQConfig.withPQCodebook(PQSourcePolicy.LARGEST_LIVE)
```

---

# 3.2 Providing PQ Vectors

If PQ vectors are already available for the entire dataset, they can be provided directly:

```java
pqConfig.withPQVectors(PQVectors pqVectors)
```

Example:

```java
PQConfig.withPQCodebook(pq)
        .withPQVectors(globalPQVectors)
```

Requirements:

- `PQVectors` must cover **all nodes across all sources**
- Index should be based on the new ordinals (ordinals where remappers map to).
- vectors must be encoded using the **same PQ codebook**

Providing PQ vectors allows compaction to:

- avoid loading full vectors
- reduce memory usage during compaction

---

# 3.3 Writing PQ Vectors

PQ vectors can optionally be written to a separate file:

```java
pqConfig.withPQVectorsOutput(Path outputPath)
```

Example:

```java
PQConfig.withPQCodebook(pq)
        .withPQVectorsOutput(path)
```

This produces a standalone PQ vector file alongside the compacted index.

---

# Relationship Between the Three Concepts

The three configuration dimensions interact as follows:

| Concept | Responsibility |
|---|---|
| Output Format | defines what the compacted index stores |
| Compaction Precision | defines how neighbors are evaluated |
| PQ Configuration | defines PQ resource availability |

---

# Example Configurations

## Exact Compaction With FusedPQ Output

```java
CompactOptions opts = CompactOptions.builder()
    .writeFeatures(EnumSet.of(INLINE_VECTORS, FUSED_PQ))
    .precision(CompactionPrecision.EXACT)
    .pqConfig(
        PQConfig.withPQCodebook(pq)
    )
    .build();
```

Behavior:

- compaction uses **full vectors**
- output index stores **FusedPQ**
- caller provides PQ codebook
---

## Memory-Efficient Compaction

```java
CompactOptions opts = CompactOptions.builder()
    .writeFeatures(EnumSet.of(INLINE_VECTORS))
    .precision(CompactionPrecision.COMPRESSED)
    .pqConfig(
        PQConfig.withPQCodebook(pq)
                .withPQVectors(globalPQVectors)
    )
    .build();
```

Behavior:

- compaction uses **PQ vectors**
- output index stores **only full vectors**

---

# Summary

`CompactOptions` separates configuration into three orthogonal dimensions:

| Dimension | Purpose |
|---|---|
| Output Format | what the compacted index stores |
| Compaction Precision | how neighbors are evaluated |
| PQ Configuration | where PQ resources come from |

This separation ensures that:

- output format does not implicitly control compaction behavior
- PQ resources can be reused flexibly
- compaction can be tuned for **accuracy, memory usage, or performance** based on user's requirements.
