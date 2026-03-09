# CompactOptions

`CompactOptions` controls how multiple `OnDiskGraphIndex` sources are compacted into a single output index.

The configuration is divided into **three independent concepts**:

1. **Output Format** – what features the resulting index stores  
2. **Compaction Precision** – how neighbors are evaluated during compaction  
3. **PQ Configuration** – where PQ resources come from and whether PQ vectors should be written

Separating these concerns keeps the API predictable and avoids hidden interactions between **index storage**, **compaction behavior**, and **quantization resources**.

---

# 1. Output Format

Output format determines **what the resulting index stores**.

```java
EnumSet<FeatureId> writeFeatures;
```

Currently supported features:

| Feature | Meaning |
|---|---|
| `INLINE_VECTORS` | Store full vectors inline in the index |
| `FUSED_PQ` | Store fused PQ codes alongside neighbors |

Example:

```java
EnumSet.of(FeatureId.INLINE_VECTORS)
EnumSet.of(FeatureId.INLINE_VECTORS, FeatureId.FUSED_PQ)
```

## Requirement

`INLINE_VECTORS` must currently always be present.

```java
writeFeatures.contains(INLINE_VECTORS) == true
```

This ensures that:

- the compacted index always contains full vectors
- future index transformations remain safe

Indexes storing **only `FUSED_PQ`** are not currently supported.

---

# 2. Compaction Precision

Compaction precision determines **how candidate neighbors are evaluated during compaction**.

```java
enum CompactionPrecision {
    EXACT,
    COMPRESSED
}
```

---

## EXACT

Compaction evaluates neighbors using **full vectors**.

Properties:

- highest accuracy
- higher memory usage
- does not require PQ resources

Example:

```java
precision = EXACT
```

---

## COMPRESSED

Compaction evaluates neighbors using **PQ-encoded vectors**.

Properties:

- significantly lower memory footprint
- faster candidate scoring
- requires compressed vectors

Compressed vectors may come from:

- caller-provided `PQVectors`
- source indexes containing PQ features

Example:

```java
precision = COMPRESSED
```

---

# 3. PQ Configuration

`CompressionConfig` specifies **where PQ resources come from** and **whether PQVectors should be written**.

Exactly **one PQ resource source** may be configured.

```java
CompactOptions.CompressionConfig
```

Possible PQ resource sources:

| Method | Meaning |
|---|---|
| `withPQVectors(pqVectors)` | Use caller-provided PQ vectors |
| `withPQCodebook(pq)` | Use provided PQ codebook for encoding |
| `withSourcePQ(policy)` | Retrieve PQ resources from source indexes |
| `none()` | No PQ resources required |

These options are **mutually exclusive**.

---

# 3.1 Using Caller-Provided PQVectors

```java
CompressionConfig.withPQVectors(PQVectors pqVectors)
```

Meaning:

- PQ vectors are already available
- the PQ codebook is retrieved from `pqVectors.getCompressor()`
- compaction may use these vectors for compressed search

This configuration supports **compressed compaction**.

Example:

```java
CompactOptions opts = CompactOptions.builder()
    .writeFeatures(EnumSet.of(INLINE_VECTORS))
    .precision(COMPRESSED)
    .compressionConfig(CompressionConfig.withPQVectors(pqVectors))
    .build();
```

---

# 3.2 Using a Provided PQ Codebook

```java
CompressionConfig.withPQCodebook(ProductQuantization pq)
```

or

```java
CompressionConfig.withPQCodebook(ProductQuantization pq, Path pqVectorsOutputPath)
```

Meaning:

- the caller provides the PQ model
- the compactor may use it to **encode PQ-based output features**

Typical uses:

- encoding `FUSED_PQ` output
- generating PQVectors during compaction

If `pqVectorsOutputPath` is provided, the compactor will also **write PQVectors to a separate file**.

Example:

```java
compressionConfig = CompressionConfig.withPQCodebook(pq)
```

or

```java
compressionConfig = CompressionConfig.withPQCodebook(pq, pqVectorsOutputPath)
```

Note:

This configuration **does not by itself enable compressed compaction**, since compressed vectors are not yet available.

---

# 3.3 Reusing PQ From Source Indexes

```java
CompressionConfig.withSourcePQ(PQSourcePolicy policy)
```

or

```java
CompressionConfig.withSourcePQ(PQSourcePolicy policy, Path pqVectorsOutputPath)
```

Meaning:

- PQ resources are derived from the source indexes
- both codebook and compressed search capability may be reused

If `pqVectorsOutputPath` is provided, the compactor will also **emit PQVectors**.

Example:

```java
pqConfig = PQConfig.withSourcePQ(PQSourcePolicy.LARGEST_LIVE)
```

---

## PQSourcePolicy

Policies control how PQ resources are selected from the sources.

```java
enum PQSourcePolicy {
    AUTO,
    LARGEST_LIVE,
    FIRST
}
```

| Policy | Behavior |
|---|---|
| `AUTO` | automatically choose a suitable source |
| `LARGEST_LIVE` | use the source with the most live nodes |
| `FIRST` | use the first source index |

---

# 4. Relationship Between the Three Concepts

The three configuration dimensions are independent:

| Dimension | Responsibility |
|---|---|
| Output Format | defines what the compacted index stores |
| Compaction Precision | defines how neighbors are evaluated |
| PQ Configuration | defines where PQ resources come from |

This separation ensures that:

- output format does not implicitly change compaction behavior
- PQ resources can be reused flexibly
- compaction can be tuned for **accuracy**, **memory usage**, or **performance**

---

# 5. Example Configurations

---

## Exact Compaction With Inline Output

```java
CompactOptions opts = CompactOptions.builder()
    .writeFeatures(EnumSet.of(INLINE_VECTORS))
    .precision(EXACT)
    .pqConfig(PQConfig.none())
    .build();
```

---

## Exact Compaction With FusedPQ Output

```java
CompactOptions opts = CompactOptions.builder()
    .writeFeatures(EnumSet.of(INLINE_VECTORS, FUSED_PQ))
    .precision(EXACT)
    .pqConfig(PQConfig.withPQCodebook(pq))
    .build();
```

---

## Compressed Compaction Using PQVectors

```java
CompactOptions opts = CompactOptions.builder()
    .writeFeatures(EnumSet.of(INLINE_VECTORS))
    .precision(COMPRESSED)
    .pqConfig(PQConfig.withPQVectors(pqVectors))
    .build();
```

---

## Compressed Compaction Using Source PQ

```java
CompactOptions opts = CompactOptions.builder()
    .writeFeatures(EnumSet.of(INLINE_VECTORS))
    .precision(COMPRESSED)
    .pqConfig(PQConfig.withSourcePQ(PQSourcePolicy.AUTO))
    .build();
```

---

## Compressed Compaction With FusedPQ Output

```java
CompactOptions opts = CompactOptions.builder()
    .writeFeatures(EnumSet.of(INLINE_VECTORS, FUSED_PQ))
    .precision(COMPRESSED)
    .pqConfig(PQConfig.withSourcePQ(PQSourcePolicy.AUTO))
    .build();
```

---

# Summary

`CompactOptions` separates compaction configuration into three orthogonal dimensions:

| Concept | Purpose |
|---|---|
| Output Format | what the compacted index stores |
| Compaction Precision | how neighbors are evaluated |
| PQ Configuration | where PQ resources come from |

Key rules:

- `INLINE_VECTORS` must currently be included in output features
- only **one PQ resource source** may be configured
- compressed compaction requires compressed vectors
- PQ configuration does not implicitly change output format

This design keeps the compaction pipeline predictable and easy to extend.
