# Memory and disk sizing

Use these figures to estimate how much memory a build needs, how much memory a search needs, and
how large the index file will be. They are per vector; multiply by the vector count. To confirm,
build a sample of your own data; the graph writer's `describeRecordLayout()` prints the size of
each part of a record.

## Vector and code sizes

`D` is the vector dimension.

| Representation | Bytes per vector | Notes |
| --- | --- | --- |
| FP32 vector | 4 × D | The uncompressed vector |
| NVQ (reranking) | 4 + Σ over subvectors of (subvector dimensions + 28) | D + 60 with two subvectors, about a quarter of FP32 |
| PQ code | M (the number of subspaces) | 256 clusters per subspace, one byte each; the codebooks add about 4 × 256 × D bytes once |
| ASH code | `encodedBits` / 8 | Includes a 5-byte header; `encodedBits = 40 + d × b` for projected dimension `d` and `b` bits per dimension |
| BQ code | 8 × ⌈D / 64⌉ | One bit per dimension, in 64-bit words |

Compression ratio is FP32 size divided by code size: at 32×, a 1,024-dimension vector
(4,096 bytes) becomes a code of about 128 bytes. See
[ASH stored codes](ash-configuration.md#stored-codes-and-training) for choosing `d` and `b`.

## On-disk index size

Each base-layer node is one record:

| Part | Bytes |
| --- | --- |
| Node id | 4 |
| Fused codes, if fused | degree × code bytes |
| Inline reranking vector, if inline | NVQ or FP32 size from the table above |
| Neighbor count and list | 4 × (degree + 1) |

Separated reranking vectors (see [Fused and separated layouts](fused-and-separated.md)) move the
vector out of the record into its own block, with the same total size. Page-aligned records can
add padding so that no record crosses a page; see
[Page-aligned records](larger-than-memory-indexes.md#page-aligned-records). Upper graph layers and
the header add little.

## Memory

**Building.** The graph's neighbor lists live on the Java heap: about 430 bytes per node at degree
32 with neighbor overflow 1.2, growing roughly in proportion to degree × overflow. Add what
construction scores with:

- compact codes (symmetric ASH construction): code bytes per node;
- full vectors: 4 × D per node, unless they are read from a file-backed source.

Give `-Xmx` headroom above that total and leave the rest of memory to the operating system's page
cache.

**Searching an in-memory index.** The graph (as above) plus the traversal codes or vectors.

**Searching an on-disk index.** The upper layers, plus any codes held in memory for separate
traversal (code bytes per node). Fused codes and reranking vectors are read through the page
cache: the more of the index the cache holds, the fewer reads reach the device, so throughput
rises with memory even when the index does not fit.

## Worked example

10 million vectors of 1,024 dimensions, degree 32, ASH codes of about 128 bytes (32×), NVQ
reranking with two subvectors:

| Item | Per vector | Total |
| --- | --- | --- |
| FP32 vectors, for comparison | 4,096 B | 41 GB |
| ASH codes | 128 B | 1.3 GB |
| NVQ vector | 1,084 B | 10.8 GB |
| Neighbor list (4 × 33) + node id | 136 B | 1.4 GB |
| Fused codes (32 × 128) | 4,096 B | 41 GB |
| **Fused index file** (id + fused codes + NVQ + neighbors) | 5,316 B | **53 GB** before padding |
| **Separate-codes index file** (id + NVQ + neighbors) | 1,220 B | **12.2 GB**, plus 1.3 GB of codes in memory |
| **Build heap** (graph + ASH codes, symmetric construction) | about 560 B | **about 5.6 GB**, plus headroom |

The fused file is larger because every record carries its 32 neighbors' codes; in exchange, a
search holds no codes in memory. [Fused and separated layouts](fused-and-separated.md) explains
when each is the better trade.
