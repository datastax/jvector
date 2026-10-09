### Optional Batched Access to File-Backed Input Vectors

**Description**

Add optional `BatchedVectorValues` range and selection cursors alongside the existing
`RandomAccessVectorValues` contract. Algorithms describe the vectors they need;
sources control batching and read-ahead. `VectorAccess` utilities preserve ordinary
point-access fallback. PQ training and PQ/NVQ encoding can use this capability without
changing their public quantization calls or requiring existing integrations to adopt it.

**How to Enable**

Supply `FvecFileVectorValues.open(path)` as the input source. Immutable `Options`
customize I/O workers, batch size, read-ahead and the shared payload budget. I/O workers
default to JVector's physical-core compute sizing used by Grid, on a separate owned
pool for blocking reads. The default
budget is at most 1% of file bytes or 64 MiB, with one record as the minimum. See
[Batched input vector access](../../batched-vector-access.md) for runnable 100k examples.

**Notes**

Files must remain immutable. Cursor vectors can be borrowed; copy before retaining
them and close cursors before their source. Buffer limits cover source payloads,
not retained training vectors, graph data, or OS page cache. This input-source feature
does not change stored graph formats or graph readers/writers. Sparse selections
preserve request order and duplicates; PQ sample policy and centroid learning are unchanged.
