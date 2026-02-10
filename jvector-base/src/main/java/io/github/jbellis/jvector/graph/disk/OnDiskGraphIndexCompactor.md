<!--
  ~ Copyright DataStax, Inc.
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~ http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
  -->

# OnDiskGraphIndex Compaction


The `aaron/compaction` branch introduces `OnDiskGraphIndexCompactor`, a tool for merging and optimizing multiple on-disk index segments into a single, efficient index.

## Impetus for Compaction

*   **Segment Fragmentation:** In systems that write data in immutable segments (like LSM-trees), search performance degrades as the number of segments increases. Each segment must be searched independently.
*   **Stale Data:** Deleted or updated nodes in older segments continue to occupy space and waste search cycles.
*   **Unified Graph Navigability:** Simply concatenating files is insufficient. To maintain high search performance, the graph structure must be unified and re-optimized.

## Key Improvements & Benefits

1.  **Index Consolidation:** Merges multiple `OnDiskGraphIndex` instances into a single file, reducing I/O overhead and improving query latency.
2.  **Space Reclamation:** Filters out deleted nodes during the compaction process, resulting in a smaller, more dense index.
3.  **Vamana-Style Re-Optimization:** Uses a diversity-aware edge selection heuristic (`CompactVamanaDiversityProvider`) to reconstruct connections. This ensures the merged graph remains high-quality and efficiently navigable.
4.  **Hierarchical Reconstruction:** Automatically rebuilds the upper layers of the HNSW structure for the unified node set, ensuring optimal entry-point navigation.
5.  **Direct Disk Serialization:** Employs `CompactWriter` to stream the new index directly to disk, avoiding the need to load the entire merged graph into heap memory during the final write phase.
6.  **Integration with Sparse Writing:** Leverages `writeSparseLevels` to persist the new hierarchy, ensuring that Version 6+ optimizations (like fused features for fast upper-layer scoring) are preserved in the compacted output.

## Detailed Workflow

The compaction process is executed in four distinct stages to ensure both memory efficiency and high graph quality.

### Stage 1: Initialization & Node Selection
*   **Segment Validation:** Verify that all source indexes share the same vector dimension and similarity function.
*   **Global Ordinal Mapping:** Each source index is assigned an `OrdinalMapper` to translate its local node IDs into a new, continuous global ID space (reclaiming space from deleted nodes).
*   **Stochastic Level Assignment:** For every "live" node in the source segments, a target graph level is assigned using a logarithmic distribution (HNSW-style). Nodes assigned to Level 1 or higher are added to the `upperLayerNodeList`.

### Stage 2: Constructing the Upper Layer Hierarchy
*   **In-Memory Construction:** An `OnHeapGraphIndex` is initialized to hold the sparse upper layers (Level 1 to `maxLevel`).
*   **Incremental Building:** The compactor iterates through the `upperLayerNodeList`. For each node, it performs a search against the *already built* portion of the upper layers to find its neighbors, maintaining the navigable small-world property.
*   **Degree Enforcement:** Once all upper-layer nodes are added, the compactor enforces the maximum degree constraints for each layer, ensuring the metadata remains consistent.

### Stage 3: Base Layer (Level 0) Search & Serialization
*   **Parallel Candidate Discovery:** The compactor iterates through all live nodes across all source segments. For each node, it initiates a parallel search (using a `ForkJoinPool`) across **all** source indexes simultaneously.
*   **Cross-Segment Neighbor Selection:**
    *   Candidates are gathered from all contributing segments.
    *   The `CompactVamanaDiversityProvider` applies the Vamana diversity heuristic to the combined candidate pool. This ensures that the new neighbor list for the compacted node is diverse and provides high reachability across the entire merged dataset.
*   **Asynchronous Writing:** Results are collected into an `inFlight` queue and written to disk sequentially via `CompactWriter.writeInlineNode` to maintain I/O efficiency while the CPU is busy searching other nodes.

### Stage 4: Finalization & Metadata Persistence
*   **Sparse Level Writing:** The `CompactWriter` invokes `writeSparseLevels`, which serializes the in-memory upper layers (constructed in Stage 2) to the file.
*   **Feature Integration:** Any additional features (like PQ-compressed vectors or separated features) are processed and written.
*   **Header/Footer Synchronization:** The file header is updated with the final node counts and layer offsets, and the footer is written (for Version 5+ compatibility), allowing the new index to be opened as a standard `OnDiskGraphIndex`.
