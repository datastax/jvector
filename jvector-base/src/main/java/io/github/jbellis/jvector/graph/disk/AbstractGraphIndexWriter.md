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

# AbstractGraphIndexWriter: `writeSparseLevels` Guide


The `writeSparseLevels` method is a critical component for persisting hierarchical graph indexes (HNSW) to disk. It specifically handles writing the upper layers (Level 1 and above) of the graph, which are sparser than the base layer (Level 0).

## 1. Overview of `writeSparseLevels`

This method iterates through all levels of the graph starting from Level 1 up to the maximum level. For each level, it writes the graph structure (connectivity) and, in newer versions (v6+), specific feature data required for search.

**Signature:**
```java
void writeSparseLevels(ImmutableGraphIndex.View view, Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException
```

**What it writes (per level):**
1.  **Graph Structure:**
    *   Iterates through all nodes in the current level.
    *   **Node ID:** Writes the remapped ordinal (new ID).
    *   **Neighbors:** Writes the count of neighbors, followed by the neighbor ordinals.
    *   **Padding:** Pads the remaining space with `-1` up to the layer's maximum degree (`maxDegree`).

2.  **Fused Features (Version 6+ only):**
    *   If the graph format version is 6 or higher, it performs an additional pass to write "fused features" (e.g., quantized vectors used for fast approximate scoring).
    *   **Optimization:** Since Level 1 contains all nodes present in higher levels (Level 2, 3, etc.), it only writes these features for nodes in Level 1.
    *   **Data Layout:** For each node in Level 1, it writes the node ID and then the fused feature data (via `fusedFeature.writeSourceFeature`).

---

## 2. Graph Format Versions

The behavior of the writer and the structure of the output file depend heavily on the `version` parameter.

| Version | Key Characteristics |
| :--- | :--- |
| **< 3** | Only supports `INLINE_VECTORS`. |
| **< 4** | Does **not** support multilayer (hierarchical) graphs. Throws exception if `maxLevel > 0`. |
| **4** | Introduced support for multilayer graphs. |
| **5** | Introduced the **Footer** metadata block (replacing the header for metadata storage), allowing append-only writing. Uses old feature ordering. |
| **6 (Current)** | **Fused Features:** Adds support for `FusedFeature` (e.g., Fused PQ). <br> **Feature Ordering:** Reorders features so fused features come last. <br> **Sparse Levels Change:** Writes fused feature data block after the graph structure in `writeSparseLevels`. |

---

## 3. Callers and Usage Context

`writeSparseLevels` is designed to be called by specific implementations of `AbstractGraphIndexWriter` during the serialization process.

### A. `OnDiskGraphIndexWriter`
The standard writer for creating on-disk indexes. Supports random access and parallel writing.
1. `writeHeader`: Writes a placeholder header.
2. `writeL0Records`: Writes Level 0 (dense layer) containing all nodes and their inline features.
3. **`writeSparseLevels`**: Writes Level 1+ structure and v6+ fused features.
4. `writeSeparatedFeatures`: Writes features stored separately.
5. `writeFooter` (v5+): Writes metadata at the end of the file.
6. Updates the header at the beginning of the file with correct offsets.

### B. `OnDiskSequentialGraphIndexWriter`
Designed for append-only environments (e.g., Object Storage, streaming) where seeking back to update a header is impossible or expensive.
1. `writeHeader`: Writes the header once at the start.
2. `writeL0Records`: Writes Level 0 sequentially.
3. **`writeSparseLevels`**: Writes Level 1+ structure.
4. `writeSeparatedFeatures`: Writes separated features.
5. `writeFooter`: Writes metadata at the end.

### C. `OnDiskGraphIndexCompactor`
Used to compact and optimize existing indexes. Calls `writeSparseLevels` as part of the compaction process to rewrite the graph structure efficiently.

---

## 4. Key Logic Summary

*   **Data Locality:**
    *   **Level 0:** Stored with inline features (vectors) for cache locality during the initial search phase.
    *   **Levels 1+:** Stored densely in `writeSparseLevels`. In Version 6, the "compressed" representation (Fused Feature) needed for upper-layer navigation is stored here to avoid seeking back to Level 0.

*   **Version Impact:**
    *   In **Version 5 or lower**, `writeSparseLevels` strictly writes the graph connectivity (IDs + Neighbors).
    *   In **Version 6**, it *also* writes the quantized/fused representation of vectors for the upper layers to optimize search speed.
