## Bug Fixes and Issue Resolutions

### Fix: NullPointerException in `OnDiskGraphIndex#ramBytesUsed`

**Problem**  
Now that we lazily load the inMemoryNeighbors and the inMemoryFeatures, we need to handle the case in `OnDiskGraphIndex` where they are null or have values that are null when the `ramBytesUsed()` method is called.

**Resolution**  
Added appropriate null checks and safeguards to ensure `ramBytesUsed()` can be safely invoked in all valid states.

**Related Issues**
- [#586](https://github.com/datastax/jvector/issues/586)

---

### Fix: Protection Against Invalid Ordinal Mappings

**Problem**  
JVector relies on the calling source code to pass in ordinal maps constructed outside of the JVector library. Improper or inconsistent ordinal mappings can lead to failures when the Graph is built or incorrect indexing or search results.

**Resolution**  
Added safeguards to detect invalid ordinal mappings.

**Notes**
Full validation of ordinal mapping requires iterating over the entire set of ordinals and can be a costly operation. This safeguard will only be activated if debug logging is enabled or if `System.getProperties().containsKey("VECTOR_DEBUG")`

**Related Issues**
- [568](https://github.com/datastax/jvector/issues/568)

---


### Fix: extractTrainingVectors may produce more than MAX_PQ_TRAINING_SET_SIZE vectors

**Problem**  
`extractTrainingVectors` could return more vectors than the intended maximum (`MAX_PQ_TRAINING_SET_SIZE`), leading to excessive memory usage during PQ training.

**Resolution**  
Uses floyd's random sampling algorithm to select random training vectors from the RandomAccessVectorValues. The solution has two phases. The first is to select MAX_PQ_TRAINING_SET_SIZE random ordinals. Then, it maps those ordinals to vectors.

**Related Issues**
- [590](https://github.com/datastax/jvector/issues/590)

---
