# Changelog

## [4.0.0-beta.4](https://github.com/datastax/jvector/tree/4.0.0-beta.4) (2025-04-15)

[Full Changelog](https://github.com/datastax/jvector/compare/4.0.0-beta.3...4.0.0-beta.4)

**Merged pull requests:**

- Release 4.0.0-beta.4 [\#457](https://github.com/datastax/jvector/pull/457) ([tlwillke](https://github.com/tlwillke))
- Creating starting point for changelog tracking. [\#456](https://github.com/datastax/jvector/pull/456) ([tlwillke](https://github.com/tlwillke))
- Adjust changelog generation steps. [\#437](https://github.com/datastax/jvector/pull/437) ([msmygit](https://github.com/msmygit))
- Fix minor bug in getNodes. [\#434](https://github.com/datastax/jvector/pull/434) ([marianotepper](https://github.com/marianotepper))
- Fix/refactor NodeScoreIterator, BoundedLongHeap, and GrowableLongHeap bulk addition implementations [\#433](https://github.com/datastax/jvector/pull/433) ([michaeljmarshall](https://github.com/michaeljmarshall))
- Remove extra prefix of v from tag version [\#432](https://github.com/datastax/jvector/pull/432) ([msmygit](https://github.com/msmygit))
- Update only the root level pom.xml as part of the GHA workflow [\#431](https://github.com/datastax/jvector/pull/431) ([msmygit](https://github.com/msmygit))
- Eliminate maven-resources-plugin warning [\#429](https://github.com/datastax/jvector/pull/429) ([msmygit](https://github.com/msmygit))

## [4.0.0-beta.3](https://github.com/datastax/jvector/tree/4.0.0-beta.3) (2025-04-09)

[Full Changelog](https://github.com/datastax/jvector/compare/4.0.0-beta.2...4.0.0-beta.3)

**Implemented enhancements:**

- Implement NodeQueue\#pushAll and AbstractLongHeap\#addAll [\#415](https://github.com/datastax/jvector/pull/415) ([michaeljmarshall](https://github.com/michaeljmarshall))

**Merged pull requests:**

- Release 4.0.0-beta.3 [\#427](https://github.com/datastax/jvector/pull/427) ([marianotepper](https://github.com/marianotepper))
- Fix calls to deprecated GraphIndex.size\(\) [\#426](https://github.com/datastax/jvector/pull/426) ([marianotepper](https://github.com/marianotepper))
- Fix NPE in GraphIndexBuilder.load [\#425](https://github.com/datastax/jvector/pull/425) ([marianotepper](https://github.com/marianotepper))
- Merge latest commits from hnsw-3 [\#423](https://github.com/datastax/jvector/pull/423) ([marianotepper](https://github.com/marianotepper))
- Update test resume [\#422](https://github.com/datastax/jvector/pull/422) ([marianotepper](https://github.com/marianotepper))
- Fix native implementations of PQ assembleAndSum and pqDecodedCosineSimilarity [\#420](https://github.com/datastax/jvector/pull/420) ([jkni](https://github.com/jkni))
- Reduce the number of vector allocations in BuildScoreProvider.pqBuilderScoreProvider [\#419](https://github.com/datastax/jvector/pull/419) ([marianotepper](https://github.com/marianotepper))
- Fix FusedADC.writeInline [\#417](https://github.com/datastax/jvector/pull/417) ([marianotepper](https://github.com/marianotepper))
- Rework the computation of accuracy [\#408](https://github.com/datastax/jvector/pull/408) ([marianotepper](https://github.com/marianotepper))

## [4.0.0-beta.2](https://github.com/datastax/jvector/tree/4.0.0-beta.2) (2025-04-02)

[Full Changelog](https://github.com/datastax/jvector/compare/4.0.0-beta.1...4.0.0-beta.2)

**Merged pull requests:**

- Release 4.0.0-beta.2 [\#414](https://github.com/datastax/jvector/pull/414) ([marianotepper](https://github.com/marianotepper))
- Count expanded nodes [\#406](https://github.com/datastax/jvector/pull/406) ([marianotepper](https://github.com/marianotepper))
- Search pruning & fix the reported number of visited nodes [\#405](https://github.com/datastax/jvector/pull/405) ([marianotepper](https://github.com/marianotepper))
- Fix flaky tests and eliminate console output [\#404](https://github.com/datastax/jvector/pull/404) ([marianotepper](https://github.com/marianotepper))
- Remove query-time usage of ByteSequence::slice in PQVectors to reduce object allocations [\#403](https://github.com/datastax/jvector/pull/403) ([michaeljmarshall](https://github.com/michaeljmarshall))
- Fix TestOnDiskGraphIndex.testReorderingWithHoles [\#400](https://github.com/datastax/jvector/pull/400) ([marianotepper](https://github.com/marianotepper))
- add index construction benchmark [\#398](https://github.com/datastax/jvector/pull/398) ([sam-herman](https://github.com/sam-herman))
- Optimization for latency reduction in Product Quantization [\#397](https://github.com/datastax/jvector/pull/397) ([AbhijitKulkarni1](https://github.com/AbhijitKulkarni1))
- Add jmh benchmarks [\#396](https://github.com/datastax/jvector/pull/396) ([sam-herman](https://github.com/sam-herman))
- Fix MutableBQVectors parameterization/encoding [\#395](https://github.com/datastax/jvector/pull/395) ([jkni](https://github.com/jkni))
- Fix BQVectors\#ramBytesUsed and BQVectors\#getCompressedSize on empty BQVectors [\#394](https://github.com/datastax/jvector/pull/394) ([jkni](https://github.com/jkni))
- make example use index view [\#392](https://github.com/datastax/jvector/pull/392) ([sam-herman](https://github.com/sam-herman))
- Update Test2DThreshold to control for averages instead of worst-case statistics [\#391](https://github.com/datastax/jvector/pull/391) ([marianotepper](https://github.com/marianotepper))
- Fix distance computations in Native provider [\#389](https://github.com/datastax/jvector/pull/389) ([marianotepper](https://github.com/marianotepper))
- Change variable names to improve readability [\#388](https://github.com/datastax/jvector/pull/388) ([marianotepper](https://github.com/marianotepper))
- Improved use ScoreTracker to avoid wasteful searching for very large k [\#387](https://github.com/datastax/jvector/pull/387) ([marianotepper](https://github.com/marianotepper))
- Use ScoreTracker to avoid wasteful searching for very large k [\#384](https://github.com/datastax/jvector/pull/384) ([jbellis](https://github.com/jbellis))

## [4.0.0-beta.1](https://github.com/datastax/jvector/tree/4.0.0-beta.1) (2025-01-09)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.6...4.0.0-beta.1)

**Merged pull requests:**

- Release 4.0.0-beta.1 [\#386](https://github.com/datastax/jvector/pull/386) ([jkni](https://github.com/jkni))
- Fix CI on Windows due to missing posix\_madvise support [\#383](https://github.com/datastax/jvector/pull/383) ([jkni](https://github.com/jkni))
- add MADV\_RANDOM [\#382](https://github.com/datastax/jvector/pull/382) ([jbellis](https://github.com/jbellis))
- Make ravv usage thread-safe [\#381](https://github.com/datastax/jvector/pull/381) ([marianotepper](https://github.com/marianotepper))
- Hand-unroll the SIMD dot product loop [\#380](https://github.com/datastax/jvector/pull/380) ([blambov](https://github.com/blambov))
- Fix regression in assembleAndSum PQ decoder performance [\#379](https://github.com/datastax/jvector/pull/379) ([jkni](https://github.com/jkni))
- Non-uniform vector quantization [\#374](https://github.com/datastax/jvector/pull/374) ([marianotepper](https://github.com/marianotepper))

## [3.0.6](https://github.com/datastax/jvector/tree/3.0.6) (2024-12-24)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.5...3.0.6)

**Merged pull requests:**

- Release 3.0.6 [\#378](https://github.com/datastax/jvector/pull/378) ([jkni](https://github.com/jkni))
- More PQVectors fixes [\#377](https://github.com/datastax/jvector/pull/377) ([jbellis](https://github.com/jbellis))

## [3.0.5](https://github.com/datastax/jvector/tree/3.0.5) (2024-12-23)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.4...3.0.5)

**Merged pull requests:**

- Release 3.0.5 [\#376](https://github.com/datastax/jvector/pull/376) ([jkni](https://github.com/jkni))
- add GraphIndexBuilder.rescore\(\) for use by C\* CompactionGraph [\#375](https://github.com/datastax/jvector/pull/375) ([jbellis](https://github.com/jbellis))

## [3.0.4](https://github.com/datastax/jvector/tree/3.0.4) (2024-12-03)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.1...3.0.4)

**Merged pull requests:**

- Release 3.0.4 [\#373](https://github.com/datastax/jvector/pull/373) ([jkni](https://github.com/jkni))
- Don't use segment hashCode in MemorySegmentVectorFloat [\#372](https://github.com/datastax/jvector/pull/372) ([jkni](https://github.com/jkni))
- Release 3.0.3 [\#371](https://github.com/datastax/jvector/pull/371) ([jkni](https://github.com/jkni))
- Store compressed vectors in dense ByteSequence for PQVectors [\#370](https://github.com/datastax/jvector/pull/370) ([michaeljmarshall](https://github.com/michaeljmarshall))
- Reenable SimdOps.assembleAndSum; implement Panama/Native equivalent for CosineDecoder acceleration [\#368](https://github.com/datastax/jvector/pull/368) ([michaeljmarshall](https://github.com/michaeljmarshall))
- Use fma in VectorSimdOps.cosineSimilarity [\#366](https://github.com/datastax/jvector/pull/366) ([jkni](https://github.com/jkni))
- Release 3.0.2 [\#364](https://github.com/datastax/jvector/pull/364) ([jkni](https://github.com/jkni))
- Use fma in SimdOps.cosineSimilarity sum vector [\#363](https://github.com/datastax/jvector/pull/363) ([jkni](https://github.com/jkni))
- Remove max JDK version check [\#362](https://github.com/datastax/jvector/pull/362) ([jkni](https://github.com/jkni))

## [3.0.1](https://github.com/datastax/jvector/tree/3.0.1) (2024-09-30)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0...3.0.1)

**Merged pull requests:**

- Release 3.0.1 [\#361](https://github.com/datastax/jvector/pull/361) ([jkni](https://github.com/jkni))
- Improve performance of reconnectOrphanedNodes [\#359](https://github.com/datastax/jvector/pull/359) ([jkni](https://github.com/jkni))
- Use float in cosine metric final calculation in default vectorization provider [\#358](https://github.com/datastax/jvector/pull/358) ([k-jamroz](https://github.com/k-jamroz))
- approximateMediod returns a random node when the graph is too disconnected to search for the centroid [\#356](https://github.com/datastax/jvector/pull/356) ([jbellis](https://github.com/jbellis))
- Remove check for VBMI on CPU [\#352](https://github.com/datastax/jvector/pull/352) ([jkni](https://github.com/jkni))
- Set IdentityMapper maxOrdinal correctly in Grid/SiftSmall. [\#351](https://github.com/datastax/jvector/pull/351) ([jkni](https://github.com/jkni))

## [3.0.0](https://github.com/datastax/jvector/tree/3.0.0) (2024-08-13)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.16...3.0.0)

**Merged pull requests:**

- Release 3.0.0 [\#353](https://github.com/datastax/jvector/pull/353) ([jkni](https://github.com/jkni))

## [3.0.0-beta.16](https://github.com/datastax/jvector/tree/3.0.0-beta.16) (2024-08-01)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.15...3.0.0-beta.16)

**Merged pull requests:**

- Release 3.0.0-beta.16 [\#350](https://github.com/datastax/jvector/pull/350) ([jkni](https://github.com/jkni))
- add support for non-sequential remapped ordinals [\#349](https://github.com/datastax/jvector/pull/349) ([jbellis](https://github.com/jbellis))
- fix global centering and add test that raw computation equals precomputed [\#346](https://github.com/datastax/jvector/pull/346) ([jbellis](https://github.com/jbellis))

## [3.0.0-beta.15](https://github.com/datastax/jvector/tree/3.0.0-beta.15) (2024-07-03)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.14...3.0.0-beta.15)

**Merged pull requests:**

- Release 3.0.0-beta.15 [\#344](https://github.com/datastax/jvector/pull/344) ([jkni](https://github.com/jkni))
- Fixes for graphs with deletes [\#343](https://github.com/datastax/jvector/pull/343) ([jbellis](https://github.com/jbellis))

## [3.0.0-beta.14](https://github.com/datastax/jvector/tree/3.0.0-beta.14) (2024-07-02)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.13...3.0.0-beta.14)

**Merged pull requests:**

- Release 3.0.0-beta.14 [\#342](https://github.com/datastax/jvector/pull/342) ([jkni](https://github.com/jkni))
- cache reranked scores to avoid redoing expensive work when resuming [\#341](https://github.com/datastax/jvector/pull/341) ([jbellis](https://github.com/jbellis))
- extract RandomAccessWriter interface from BRAW [\#340](https://github.com/datastax/jvector/pull/340) ([jbellis](https://github.com/jbellis))

## [3.0.0-beta.13](https://github.com/datastax/jvector/tree/3.0.0-beta.13) (2024-06-07)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.12...3.0.0-beta.13)

**Merged pull requests:**

- Release 3.0.0-beta.13 [\#338](https://github.com/datastax/jvector/pull/338) ([jkni](https://github.com/jkni))
- Clear scratch structures if search terminates exceptionally [\#337](https://github.com/datastax/jvector/pull/337) ([jkni](https://github.com/jkni))
- Reduce tendency of reconnectOrphanedNodes to leave orphaned nodes [\#335](https://github.com/datastax/jvector/pull/335) ([jkni](https://github.com/jkni))

## [3.0.0-beta.12](https://github.com/datastax/jvector/tree/3.0.0-beta.12) (2024-05-29)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.11...3.0.0-beta.12)

**Merged pull requests:**

- Release 3.0.0-beta.12 [\#334](https://github.com/datastax/jvector/pull/334) ([jkni](https://github.com/jkni))
- add writeHeader and getPath methods to OnDiskGraphIndexWriter [\#332](https://github.com/datastax/jvector/pull/332) ([jbellis](https://github.com/jbellis))

## [3.0.0-beta.11](https://github.com/datastax/jvector/tree/3.0.0-beta.11) (2024-05-28)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.10...3.0.0-beta.11)

**Merged pull requests:**

- Release 3.0.0-beta.11 [\#331](https://github.com/datastax/jvector/pull/331) ([jkni](https://github.com/jkni))
- Implement support for COSINE in fused ADC [\#329](https://github.com/datastax/jvector/pull/329) ([jkni](https://github.com/jkni))
- Ecapsulate NodeArray internals [\#328](https://github.com/datastax/jvector/pull/328) ([jbellis](https://github.com/jbellis))
- Remove on-disk reranking [\#327](https://github.com/datastax/jvector/pull/327) ([jbellis](https://github.com/jbellis))
- Reduce the per-node overhead of edge lists in OnHeapGraphIndex [\#325](https://github.com/datastax/jvector/pull/325) ([jbellis](https://github.com/jbellis))
- standardize ReaderSupplier implementations as inner classes of their respective RandomAccessReaders, and add a Supplier for SimpleReader [\#323](https://github.com/datastax/jvector/pull/323) ([jbellis](https://github.com/jbellis))

## [3.0.0-beta.10](https://github.com/datastax/jvector/tree/3.0.0-beta.10) (2024-05-17)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.9...3.0.0-beta.10)

**Merged pull requests:**

- Release 3.0.0-beta.10 [\#321](https://github.com/datastax/jvector/pull/321) ([jkni](https://github.com/jkni))
- Fix InlineVectorValues.size [\#320](https://github.com/datastax/jvector/pull/320) ([jkni](https://github.com/jkni))

## [3.0.0-beta.9](https://github.com/datastax/jvector/tree/3.0.0-beta.9) (2024-05-13)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.7...3.0.0-beta.9)

**Merged pull requests:**

- Release 3.0.0-beta.9 [\#319](https://github.com/datastax/jvector/pull/319) ([jkni](https://github.com/jkni))
- Release 3.0.0-beta.8 [\#318](https://github.com/datastax/jvector/pull/318) ([jkni](https://github.com/jkni))
- always rerank at least the best result found so that caller will have something with which to compare this index's results to others' [\#317](https://github.com/datastax/jvector/pull/317) ([jbellis](https://github.com/jbellis))
- improve memory usage during construction [\#316](https://github.com/datastax/jvector/pull/316) ([jbellis](https://github.com/jbellis))
- add getFeatureSet methods to ODGI and ODGIW [\#315](https://github.com/datastax/jvector/pull/315) ([jbellis](https://github.com/jbellis))

## [3.0.0-beta.7](https://github.com/datastax/jvector/tree/3.0.0-beta.7) (2024-05-08)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.6...3.0.0-beta.7)

**Merged pull requests:**

- Release 3.0.0-beta.7 [\#314](https://github.com/datastax/jvector/pull/314) ([jkni](https://github.com/jkni))
- Fix writing jvector2-compatible indexes incrementally [\#313](https://github.com/datastax/jvector/pull/313) ([jbellis](https://github.com/jbellis))
- remove BQ centering [\#312](https://github.com/datastax/jvector/pull/312) ([jbellis](https://github.com/jbellis))

## [3.0.0-beta.6](https://github.com/datastax/jvector/tree/3.0.0-beta.6) (2024-05-06)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.5...3.0.0-beta.6)

**Merged pull requests:**

- Release 3.0.0-beta.6 [\#311](https://github.com/datastax/jvector/pull/311) ([jkni](https://github.com/jkni))
- add rerankK to GraphSearcher::search, and worstApproximateScoreInTopK to SearchResult [\#310](https://github.com/datastax/jvector/pull/310) ([jbellis](https://github.com/jbellis))

## [3.0.0-beta.5](https://github.com/datastax/jvector/tree/3.0.0-beta.5) (2024-05-03)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.4...3.0.0-beta.5)

**Merged pull requests:**

- Release 3.0.0-beta.5 [\#309](https://github.com/datastax/jvector/pull/309) ([jkni](https://github.com/jkni))
- add ability to write old versions of PQ and ODGI.  current version standardized as 3 to avoid confusion [\#308](https://github.com/datastax/jvector/pull/308) ([jbellis](https://github.com/jbellis))
- Check for GCC 11+ instead of failing compile [\#307](https://github.com/datastax/jvector/pull/307) ([jkni](https://github.com/jkni))
- Fix usage of Float.MIN\_VALUE [\#306](https://github.com/datastax/jvector/pull/306) ([jkni](https://github.com/jkni))

## [3.0.0-beta.4](https://github.com/datastax/jvector/tree/3.0.0-beta.4) (2024-05-02)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.3...3.0.0-beta.4)

**Merged pull requests:**

- Release 3.0.0-beta.4 [\#305](https://github.com/datastax/jvector/pull/305) ([jkni](https://github.com/jkni))
- Switch Fused ADC from 32-cluster to 256-cluster PQ, maxDegree 32 graphs [\#304](https://github.com/datastax/jvector/pull/304) ([jkni](https://github.com/jkni))
- GraphIndexBuilder implements Closeable instead of AutoCloseable [\#302](https://github.com/datastax/jvector/pull/302) ([jbellis](https://github.com/jbellis))

## [3.0.0-beta.3](https://github.com/datastax/jvector/tree/3.0.0-beta.3) (2024-04-22)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.2...3.0.0-beta.3)

**Merged pull requests:**

- Release 3.0.0-beta.3 [\#301](https://github.com/datastax/jvector/pull/301) ([jkni](https://github.com/jkni))
- Flesh out support for reranking from inline full/lvq vectors while building graph using pq [\#300](https://github.com/datastax/jvector/pull/300) ([jbellis](https://github.com/jbellis))
- add OrdinalMapper for when can't compute a static map before constructing the OnDiskGraphIndexWriter [\#299](https://github.com/datastax/jvector/pull/299) ([jbellis](https://github.com/jbellis))
- Reuse backing ByteSequence for LVQ vectors in ODGI [\#298](https://github.com/datastax/jvector/pull/298) ([jkni](https://github.com/jkni))
- Add a memory-mapped RandomAccessReader using MemorySegment api [\#296](https://github.com/datastax/jvector/pull/296) ([mdogan](https://github.com/mdogan))

## [3.0.0-beta.2](https://github.com/datastax/jvector/tree/3.0.0-beta.2) (2024-04-16)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-beta.1...3.0.0-beta.2)

**Merged pull requests:**

- Release 3.0.0-beta.2 [\#297](https://github.com/datastax/jvector/pull/297) ([jkni](https://github.com/jkni))
- tweaks to make life easier for upgraders [\#290](https://github.com/datastax/jvector/pull/290) ([jbellis](https://github.com/jbellis))

## [3.0.0-beta.1](https://github.com/datastax/jvector/tree/3.0.0-beta.1) (2024-04-12)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-alpha.7...3.0.0-beta.1)

**Merged pull requests:**

- Release 3.0.0-beta.1 [\#276](https://github.com/datastax/jvector/pull/276) ([jkni](https://github.com/jkni))
- Incremental graph writes, ODGI abstraction allowing features to be combined [\#275](https://github.com/datastax/jvector/pull/275) ([jkni](https://github.com/jkni))
- remove \(broken\) concurrency support from removeDeletedNodes [\#273](https://github.com/datastax/jvector/pull/273) ([jbellis](https://github.com/jbellis))
- Reduce allocation by pooling GraphSearcher objects [\#270](https://github.com/datastax/jvector/pull/270) ([jbellis](https://github.com/jbellis))
- Optimize GraphSeacher.visited using inthashset [\#269](https://github.com/datastax/jvector/pull/269) ([jbellis](https://github.com/jbellis))
- Fix packing of LVQ vector dimensionalities not divisible by 64. Introduce test for LVQ similarity. [\#268](https://github.com/datastax/jvector/pull/268) ([jkni](https://github.com/jkni))
- Remove PQVectors from ADCGraphIndex [\#267](https://github.com/datastax/jvector/pull/267) ([jkni](https://github.com/jkni))
- make Test2DThreshold less fragile [\#266](https://github.com/datastax/jvector/pull/266) ([jbellis](https://github.com/jbellis))
- Enable JDK 22 CI [\#258](https://github.com/datastax/jvector/pull/258) ([jkni](https://github.com/jkni))
- Merge 3.0-alpha [\#256](https://github.com/datastax/jvector/pull/256) ([jkni](https://github.com/jkni))
- Use AVX-512 reduce-add intrinsics [\#252](https://github.com/datastax/jvector/pull/252) ([jkni](https://github.com/jkni))
- Remove conditional dotProduct/squareDistance in NativeVectorUtilSupport [\#251](https://github.com/datastax/jvector/pull/251) ([jkni](https://github.com/jkni))
- only compute aMagnitude for cosines once, since it is independent of the query vector [\#250](https://github.com/datastax/jvector/pull/250) ([jbellis](https://github.com/jbellis))
- Fix bytesToRead math in MMapReader.read\(int\[\], int, int\) [\#249](https://github.com/datastax/jvector/pull/249) ([jkni](https://github.com/jkni))
- Concurrent deletes [\#248](https://github.com/datastax/jvector/pull/248) ([jbellis](https://github.com/jbellis))
- Replace NormalDistributionTracker with TwoPhaseTracker [\#247](https://github.com/datastax/jvector/pull/247) ([jbellis](https://github.com/jbellis))
- Use on-heap MemorySegments for native vectors/sequences [\#246](https://github.com/datastax/jvector/pull/246) ([jkni](https://github.com/jkni))
- Build indexes using compressed vectors [\#244](https://github.com/datastax/jvector/pull/244) ([jbellis](https://github.com/jbellis))
- Reduce repetition in Bench. Factor out more concise methods. [\#243](https://github.com/datastax/jvector/pull/243) ([jkni](https://github.com/jkni))
- add a diverseBefore marker to avoid recomputing diversity that hasn't changed [\#242](https://github.com/datastax/jvector/pull/242) ([jbellis](https://github.com/jbellis))
- Fix Bench timings on paths performing multiple types of queries [\#240](https://github.com/datastax/jvector/pull/240) ([jkni](https://github.com/jkni))
- Corrected relative path to access Siftsmall example files [\#239](https://github.com/datastax/jvector/pull/239) ([rd-99](https://github.com/rd-99))
- Fix TestVectorSimilarityFunction test. Add missing license. [\#237](https://github.com/datastax/jvector/pull/237) ([jkni](https://github.com/jkni))
- Fix indexing into MemorySegment when loading/storing from FloatVector in scale [\#236](https://github.com/datastax/jvector/pull/236) ([jkni](https://github.com/jkni))
- Bump version to account for 3.0-alpha branch release [\#234](https://github.com/datastax/jvector/pull/234) ([jkni](https://github.com/jkni))
- Improve quality of bindings with newer jextract version [\#229](https://github.com/datastax/jvector/pull/229) ([jkni](https://github.com/jkni))
- Improve performance of native vectorization provider [\#224](https://github.com/datastax/jvector/pull/224) ([jkni](https://github.com/jkni))
- Anisotropic PQ [\#201](https://github.com/datastax/jvector/pull/201) ([jbellis](https://github.com/jbellis))
- Vector abstractions, native code, and fused graphs [\#191](https://github.com/datastax/jvector/pull/191) ([jkni](https://github.com/jkni))

## [3.0.0-alpha.7](https://github.com/datastax/jvector/tree/3.0.0-alpha.7) (2024-03-19)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-alpha.6...3.0.0-alpha.7)

**Merged pull requests:**

- Release 3.0.0-alpha.7 [\#255](https://github.com/datastax/jvector/pull/255) ([jkni](https://github.com/jkni))
- use euclidean similarity for centroid search if the centroid is zero and the index uses cosine similarity [\#221](https://github.com/datastax/jvector/pull/221) ([jbellis](https://github.com/jbellis))
- DenseIntMap concurrency [\#219](https://github.com/datastax/jvector/pull/219) ([jbellis](https://github.com/jbellis))
- Fix flaky ProductQuantization test [\#217](https://github.com/datastax/jvector/pull/217) ([jkni](https://github.com/jkni))

## [3.0.0-alpha.6](https://github.com/datastax/jvector/tree/3.0.0-alpha.6) (2024-02-26)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-alpha.5...3.0.0-alpha.6)

**Merged pull requests:**

- Release 3.0.0-alpha.6 [\#233](https://github.com/datastax/jvector/pull/233) ([jkni](https://github.com/jkni))

## [3.0.0-alpha.5](https://github.com/datastax/jvector/tree/3.0.0-alpha.5) (2024-02-09)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-alpha.4...3.0.0-alpha.5)

**Merged pull requests:**

- Release 3.0.0-alpha.5 [\#216](https://github.com/datastax/jvector/pull/216) ([jkni](https://github.com/jkni))
- Fix GraphIndexBuilder.reconnectOrphanedNodes [\#215](https://github.com/datastax/jvector/pull/215) ([mdogan](https://github.com/mdogan))

## [3.0.0-alpha.4](https://github.com/datastax/jvector/tree/3.0.0-alpha.4) (2024-02-08)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-alpha.3...3.0.0-alpha.4)

**Merged pull requests:**

- Release 3.0.0-alpha.4 [\#213](https://github.com/datastax/jvector/pull/213) ([jkni](https://github.com/jkni))
- Expose MAX\_PQ\_TRAINING\_SET\_SIZE. [\#212](https://github.com/datastax/jvector/pull/212) ([jkni](https://github.com/jkni))
- add `getCompressor` to CompressedVectors [\#211](https://github.com/datastax/jvector/pull/211) ([jbellis](https://github.com/jbellis))

## [3.0.0-alpha.3](https://github.com/datastax/jvector/tree/3.0.0-alpha.3) (2024-02-08)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-alpha.2...3.0.0-alpha.3)

**Merged pull requests:**

- Release 3.0.0-alpha.3 [\#210](https://github.com/datastax/jvector/pull/210) ([jkni](https://github.com/jkni))
- add PQ.refine for when you already built PQ once for a similar set of vectors [\#209](https://github.com/datastax/jvector/pull/209) ([jbellis](https://github.com/jbellis))
- optimize reconnectOrphanedNodes [\#208](https://github.com/datastax/jvector/pull/208) ([jbellis](https://github.com/jbellis))
- Attempt 3 at cleaning up the threadlocal leaks [\#206](https://github.com/datastax/jvector/pull/206) ([jbellis](https://github.com/jbellis))
- Auto-boxing/unboxing reductions, reduce boxed footprint on heap [\#202](https://github.com/datastax/jvector/pull/202) ([jkni](https://github.com/jkni))
- Remove overzealous dependency exclusion affecting util-mmap [\#200](https://github.com/datastax/jvector/pull/200) ([jkni](https://github.com/jkni))
- Fix/document unclosed views [\#195](https://github.com/datastax/jvector/pull/195) ([jkni](https://github.com/jkni))
- Improve GraphIndexBuilder\#cleanup doc for concurrent searches [\#190](https://github.com/datastax/jvector/pull/190) ([mdogan](https://github.com/mdogan))

## [3.0.0-alpha.2](https://github.com/datastax/jvector/tree/3.0.0-alpha.2) (2024-01-10)

[Full Changelog](https://github.com/datastax/jvector/compare/3.0.0-alpha.1...3.0.0-alpha.2)

**Merged pull requests:**

- Release 3.0.0-alpha.2 [\#186](https://github.com/datastax/jvector/pull/186) ([jkni](https://github.com/jkni))
- Add GraphSearcher::resume [\#185](https://github.com/datastax/jvector/pull/185) ([jbellis](https://github.com/jbellis))
- Search parameters [\#183](https://github.com/datastax/jvector/pull/183) ([jbellis](https://github.com/jbellis))
- Cleanup for 3.0 release [\#182](https://github.com/datastax/jvector/pull/182) ([jbellis](https://github.com/jbellis))
- Handle the case of removing all nodes in the cleanup [\#181](https://github.com/datastax/jvector/pull/181) ([mdumandag](https://github.com/mdumandag))

## [3.0.0-alpha.1](https://github.com/datastax/jvector/tree/3.0.0-alpha.1) (2024-01-03)

[Full Changelog](https://github.com/datastax/jvector/compare/2.0.5...3.0.0-alpha.1)

**Merged pull requests:**

- Release 3.0.0-alpha.1 [\#179](https://github.com/datastax/jvector/pull/179) ([jkni](https://github.com/jkni))
- Don't create new ScoreTracker.NoOpTracker each time [\#178](https://github.com/datastax/jvector/pull/178) ([dlg99](https://github.com/dlg99))
- Fix DenseIntMap size [\#177](https://github.com/datastax/jvector/pull/177) ([mdogan](https://github.com/mdogan))
- Make GraphIndexBuilder.markNodeDeleted thread-safe [\#175](https://github.com/datastax/jvector/pull/175) ([mdogan](https://github.com/mdogan))
- Remove caching of vectors encountered during search, update reranker interface accordingly [\#173](https://github.com/datastax/jvector/pull/173) ([jbellis](https://github.com/jbellis))
- Parallelize GraphIndexBuilder.removeDeletedNodes\(\) [\#172](https://github.com/datastax/jvector/pull/172) ([mdogan](https://github.com/mdogan))

## [2.0.5](https://github.com/datastax/jvector/tree/2.0.5) (2023-12-21)

[Full Changelog](https://github.com/datastax/jvector/compare/2.0.4...2.0.5)

**Merged pull requests:**

- Release 2.0.5 [\#171](https://github.com/datastax/jvector/pull/171) ([jkni](https://github.com/jkni))
- Replace CHM with HashMap in CHMGraphCache [\#170](https://github.com/datastax/jvector/pull/170) ([michaeljmarshall](https://github.com/michaeljmarshall))
- Remove ThreadPooling self reference in Pooled object to prevent memory leak [\#169](https://github.com/datastax/jvector/pull/169) ([michaeljmarshall](https://github.com/michaeljmarshall))
- Add optional FJP args for indexing and quantization [\#162](https://github.com/datastax/jvector/pull/162) ([mdogan](https://github.com/mdogan))
- PhysicalCoreExecutor can exit gracefully; CachingGraphIndex's cacheDistance can be customized [\#160](https://github.com/datastax/jvector/pull/160) ([xjtushilei](https://github.com/xjtushilei))
- Fix some bugs [\#156](https://github.com/datastax/jvector/pull/156) ([chengsecret](https://github.com/chengsecret))
- Minor typo fix in README.md [\#155](https://github.com/datastax/jvector/pull/155) ([FRosner](https://github.com/FRosner))
- Scrub both fvec/ivec and hdf5 dot product datasets [\#154](https://github.com/datastax/jvector/pull/154) ([jkni](https://github.com/jkni))
- Use fma in SIMD Euclidean/cosine [\#153](https://github.com/datastax/jvector/pull/153) ([jkni](https://github.com/jkni))
- Fix usage of null acceptOrds in SiftSmall example [\#152](https://github.com/datastax/jvector/pull/152) ([vbekiaris](https://github.com/vbekiaris))

## [2.0.4](https://github.com/datastax/jvector/tree/2.0.4) (2023-11-10)

[Full Changelog](https://github.com/datastax/jvector/compare/2.0.3...2.0.4)

**Merged pull requests:**

- Release 2.0.4 [\#151](https://github.com/datastax/jvector/pull/151) ([jkni](https://github.com/jkni))
- vectorsEncountered not always in sync with resultQueue, causing NPE when breaking out of loop due to threshold probability [\#150](https://github.com/datastax/jvector/pull/150) ([jbellis](https://github.com/jbellis))
- Run verify phase in CI \(which includes license checks\) [\#149](https://github.com/datastax/jvector/pull/149) ([jkni](https://github.com/jkni))

## [2.0.3](https://github.com/datastax/jvector/tree/2.0.3) (2023-11-08)

[Full Changelog](https://github.com/datastax/jvector/compare/2.0.2...2.0.3)

**Merged pull requests:**

- Release 2.0.3 [\#148](https://github.com/datastax/jvector/pull/148) ([jkni](https://github.com/jkni))
- add upgrade guide [\#147](https://github.com/datastax/jvector/pull/147) ([jbellis](https://github.com/jbellis))
- Split LongHeap into Growable and Bounded flavors [\#146](https://github.com/datastax/jvector/pull/146) ([jbellis](https://github.com/jbellis))

## [2.0.2](https://github.com/datastax/jvector/tree/2.0.2) (2023-11-07)

[Full Changelog](https://github.com/datastax/jvector/compare/2.0.1...2.0.2)

**Merged pull requests:**

- Release 2.0.2 [\#145](https://github.com/datastax/jvector/pull/145) ([jkni](https://github.com/jkni))

## [2.0.1](https://github.com/datastax/jvector/tree/2.0.1) (2023-11-07)

[Full Changelog](https://github.com/datastax/jvector/compare/2.0.0...2.0.1)

**Merged pull requests:**

- Release 2.0.1 [\#144](https://github.com/datastax/jvector/pull/144) ([jkni](https://github.com/jkni))
- add getOriginalSize and getCompressedSize to CompressedVectors interface [\#143](https://github.com/datastax/jvector/pull/143) ([jbellis](https://github.com/jbellis))
- Cherry-pick various bench improvements from PR \#76. [\#133](https://github.com/datastax/jvector/pull/133) ([jkni](https://github.com/jkni))

## [2.0.0](https://github.com/datastax/jvector/tree/2.0.0) (2023-11-06)

[Full Changelog](https://github.com/datastax/jvector/compare/1.0.2...2.0.0)

**Merged pull requests:**

- Release 2.0.0 [\#142](https://github.com/datastax/jvector/pull/142) ([jkni](https://github.com/jkni))
- Fix running single test using Maven commandline [\#141](https://github.com/datastax/jvector/pull/141) ([jkni](https://github.com/jkni))
- Reconnect orphaned nodes in cleanup\(\) [\#138](https://github.com/datastax/jvector/pull/138) ([jbellis](https://github.com/jbellis))
- Add binary quantization [\#135](https://github.com/datastax/jvector/pull/135) ([jbellis](https://github.com/jbellis))
- Updated download helper [\#134](https://github.com/datastax/jvector/pull/134) ([msmygit](https://github.com/msmygit))
- CI improvement [\#131](https://github.com/datastax/jvector/pull/131) ([jkni](https://github.com/jkni))
- downloads wikipedia fvec files for 100k, switched to squad based query vectors [\#130](https://github.com/datastax/jvector/pull/130) ([phact](https://github.com/phact))
- Addresses issue \#36 by adding license header checks. Added headers on… [\#129](https://github.com/datastax/jvector/pull/129) ([zznate](https://github.com/zznate))
- Adds DenseIntMap for building graph with much less contention. back to zero dependency! [\#128](https://github.com/datastax/jvector/pull/128) ([tjake](https://github.com/tjake))
- Ipcexample [\#127](https://github.com/datastax/jvector/pull/127) ([tjake](https://github.com/tjake))
- Fix num/denom distortion for first round of assignments when clustering [\#121](https://github.com/datastax/jvector/pull/121) ([jkni](https://github.com/jkni))
- Deletes [\#117](https://github.com/datastax/jvector/pull/117) ([jbellis](https://github.com/jbellis))

## [1.0.2](https://github.com/datastax/jvector/tree/1.0.2) (2023-10-09)

[Full Changelog](https://github.com/datastax/jvector/compare/1.0.1...1.0.2)

**Merged pull requests:**

- Release 1.0.2 [\#120](https://github.com/datastax/jvector/pull/120) ([jkni](https://github.com/jkni))
- fix mergeNeighbors to not add duplicate nodes, and fix test to check for duplicates [\#119](https://github.com/datastax/jvector/pull/119) ([jbellis](https://github.com/jbellis))
- Mt index build fixes [\#113](https://github.com/datastax/jvector/pull/113) ([tjake](https://github.com/tjake))
- Fork test VM per core [\#111](https://github.com/datastax/jvector/pull/111) ([jkni](https://github.com/jkni))
- Add improved test coverage for on-disk graph caching [\#109](https://github.com/datastax/jvector/pull/109) ([jkni](https://github.com/jkni))

## [1.0.1](https://github.com/datastax/jvector/tree/1.0.1) (2023-10-02)

[Full Changelog](https://github.com/datastax/jvector/compare/1.0.0...1.0.1)

**Merged pull requests:**

- Release 1.0.1 [\#108](https://github.com/datastax/jvector/pull/108) ([jkni](https://github.com/jkni))

## [1.0.0](https://github.com/datastax/jvector/tree/1.0.0) (2023-09-29)

[Full Changelog](https://github.com/datastax/jvector/compare/0.9.3...1.0.0)

**Merged pull requests:**

- Release 1.0.0 [\#107](https://github.com/datastax/jvector/pull/107) ([jkni](https://github.com/jkni))
- Fix SimpleMappedReader to respect offset [\#106](https://github.com/datastax/jvector/pull/106) ([jkni](https://github.com/jkni))
- Adjust PQ clustering parameters based on experimentation [\#105](https://github.com/datastax/jvector/pull/105) ([jkni](https://github.com/jkni))
- Add simd approach for summing the cached PQ products of each encoded vector [\#104](https://github.com/datastax/jvector/pull/104) ([tjake](https://github.com/tjake))
- README tweaks [\#101](https://github.com/datastax/jvector/pull/101) ([bradfordcp](https://github.com/bradfordcp))
- KMeansPlusPlusClusterer optimizations [\#100](https://github.com/datastax/jvector/pull/100) ([jkni](https://github.com/jkni))

## [0.9.3](https://github.com/datastax/jvector/tree/0.9.3) (2023-09-25)

[Full Changelog](https://github.com/datastax/jvector/compare/0.9.2...0.9.3)

**Merged pull requests:**

- Release 0.9.3 [\#99](https://github.com/datastax/jvector/pull/99) ([jkni](https://github.com/jkni))
- Remove triangle inequality from k means plus plus [\#98](https://github.com/datastax/jvector/pull/98) ([jkni](https://github.com/jkni))
- Clean up all build warnings related to multimodule versioning. [\#97](https://github.com/datastax/jvector/pull/97) ([jkni](https://github.com/jkni))
- wikipedia datasets in readme [\#95](https://github.com/datastax/jvector/pull/95) ([phact](https://github.com/phact))
- Fragment cache [\#94](https://github.com/datastax/jvector/pull/94) ([jbellis](https://github.com/jbellis))
- Add decodedCosine fast path [\#91](https://github.com/datastax/jvector/pull/91) ([jkni](https://github.com/jkni))
- Implement optimized decoded square distance [\#89](https://github.com/datastax/jvector/pull/89) ([jkni](https://github.com/jkni))
- Fix code coverage in IntelliJ [\#88](https://github.com/datastax/jvector/pull/88) ([jkni](https://github.com/jkni))
- Fix recall regression for centered PQ with non-dot product metrics [\#84](https://github.com/datastax/jvector/pull/84) ([jkni](https://github.com/jkni))
- move sharing from annotation to method; use that in PQ [\#83](https://github.com/datastax/jvector/pull/83) ([jbellis](https://github.com/jbellis))
- Refactor tests into jvector-tests module. Set up configurations to be able to run tests with JDK11 features and JDK20 features. [\#75](https://github.com/datastax/jvector/pull/75) ([jkni](https://github.com/jkni))

## [0.9.2](https://github.com/datastax/jvector/tree/0.9.2) (2023-09-18)

[Full Changelog](https://github.com/datastax/jvector/compare/0.9.1...0.9.2)

**Merged pull requests:**

- Release 0.9.2 [\#82](https://github.com/datastax/jvector/pull/82) ([jkni](https://github.com/jkni))
- JVECTOR-78 Make Graph and Graph.View AutoCloseable [\#79](https://github.com/datastax/jvector/pull/79) ([mike-tr-adamson](https://github.com/mike-tr-adamson))
- Minor documentation updates [\#77](https://github.com/datastax/jvector/pull/77) ([msmygit](https://github.com/msmygit))
- Bump readme version [\#73](https://github.com/datastax/jvector/pull/73) ([jkni](https://github.com/jkni))

## [0.9.1](https://github.com/datastax/jvector/tree/0.9.1) (2023-09-15)

[Full Changelog](https://github.com/datastax/jvector/compare/0.9.0...0.9.1)

**Merged pull requests:**

- Release 0.9.1 [\#72](https://github.com/datastax/jvector/pull/72) ([jkni](https://github.com/jkni))
- Change package from com.github.jbellis to io.github.jbellis [\#71](https://github.com/datastax/jvector/pull/71) ([jkni](https://github.com/jkni))

## [0.9.0](https://github.com/datastax/jvector/tree/0.9.0) (2023-09-14)

**Merged pull requests:**

- Initial Release 0.9.0 [\#69](https://github.com/datastax/jvector/pull/69) ([jkni](https://github.com/jkni))
