# Embedded IO contract (draft)

> **Status:** draft for review. Implemented in `jvector-base` (package
> `io.github.jbellis.jvector.disk`) with tests in `jvector-tests`; the consumer changes in
> section 6 are not. Motivated by [#722](https://github.com/datastax/jvector/issues/722) and the
> review threads on [#710](https://github.com/datastax/jvector/pull/710). Every claim about
> current behaviour is cross-referenced as `path:line` against `origin/main` @ `098d131c`.

## 1. Problem

jvector is embedded by systems that own their storage: Cassandra SAI and CNDB (a graph is one
region of a component file, wrapped in an SAI header and footer), OpenSearch via Lucene (a graph
is a codec file whose `IndexOutput` is sequential, append-only, with an incremental checksum),
and standalone users (a graph is a file). Today the compactor and the parallel writer only know
how to talk to a `Path`, and even the random-access writer assumes it owns a whole file. The
result, as described on #722, is that hosts copy and re-copy jvector's bytes to fit them into
their own containers, or cannot use the compactor at all.

The contract below lets a host say **where** jvector's bytes go and **when** they become real,
without any host resource type (`Path`, `File`, `FileChannel`, `IndexOutput`) appearing in the
embedded view, and without slowing down the plain-file case.

### 1.1 Scope

**In:** a host-owned location, possibly interior to a host file; strictly sequential
(append-only) writes; more than one artifact per operation; a per-artifact completion point
and a per-operation commit-or-abort boundary.

**Out, deliberately:** positional writes, read-back of the output while it is being written,
and scratch space. Section 6 shows why none of them is needed: nothing in the format or in the
compactor's algorithm requires out-of-order writes; the only in-place rewrite is the optional
post-compaction refinement, which is off by default. Should a host ever want to offer
positional access as an optimization, it can be added later as an optional capability without
changing the shapes here.

## 2. Concerns catalog

Everything raised on #722, on the #710 threads, and by the code, with where it lands.

| # | Concern | Evidence | Raised by | Lands in |
|---|---|---|---|---|
| C1 | Output location must be host-owned, possibly a region inside a host file | `CompactWriter` opens its own file: `graph/disk/CompactWriter.java:96` | #722; #710 | `IndexDestination`, `OutputReservation.stream()` with region-relative positions |
| C2 | No `Path`/`File`/channel type may leak into the embedded view | `OutputReservation.file()` on #710 | reta on #722 | no host type on any vended interface; `Path` only in the file-backed static factories |
| C3 | Lucene `IndexOutput` is sequential-only with an incremental CRC; random writes were declined upstream ([lucene#15420](https://github.com/apache/lucene/issues/15420)) | the OpenSearch plugin adapts `IndexOutput` as a plain `IndexWriter` (`JVectorIndexWriter`) | reta on #722 | the contract is sequential by construction |
| C4 | The compactor writes level-0 records positionally from a second handle on the same file | `graph/disk/OnDiskGraphIndexCompactor.java:908-931` | code | section 6: in-order emission with a reorder buffer |
| C5 | The compactor reads its own output back during refinement and rewrites records in place | `OnDiskGraphIndexCompactor.java:497-503`, `:728-731` | code | section 6: refinement as a second pass, or off (its default) |
| C6 | The fused-PQ pre-encode cache is mapped past the projected EOF of the output file and truncated away | `graph/disk/FusedCompactionStrategy.java:208-216`, `:190` | code | out of scope: a placement question for adoption |
| C7 | One operation can produce several artifacts (graph plus compressed-vectors sidecar) placed differently by each host | `compact(Path, Path)` at `OnDiskGraphIndexCompactor.java:333`; `graph/disk/SidecarCompactionStrategy.java:98` | code; host code | `OutputArtifact`, one reservation per artifact, session-level commit |
| C8 | Transactional boundary: commit at most once, close exactly once, close-without-commit aborts and discards partial output | #710 `Target` thread | jshook | the state machines in 3.3 |
| C9 | Configuration (where) stays separate from the operation (one live, single-use handle) | same thread | ashkrisk asked; jshook answered | `IndexDestination` (config) vs `OutputSession` / `OutputReservation` (operation) |
| C10 | The offsets a graph stores are relative to the writer's coordinate origin; a reader must use the same origin | `graph/disk/OnDiskGraphIndex.java:305` (footer search relative to `length()`); `AbstractGraphIndexWriter.writeFooter` stores `out.position()` | #710 memory-safety commit | 3.4: positions start at `0` at the region start; hosts read back from the same origin |
| C11 | Durability and checksums are host policy | Cassandra re-reads the range for its footer CRC (`graph/disk/RandomAccessOnDiskGraphIndexWriter.java:210-212`); Lucene computes its CRC incrementally | host code | `complete()` is where the host finalizes; jvector never forces or checksums the host's stream |
| C12 | Long operations must report progress, accept throttling in bytes, and honour cancellation | `util/work/ProgressTracker.java`, `util/work/WorkLimiter.java` on #710 | #710 | 3.5: existing seams, applied to output IO; abort on interrupt |
| C13 | No work may escape to pools the host does not own | `graph/ParallelExecutor.java` on #710 | #710 threads | unchanged: execution seams carry that; this contract carries IO only |
| C14 | The plain-file path must stay as fast as today | requirement | this task | 3.6 |
| C15 | The build-path writers need the same seam; the parallel writer hard-requires a `Path` | `graph/disk/OnDiskGraphIndexWriter.java:172`; `OnDiskParallelGraphIndexWriter.java:123-126` | code; Cassandra | `OnDiskSequentialGraphIndexWriter.Builder(graph, reservation.stream())` works today |

## 3. Contract

### 3.1 Shape

```
IndexDestination            stateless "where", built once by the host, held
   open() ----------------> OutputSession                 one write operation; the transactional unit
                              reserve(GRAPH) ----------> OutputReservation   one artifact's output
                                                             stream()      IndexWriter, append-only
                                                             complete() / close()
                              reserve(COMPRESSED_VECTORS) -> OutputReservation
                              commit() / close()
```

Two levels of "done" mirror what every host already has: each artifact is **completed** (Lucene
writes its file footer here, SAI writes its component footer here), and the session is
**committed** once as a set (Lucene's segment commit, SAI's component completion, the standalone
rename into place). Close without commit is an abort at either level, so try-with-resources is
the whole error-handling story.

The only IO type jvector sees is the existing `IndexWriter`: a `DataOutput` with `position()`.
The contract adds lifecycle around it, not a new IO primitive.

### 3.2 Interfaces

All in `io.github.jbellis.jvector.disk`, `@Experimental`, JDK 11. Shown here without javadoc;
the sources carry the full contract.

```java
@FunctionalInterface
public interface IndexDestination {
    OutputSession open() throws IOException;

    static IndexDestination toFile(Path path);                         // standalone: temp file + rename on commit
    static IndexDestination toFiles(Map<OutputArtifact, Path> paths);  // one standalone file per artifact
    static IndexDestination inFile(Path path, long offset);            // a region inside a caller-owned file
}

public interface OutputSession extends AutoCloseable {
    OutputReservation reserve(OutputArtifact artifact) throws IOException;  // at most once per artifact
    void commit() throws IOException;    // all reservations completed and closed; at most once
    @Override void close() throws IOException;   // idempotent; no commit => abort, partial output discarded
}

public interface OutputReservation extends AutoCloseable {
    OutputArtifact artifact();
    IndexWriter stream() throws IOException;   // append-only; same instance each call; position() starts at 0
    void complete() throws IOException;        // flushes the stream; host finalizes (footer, checksum, durability)
    @Override void close() throws IOException; // idempotent; no complete => artifact discarded
}

public enum OutputArtifact { GRAPH, COMPRESSED_VECTORS }
```

### 3.3 Lifecycle state machines

`OutputSession`

| From | Call | To | Notes |
|---|---|---|---|
| OPEN | `reserve(a)` | OPEN | at most once per artifact; `IllegalArgumentException` if the destination has no placement for `a` |
| OPEN | `commit()` ok | COMMITTED | requires every reservation completed and closed |
| OPEN | `commit()` throws | ABORTED | partial artifacts discarded on `close()` |
| OPEN | `close()` | ABORTED | discard everything |
| COMMITTED | `close()` | CLOSED | release handles only |
| ABORTED / CLOSED | `close()` | same | idempotent |
| any non-open state | anything else | `IllegalStateException` | |

`OutputReservation`

| From | Call | To | Notes |
|---|---|---|---|
| OPEN | `stream()` | OPEN | returns the same instance every time |
| OPEN | `complete()` ok | COMPLETED | flushes the stream; host finalizes |
| OPEN | `complete()` throws | OPEN (not completed) | session cannot commit |
| OPEN | `close()` | DISCARDED | session cannot commit |
| COMPLETED | `close()` | CLOSED | release the stream |
| DISCARDED / CLOSED | `close()` | same | idempotent |
| COMPLETED / DISCARDED / CLOSED | `stream`, `complete` | `IllegalStateException` | |

### 3.4 Coordinates, threading, failure

- **Positions start at 0 at the region start, and the format stores offsets in those
  coordinates.** The footer's header offset and any separated-feature offsets are relative to
  the first byte jvector wrote. A host therefore reads the artifact back with a reader whose
  origin is the same place: a slice or window starting at the region, or, for a whole-container
  reader, `OnDiskGraphIndex.load(supplier, regionStart, false)`, which reads the header at the
  region start instead of trusting the container's end. Graphs written today with
  `withStartOffset(base)` over an absolute writer store absolute offsets and keep their
  absolute-origin reader; the origin is a property of the written artifact, so a host that
  migrates changes its writer and its reader together.
- The stream is single-threaded. jvector may compute in parallel but writes through one thread.
- Lifecycle calls (`complete`, `commit`, `close`) happen on the owning thread after every worker
  has finished. jvector drains its executors before completing, so a host never has to defend
  `close()` against in-flight writes.
- Any `IOException` or `InterruptedException` propagates unchanged; the try-with-resources unwind
  closes reservations and the session without completing or committing, which is the abort.
  Cancellation (C12) is the same path.
- Abort-time `close()` must succeed from any state. If it throws anyway, jvector attaches that
  exception as suppressed on the original cause and rethrows the original.
- `complete` or `commit` failing is an abort, never a partial success. Hosts must not publish
  anything before `commit` returns.
- No consumer truncates, deletes, or renames a host resource. What it may touch is exactly the
  bytes it streams.

### 3.5 Instrumentation

- Every byte jvector writes goes through an `IndexWriter` the host provided, so metering is
  decoration on the host side with zero cost when absent. No counters live in the contract.
- jvector reports phases through the `ProgressTracker`/`WorkStage` seam and asks
  `WorkLimiter.acquire(bytes)` before each block it streams, so a host's byte-rate throttle
  applies to output IO.

### 3.6 Performance

- The contract adds no IO layer: the host's stream is the only thing between jvector's writer
  and storage. Standalone, that is one buffered channel writer plus one `rename` at commit.
- Streaming is sequential IO, which every storage tier prefers, and it is the only IO shape
  cloud-backed hosts (Lucene directories, CNDB staging) can give without a local copy.
- The compactor's cost of emitting in order is a bounded reorder buffer (section 6), not extra IO.

## 4. Reference implementation: `FileIndexDestination` (the fast path)

Package-private, backs the three static factories. This is the normative Java IO mapping and
the standalone fast path; it uses nothing a host could not also use.

| Element | Standalone (`toFile`, `toFiles`) | In-file region (`inFile`) |
|---|---|---|
| `reserve(a)` | create `<name>.<random>.tmp` beside the target, open it for writing | open `path` (`WRITE, CREATE`, no truncate) |
| `stream()` | buffered append-only `IndexWriter` over the channel from position `0` | same, from `offset`; `position()` still starts at `0` |
| `complete()` | flush the stream, `channel.force(false)` | same |
| `commit()` | for each artifact: `Files.move(tmp, path, ATOMIC_MOVE, REPLACE_EXISTING)` (a rename on POSIX) | no-op; the caller writes its footer through its own handle after commit |
| abort | delete every temp file; `path` is untouched | nothing to undo beyond releasing the channel; the region's bytes are the caller's to ignore |

Compared with `FileCompactionDestination` on #710 (which wrote the final path directly and
deleted it on abort), the standalone default gains a real transactional boundary: a reader of
`path` never observes a partial graph, and a failed compaction leaves the previous file intact.

## 5. Host mappings

### 5.1 Standalone

```java
compactor.compact(IndexDestination.toFile(Path.of("compacted.index")));
compactor.compact(IndexDestination.toFiles(Map.of(
        OutputArtifact.GRAPH, graphPath,
        OutputArtifact.COMPRESSED_VECTORS, pqPath)));
```

`compact(Path)` and `compact(Path, Path)` remain as one-line delegations to these.

### 5.2 Cassandra SAI

Today `CompactionGraph` builds `new OnDiskGraphIndexWriter.Builder(graph, path).withStartOffset(termsOffset)`,
seeks the writer's output to the end of the terms file to write the SAI header, and after the
graph writes an SAI footer with `writer.checksum()`. PQ goes to a separate component through
`pqOutput.asSequentialWriter()`.

| Call | Host behaviour |
|---|---|
| `open()` | one session per index build or compaction |
| `reserve(GRAPH)` | write the SAI header through its own writer, then vend the terms component's `SequentialWriter` adapted as `IndexWriter`, with `position()` counting from the region start |
| `complete()` | CRC over the region (re-read, as today), write the SAI footer |
| `reserve(COMPRESSED_VECTORS)` | open the PQ component output, write its SAI header, vend it as `IndexWriter`; `complete()` writes the PQ footer |
| `commit()` | mark the components complete |
| abort | mark the components failed |

Because the format stores offsets in the stream's coordinates (3.4), the SAI reader side
changes with the writer: the graph is loaded from `termsOffset` with header-first loading, or
through a reader whose origin is `termsOffset`, rather than through a whole-component footer
search. Files written before the migration keep their absolute-origin reader; the component
metadata can record which convention a file uses.

### 5.3 CNDB

Same as SAI at the stream level. Where the container is remotely backed, the host's stream
writes a local staging file and `commit()` is where the upload or publish happens; abort
discards the staging file. The commit boundary is exactly the point CNDB needs to make the
change visible, and nothing in jvector observes the difference.

### 5.4 OpenSearch / Lucene

Today `JVectorWriter` builds with `OnDiskSequentialGraphIndexWriter` over `JVectorIndexWriter`
(an `IndexWriter` adapter on `IndexOutput` with `position()` but no seek), wraps it in
`CodecUtil.writeIndexHeader`/`writeFooter`, and writes PQ as a blob after the graph in the same
file, recording lengths in its metadata. Merges rebuild with `GraphIndexBuilder`.

| Call | Host behaviour |
|---|---|
| `reserve(GRAPH)` | `directory.createOutput(name)`, `CodecUtil.writeIndexHeader`, then `JVectorIndexWriter` positioned so that region position `0` is the byte after the codec header |
| `complete()` | record the length for the metadata file, `CodecUtil.writeFooter(out)`, close the output |
| `reserve(COMPRESSED_VECTORS)` | either a second codec file, or a second region in the same file after the graph (the plugin's current layout) |
| `commit()` | nothing more: the segment commit is Lucene's |
| abort | `IOUtils.deleteFilesIgnoringExceptions` on the created outputs |

The `IndexOutput` only ever sees sequential writes, so Lucene's incremental checksum and its
`IndexOutput` contract are both respected without any change on the Lucene side, and without a
temporary file or a copy. This answers the question reta left open on #722 about whether two
API flavours are needed: one contract, sequential.

## 6. Consumer changes (separate PR)

Nothing here is part of the contract PR; it shows the contract is sufficient. The key fact,
verified against the code, is that nothing fundamental prevents jvector from writing index data
in order.

**Build path.** `OnDiskSequentialGraphIndexWriter` already writes header, records and footer in
order with no seeks (`graph/disk/OnDiskSequentialGraphIndexWriter.java:103-160`) and is what
the OpenSearch plugin uses. `new Builder(graph, reservation.stream())` works with no new
constructor. The random-access writer's only backward seek rewrites the header at the start so
that separated-feature offsets are correct for readers that ignore the footer
(`RandomAccessOnDiskGraphIndexWriter.java:172-173`); footer-based readers do not need it.

**Compactor.** Its output depends only on its inputs, all known before the first byte:

- The header is complete up front: layer sizes and degrees come from the sources, the entry node
  is resolved, the retrained codebook exists, and only inline features are supported
  (`OnDiskGraphIndexCompactor.java:420-450`).
- Level-0 batches are contiguous slices of each source's node list, submitted in order through
  a bounded window and consumed in completion order (`:1315-1345`); each batch reads only the
  source graphs (`:1010-1070`) and yields fully materialized records with their final offsets.
  Emitting in order means holding finished batches until their predecessors finish. The memory
  bound is the window already in flight today, since the bytes are materialized before the
  write; the cost is head-of-line latency on a slow batch.
- With the offset mapper used in practice, source order is output-ordinal order. With an
  arbitrary mapper, records are produced by iterating new ordinals and resolving back to the
  source, as the sidecar strategy already does (`SidecarCompactionStrategy.java:161`).
- Dead ordinal slots are skipped today and left as zero pages in a sparse file; a stream writes
  them explicitly. Same file size, more bytes through the stream.
- Upper layers, the level-1 PQ records, and the footer are already appended in order.
- Refinement (`refineAfterCompaction`, default `false`, `:87`) is the only in-place rewrite. In
  a streaming world it becomes a second pass producing a new stream, or stays off.
- The sidecar is already written sequentially (`SidecarCompactionStrategy.java:98`); it moves
  from a `Path` to `reserve(COMPRESSED_VECTORS).stream()`.
- Where the fused-PQ pre-encode cache lives when the output is not a jvector-owned file is an
  adoption question (C6), outside this contract.

```java
public void compact(IndexDestination destination) throws IOException {
    try (OutputSession session = destination.open()) {
        try (OutputReservation out = session.reserve(OutputArtifact.GRAPH)) {
            writeGraphInOrder(out.stream());          // header, L0 in ordinal order, upper layers, footer
            out.complete();
        }
        if (sidecarStrategy.writesCodesSidecar()) {
            try (OutputReservation out = session.reserve(OutputArtifact.COMPRESSED_VECTORS)) {
                sidecarStrategy.writeSidecar(out.stream());
                out.complete();
            }
        }
        session.commit();
    }
}
```

## 7. What the reviewers asked, and where it landed

| Thread | Resolution in this draft |
|---|---|
| ashkrisk: does `SeekableSink` duplicate the reader/writer interfaces? | `SeekableSink` is gone. The contract vends the existing `IndexWriter` and adds lifecycle around it; nothing is duplicated. |
| ashkrisk: why `destination.open()` rather than passing the target? | Section 3.3: the destination is stateless configuration; the session and reservation carry single-use state with an enforced state machine. The split is what makes commit-at-most-once and close-exactly-once checkable. |
| MarkWolters: what does `readAt` return at the end? | Moot: there is no positional read in the contract. |
| reta: `OutputReservation` is still path-biased | No `Path` on any vended type; the output is an `IndexWriter`. `Path` survives only in the file-backed static factories. |
| reta: Lucene has no random IO | The contract is sequential by construction (3.1, 5.4); the compactor emits in order (6). |
| ashkrisk: split the memory-safety edits from the interfaces | This contract is its own PR, per jshook's note on #722. The `OnDiskGraphIndex`/`ReaderSupplier` edits stay on #710. |

## 8. Migration from the #710 types

| #710 | This draft |
|---|---|
| `graph.disk.CompactionDestination` | `disk.IndexDestination` (generic: also serves the build-path writers) |
| `CompactionDestination.reserve()` -> `OutputReservation` | `IndexDestination.open()` -> `OutputSession.reserve(artifact)` -> `OutputReservation` |
| `OutputReservation.file()`, `startOffset()` | removed; `stream()` |
| `OutputReservation.commit(bodyLength)` | `OutputReservation.complete()` per artifact plus `OutputSession.commit()` per operation |
| `CompactionDestination.toFile(path)` | `IndexDestination.toFile(path)` with temp-file-and-rename; `toFiles`, `inFile` added |
| `SeekableSink`, `FileChannelSeekableSink`, `TestSeekableSink` | dropped: positional IO is out of scope |
| `FileCompactionDestination` | `FileIndexDestination` |

## 9. Open questions

1. **Size hint.** `reserve(artifact)` carries no projected size. A host doing space accounting
   (CNDB) may want one; it is an additive parameter when a host asks.
2. **Completion payload.** `complete()` carries nothing. Every host computes its own length from
   its stream and its own checksum; if one wants jvector's view for a cross-check, a length or a
   CRC can be added without changing the lifecycle.
3. **Positional access as an optional capability.** If a host wants to let the compactor skip
   the reorder buffer, a capability-gated positional view can be added later. Nothing here
   precludes it.
4. **Refinement.** Off by default; as a second pass it costs one extra copy. Whether it stays a
   positional-only feature or becomes a two-pass streaming feature is an adoption decision.
5. **`Builder(graph, OutputReservation)` convenience** for the build-path writers, and whether
   `OnDiskParallelGraphIndexWriter` should be folded into the sequential writer with a reorder
   buffer once its `Path` requirement is gone.
