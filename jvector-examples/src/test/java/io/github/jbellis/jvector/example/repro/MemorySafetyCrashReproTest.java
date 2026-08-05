/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.example.repro;

import org.junit.Assume;
import org.junit.Test;

import java.nio.file.Path;
import java.util.Locale;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

/// Reader-level reproductions for the compaction memory-safety theories (see
/// `local/memory_safety.md` on the cooperative-embedding branch). Each scenario runs in a child
/// JVM via [ChildJvm] because the expected outcome of several of them is a native JVM crash; the
/// parent asserts on exit status, harness outcome markers, and the hs_err evidence (`siginfo`
/// si_code and the problematic frame — the discriminators the bug doc's trigger table needs).
///
/// The matrix is reader-kind x invalidation-trigger:
///
/// | theory | reader                          | trigger        | expectation under test                       |
/// |--------|---------------------------------|----------------|----------------------------------------------|
/// | T1     | Arena (`MemorySegmentReader`)   | clean close    | Java exception, never a native crash         |
/// | T2     | Arena (`MemorySegmentReader`)   | truncate       | catastrophic-or-loud (SIGBUS crash or error) |
/// | T2b    | host-style NIO mmap             | truncate       | catastrophic-or-loud (SIGBUS crash or error) |
/// | T3     | host-style NIO mmap             | clean unmap    | native SIGSEGV in the swap-copy leaf         |
/// | T3b    | `SimpleMappedReader` (shipped)  | supplier close | native crash (its close IS a raw unmap)      |
///
/// T1 validates the bug doc's §3 deduction for the arena reader. T3/T3b test the refinement that
/// the deduction does NOT extend to NIO-mapped readers: for those, a clean close of a source
/// while a read is in flight is itself sufficient to produce the production core dump — no file
/// truncation or rewrite required.
public class MemorySafetyCrashReproTest {

    private static void assumeLinux() {
        Assume.assumeTrue("crash reproductions rely on Linux mmap semantics and hs_err parsing",
                System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("linux"));
    }

    private static void assumeArenaReader() {
        Assume.assumeTrue("jvector-native MemorySegmentReader (JDK 22+) not available on this classpath",
                MemorySafetyReproHarness.arenaReaderAvailable());
    }

    private static ChildJvm.Result runReader(String label, String kind, String trigger) throws Exception {
        Path dir = ReproGraphs.newWorkDir(label);
        return ChildJvm.run(dir, 180, "reader", kind, trigger, dir.toString());
    }

    private static void verdict(String theory, String text) {
        System.out.println("VERDICT|" + theory + "|" + text);
    }

    /// T1: closing the Arena-managed supplier while reads are in flight must degrade to a Java
    /// exception (or a refused close) — never a native crash. This is the safe half of the
    /// reader matrix and validates the doc's "clean close cannot core-dump" deduction *for this
    /// reader implementation*.
    @Test(timeout = 240_000)
    public void t1_arenaReaderCloseUnderLiveReadsIsSafe() throws Exception {
        assumeLinux();
        assumeArenaReader();
        ChildJvm.Result r = runReader("t1-arena-close", "arena", "close");

        assertFalse("T1 DISPROVEN: clean Arena close crashed natively: " + r.summary(), r.crashed());
        assertTrue("arena close must stop readers with an exception or refuse to close: " + r.summary(),
                r.exitCode == 0 || r.exitCode == 43);
        verdict("T1", "PROVEN safe: " + r.summary());
    }

    /// T2: truncating the file under the Arena reader's live byte-swapping reads. The doc's §5a
    /// predicts a SIGBUS crash in `Copy::conjoint_swap`; modern JDKs may instead rescue the fault
    /// into `java.lang.InternalError`. Either way the read must not silently keep succeeding —
    /// which outcome actually occurs decides how the doc's trigger table should read for this
    /// reader, so the verdict line records it.
    @Test(timeout = 240_000)
    public void t2_arenaReaderTruncationIsCatastrophicOrLoud() throws Exception {
        assumeLinux();
        assumeArenaReader();
        ChildJvm.Result r = runReader("t2-arena-truncate", "arena", "truncate");

        assertNotEquals("T2 DISPROVEN: truncation had no effect on live reads: " + r.summary(), 42, r.exitCode);
        boolean sigbusCrash = r.crashed() && r.hsErrContains("SIGBUS");
        boolean survivedWithError = !r.crashed() && r.exitCode == 0;
        assertTrue("truncation under live mmap reads must crash with SIGBUS or surface a Java error: " + r.summary(),
                sigbusCrash || survivedWithError);
        verdict("T2", (sigbusCrash ? "SIGBUS crash (doc §5a expectation holds): " : "no crash — fault surfaced as a Java error (doc §5a needs revision for this JDK): ")
                + r.summary());
    }

    /// T2b: the same truncation, through the host-style NIO reader.
    @Test(timeout = 240_000)
    public void t2b_hostStyleReaderTruncationIsCatastrophicOrLoud() throws Exception {
        assumeLinux();
        ChildJvm.Result r = runReader("t2b-hoststyle-truncate", "hoststyle", "truncate");

        assertNotEquals("T2b DISPROVEN: truncation had no effect on live reads: " + r.summary(), 42, r.exitCode);
        boolean sigbusCrash = r.crashed() && r.hsErrContains("SIGBUS");
        boolean survivedWithError = !r.crashed() && r.exitCode == 0;
        assertTrue("truncation under live NIO mmap reads must crash with SIGBUS or surface a Java error: " + r.summary(),
                sigbusCrash || survivedWithError);
        verdict("T2b", (sigbusCrash ? "SIGBUS crash: " : "no crash — fault surfaced as a Java error: ") + r.summary());
    }

    /// T3: a *clean close* (raw munmap, no handshake) of the host-style NIO mapping under live
    /// byte-swapping reads must crash natively with SIGSEGV in the same swap-copy leaf as the
    /// production core dump. This is the refinement of the bug doc: for host-style readers, early
    /// release of a source is sufficient — no truncation or in-place rewrite is required.
    @Test(timeout = 240_000)
    public void t3_hostStyleReaderCleanUnmapCrashesNatively() throws Exception {
        assumeLinux();
        ChildJvm.Result r = runReader("t3-hoststyle-close", "hoststyle", "close");

        assertTrue("T3 DISPROVEN: clean unmap under live reads did not crash: " + r.summary(), r.crashed());
        assertTrue("expected SIGSEGV (unmapped pages), got: " + r.siginfo() + " / " + r.summary(),
                r.hsErrContains("SIGSEGV"));
        assertTrue("expected the byte-swapping copy leaf in the crash evidence: frame=" + r.problematicFrame(),
                r.hsErrContains("copySwapMemory") || r.hsErrContains("conjoint_swap") || r.hsErrContains("FloatBufferS"));
        verdict("T3", "PROVEN: clean unmap of a host-style mapping crashes natively: " + r.summary());
    }

    /// T3b: jvector's own shipped [io.github.jbellis.jvector.disk.SimpleMappedReader] has the same
    /// property — its `Supplier.close()` is `Unsafe.invokeCleaner`, a raw munmap. Any caller that
    /// closes it while a straggler read is in flight gets a native crash, not an exception.
    @Test(timeout = 240_000)
    public void t3b_simpleMappedReaderSupplierCloseCrashesNatively() throws Exception {
        assumeLinux();
        ChildJvm.Result r = runReader("t3b-simplemapped-close", "simplemapped", "close");

        assertTrue("T3b DISPROVEN: SimpleMappedReader.Supplier.close() under live reads did not crash: " + r.summary(),
                r.crashed());
        assertTrue("expected a memory fault signal, got: " + r.siginfo(),
                r.hsErrContains("SIGSEGV") || r.hsErrContains("SIGBUS"));
        verdict("T3b", "PROVEN: shipped SimpleMappedReader close is a raw unmap and crashes live readers: " + r.summary());
    }
}
