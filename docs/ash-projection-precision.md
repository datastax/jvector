# Choosing ASH query-projection precision

Query-projection precision is independent of ASH's code parameters `b` and `d`.
Selecting INT8 projection does not change the stored two-bit or four-bit ASH
codes. Training and base-vector encoding retain their existing FP32 behavior.

## Recommended setting

These recommendations concern CPU query throughput and latency. Validate recall
on your data before deploying FP16 or INT8.

Use the default, `auto`, on every processor. What it selects depends on the
instructions the processor supports:

| Processor support | `auto` selects | Notes |
|---|---|---|
| x86-64 with AVX-512 VNNI and AVX-512 FP16 | INT8, native kernel, 512-bit tiles | Faster than FP16 projection. Use `fp16` where recall at low overquery matters more than throughput. |
| x86-64 with AVX-512 VNNI, without AVX-512 FP16 | INT8, native kernel, 256-bit tiles | FP16 projection is unavailable. |
| x86-64 with AVX-VNNI, without AVX-512 | INT8 through the oneDNN add-on | Without the add-on, `auto` uses FP32. |
| x86-64 without VNNI | FP32 | |
| AArch64 | FP32 | |
| Panama or scalar provider | FP32 | `int8` and `fp16` run in Java there, more slowly than FP32; the native provider is recommended for ASH. |

To require INT8 and fail if it is unavailable, rather than fall back:

```text
-Djvector.ash.queryProjection=int8
```

## Available settings

Set the precision per model in code with
`AsymmetricHashing.setQueryOptions(ASHQueryProjection, maxBatchSize)`, or for
every model in the JVM with `jvector.ash.queryProjection`. Either must take
effect before the model's first query:

| Setting | Behavior |
|---|---|
| `auto` (default) | Use INT8 where an integer dot-product backend is available (x86-64 VNNI); otherwise try optimized FP16, with the existing FP32 fallback if FP16 is unavailable or outside its supported range. |
| `fp32` | Use the existing FP32 projection. |
| `fp16` | Require optimized FP16 projection with FP32 accumulation and output. Report an error if unavailable or outside the supported range. |
| `int8` | Require optimized INT8 projection with signed 32-bit integer accumulation and FP32 output. Report an error if unavailable; never silently substitute FP16 or FP32. |

`auto` selects INT8 first because it was the fastest projection on every CPU
measured: 1.03-1.10x FP16 on a CPU with AVX-512 FP16 and AMX, and 1.2-1.5x FP32 on
AVX-512 VNNI CPUs without FP16, across nine 1M datasets. The query is quantized to two INT8 digits with one scale:
a coarse digit and its rounding remainder, times 254. Nearly all INT8 projection
error came from rounding the query; with the second digit, exhaustive recall
matches FP32 projection, while the matrix keeps one INT8 digit per row.

On AVX-512 VNNI CPUs, INT8 uses JVector's exact-integer Highway kernel for single
queries and batches, in the native library itself: 512-bit tiles on CPUs that also
have AVX-512 FP16, and 256-bit tiles on other AVX-512 VNNI CPUs, where 512-bit integer
dot products slowed the graph search that follows them. The oneDNN add-on serves INT8
on CPUs with AVX-VNNI but not AVX-512, and chooses the CPU implementation within FP16
at runtime. Setting
the environment variable `JVECTOR_ASH_INT8_ONEDNN` makes INT8 use the add-on
everywhere, for example to compare its AMX kernels; both compute the same integers,
so the results are identical.
Explicit FP16 and INT8 settings fail clearly if the required backend is unavailable, including
when a virtual machine does not expose the needed hardware features. `setQueryOptions` checks
its precision when called; a precision from the `jvector.ash.queryProjection` property is checked at
the model's first query, because training, encoding and compaction never depend on query settings.
To check before building anything, call `ASHQueryProjection.FP16.isSupported()` (or
`requireSupported()`, which throws with the reason and what to use instead).

The model caches its query-projection view on first use. Changing the property
afterwards does not reconfigure an existing view, and `setQueryOptions` throws
once the view exists.

## BenchYAML runs

Set `queryProjection` in the `parameters` map of an ASH compression entry.
The [BenchYAML Guide](benchyaml-guide.md#ash-query-parameters) defines the
accepted values. An explicit `-Djvector.ash.queryProjection=...` overrides YAML.
The default is `auto`. Use one projection mode per JVM.

`experiments.csv` adds two columns for ASH cases:

- `ash_query_projection_requested`: the selected setting after JVM/YAML precedence.
- `ash_query_projection_effective`: the prepared projection's actual precision.
  `auto` can resolve to `int8`, `fp16` or `fp32`; `fp16+fp32` means the automatic FP16
  view has used its FP32 range fallback.

The columns are blank for non-ASH cases. Effective precision is blank if no
query projection was initialized. Reporting reads the existing view after
measurement; it does not prepare another view or infer precision from the CPU.

## Effect on graph construction

BenchYAML defaults to `symmetric-all` construction: insertion and diversity use
encoded symmetric scoring, without dense query projection for insertion. Query
precision therefore affects application queries, not insertion under this default.

Select `scoring: mixed` on the ASH construction entry to use asymmetric
insertion and symmetric diversity scoring. Query projection then applies
during insertion.
Library callers explicitly choose an encoded node score provider or an
original-vector insertion overload for their graph builders. See [ASH configuration and defaults](ash-configuration.md#graph-construction).

Neither construction mode changes training or base-encoding precision. To
measure query-projection effects independently, query the same saved graph
under each precision in separate JVM runs.

## Deployment

FP16 requires the native vector provider, a CPU with AVX-512 FP16, and the oneDNN
add-on. INT8 requires the native vector provider, and the add-on only on CPUs
without AVX-512 VNNI. Published release JARs include the add-on; source builds
add it with `-Dnative.onednn=true`, which also needs CMake. See the [native development guide](../jvector-native/src/main/native/README.md#recommended-onednn-build-and-package)
for build commands, supported runtime behavior, diagnostics and tests.

## Validate a selection

Use `DistancesASH` to check projection throughput and numerical behavior.
Use standard BenchYAML configurations to measure graph QPS, recall and latency
at the application's physical-core budget. Compare query precisions on the
same graph; measure construction separately when insertion precision changes.
A projection-throughput gain does not imply the same graph-query speedup.

## References

- [Processor instruction sets](https://gcc.gnu.org/onlinedocs/gcc-15.1.0/gcc/x86-Options.html).
- [oneDNN CPU runtime dispatch](https://github.com/uxlfoundation/oneDNN/blob/main/README.md).
- [oneDNN INT8 arithmetic](https://uxlfoundation.github.io/oneDNN/dev_guide_int8_computations.html).

The repository pins its oneDNN version in
`jvector-native/src/main/onednn/subprojects/onednn.wrap`.
