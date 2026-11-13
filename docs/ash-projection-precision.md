# Choosing ASH query-projection precision

Query-projection precision is independent of ASH's code parameters `b` and `d`.
Selecting INT8 projection does not change the stored two-bit or four-bit ASH
codes. Training and base-vector encoding retain their existing FP32 behavior.

## Recommended setting

These recommendations concern CPU query throughput and latency. Validate recall
on your data before deploying FP16 or INT8.

| Processor family | Setting | Why |
|---|---|---|
| Intel Xeon Cascade Lake, Cooper Lake, Ice Lake | `auto` (INT8) | Fast integer projection with 256-bit tiles; the required FP16 path is unavailable. |
| Intel Xeon Sapphire Rapids, Emerald Rapids, Granite Rapids | `auto` (INT8) | INT8 with 512-bit tiles is faster than AMX FP16; use `fp16` where recall at low overquery matters more. |
| Intel Xeon Sierra Forest | `auto` (INT8) | Integer acceleration (AVX-VNNI through oneDNN) is the useful low-precision path. |
| AMD Zen 4 and Zen 5: EPYC, Ryzen, Threadripper | `auto` (INT8) | AVX-512 VNNI integer acceleration; these CPUs lack the FP16 instructions required by this backend. |
| Older Intel Xeon and AMD Zen 1–3 | `auto` (FP32) | Without VNNI or FP16 support, `auto` keeps the established FP32 path. |
| Arm Neoverse / AWS Graviton and Apple Silicon | `auto` (FP32) | The established supported path for this release. |

For query throughput and latency on Intel Ice Lake:

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
| `auto` (default) | Use INT8 where an integer dot-product backend is available (x86 VNNI or AMX); otherwise try optimized FP16, with the existing FP32 fallback if FP16 is unavailable or outside its supported range. |
| `fp32` | Use the existing FP32 projection. |
| `fp16` | Require optimized FP16 projection with FP32 accumulation and output. Report an error if unavailable or outside the supported range. |
| `int8` | Require optimized INT8 projection with signed 32-bit integer accumulation and FP32 output. Report an error if unavailable; never silently substitute FP16 or FP32. |

`auto` selects INT8 first because it was the fastest projection on every CPU
measured: 1.03-1.10x FP16 on an AMX CPU and 1.2-1.5x FP32 on Ice Lake class CPUs,
across nine 1M datasets. The query is quantized to two INT8 digits with one scale:
a coarse digit and its rounding remainder, times 254. Nearly all INT8 projection
error came from rounding the query; with the second digit, exhaustive recall
matches FP32 projection, while the matrix keeps one INT8 digit per row.

On AVX-512 VNNI CPUs, INT8 uses JVector's exact-integer tile kernel for single queries
and batches: 512-bit tiles on CPUs with AMX, and 256-bit tiles otherwise, where wide
integer dot products slow the graph search that follows. oneDNN serves INT8 on other
CPUs and chooses the CPU implementation within FP16 at runtime.
Explicit FP16 and INT8 settings fail clearly if the required backend is unavailable, including
when a virtual machine does not expose the needed hardware features.

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

FP16 and INT8 require the native vector provider and the oneDNN add-on included
by the standard native build. See the [native development guide](../jvector-native/src/main/native/README.md#recommended-onednn-build-and-package)
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
