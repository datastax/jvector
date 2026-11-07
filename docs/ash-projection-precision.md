<!--
Copyright DataStax, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Choosing ASH query-projection precision

Query-projection precision is independent of ASH's code parameters `b` and `d`.
Selecting INT8 projection does not change the stored two-bit or four-bit ASH
codes. Training and base-vector encoding retain their existing FP32 behavior.

## Recommended setting

These recommendations concern CPU query throughput and latency. Validate recall
on your data before deploying FP16 or INT8.

| Processor family | Setting | Why |
|---|---|---|
| Intel Xeon Cascade Lake, Cooper Lake, Ice Lake | `int8` | Fast integer projection; the required FP16 path is unavailable. |
| Intel Xeon Sapphire Rapids, Emerald Rapids, Granite Rapids | `fp16` | Fast FP16 projection with less quantization than INT8. |
| Intel Xeon Sierra Forest | `int8` | Integer acceleration is the useful low-precision path. |
| AMD Zen 4 and Zen 5: EPYC, Ryzen, Threadripper | `int8` | VNNI integer acceleration; these CPUs lack the FP16 instructions required by this backend. |
| Older Intel Xeon and AMD Zen 1–3 | `fp32` | Use the established FP32 path. |
| Arm Neoverse / AWS Graviton and Apple Silicon | `fp32` | Use the established supported path for this release. |

For query throughput and latency on Intel Ice Lake:

```text
-Djvector.ash.queryProjection=int8
```

## Available settings

Set `jvector.ash.queryProjection` before the model's first query use:

| Setting | Behavior |
|---|---|
| `auto` (default) | Try optimized FP16; retain the existing FP32 fallback if FP16 is unavailable or outside its supported range. Never select INT8 automatically. |
| `fp32` | Use the existing FP32 projection. |
| `fp16` | Require optimized FP16 projection with FP32 accumulation and output. Report an error if unavailable or outside the supported range. |
| `int8` | Require optimized INT8 projection with signed 32-bit integer accumulation and FP32 output. Report an error if unavailable; never silently substitute FP16 or FP32. |

`auto` preserves the existing precision policy. It does not promise to select
the fastest precision. INT8 remains an explicit accuracy/performance choice.

oneDNN chooses the CPU implementation within the selected precision at runtime.
Explicit FP16 and INT8 settings fail clearly if the required backend is unavailable, including
when a virtual machine does not expose the needed hardware features.

The model caches its query-projection view on first use. Changing the property
after initialization does not reconfigure an existing view.

## BenchYAML runs

Set `queryProjection` in the `parameters` map of an ASH compression entry.
The [BenchYAML Guide](benchyaml-guide.md#ash-query-parameters) defines the
accepted values. An explicit `-Djvector.ash.queryProjection=...` overrides YAML.
The default is `auto`. Use one projection mode per JVM.

`experiments.csv` adds two columns for ASH cases:

- `ash_query_projection_requested`: the selected setting after JVM/YAML precedence.
- `ash_query_projection_effective`: the prepared projection's actual precision.
  `auto` can resolve to `fp16` or `fp32`; `fp16+fp32` means the automatic FP16
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
`jvector-native/src/main/native/subprojects/onednn.wrap`.
