ASH supports explicit INT8 query projection through the existing sequential
oneDNN add-on on qualified x86 VNNI/AMX processors. Select it with
`-Djvector.ash.queryProjection=int8`. Training, base encoding, model/index formats,
and the existing automatic FP16/FP32 policy are unchanged. The implementation
reuses the projection API, batching and staging, with explicit capability,
numerical-range and ABI guards.

See [Choosing ASH query-projection precision](../../ash-projection-precision.md)
for processor-family recommendations and selection behavior.
