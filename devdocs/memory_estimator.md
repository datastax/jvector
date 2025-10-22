## Estimating Memory Requirements

JVector includes a core `MemoryCostEstimator` utility that projects the RAM footprint of a graph build before you invest in the full construction. Unlike the generic `RamUsageEstimator` (which inspects existing Java objects), this tool is domain-aware: it spins up a *representative* mini-index with `GraphIndexBuilder`, records the bytes consumed by graph structures, PQ vectors/codebooks, and thread-local buffers, then extrapolates what a full-scale build would cost. In other words, it answers “how much RAM will this configuration require?” rather than “how much does this in-memory object currently use?”.

Key differences from `RamUsageEstimator`:

- **Configuration-driven.** You supply an `IndexConfig`; the estimator knows about graph degree, hierarchy, overflow ratio, PQ knobs, etc.
- **Includes build/serving buffers.** It models the per-thread scratch space `GraphIndexBuilder` and `GraphSearcher` allocate, which a generic object walk would miss.
- **Deployment profiles.** Through `MemoryModel.breakdownForProfile` you can inspect scenarios like MINIMAL_MEMORY, GRAPH_IN_MEMORY_PQ, etc.
- **Predictive rather than introspective.** You can estimate requirements before ever building the full index.

Use `RamUsageEstimator` when you need an exact footprint of *current* objects (e.g., debugging) and `MemoryCostEstimator` when planning or sizing JVector indexes.

### Quick Start

```java
int dimension = 768;

MemoryCostEstimator.IndexConfig config = MemoryCostEstimator.IndexConfig.defaultConfig(dimension);
MemoryCostEstimator.MemoryModel model = MemoryCostEstimator.createModel(config, 2_000);

MemoryCostEstimator.Estimate serving = model.estimateBytes(10_000_000);                 // 10M vectors, steady state
MemoryCostEstimator.Estimate indexing = model.estimateBytesWithIndexingBuffers(10_000_000, 16); // 16 build threads
MemoryCostEstimator.Estimate searching = model.estimateBytesWithSearchBuffers(10_000_000, 64);  // 64 query threads

long centralBytes = serving.value();
long margin = serving.marginBytes();
System.out.printf("Servicing: %d bytes ± %d bytes (%.0f%%)\n", centralBytes, margin, serving.marginFraction() * 100);
```

Pick a sample size between 1,000 and 10,000 vectors; larger samples tighten the projection at the cost of a longer warm-up build. The returned `MemoryModel` offers `estimateBytes` for steady-state usage plus helpers to account for thread-local scratch space during indexing or query serving. Each method now returns an `Estimate` (central value and ±margin so you can reason about best/worst cases directly).

### Accuracy

`MemoryCostEstimatorAccuracyTest` exercises the estimator against real graph builds with and without PQ. The measured footprint must stay within a 20% tolerance of the prediction. Current coverage includes

* Hierarchical, M=16, no PQ — 64–4096 dims show +7.8–12.1% relative error (still within the 20% guardrail)
* Hierarchical, M=16, PQ (16×256 up to 1024d, 32×256 for ≥2048d) — 64–4096 dims land between +0.9% and +9.3%
* Flat, M=32, no PQ — 64–4096 dims sit in the +0.9–1.8% range
* Flat, M=24, PQ (16×256 up to 1024d, 32×256 for ≥2048d) — 64–4096 dims stay within +0.04–2.4%

Feel free to add more scenarios if you tune different configurations regularly—the test harness makes it straightforward to expand the matrix.
