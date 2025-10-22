## Estimating Memory Requirements

JVector includes a core `MemoryCostEstimator` utility that projects the RAM footprint of a graph build before you invest in the full construction. Unlike the generic `RamUsageEstimator` (which inspects existing Java objects), this tool is domain-aware: it spins up a *representative* mini-index with `GraphIndexBuilder`, records the bytes consumed by graph structures, PQ vectors/codebooks, and thread-local buffers, then extrapolates what a full-scale build would cost. In other words, it answers “how much RAM will this configuration require?” rather than “how much does this in-memory object currently use?”.

Key differences from `RamUsageEstimator`:

- **Configuration-driven.** You supply an `IndexConfig`; the estimator knows about graph degree, hierarchy, overflow ratio, PQ knobs, etc.
- **Includes build/serving buffers.** It models the per-thread scratch space that `GraphIndexBuilder` and `GraphSearcher` allocate, which a generic object walk would miss.
- **Ram bytes roll-up.** Instead of calling `ramBytesUsed()` on each structure and summing blindly, the estimator samples per-layer node measurements, separates fixed from per-node overhead, and extrapolates—giving more nuance than the aggregate numbers you get from `RamUsageEstimator`.
- **Predictive rather than introspective.** You can estimate requirements before ever building the full index; `RamUsageEstimator` only reports on objects you already have.

Use `RamUsageEstimator` when you need an exact footprint of *current* objects (e.g., debugging) and `MemoryCostEstimator` when planning or sizing JVector indexes.

### Quick Start

```java
int dimension = 768;

MemoryCostEstimator.IndexConfig config = MemoryCostEstimator.IndexConfig.defaultConfig(dimension);
MemoryCostEstimator.MemoryModel model = MemoryCostEstimator.createModel(config, 2_000);

MemoryCostEstimator.Estimate serving = model.estimateBytes(10_000_000); // 10M vectors, steady state
MemoryCostEstimator.Estimate indexing = model.estimateBytesWithIndexingBuffers(10_000_000, 16); // 16 build threads
MemoryCostEstimator.Estimate searching = model.estimateBytesWithSearchBuffers(10_000_000, 64);  // 64 query threads

long centralBytes = serving.value();
long margin = serving.marginBytes();
System.out.printf("Servicing: %d bytes ± %d bytes (%.0f%%)\n", centralBytes, margin, serving.marginFraction() * 100);
```

Pick a sample size between 1,000 and 10,000 vectors; larger samples tighten the projection at the cost of a longer warm-up build. The returned `MemoryModel` offers `estimateBytes` for steady-state usage plus helpers to account for thread-local scratch space during indexing or query serving. Each method now returns an `Estimate` (central value and ±margin so you can reason about best/worst cases directly).

### Accuracy

`MemoryCostEstimatorAccuracyTest` exercises the estimator against real graph builds with and without PQ. The measured footprint must stay within a 20% tolerance of the prediction. Current coverage includes

* Hierarchical, M=16, no PQ — 64–4096 dims stay within +0.6–1.7% relative error (well below the 20% guardrail)
* Hierarchical, M=16, PQ (16×256 up to 1024d, 32×256 for ≥2048d) — 64–4096 dims land between +0.04% and +1.45%
* Flat, M=32, no PQ — 64–4096 dims sit in the +0.0–1.0% range
* Flat, M=24, PQ (16×256 up to 1024d, 32×256 for ≥2048d) — 64–4096 dims stay within +0.0–1.2%
* Hierarchical, high degree (M=48, no PQ) — 64–4096 dims remain inside +0.2–1.0%
* Hierarchical, cosine similarity (M=16, no PQ) — 64–4096 dims land between +0.6–2.3%

If you tune different configurations regularly (e.g., larger degrees or alternative similarity functions), extend the matrix so the backing data stays relevant.
