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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.disk.FvecFileVectorValues;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSetLoaderSimpleMFD;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.VectorAccess;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.ref.Reference;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/** Measures sample retrieval, quantizer training, full-set encoding and an encoded-score sink. */
public final class VectorPrefetchBenchmark {
    private VectorPrefetchBenchmark() {}

    public static void main(String[] args) throws Exception {
        run(args);
    }

    private static void run(String[] args) throws Exception {
        var options = Options.parse(args);
        if (options.contains("help")) {
            System.out.println("Usage: VectorPrefetchBenchmark"
                    + " [--dataset catalog-name | --file vectors.fvecs] [--samples 128000]"
                    + " [--mode all|preload|demand|prefetch] [--quantizer nvq|pq|adapter-class] [--quantizer-options key=value,...] [--loader-options key=value,...]"
                    + " [--metric DOT_PRODUCT|EUCLIDEAN|COSINE] [--query-file queries.fvecs]"
                    + " [--cache cold|uncontrolled] [--seed 42] [--progress-seconds 30]");
            return;
        }
        if (!options.contains("child")) {
            Path file = options.file(); // Finish catalog/download preparation before launching timed arms.
            var modes = options.get("mode", "all").equals("all")
                    ? List.of("preload", "demand", "prefetch") : List.of(options.get("mode", "all"));
            var rows = new ArrayList<String>();
            for (var mode : modes) {
                options.checkMode(mode);
                var command = new ArrayList<String>();
                command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
                // Preserve IDE/CLI heap, provider, CPU and native-access settings in each isolated arm.
                for (var arg : ManagementFactory.getRuntimeMXBean().getInputArguments()) {
                    if (!arg.startsWith("-agentlib:") && !arg.startsWith("-javaagent:")) command.add(arg);
                }
                command.add("-classpath"); command.add(System.getProperty("java.class.path"));
                command.add(VectorPrefetchBenchmark.class.getName());
                command.addAll(options.childArguments(file, mode));
                System.out.println("\nStarting " + mode + " in a fresh JVM");
                var child = new ProcessBuilder(command).redirectErrorStream(true).start();
                var stopChild = new Thread(child::destroy, "stop-input-benchmark-arm");
                Runtime.getRuntime().addShutdownHook(stopChild);
                try {
                    try (var output = new java.io.BufferedReader(new java.io.InputStreamReader(child.getInputStream()))) {
                        String line;
                        while ((line = output.readLine()) != null) {
                            System.out.println(line);
                            if (line.startsWith("| pipeline |")) rows.add(line);
                        }
                    }
                    int exit = child.waitFor(); // One arm at a time, including when launched from IntelliJ.
                    if (exit != 0) throw new IllegalStateException(mode + " failed (exit " + exit + "); remaining arms not run");
                } catch (InterruptedException e) {
                    child.destroy();
                    child.waitFor();
                    Thread.currentThread().interrupt();
                    throw e;
                } finally { Runtime.getRuntime().removeShutdownHook(stopChild); }
            }
            System.out.println("\nComparison (seconds; quantizer=" + options.get("quantizer", "nvq")
                    + ", options=" + options.get("quantizer-options", "defaults") + ", metric=" + options.get("metric", "DOT_PRODUCT")
                    + ", fresh JVM per arm, cache=" + options.get("cache", "cold") + "):");
            printHeader();
            rows.forEach(System.out::println);
            return;
        }
        measure(options);
    }

    private static void measure(Options options) throws Exception {
        Path file = Path.of(options.get("file", ""));
        String mode = options.get("mode", "all");
        options.checkMode(mode);
        var metric = VectorSimilarityFunction.valueOf(options.get("metric", "DOT_PRODUCT").toUpperCase(Locale.ROOT));
        String quantizer = options.get("quantizer", "nvq");
        Quantizer quantizerFactory = quantizer(quantizer);
        var quantizerOptions = parameters(options.get("quantizer-options", ""));
        var loaderOptions = parameters(options.get("loader-options", ""));
        String cache = options.get("cache", "cold");
        if (!cache.equals("cold") && !cache.equals("uncontrolled")) throw new IllegalArgumentException("Unknown cache policy: " + cache);
        var executor = PhysicalCoreExecutor.pool();
        int count, dimension;
        io.github.jbellis.jvector.vector.types.VectorFloat<?> query;
        // Prepare one query outside timing. By default hold out the last base vector in every arm.
        boolean heldOut = !options.contains("query-file");
        try (var probe = new DemandFvecVectorValues(file)) {
            dimension = probe.dimension();
            count = probe.size() - (heldOut ? 1 : 0);
            if (count <= 0) throw new IllegalArgumentException("Need at least one candidate and a query");
            if (heldOut) query = probe.getVector(probe.size() - 1).copy();
            else try (var queries = new DemandFvecVectorValues(Path.of(options.get("query-file", "")))) {
                if (queries.size() == 0 || queries.dimension() != dimension) throw new IllegalArgumentException("Invalid query file");
                query = queries.getVector(0).copy();
            }
        }
        int[] ordinals = sample(count, options.positiveInt("samples", Math.min(count, 128000)), options.longValue("seed", 42));
        int selectedCount = ordinals.length;
        if (quantizer.equals("pq") && selectedCount < integer(quantizerOptions, "clusters", 256)) throw new IllegalArgumentException("PQ requires at least the configured cluster count of selected vectors");
        System.out.printf("Pipeline: %s, encoding candidates=%,d, training samples=%,d, dimensions=%d, compute workers=%d%n",
                mode, count, selectedCount, dimension, executor.getParallelism());
        System.out.printf("Quantizer=%s, metric=%s, sample seed=%d, query=%s, provider=%s%n",
                quantizer, metric, options.longValue("seed", 42), heldOut ? "held-out final base vector" : "first query-file vector",
                VectorizationProvider.getInstance().getClass().getSimpleName());
        System.out.println("Quantizer options: " + quantizerOptions + "; loader options: " + loaderOptions);
        System.out.println("Input: " + file.toAbsolutePath());
        if (cache.equals("cold")) ColdFileCache.evictAndVerify(file);
        else System.out.println("OS file cache: uncontrolled (NOT a cold-cache measurement)");

        try (var progress = new Progress(options.nonnegativeInt("progress-seconds", 30))) {
            long begin = System.nanoTime();
            double preloadSeconds = 0;
            progress.phase(mode.equals("preload") ? "preload preparation (excluded)" : "sample retrieval");
            try (var input = loader(mode).open(file, loaderOptions)) {
                var original = input.values();
                if (original.size() != count + (heldOut ? 1 : 0) || original.dimension() != dimension)
                    throw new IllegalStateException("Loader changed source dimensions or vector count");
                RandomAccessVectorValues vectors = prefix(original, count);
                if (mode.equals("preload")) {
                    preloadSeconds = seconds(System.nanoTime() - begin);
                    System.out.printf(Locale.ROOT, "Resident preload preparation: %.3f s (excluded from comparison)%n", preloadSeconds);
                    // The resident baseline measures work after all input vectors are available.
                    begin = System.nanoTime();
                    progress.phase("sample retrieval");
                }
                RandomAccessVectorValues trainingVectors = new ListRandomAccessVectorValues(
                        VectorAccess.copySelected(vectors, ordinals, executor), dimension);
                long loaded = System.nanoTime();
                progress.phase("train");
                Encoder encoder = quantizerFactory.train(trainingVectors, quantizerOptions, executor);
                long trained = System.nanoTime();
                progress.phase("encode");
                var encoded = encoder.encode(vectors, executor);
                if (encoded.count() != vectors.size()) throw new IllegalStateException("Encoding changed candidate count");
                long encodedAt = System.nanoTime();
                progress.phase("nearest-vector sink");
                var scorer = encoded.scorer(query, metric);
                int nearest = -1;
                double best = Double.NEGATIVE_INFINITY;
                double checksum = 0;
                for (int i = 0; i < vectors.size(); i++) {
                    double score = scorer.applyAsDouble(i);
                    if (!Double.isFinite(score)) throw new IllegalStateException("Non-finite score at selected position " + i);
                    checksum += score;
                    if (score > best) { best = score; nearest = i; }
                }
                long finished = System.nanoTime();
                progress.phase("complete");
                int ordinal = nearest; // Encoding and scoring always retain full base ordinals.
                printHeader();
                System.out.printf(Locale.ROOT, "| %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %d | %.8f | %.9f |%n",
                        "pipeline", mode, mode.equals("preload") ? String.format(Locale.ROOT, "%.3f", preloadSeconds) : "—", seconds(loaded - begin), seconds(trained - loaded),
                        seconds(encodedAt - trained), seconds(finished - encodedAt), seconds(finished - begin), ordinal, best, checksum);
                String statistics = input.statistics();
                if (!statistics.isEmpty()) System.out.println(statistics);
                Reference.reachabilityFence(input); // Keep all resident preload vectors live through the sink.
            }

        }
    }


    /** Factory for any representation (PQ, NVQ, ASH or future types), without changing core APIs. */
    public interface Quantizer {
        Encoder train(RandomAccessVectorValues vectors, java.util.Map<String, String> parameters,
                      java.util.concurrent.ForkJoinPool executor);
    }

    /** Prepared representation: encode the candidate set through its normal production API. */
    @FunctionalInterface
    public interface Encoder {
        Encoded encode(RandomAccessVectorValues vectors, java.util.concurrent.ForkJoinPool executor);
    }

    /** A minimal nearest-vector sink contract, independent of a particular vector class. */
    public interface Encoded {
        int count();
        java.util.function.IntToDoubleFunction scorer(
                io.github.jbellis.jvector.vector.types.VectorFloat<?> query, VectorSimilarityFunction metric);
    }

    /** Adapter for existing production compressors; other vector types can supply Encoded directly. */
    public static Encoder compressed(VectorCompressor<?> compressor) {
        return (vectors, executor) -> {
            var encoded = compressor.encodeAll(vectors, executor);
            return new Encoded() {
                public int count() { return encoded.count(); }
                public java.util.function.IntToDoubleFunction scorer(
                        io.github.jbellis.jvector.vector.types.VectorFloat<?> query, VectorSimilarityFunction metric) {
                    var function = encoded.scoreFunctionFor(query, metric);
                    return function::similarityTo;
                }
            };
        };
    }

    /** Future vector sources can expose BatchedVectorValues; ordinary RAVV also works. */
    public interface Loader {
        Input open(Path file, java.util.Map<String, String> parameters) throws Exception;
    }

    /** Owns source resources until both quantization and the sink have finished. */
    public interface Input extends AutoCloseable {
        RandomAccessVectorValues values();
        default String statistics() { return ""; }
        @Override void close() throws Exception;
    }

    private static Quantizer quantizer(String name) throws Exception {
        if (name.equals("nvq")) return (vectors, parameters, executor) -> {
            requireKeys(parameters, Set.of("subvectors", "learn"));
            var nvq = NVQuantization.compute(vectors, integer(parameters, "subvectors", 1));
            nvq.learn = bool(parameters, "learn", true);
            return compressed(nvq);
        };
        if (name.equals("pq")) return (vectors, parameters, executor) -> {
            requireKeys(parameters, Set.of("subspaces", "clusters", "center", "anisotropic"));
            return compressed(ProductQuantization.compute(vectors, integer(parameters, "subspaces", Math.max(1, vectors.dimension() / 8)),
                    integer(parameters, "clusters", 256), bool(parameters, "center", false),
                    Float.parseFloat(parameters.getOrDefault("anisotropic", "-1")), executor, executor));
        };
        return (Quantizer) Class.forName(name).getConstructor().newInstance();
    }

    private static Loader loader(String name) throws Exception {
        if (name.equals("preload")) return (file, parameters) -> {
            requireKeys(parameters, Set.of());
            var all = SiftLoader.readFvecs(file.toString());
            var values = new ListRandomAccessVectorValues(all, all.get(0).length());
            return new Input() {
                public RandomAccessVectorValues values() { return values; }
                public void close() { }
            };
        };
        if (name.equals("demand")) return (file, parameters) -> {
            requireKeys(parameters, Set.of());
            var source = new DemandFvecVectorValues(file);
            return new Input() {
                public RandomAccessVectorValues values() { return source; }
                public void close() throws IOException { source.close(); }
                public String statistics() { return String.format("Source logical reads=%,d, bytes=%,d (demand-only)", source.reads(), source.bytes()); }
            };
        };
        if (name.equals("prefetch")) return (file, parameters) -> {
            requireKeys(parameters, Set.of("ioThreads", "batchVectors", "readAhead", "maxBufferBytes"));
            var options = FvecFileVectorValues.Options.defaults();
            if (parameters.containsKey("ioThreads")) options = options.withIoThreads(integer(parameters, "ioThreads", 1));
            if (parameters.containsKey("batchVectors")) options = options.withBatchVectors(integer(parameters, "batchVectors", 64));
            if (parameters.containsKey("readAhead")) options = options.withReadAhead(integer(parameters, "readAhead", 3));
            if (parameters.containsKey("maxBufferBytes")) options = options.withMaxBufferBytes(Long.parseLong(parameters.get("maxBufferBytes")));
            var source = FvecFileVectorValues.open(file, options);
            return new Input() {
                public RandomAccessVectorValues values() { return source; }
                public void close() throws IOException { source.close(); }
                public String statistics() {
                    var stats = source.statistics();
                    return String.format("Source logical reads=%,d, bytes=%,d, payload bytes=%,d, payload limit=%,d",
                            stats.reads, stats.bytesRead, stats.bufferBytes, stats.bufferLimitBytes);
                }
            };
        };
        return (Loader) Class.forName(name).getConstructor().newInstance();
    }

    private static java.util.Map<String, String> parameters(String input) {
        var values = new LinkedHashMap<String, String>();
        if (!input.isEmpty()) for (var entry : input.split(",", -1)) {
            var pair = entry.split("=", 2);
            if (pair.length != 2 || pair[0].isEmpty() || pair[1].isEmpty() || values.putIfAbsent(pair[0], pair[1]) != null)
                throw new IllegalArgumentException("Expected unique key=value parameters: " + input);
        }
        return java.util.Collections.unmodifiableMap(values);
    }
    private static void requireKeys(java.util.Map<String, String> parameters, Set<String> keys) {
        for (var key : parameters.keySet()) if (!keys.contains(key)) throw new IllegalArgumentException("Unknown parameter: " + key);
    }
    private static int integer(java.util.Map<String, String> parameters, String key, int fallback) {
        return Integer.parseInt(parameters.getOrDefault(key, Integer.toString(fallback)));
    }
    private static boolean bool(java.util.Map<String, String> parameters, String key, boolean fallback) {
        String value = parameters.getOrDefault(key, Boolean.toString(fallback));
        if (!value.equals("true") && !value.equals("false")) throw new IllegalArgumentException("Invalid boolean for " + key);
        return Boolean.parseBoolean(value);
    }
    private static void printHeader() {
        System.out.println("| Workload | Arm | Preload s (excluded) | Sample s | Train s | Encode s | Sink s | Comparison total s | Nearest ordinal | Score | Score sum |");
        System.out.println("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    }

    private static double seconds(long nanos) { return nanos / 1e9; }

    /** Uniform selection without replacement; preserve the same shuffled request order in every arm. */
    static int[] sample(int population, int count, long seed) {
        if (count <= 0 || count > population) throw new IllegalArgumentException("Sample count must be in [1, " + population + "]");
        var random = new SplittableRandom(seed);
        var used = new HashSet<Integer>();
        var selected = new int[count];
        for (int j = population - count, i = 0; j < population; j++, i++) {
            int candidate = random.nextInt(j + 1);
            int ordinal = used.add(candidate) ? candidate : j;
            used.add(ordinal); selected[i] = ordinal;
        }
        for (int i = count - 1; i > 0; i--) {
            int j = random.nextInt(i + 1), swap = selected[i];
            selected[i] = selected[j]; selected[j] = swap;
        }
        return selected;
    }

    private static RandomAccessVectorValues prefix(RandomAccessVectorValues source, int count) {
        if (source.size() == count) return source;
        if (source instanceof io.github.jbellis.jvector.graph.BatchedVectorValues)
            return new Prefix((io.github.jbellis.jvector.graph.BatchedVectorValues) source, count);
        return new RandomAccessVectorValues() {
            public int size() { return count; }
            public int dimension() { return source.dimension(); }
            public boolean isValueShared() { return source.isValueShared(); }
            public RandomAccessVectorValues copy() { return prefix(source.copy(), count); }
            public io.github.jbellis.jvector.vector.types.VectorFloat<?> getVector(int ordinal) {
                java.util.Objects.checkIndex(ordinal, count); return source.getVector(ordinal);
            }
        };
    }

    /** Restrict a file source without losing its batched range/selection capability. */
    private static final class Prefix implements io.github.jbellis.jvector.graph.BatchedVectorValues {
        private final io.github.jbellis.jvector.graph.BatchedVectorValues source;
        private final int size;
        Prefix(io.github.jbellis.jvector.graph.BatchedVectorValues source, int size) { this.source = source; this.size = size; }
        public int size() { return size; }
        public int dimension() { return source.dimension(); }
        public boolean isValueShared() { return true; }
        public RandomAccessVectorValues copy() { return prefix(source.copy(), size); }
        public io.github.jbellis.jvector.vector.types.VectorFloat<?> getVector(int ordinal) {
            java.util.Objects.checkIndex(ordinal, size); return source.getVector(ordinal);
        }
        public io.github.jbellis.jvector.graph.VectorCursor openRange(int start, int end) {
            java.util.Objects.checkFromToIndex(start, end, size); return source.openRange(start, end);
        }
        public io.github.jbellis.jvector.graph.VectorCursor openSelection(int[] ordinals, int offset, int count) {
            java.util.Objects.checkFromIndexSize(offset, count, ordinals.length);
            for (int i = offset; i < offset + count; i++) java.util.Objects.checkIndex(ordinals[i], size);
            return source.openSelection(ordinals, offset, count);
        }
    }

    private static final class Progress implements AutoCloseable {
        private volatile String phase = "starting";
        private volatile long since = System.nanoTime();
        private final java.util.concurrent.ScheduledExecutorService timer;
        Progress(int seconds) {
            timer = seconds == 0 ? null : Executors.newSingleThreadScheduledExecutor(r -> {
                var thread = new Thread(r, "input-benchmark-progress"); thread.setDaemon(true); return thread;
            });
            if (timer != null) timer.scheduleAtFixedRate(() -> System.out.printf(Locale.ROOT, "Still %s (%.0f s)%n",
                    phase, seconds(System.nanoTime() - since)), seconds, seconds, TimeUnit.SECONDS);
        }
        void phase(String name) {
            phase = name; since = System.nanoTime(); System.out.println("Phase: " + name);
        }
        public void close() { if (timer != null) timer.shutdownNow(); }
    }

    private static final class Options {
        private final LinkedHashMap<String, String> values;
        Options(LinkedHashMap<String, String> values) { this.values = values; }
        private static final Set<String> KEYS = Set.of("dataset", "file", "samples", "mode", "quantizer", "metric",
                "query-file", "cache", "seed", "progress-seconds", "help", "child", "quantizer-options", "loader-options");
        static Options parse(String[] args) {
            var values = new LinkedHashMap<String, String>();
            for (int i = 0; i < args.length; i++) {
                if (!args[i].startsWith("--") || !KEYS.contains(args[i].substring(2)))
                    throw new IllegalArgumentException("Unknown option: " + args[i]);
                String key = args[i].substring(2);
                String value = key.equals("help") || key.equals("child") ? "true"
                        : ++i < args.length ? args[i] : null;
                if (value == null || value.startsWith("--")) throw new IllegalArgumentException("Missing value for --" + key);
                if (values.putIfAbsent(key, value) != null) throw new IllegalArgumentException("Duplicate --" + key);
            }
            if (values.containsKey("file") && values.containsKey("dataset")) throw new IllegalArgumentException("Choose --file or --dataset");
            return new Options(values);
        }
        boolean contains(String key) { return values.containsKey(key); }
        String get(String key, String fallback) { return values.getOrDefault(key, fallback); }
        long longValue(String key, long fallback) { return Long.parseLong(get(key, Long.toString(fallback))); }
        int nonnegativeInt(String key, int fallback) {
            int n = Integer.parseInt(get(key, Integer.toString(fallback)));
            if (n < 0) throw new IllegalArgumentException("--" + key + " must be nonnegative"); return n;
        }
        int positiveInt(String key, int fallback) {
            int n = nonnegativeInt(key, fallback);
            if (n == 0) throw new IllegalArgumentException("--" + key + " must be positive"); return n;
        }
        void checkMode(String mode) {
            if (mode.equals("all")) throw new IllegalArgumentException("A child must select one loader");
            try { loader(mode); } catch (Exception e) { throw new IllegalArgumentException("Unknown loader: " + mode, e); }
        }
        Path file() throws IOException {
            if (contains("file")) return Path.of(get("file", "")).toAbsolutePath();
            var loader = new DataSetLoaderSimpleMFD("jvector-examples/yaml-configs/dataset-catalogs");
            String dataset = get("dataset", "e5-small-v2-100k");
            return loader.loadBaseVectorFile(dataset).orElseThrow(() -> new IllegalArgumentException("Unknown catalog dataset: " + dataset)).toAbsolutePath();
        }
        List<String> childArguments(Path file, String mode) {
            var args = new ArrayList<String>();
            values.forEach((key, value) -> {
                if (!Set.of("dataset", "file", "mode", "child").contains(key)) { args.add("--" + key); args.add(value); }
            });
            args.addAll(List.of("--file", file.toString(), "--mode", mode, "--child")); return args;
        }
    }
}
