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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.disk.CommonHeader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedNVQ;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedVectors;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.InputStream;
import java.lang.instrument.Instrumentation;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.function.IntFunction;
import java.util.jar.Attributes;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.jar.Manifest;

import static org.junit.Assert.*;

/** Independent JVM sizing, including actual collection capacities and shared object identities. */
public class TestRetainedMemoryAccounting {
    /** A test-only startup agent: no agent or module-opening options are needed by production. */
    public static class Agent {
        static Instrumentation instrumentation;
        public static void premain(String arguments, Instrumentation value) {
            instrumentation = value;
        }
    }

    @Test
    public void matchesJvmFootprintsAcrossLayouts() throws Exception {
        var directory = Files.createTempDirectory("pq-retained-memory-");
        try {
            var agent = directory.resolve("agent.jar");
            var manifest = new Manifest();
            manifest.getMainAttributes().put(Attributes.Name.MANIFEST_VERSION, "1.0");
            manifest.getMainAttributes().putValue("Premain-Class", Agent.class.getName());
            String resource = Agent.class.getName().replace('.', '/') + ".class";
            try (var out = new JarOutputStream(Files.newOutputStream(agent), manifest);
                 InputStream in = Agent.class.getClassLoader().getResourceAsStream(resource)) {
                assertNotNull(in);
                out.putNextEntry(new JarEntry(resource));
                in.transferTo(out);
                out.closeEntry();
            }
            var layouts = new ArrayList<List<String>>();
            for (int alignment : new int[]{8, 16}) {
                for (String references : List.of("+", "-")) {
                    for (String classPointers : List.of("+", "-")) {
                        // Before JDK-8241825 (JDK 15), disabling compressed oops also
                        // disables compressed class pointers; this combination cannot be tested.
                        if (Runtime.version().feature() < 15 && references.equals("-") && classPointers.equals("+")) continue;
                        layouts.add(List.of("-XX:" + references + "UseCompressedOops",
                                "-XX:" + classPointers + "UseCompressedClassPointers",
                                "-XX:ObjectAlignmentInBytes=" + alignment));
                    }
                    if (Runtime.version().feature() >= 25) {
                        layouts.add(List.of("-XX:+UseCompactObjectHeaders", "-XX:" + references + "UseCompressedOops",
                                "-XX:ObjectAlignmentInBytes=" + alignment));
                    }
                }
            }
            for (int i = 0; i < layouts.size(); i++) {
                var command = new ArrayList<String>();
                command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
                command.addAll(List.of("-Xmx2g", "-XX:ActiveProcessorCount=4", "-javaagent:" + agent,
                        "--add-opens=java.base/java.util=ALL-UNNAMED",
                        "--add-opens=java.base/java.lang=ALL-UNNAMED",
                        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
                        "-Djvector.vectorization_provider=" + System.getProperty("jvector.vectorization_provider", "default")));
                if (Runtime.version().feature() >= 20) command.add("--add-modules=jdk.incubator.vector");
                if (Runtime.version().feature() >= 22) {
                    command.add("--enable-native-access=ALL-UNNAMED");
                    command.add("--add-opens=java.base/jdk.internal.foreign=ALL-UNNAMED");
                }
                command.addAll(layouts.get(i));
                command.add("-cp");
                command.add(System.getProperty("java.class.path"));
                command.add(getClass().getName());
                var output = directory.resolve("layout-" + i + ".log");
                Process process = new ProcessBuilder(command).redirectErrorStream(true).redirectOutput(output.toFile()).start();
                try {
                    assertTrue("Memory verification timed out: " + layouts.get(i), process.waitFor(120, TimeUnit.SECONDS));
                    if (process.exitValue() != 0) fail(layouts.get(i) + "\n" + Files.readString(output));
                    try (var lines = Files.lines(output)) {
                        lines.filter(line -> line.startsWith("Footprint ")).forEach(System.out::println);
                    }
                    System.out.println("Verified JVM footprints: JDK " + Runtime.version().feature()
                            + ", " + layouts.get(i));
                } finally {
                    if (process.isAlive()) process.destroyForcibly().waitFor();
                }
            }
        } finally {
            try (var files = Files.walk(directory)) {
                for (Path file : (Iterable<Path>) files.sorted(Collections.reverseOrder())::iterator) Files.delete(file);
            }
        }
    }

    /** Executed in forked VMs, with instrumentation rather than the production sizing utility. */
    public static void main(String[] args) throws Exception {
        assertNotNull("Startup agent was not loaded", Agent.instrumentation);
        verifyRequestedLayout();
        verifyBuffers();
        verifyObjectSharing();
        verifyGraphs();
        System.out.println("Retained-memory footprints match the JVM");
    }

    private static void verifyRequestedLayout() throws Exception {
        Class<?> beanClass = Class.forName("com.sun.management.HotSpotDiagnosticMXBean");
        Object bean = ManagementFactory.class.getMethod("getPlatformMXBean", Class.class).invoke(null, beanClass);
        for (String argument : ManagementFactory.getRuntimeMXBean().getInputArguments()) {
            if (!argument.startsWith("-XX:")) continue;
            String option = argument.substring(4);
            if (!(option.contains("UseCompressed") || option.contains("UseCompactObjectHeaders")
                    || option.startsWith("ObjectAlignmentInBytes="))) continue;
            String expected;
            String name;
            if (option.charAt(0) == '+' || option.charAt(0) == '-') {
                name = option.substring(1);
                expected = Boolean.toString(option.charAt(0) == '+');
            } else {
                int equals = option.indexOf('=');
                name = option.substring(0, equals);
                expected = option.substring(equals + 1);
            }
            Object value = beanClass.getMethod("getVMOption", String.class).invoke(bean, name);
            assertEquals("Requested layout was not enabled: " + argument, expected,
                    value.getClass().getMethod("getValue").invoke(value));
        }
    }

    private static void verifyBuffers() throws Exception {
        for (String implementation : List.of("ArrayByteSequence", "MemorySegmentByteSequence")) {
            final Class<?> type;
            try {
                type = Class.forName("io.github.jbellis.jvector.vector." + implementation);
            } catch (ClassNotFoundException e) {
                // Native module is not on the scalar/Panama-only reactor classpath.
                continue;
            }
            var constructor = type.getDeclaredConstructor(int.class);
            constructor.setAccessible(true);
            for (int size : new int[]{0, 1, 4, 7, 8, 15, 16, 17, 31, 32, 33, 48, 96, 192, 384, 768}) {
                var code = (ByteSequence<?>) constructor.newInstance(size);
                assertEquals(implementation + " size=" + size, measured(code, Collections.emptySet()), code.ramBytesUsed());
                var source = new FusedPQ.FusedPQInlineSource(code);
                long measuredSource = measured(source, Collections.emptySet());
                assertEquals(measuredSource, source.ramBytesUsed());
                if (size == 384) System.out.println("Footprint " + implementation
                        + " source384 reported=" + source.ramBytesUsed() + " measured=" + measuredSource);
                if (size >= 4) {
                    var slice = code.slice(1, size - 2);
                    assertEquals("Slice must retain its full backing storage", measured(slice, Collections.emptySet()), slice.ramBytesUsed());
                    var nested = slice.slice(1, 0);
                    assertEquals(measured(nested, Collections.emptySet()), nested.ramBytesUsed());
                    Object[] shared = {code, slice, nested, slice};
                    assertEquals(measured(shared, Collections.emptySet()),
                            RamUsageEstimator.sizeOfObjectGraph(shared, TestRetainedMemoryAccounting::isOwned));
                }
            }
        }
    }

    private static void verifyObjectSharing() throws Exception {
        Object[] cycle = new Object[4];
        var buffer = VectorizationProvider.getInstance().getVectorTypeSupport().createByteSequence(33);
        cycle[0] = cycle;
        cycle[1] = buffer;
        cycle[2] = buffer;
        cycle[3] = new Object[]{cycle, buffer};
        assertEquals(measured(cycle, Collections.emptySet()),
                RamUsageEstimator.sizeOfObjectGraph(cycle, TestRetainedMemoryAccounting::isOwned));
        assertEquals(0, RamUsageEstimator.sizeOfObjectGraph(null, object -> true));
        assertThrows(NullPointerException.class, () -> RamUsageEstimator.sizeOfObjectGraph(new Object(), null));
        for (Object collection : List.of(List.of(new byte[17]), new ArrayList<>(), List.of())) {
            assertEquals("Shared collection sentinels must not be charged to their owner",
                    measured(collection, Collections.emptySet()),
                    RamUsageEstimator.sizeOfObjectGraph(collection, TestRetainedMemoryAccounting::isOwned));
        }
    }

    private static void verifyGraphs() throws Exception {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        int[][] shapes = {{0, 0, 1}, {1, 0, 1}, {7, 2, 1}, {37, 8, 1}, {257, 32, 1},
                {37, 8, 2}, {257, 32, 4}, {65, 2, 12}, {37, 2, 32}};
        for (int version = 2; version <= 6; version++) {
            for (int[] shape : shapes) {
                if (version < 4 && shape[2] > 1) continue;
                verifyGraph(version, shape, null);
                if (version < 6 || shape[0] == 0) continue;
                for (int codeSize : new int[]{1, 7, 17, 48, 96, 192, 384, 768}) {
                    var books = new VectorFloat<?>[codeSize];
                    var subspaces = new int[codeSize][2];
                    for (int j = 0; j < codeSize; j++) {
                        books[j] = types.createFloatVector(256);
                        subspaces[j] = new int[]{1, j};
                    }
                    var pq = new ProductQuantization(books, 256, subspaces, null, KMeansPlusPlusClusterer.UNWEIGHTED);
                    verifyGraph(version, shape, pq);
                    if (codeSize == 48) {
                        for (FeatureId feature : List.of(FeatureId.NVQ_VECTORS, FeatureId.SEPARATED_VECTORS, FeatureId.SEPARATED_NVQ)) {
                            verifyGraph(version, shape, pq, feature);
                        }
                    }
                }
            }
        }
    }

    private static void verifyGraph(int version, int[] shape, ProductQuantization pq) throws Exception {
        verifyGraph(version, shape, pq, FeatureId.INLINE_VECTORS);
    }

    private static void verifyGraph(int version, int[] shape, ProductQuantization pq, FeatureId reranking) throws Exception {
        int dimension = pq == null ? 8 : pq.compressedVectorSize();
        var layers = new ArrayList<CommonHeader.LayerInfo>();
        for (int level = 0; level < shape[2]; level++) {
            int count = Math.max(1, shape[0] >>> Math.min(level, 30));
            layers.add(new CommonHeader.LayerInfo(count, Math.min(shape[1], count - 1)));
        }
        ImmutableGraphIndex graph = shape[0] == 0 ? new TestUtil.EmptyGraphIndex(dimension, new Random(7381))
                : new TestUtil.RandomlyConnectedGraphIndex(layers, new Random(7381));
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        var vector = types.createFloatVector(dimension);
        var path = Files.createTempFile("pq-owned-graph-", ".index");
        String description = "v" + version + " " + Arrays.toString(shape) + " code=" + dimension + " " + reranking;
        try {
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            try (var view = graph.getView()) {
                var builder = new OnDiskGraphIndexWriter.Builder(graph, path).withVersion(version);
                switch (reranking) {
                    case INLINE_VECTORS:
                        builder.with(new InlineVectors(dimension));
                        suppliers.put(reranking, node -> new InlineVectors.State(vector));
                        break;
                    case SEPARATED_VECTORS:
                        builder.with(new SeparatedVectors(dimension, 0));
                        suppliers.put(reranking, node -> new InlineVectors.State(vector));
                        break;
                    case NVQ_VECTORS:
                    case SEPARATED_NVQ:
                        var nvq = NVQuantization.create(types.createFloatVector(dimension), 2);
                        if (reranking == FeatureId.NVQ_VECTORS) builder.with(new NVQ(nvq));
                        else builder.with(new SeparatedNVQ(nvq, 0));
                        var encoded = nvq.encode(vector);
                        suppliers.put(reranking, node -> new NVQ.State(encoded));
                        break;
                    default:
                        throw new AssertionError(reranking);
                }
                if (pq != null) {
                    var code = types.createByteSequence(pq.compressedVectorSize());
                    builder.with(new FusedPQ(shape[1], pq));
                    suppliers.put(FeatureId.FUSED_PQ, node -> new FusedPQ.State(view, ignored -> code, node));
                }
                try (var writer = builder.build()) { writer.write(suppliers); }
            }
            try (var reader = new SimpleMappedReader.Supplier(path);
                 var loaded = OnDiskGraphIndex.load(reader, 0);
                 var view = loaded.getView()) {
                Set<Object> external = Collections.newSetFromMap(new IdentityHashMap<>());
                external.add(reader);
                var format = OnDiskGraphIndex.class.getDeclaredField("format");
                format.setAccessible(true);
                external.add(format.get(loaded));
                assertGraphEquals(description, loaded, external);
                // Allocating collection views changes retention; measurement must not create them.
                loaded.getFeatureSet().iterator().hasNext();
                if (pq != null) {
                    var sources = view.getInlineSourceFeatures();
                    sources.keySet(); sources.values(); sources.entrySet();
                    assertGraphEquals(description + " views", loaded, external);
                    var saved = new ArrayList<>(sources.values());
                    sources.clear();
                    assertGraphEquals(description + " cleared", loaded, external);
                    // Duplicate identities and shared backing storage must not be charged twice.
                    var code = types.createByteSequence(33);
                    sources.put(1001, new FusedPQ.FusedPQInlineSource(code));
                    sources.put(1002, new FusedPQ.FusedPQInlineSource(code.slice(1, 17)));
                    sources.put(1003, sources.get(1001));
                    assertGraphEquals(description + " shared", loaded, external);
                    for (int i = 0; i < 40; i++) sources.put(2000 + i, saved.get(0));
                    assertGraphEquals(description + " grown", loaded, external);
                    sources.clear(); sources.compact();
                    assertGraphEquals(description + " compacted", loaded, external);
                }
                assertGraphEquals(description + " stable", loaded, external);
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

    private static void assertGraphEquals(String message, OnDiskGraphIndex graph, Set<Object> external) throws Exception {
        long expected = measured(graph, external);
        assertEquals(message, expected, graph.ramBytesUsed());
        assertEquals("Accounting must not allocate retained graph storage", expected, measured(graph, external));
    }

    private static boolean isOwned(Object value) {
        if (value instanceof Class<?> || value instanceof Enum<?> || value instanceof VectorCompressor<?>
                || value instanceof ThreadLocal<?> || value instanceof ExplicitThreadLocal<?>) return false;
        if (value.getClass().isArray() && value.getClass().getComponentType().isEnum()) return false;
        if (value instanceof Integer && value == Integer.valueOf((Integer) value)) return false;
        return true;
    }

    private static final ClassValue<Set<Object>> STATIC_REFERENCES = new ClassValue<>() {
        @Override
        protected Set<Object> computeValue(Class<?> type) {
            Set<Object> result = Collections.newSetFromMap(new IdentityHashMap<>());
            for (Class<?> current = type; current != null; current = current.getSuperclass()) {
                for (Class<?> outer = current; outer != null; outer = outer.getDeclaringClass()) {
                    for (Field field : outer.getDeclaredFields()) {
                        if (!Modifier.isStatic(field.getModifiers()) || !Modifier.isFinal(field.getModifiers())
                                || field.getType().isPrimitive()) continue;
                        try {
                            field.setAccessible(true);
                            Object value = field.get(null);
                            if (value != null) result.add(value);
                        } catch (ReflectiveOperationException e) {
                            throw new AssertionError(e);
                        }
                    }
                }
            }
            return result;
        }
    };

    /** Separate reflective traversal: no production sizing, field offsets, or accounting methods. */
    private static long measured(Object root, Set<Object> external) throws Exception {
        Set<Object> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        var pending = new ArrayDeque<Object>();
        pending.add(root);
        long bytes = 0;
        while (!pending.isEmpty()) {
            Object value = pending.removeFirst();
            if (!visited.add(value) || external.contains(value) || !isOwned(value)) continue;
            if (!value.getClass().isArray() && STATIC_REFERENCES.get(value.getClass()).contains(value)) continue;
            bytes += Agent.instrumentation.getObjectSize(value);
            if (value instanceof Object[]) {
                for (Object element : (Object[]) value) if (element != null) pending.add(element);
            } else if (!value.getClass().isArray()) {
                for (Class<?> type = value.getClass(); type != null; type = type.getSuperclass()) {
                    for (Field field : type.getDeclaredFields()) {
                        if (Modifier.isStatic(field.getModifiers()) || field.getType().isPrimitive()) continue;
                        field.setAccessible(true);
                        Object child = field.get(value);
                        if (child != null && !STATIC_REFERENCES.get(field.getDeclaringClass()).contains(child)) pending.add(child);
                    }
                }
            }
        }
        return bytes;
    }
}
