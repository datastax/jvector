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

import io.github.jbellis.jvector.example.util.MMapRandomAccessVectorValues;
import io.github.jbellis.jvector.example.util.UpdatableRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.newsclub.net.unix.AFUNIXServerSocket;
import org.newsclub.net.unix.AFUNIXSocket;
import org.newsclub.net.unix.AFUNIXSocketAddress;

import java.io.File;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Simple local service to use for interaction with JVector over IPC.
 * Only handles a single connection at a time.
 */
public class IPCService
{
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    // How each command message is marked as finished
    private static final String DELIM = "\n";

    static class SessionContext {
        boolean isBulkLoad = false;

        int dimension;
        int M;
        int efConstruction;
        float neighborOverflow;
        boolean addHierarchy;
        VectorSimilarityFunction similarityFunction;
        RandomAccessVectorValues ravv;
        CompressedVectors cv;
        GraphIndexBuilder indexBuilder;
        GraphIndex index;
        GraphSearcher searcher;
        final StringBuffer result = new StringBuffer(1024);
    }

    enum Command {
        CREATE,  //DIMENSIONS SIMILARITY_TYPE M EF\n
        WRITE,  //[N,N,N] [N,N,N]...\n
        BULKLOAD, // /path/to/local/file
        OPTIMIZE, //Run once finished writing
        SEARCH, //EF-search limit [N,N,N] [N,N,N]...\n
        MEMORY, // Memory usage in kb
    }

    enum Response {
        OK,
        ERROR,
        RESULT
    }

    final Path socketFile;
    final AFUNIXServerSocket unixSocket;
    IPCService(Path socketFile) throws IOException {
        this.socketFile = socketFile;
        this.unixSocket = AFUNIXServerSocket.newInstance();
        this.unixSocket.bind(AFUNIXSocketAddress.of(socketFile));
    }

    void create(String input, SessionContext ctx) {
        String[] args = input.split("\\s+");

        if (args.length != 6)
            throw new IllegalArgumentException("Illegal CREATE statement. Expecting 'CREATE [DIMENSIONS] [SIMILARITY_TYPE] [M] [EF]'");

        int dimensions = Integer.parseInt(args[0]);
        VectorSimilarityFunction sim = VectorSimilarityFunction.valueOf(args[1]);
        int M = Integer.parseInt(args[2]);
        int efConstruction = Integer.parseInt(args[3]);
        float neighborOverflow = Float.parseFloat(args[4]);
        boolean addHierarchy = Boolean.parseBoolean(args[5]);

        ctx.ravv = new UpdatableRandomAccessVectorValues(dimensions);
        ctx.indexBuilder =  new GraphIndexBuilder(ctx.ravv, sim, M, efConstruction, neighborOverflow, 1.4f, addHierarchy);
        ctx.M = M;
        ctx.dimension = dimensions;
        ctx.efConstruction = efConstruction;
        ctx.similarityFunction = sim;
        ctx.isBulkLoad = false;
    }

    void write(String input, SessionContext ctx) {
        if (ctx.isBulkLoad)
            throw new IllegalStateException("Session is for bulk loading.  To reset call CREATE again");

        String[] args = input.split("\\s+");
        for (int i = 0; i < args.length; i++) {
            String vStr = args[i];
            if (!vStr.startsWith("[") || !vStr.endsWith("]"))
                throw new IllegalArgumentException("Invalid vector encountered. Expecting 'WRITE [F1,F2...] [F3,F4...] ...' but got " + vStr);

            String[] values = vStr.substring(1, vStr.length() - 1).split(",");
            if (values.length != ctx.ravv.dimension())
                throw new IllegalArgumentException(String.format("Invalid vector dimension: %d %d!=%d", i, values.length, ctx.ravv.dimension()));

            VectorFloat<?> vector = vectorTypeSupport.createFloatVector(ctx.dimension);
            for (int k = 0; k < vector.length(); k++)
                vector.set(k, Float.parseFloat(values[k]));

            ((UpdatableRandomAccessVectorValues)ctx.ravv).add(vector);
            var node = ctx.ravv.size() - 1;
            ctx.indexBuilder.addGraphNode(node, ctx.ravv.getVector(node));
        }
    }

    void bulkLoad(String input, SessionContext ctx) {
        String[] args = input.split("\\s+");
        if (args.length != 1)
            throw new IllegalArgumentException("Invalid arguments. Expecting 'BULKLOAD /path/to/local/file'. got " + input);

        File f = new File(args[0]);
        if (!f.exists())
            throw new IllegalArgumentException("No file at: " + f);

        long length = f.length();
        if (length % ((long) ctx.dimension * Float.BYTES) != 0)
            throw new IllegalArgumentException("File is not encoded correctly");

        ctx.index = null;
        ctx.ravv = null;
        ctx.indexBuilder = null;
        ctx.isBulkLoad = true;

        var ravv = new MMapRandomAccessVectorValues(f, ctx.dimension);
        var indexBuilder = new GraphIndexBuilder(ravv, ctx.similarityFunction, ctx.M, ctx.efConstruction, ctx.neighborOverflow, 1.4f, ctx.addHierarchy);
        System.out.println("BulkIndexing " + ravv.size());
        ctx.index = flushGraphIndex(indexBuilder.build(ravv), ravv);
        ctx.cv = pqIndex(ravv, ctx);

        //Finished with raw data we can close/cleanup
        ravv.close();
        ctx.searcher = new GraphSearcher(ctx.index);
    }

    private CompressedVectors pqIndex(RandomAccessVectorValues ravv, SessionContext ctx) {
        var pqDims = ctx.dimension > 10 ? Math.max(ctx.dimension / 4, 10) : ctx.dimension;
        long start = System.nanoTime();
        ProductQuantization pq = ProductQuantization.compute(ravv, pqDims, 256, ctx.similarityFunction == VectorSimilarityFunction.EUCLIDEAN);
        System.out.format("PQ@%s build %.2fs,%n", pqDims, (System.nanoTime() - start) / 1_000_000_000.0);
        start = System.nanoTime();
        var cv = pq.encodeAll(ravv);
        System.out.format("PQ encoded %d vectors [%.2f MB] in %.2fs,%n", ravv.size(), (cv.ramBytesUsed()/1024f/1024f) , (System.nanoTime() - start) / 1_000_000_000.0);
        return cv;
    }

    private static GraphIndex flushGraphIndex(OnHeapGraphIndex onHeapIndex, RandomAccessVectorValues ravv) {
        try {
            var testDirectory = Files.createTempDirectory("BenchGraphDir");
            var graphPath = testDirectory.resolve("graph.bin");

            OnDiskGraphIndex.write(onHeapIndex, ravv, graphPath);
            return onHeapIndex;
        } catch (IOException e) {
            throw new IOError(e);
        }
    }

    void optimize(SessionContext ctx) {
        if (!ctx.isBulkLoad) {
            if (ctx.ravv.size() > 256) {
                ctx.indexBuilder.cleanup();
                ctx.index = flushGraphIndex(ctx.indexBuilder.getGraph(), ctx.ravv);
                ctx.cv = pqIndex(ctx.ravv, ctx);
                ctx.indexBuilder = null;
                ctx.ravv = null;
            } else { //Not enough data for PQ
                ctx.indexBuilder.cleanup();
                ctx.index = ctx.indexBuilder.getGraph();
                ctx.cv = null;
            }

        }
    }

    String memory(SessionContext ctx) {
        long kb = 0;
        if (ctx.cv != null)
            kb = ctx.cv.ramBytesUsed() / 1024;
        return String.format("%s %d\n", Response.RESULT, kb);
    }

    String search(String input, SessionContext ctx) {
        String[] args = input.split("\\s+");

        if (args.length < 3)
            throw new IllegalArgumentException("Invalid arguments search-ef top-k [vector1] [vector2]...");

        int searchEf = Integer.parseInt(args[0]);
        int topK = Integer.parseInt(args[1]);

        VectorFloat<?> queryVector = vectorTypeSupport.createFloatVector(ctx.dimension);
        ctx.result.setLength(0);
        ctx.result.append(Response.RESULT);
        int[][] results = new int[args.length - 2][];
        IntStream loopStream = IntStream.range(0, args.length-2);

        //Only use parallel path if we have > 1 core
        if (ForkJoinPool.commonPool().getPoolSize() > 1)
            loopStream.parallel();

        loopStream.forEach(i -> {
            String vStr = args[i + 2]; //Skipping first 2 args which are not vectors
            if (!vStr.startsWith("[") || !vStr.endsWith("]"))
                throw new IllegalArgumentException("Invalid query vector encountered:" + vStr);

            String[] values = vStr.substring(1, vStr.length() - 1).split(",");
            if (values.length != ctx.dimension)
                throw new IllegalArgumentException(String.format("Invalid vector dimension: %d!=%d", values.length, ctx.dimension));


            for (int k = 0; k < queryVector.length(); k++)
                queryVector.set(k, Float.parseFloat(values[k]));

            SearchResult r;
            if (ctx.cv != null) {
                ScoreFunction.ApproximateScoreFunction sf = ctx.cv.precomputedScoreFunctionFor(queryVector, ctx.similarityFunction);
                try (var view = ctx.index.getView()) {
                    var rr = view instanceof GraphIndex.ScoringView
                            ? ((GraphIndex.ScoringView) view).rerankerFor(queryVector, ctx.similarityFunction)
                            : ctx.ravv.rerankerFor(queryVector, ctx.similarityFunction);
                    var ssp = new SearchScoreProvider(sf, rr);
                    r = new GraphSearcher(ctx.index).search(ssp, searchEf, Bits.ALL);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            } else {
                r = GraphSearcher.search(queryVector, topK, ctx.ravv, ctx.similarityFunction, ctx.index, Bits.ALL);
            }

            var resultNodes = r.getNodes();
            int count = Math.min(resultNodes.length, topK);

            results[i] = new int[count];
            for (int k = 0; k < count; k++)
                results[i][k] = resultNodes[k].node;
        });

        //Format Response
        for (int[] result : results) {
            ctx.result.append(" [");
            for (int k = 0; k < result.length; k++) {
                if (k > 0) ctx.result.append(",");
                ctx.result.append(result[k]);
            }
            ctx.result.append("]");
        }
        ctx.result.append("\n");
        return ctx.result.toString();
    }

    String process(String input, SessionContext ctx) {
        int delim = input.indexOf(' ');
        String command = delim < 1 ? input : input.substring(0, delim);
        String commandArgs = input.substring(delim + 1);
        String response = Response.OK.name() + "\n";
        switch (Command.valueOf(command)) {
            case CREATE: create(commandArgs, ctx); break;
            case WRITE: write(commandArgs, ctx); break;
            case BULKLOAD: bulkLoad(commandArgs, ctx); break;
            case OPTIMIZE: optimize(ctx); break;
            case SEARCH: response = search(commandArgs, ctx); break;
            case MEMORY: response = memory(ctx); break;
            default: throw new UnsupportedOperationException("No support for: '" + command + "'");
        }
        return response;
    }

    void serve() throws IOException {
        int bufferSize = unixSocket.getReceiveBufferSize();
        byte[] buffer = new byte[bufferSize];
        StringBuffer sb = new StringBuffer(1024);
        System.out.println("Service listening on " + socketFile);
        while (true) {
            AFUNIXSocket connection = unixSocket.accept();
            System.out.println("new connection!");
            SessionContext context = new SessionContext();

            try (InputStream is = connection.getInputStream();
                 OutputStream os = connection.getOutputStream()) {
                int read;
                while ((read = is.read(buffer)) != -1) {
                    String s = new String(buffer, 0, read, StandardCharsets.UTF_8);
                    if (s.contains(DELIM)) {
                        int doffset;
                        while ((doffset = s.indexOf(DELIM)) != -1) {
                            sb.append(s, 0, doffset);

                            //Save tail for next loop
                            s = s.substring(doffset + 1);
                            try {
                                String cmd = sb.toString();
                                if (!cmd.trim().isEmpty()) {
                                    String response = process(cmd, context);
                                    os.write(response.getBytes(StandardCharsets.UTF_8));
                                }
                            } catch (Throwable t) {
                                String response = String.format("%s %s\n", Response.ERROR, t.getMessage());
                                os.write(response.getBytes(StandardCharsets.UTF_8));
                                t.printStackTrace();
                            }

                            //Reset buffer
                            sb.setLength(0);
                        }
                    } else {
                        sb.append(s);
                    }
                }
            }
        }
    }

    static void help() {
        System.out.println("Usage: ipcservice.jar [/unix/socket/path.sock]");
        System.exit(1);
    }

    public static void main(String[] args) {
        String socketFile = System.getProperty("java.io.tmpdir") + "/jvector.sock";
        if (args.length > 1)
            help();

        if (args.length == 1)
            socketFile = args[0];

        try {
            IPCService service = new IPCService(Path.of(socketFile));
            service.serve();
        } catch (Throwable t) {
            t.printStackTrace();
            System.exit(1);
        }

        System.exit(0);
    }
}
