package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Set;

public class LazyFvecsLoader {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * Loads a LazyFvecsDataSet from the given fvecs file.
     * @param filePath The path to the fvecs file.
     * @param queryVectors Query vectors (loaded eagerly, as before).
     * @param groundTruth Ground truth data (loaded eagerly, as before).
     * @param similarityFunction The similarity function (e.g. COSINE or EUCLIDEAN).
     * @return A LazyFvecsDataSet that lazily loads base vectors.
     */
    public static LazyFvecsDataSet load(String filePath,
                                        List<VectorFloat<?>> queryVectors,
                                        List<? extends Set<Integer>> groundTruth,
                                        VectorSimilarityFunction similarityFunction) {
        Path path = Paths.get(filePath);
        int dimension;
        int vectorCount;
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            // Read the first 4 bytes to get the dimension.
            ByteBuffer headerBuffer = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            int bytesRead = channel.read(headerBuffer, 0);
            if (bytesRead != 4) {
                throw new IOException("Unable to read header from " + filePath);
            }
            headerBuffer.flip();
            dimension = headerBuffer.getInt();
            int recordSize = 4 + dimension * Float.BYTES;
            long fileSize = channel.size();
            vectorCount = (int) (fileSize / recordSize);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load fvecs metadata from " + filePath, e);
        }
        return new LazyFvecsDataSet("LazyFvecs: " + filePath,
                similarityFunction,
                queryVectors,
                groundTruth,
                path,
                vectorCount,
                dimension,
                vectorTypeSupport);
    }
}

