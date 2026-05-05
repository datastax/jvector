package io.github.jbellis.jvector.quantization.ash;

public interface AshDriver {

    public interface PackedVectors {}

    PackedVectors create(int n);

    int getDimension();

    int getBitDepth();

    /**
     * toPackOffset + dimension() must be less than or equal to toPack.length,
     * otherwise implementations should throw
     */
    void packInts(int[] toPack, int toPackOffset, PackedVectors out, int pvOffset);

    /** outOffset + dimension() must be less than or equal to out.length */
    void unpackInts(PackedVectors packed, int pvOffset, int[] out, int outOffset);

    /** qOffset + dimension() must be less than or equal to query.length */
    float asymmetricScorePackedInts(PackedVectors packed, int pvOffset, float[] query, int qOffset);

    float symmetricScorePackedInts(PackedVectors a, int aOffset, PackedVectors b, int bOffset);

    float getRawComponentSum(PackedVectors packed, int pvOffset);
}
