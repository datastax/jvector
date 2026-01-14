package io.github.jbellis.jvector.example.diagnostics;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

/**
 * Self-contained sanity check for ASH asymmetric vs symmetric scoring (Java 11).
 *
 * Intended usage:
 *   ASHScoreDebug.run(ds.getBaseRavv(), ashVectors, 2000, 50000, 100, new Random(42));
 *
 * Prints:
 *  - Pearson correlations: (GT vs asym), (GT vs sym), (asym vs sym)
 *  - A few top-K overlap comparisons on a subset
 */
public final class ASHScoreDebug {
    private ASHScoreDebug() { }

    public static void run(RandomAccessVectorValues floatVectors,
                           ASHVectors ashVectors,
                           int numPairs,
                           int subsetN,
                           int topK,
                           Random rng) {
        if (floatVectors == null) throw new IllegalArgumentException("floatVectors is null");
        if (ashVectors == null) throw new IllegalArgumentException("ashVectors is null");
        if (rng == null) throw new IllegalArgumentException("rng is null");
        if (numPairs <= 0) throw new IllegalArgumentException("numPairs must be > 0");
        if (subsetN <= 0) throw new IllegalArgumentException("subsetN must be > 0");
        if (topK <= 0) throw new IllegalArgumentException("topK must be > 0");

        AsymmetricHashing ash = ashVectors.getCompressor();
        if (ash.landmarkCount != 1) {
            throw new IllegalArgumentException("ASHScoreDebug assumes landmarkCount==1, got " + ash.landmarkCount);
        }

        int n = floatVectors.size();
        int D = floatVectors.dimension();
        int d = ash.quantizedDim;

        int effectiveSubsetN = Math.min(subsetN, n);

        // Constant for C=1 (included for manuscript-faithful reconstruction; does not affect ranking)
        final float muNormSq = VectorUtil.dotProduct(ash.landmarks[0], ash.landmarks[0]);

        System.out.println("=== ASHScoreDebug ===");
        System.out.println("N=" + n + " D=" + D + " d=" + d
                + " subsetN=" + effectiveSubsetN + " pairs=" + numPairs + " topK=" + topK);

        // 1) Bit-math sanity on trivial patterns (checks signDot identity)
        sanityCheckBinaryMath(d);

        // 2) Pair sampling: GT dot, asymmetric score, symmetric score
        double[] gt = new double[numPairs];
        double[] asym = new double[numPairs];
        double[] sym = new double[numPairs];

        for (int t = 0; t < numPairs; t++) {
            int i = rng.nextInt(effectiveSubsetN);
            int j = rng.nextInt(effectiveSubsetN);

            VectorFloat<?> xi = floatVectors.getVector(i);
            VectorFloat<?> xj = floatVectors.getVector(j);

            // Ground truth dot
            gt[t] = VectorUtil.dotProduct(xi, xj);

            // Asymmetric: query xi vs encoded j
            ScoreFunction.ApproximateScoreFunction f =
                    ashVectors.scoreFunctionFor(xi, VectorSimilarityFunction.DOT_PRODUCT);
            asym[t] = f.similarityTo(j);

            // Symmetric: encoded i vs encoded j (manuscript-consistent)
            sym[t] = symmetricDot(ashVectors.get(i), ashVectors.get(j), d, muNormSq);
        }

        printCorrelations(gt, asym, sym);

        // 3) Top-K overlap sanity for a few query ords (subset brute force)
        int[] queries = pickQueries(effectiveSubsetN, rng, 5);
        for (int qi = 0; qi < queries.length; qi++) {
            int q = queries[qi];
            System.out.println();
            System.out.println("--- TopK overlap for q=" + q
                    + " (subsetN=" + effectiveSubsetN + ", topK=" + topK + ") ---");
            topKOverlapForQuery(floatVectors, ashVectors, q, effectiveSubsetN, topK, d, muNormSq);
        }

        System.out.println("=== end ASHScoreDebug ===");
    }

    /**
     * Symmetric ASH–ASH dot-product approximation (C=1), consistent with:
     *  - g produces sign vectors in {-1,+1}^d (stored as bits b in {0,1}^d via s = 2b - 1)
     *  - f(z) = d^{-1/2} A z implies <tilde x, tilde y> ≈ (1/d) <g(tilde x), g(tilde y)>
     *
     * With stored header:
     *  - scale  = ||x-μ|| / sqrt(d)
     *  - offset = <x, μ> - ||μ||^2
     *
     * Centered dot term:
     *  ||x-μ|| ||y-μ|| <tilde x, tilde y> ≈ (scale_x * scale_y) * <g_x, g_y>
     *
     * And <g_x, g_y> = 2*matches - d, where matches is XNOR-popcount of bit vectors.
     *
     * Full dot:
     *  <x,y> ≈ scale_x*scale_y*(2*matches - d) + offset_x + offset_y + ||μ||^2
     */
    public static float symmetricDot(AsymmetricHashing.QuantizedVector v1,
                                     AsymmetricHashing.QuantizedVector v2,
                                     int d,
                                     float muNormSq) {
        long[] aBits = v1.binaryVector;
        long[] bBits = v2.binaryVector;

        int matches = 0;

        int bitBase = 0;
        for (int w = 0; w < aBits.length && bitBase < d; w++, bitBase += 64) {
            long aw = aBits[w];
            long bw = bBits[w];

            // Defensive tail masking (should be unnecessary when d is 64-bit aligned)
            int remaining = d - bitBase;
            if (remaining < 64) {
                long mask = (remaining == 64) ? ~0L : ((1L << remaining) - 1L);
                aw &= mask;
                bw &= mask;
            }

            matches += Long.bitCount(~(aw ^ bw)); // XNOR-popcount
        }

        // sign-dot in {-1,+1}^d
        float signDot = 2.0f * (float) matches - (float) d;

        return (v1.scale * v2.scale) * signDot + v1.offset + v2.offset + muNormSq;
    }

    private static void printCorrelations(double[] gt, double[] asym, double[] sym) {
        double cGtAsym = pearson(gt, asym);
        double cGtSym = pearson(gt, sym);
        double cAsymSym = pearson(asym, sym);

        System.out.printf("Pearson corr: GT vs ASYM = %.4f%n", cGtAsym);
        System.out.printf("Pearson corr: GT vs  SYM = %.4f%n", cGtSym);
        System.out.printf("Pearson corr: ASYM vs SYM = %.4f%n", cAsymSym);

        System.out.printf("Means: GT=%.6f ASYM=%.6f SYM=%.6f%n", mean(gt), mean(asym), mean(sym));
        System.out.printf("Stdev: GT=%.6f ASYM=%.6f SYM=%.6f%n", stdev(gt), stdev(asym), stdev(sym));
    }

    private static double pearson(double[] a, double[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("length mismatch");
        int n = a.length;

        double ma = mean(a);
        double mb = mean(b);

        double num = 0.0;
        double da = 0.0;
        double db = 0.0;
        for (int i = 0; i < n; i++) {
            double xa = a[i] - ma;
            double xb = b[i] - mb;
            num += xa * xb;
            da += xa * xa;
            db += xb * xb;
        }
        double denom = Math.sqrt(da) * Math.sqrt(db);
        return denom == 0.0 ? 0.0 : (num / denom);
    }

    private static double mean(double[] a) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += a[i];
        return s / (double) a.length;
    }

    private static double stdev(double[] a) {
        double m = mean(a);
        double s2 = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - m;
            s2 += d * d;
        }
        return Math.sqrt(s2 / (double) Math.max(1, a.length - 1));
    }

    private static int[] pickQueries(int subsetN, Random rng, int count) {
        int[] qs = new int[count];
        for (int i = 0; i < count; i++) qs[i] = rng.nextInt(subsetN);
        return qs;
    }

    private static void topKOverlapForQuery(RandomAccessVectorValues floatVectors,
                                            ASHVectors ashVectors,
                                            int q,
                                            int subsetN,
                                            int topK,
                                            int d,
                                            float muNormSq) {
        VectorFloat<?> xq = floatVectors.getVector(q);

        // Precompute asymmetric score function once
        ScoreFunction.ApproximateScoreFunction asymF =
                ashVectors.scoreFunctionFor(xq, VectorSimilarityFunction.DOT_PRODUCT);

        // Encoded q for symmetric
        AsymmetricHashing.QuantizedVector qEnc = ashVectors.get(q);

        // Score all candidates in subset (excluding self)
        Scored[] gt = new Scored[subsetN - 1];
        Scored[] asym = new Scored[subsetN - 1];
        Scored[] sym = new Scored[subsetN - 1];

        int idx = 0;
        for (int j = 0; j < subsetN; j++) {
            if (j == q) continue;

            VectorFloat<?> xj = floatVectors.getVector(j);

            double gtScore = VectorUtil.dotProduct(xq, xj);
            double asymScore = asymF.similarityTo(j);
            double symScore = (double) symmetricDot(qEnc, ashVectors.get(j), d, muNormSq);

            gt[idx] = new Scored(j, gtScore);
            asym[idx] = new Scored(j, asymScore);
            sym[idx] = new Scored(j, symScore);
            idx++;
        }

        int[] gtTop = topKIds(gt, topK);
        int[] asymTop = topKIds(asym, topK);
        int[] symTop = topKIds(sym, topK);

        System.out.printf("overlap(GT, ASYM)  = %.2f%%%n", 100.0 * overlapFraction(gtTop, asymTop));
        System.out.printf("overlap(GT,  SYM)  = %.2f%%%n", 100.0 * overlapFraction(gtTop, symTop));
        System.out.printf("overlap(ASYM, SYM) = %.2f%%%n", 100.0 * overlapFraction(asymTop, symTop));
    }

    private static int[] topKIds(Scored[] arr, int k) {
        int kk = Math.min(k, arr.length);
        Arrays.sort(arr, new Comparator<Scored>() {
            @Override
            public int compare(Scored a, Scored b) {
                return Double.compare(b.score, a.score); // descending
            }
        });
        int[] out = new int[kk];
        for (int i = 0; i < kk; i++) out[i] = arr[i].id;
        Arrays.sort(out);
        return out;
    }

    private static double overlapFraction(int[] aSorted, int[] bSorted) {
        int i = 0, j = 0, hit = 0;
        while (i < aSorted.length && j < bSorted.length) {
            int va = aSorted[i];
            int vb = bSorted[j];
            if (va == vb) { hit++; i++; j++; }
            else if (va < vb) i++;
            else j++;
        }
        return (double) hit / (double) aSorted.length;
    }

    private static final class Scored {
        final int id;
        final double score;
        Scored(int id, double score) { this.id = id; this.score = score; }
    }

    private static void sanityCheckBinaryMath(int d) {
        if ((d & 63) != 0) {
            System.out.println("[sanity] skipped: d not multiple of 64");
            return;
        }

        // For sign vectors s in {-1,+1}^d represented as bits b via s = 2b - 1:
        // zeros·zeros = d, ones·ones = d, zeros·ones = -d
        float exp00 = (float) d;
        float exp11 = (float) d;
        float exp01 = -(float) d;

        System.out.printf("[sanity] expected signDot: 00=%.1f 11=%.1f 01=%.1f%n", exp00, exp11, exp01);

        int words = (d + 63) >>> 6;
        long[] zeros = new long[words];
        long[] ones = new long[words];
        Arrays.fill(ones, ~0L);

        float got00 = computeSignDotFromBits(zeros, zeros, d);
        float got11 = computeSignDotFromBits(ones, ones, d);
        float got01 = computeSignDotFromBits(zeros, ones, d);

        System.out.printf("[sanity] computed signDot: 00=%.1f 11=%.1f 01=%.1f%n", got00, got11, got01);
    }

    private static float computeSignDotFromBits(long[] aBits, long[] bBits, int d) {
        int matches = 0;
        for (int w = 0; w < aBits.length; w++) {
            matches += Long.bitCount(~(aBits[w] ^ bBits[w]));
        }
        return 2.0f * (float) matches - (float) d;
    }
}
