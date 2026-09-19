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

package io.github.jbellis.jvector.example.diagnostics;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.ASHScorer;
import io.github.jbellis.jvector.quantization.ASHSymmetricScorer;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

/** Input-space comparison using production scorers, with all scores on the raw dot-product scale. */
public final class ASHScoreDebug {
    private static volatile float sink;
    private ASHScoreDebug() {}
    public static void run(RandomAccessVectorValues values, ASHVectors encoded,
                           int pairs, int subset, int k, Random random) {
        if (pairs <= 0 || subset < 2 || k <= 0) throw new IllegalArgumentException("Invalid sample sizes");
        int n = Math.min(subset, values.size());
        if (n < 2) throw new IllegalArgumentException("At least two vectors are required");
        k = Math.min(k, n - 1);
        var symmetric = new ASHSymmetricScorer(encoded);
        var asymmetric = new ASHScorer(encoded.getCompressor());
        // Separate views: shared RAVV implementations may overwrite returned vectors.
        var left = values.copy(); var right = values.copy();
        double[] gt = new double[pairs], sa = new double[pairs], sb = new double[pairs], ss = new double[pairs];
        int asymWins = 0, symWins = 0;
        for (int t = 0; t < pairs; t++) {
            int i = random.nextInt(n), j = random.nextInt(n - 1);
            if (j >= i) j++;
            var x = left.getVector(i); var y = right.getVector(j);
            gt[t] = VectorUtil.dotProduct(x,y);
            sa[t] = asymmetric.scoreFunctionFor(x,VectorSimilarityFunction.DOT_PRODUCT).similarityTo(encoded.get(j));
            sb[t] = asymmetric.scoreFunctionFor(y,VectorSimilarityFunction.DOT_PRODUCT).similarityTo(encoded.get(i));
            ss[t] = symmetric.dotProduct(i,j);
            if (Math.abs(sa[t]-gt[t]) < Math.abs(ss[t]-gt[t])) asymWins++;
            if (Math.abs(ss[t]-gt[t]) < Math.abs(sa[t]-gt[t])) symWins++;
        }
        System.out.printf("ASH score study: N=%d sampleN=%d pairs=%d bpd=%d d=%d raw dots, self excluded%n",
                values.size(),n,pairs,encoded.getCompressor().bitsPerDimension,encoded.getCompressor().quantizedDim);
        report("ASYM_FORWARD",gt,sa); report("ASYM_REVERSE",gt,sb); report("SYMMETRIC",gt,ss);
        System.out.printf("CLOSER_TO_EXACT asym=%d sym=%d ties=%d%n",asymWins,symWins,pairs-asymWins-symWins);
        int queries = 16;
        double overlapAsym=0, overlapSym=0, localA=0,localS=0;
        for (int q = 0; q < queries; q++) {
            int id=random.nextInt(n);
            var x=left.getVector(id);
            var asf=asymmetric.scoreFunctionFor(x,VectorSimilarityFunction.DOT_PRODUCT);
            float[][] scores = new float[3][n];
            for (int j=0;j<n;j++) {
                scores[0][j]=VectorUtil.dotProduct(x,right.getVector(j));
                scores[1][j]=asf.similarityTo(encoded.get(j));
                scores[2][j]=symmetric.dotProduct(id,j);
            }
            for (float[] row:scores) row[id]=Float.NEGATIVE_INFINITY;
            Integer[] exact=top(scores[0],Math.min(Math.max(100,k),n-1));
            Integer[] at=top(scores[1],k), st=top(scores[2],k);
            for (int i=0;i<k;i++) {
                if (Arrays.asList(at).contains(exact[i])) overlapAsym++;
                if (Arrays.asList(st).contains(exact[i])) overlapSym++;
            }
            for (int rank = 0; rank < Math.min(100, exact.length); rank++) {
                int j = exact[rank];
                int count = Math.min(100, exact.length);
                localA+=Math.abs(scores[1][j]-scores[0][j])/count;
                localS+=Math.abs(scores[2][j]-scores[0][j])/count;
            }
        }
        System.out.printf("EXACT_TOP%d_OVERLAP queries=%d subset=%d ASYM=%.6f SYMMETRIC=%.6f NEAREST100_MAE_ASYM=%.7f NEAREST100_MAE_SYM=%.7f%n",
                k,queries,n,overlapAsym/(queries*k),overlapSym/(queries*k),localA/queries,localS/queries);
        int queryId = 0;
        var query = left.getVector(queryId);
        time("SYMMETRIC_PACKED", symmetric.scoreFunctionFor(queryId), n);
        String oldMode = System.getProperty("jvector.ash.singleKernel");
        try {
            System.setProperty("jvector.ash.singleKernel", "scalar");
            time("ASYMMETRIC_SCALAR", encoded.scoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT), n);
            int bits = encoded.getCompressor().bitsPerDimension;
            if ((bits == 2 || bits == 4) && io.github.jbellis.jvector.vector.VectorizationProvider.getInstance()
                    .getVectorUtilSupport().supportsAshProjectionScoring()) {
                System.setProperty("jvector.ash.singleKernel", "simd");
                time("ASYMMETRIC_SIMD", encoded.scoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT), n);
            }
        } finally {
            if (oldMode == null) System.clearProperty("jvector.ash.singleKernel");
            else System.setProperty("jvector.ash.singleKernel", oldMode);
        }
    }

    // Single-thread sequential scans; query setup excluded. This is not a graph QPS estimate.
    private static void time(String mode, io.github.jbellis.jvector.graph.similarity.ScoreFunction sf, int n) {
        long[] times = new long[9];
        for (int pass = -5; pass < times.length; pass++) {
            float total = 0;
            long start = System.nanoTime();
            for (int j = 0; j < n; j++) total += sf.similarityTo(j);
            long elapsed = System.nanoTime()-start;
            sink = total;
            if (pass >= 0) times[pass] = elapsed;
        }
        Arrays.sort(times);
        System.out.printf("STEADY_SCORING %s median_ns_per_pair=%.3f millions_per_second=%.3f%n",
                mode, times[4]/(double)n, n*1000.0/times[4]);
    }
    private static Integer[] top(float[] a,int k) {
        Integer[] ids=new Integer[a.length]; for(int i=0;i<a.length;i++)ids[i]=i;
        Arrays.sort(ids,Comparator.<Integer>comparingDouble(i->-a[i]).thenComparingInt(i->i));
        return Arrays.copyOf(ids,k);
    }
    private static void report(String name,double[] x,double[] y) {
        double sx=0,sy=0,sxx=0,syy=0,sxy=0,mae=0,mse=0;
        for(int i=0;i<x.length;i++) {
            sx+=x[i];sy+=y[i];sxx+=x[i]*x[i];syy+=y[i]*y[i];sxy+=x[i]*y[i];
            double err=y[i]-x[i];mae+=Math.abs(err);mse+=err*err;
        }
        int n=x.length;
        double corr=(n*sxy-sx*sy)/Math.sqrt((n*sxx-sx*sx)*(n*syy-sy*sy));
        System.out.printf("SCORE_ERROR %s pearson=%.7f MAE=%.7f RMSE=%.7f bias=%.7f%n",name,corr,mae/n,Math.sqrt(mse/n),(sy-sx)/n);
    }
}
