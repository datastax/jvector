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

package io.github.jbellis.jvector.vector;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.concurrent.atomic.AtomicReference;

/**
 * SimdOpsJava24Plus contains the implementation of vectorized solutions with compatibility of apis available in Java 24 and above
 */
final class SimdOpsJava24Plus
        extends SimdOps
{
    public static void quantizePartials(float delta, ArrayVectorFloat partials, ArrayVectorFloat partialBases, AtomicReference<Object> quantizedPartials) {
        var codebookSize = partials.length() / partialBases.length();
        var codebookCount = partialBases.length();
        quantizedPartials.set(java.lang.reflect.Array.newInstance(short.class, partials.length()));
        short[] quantizedPartialsShortsArray = (short[]) (quantizedPartials.get());

        for (int i = 0; i < codebookCount; i++) {
            var vectorizedLength = FloatVector.SPECIES_512.loopBound(codebookSize);
            var codebookBase = partialBases.get(i);
            var codebookBaseVector = FloatVector.broadcast(FloatVector.SPECIES_512, codebookBase);
            int j = 0;
            for (; j < vectorizedLength; j += FloatVector.SPECIES_512.length()) {
                var partialVector = FloatVector.fromArray(FloatVector.SPECIES_512, partials.get(), i * codebookSize + j);
                var quantized = (partialVector.sub(codebookBaseVector)).div(delta);
                quantized = quantized.max(FloatVector.zero(FloatVector.SPECIES_512)).min(FloatVector.broadcast(FloatVector.SPECIES_512, 65535));
                var quantizedBytes = (ShortVector) quantized.convertShape(VectorOperators.F2S, ShortVector.SPECIES_256, 0);
                quantizedBytes.intoArray(quantizedPartialsShortsArray, i * codebookSize + j);
            }
            for (; j < codebookSize; j++) {
                var val = partials.get(i * codebookSize + j);
                var quantized = (short) Math.min((val - codebookBase) / delta, 65535);
                quantizedPartialsShortsArray[i * codebookSize + j] = quantized;
            }
        }
    }

    //--------------------------------------------------------------------------------
    // edgeLoadingSimilarity quantized similarity implementation using Java Vector API.
    // selectFrom and SaturatedAddition operations are available from Java 24 and above.
    //--------------------------------------------------------------------------------
    static final FloatVector CONSTANT1F = FloatVector.broadcast(FloatVector.SPECIES_512, 1.f);
    static final FloatVector CONSTANT2F = FloatVector.broadcast(FloatVector.SPECIES_512, 2.f);
    public static void bulkShuffleQuantizedSimilarity(ArrayByteSequence shuffles,
            int codebookCount,
            short[] quantizedPartialsShorts,
            float delta,
            float minDistance,
            VectorSimilarityFunction vsf, ArrayVectorFloat results)
    {
        VectorSpecies<Short> shortSpecies512 = ShortVector.SPECIES_512;
        VectorSpecies<Byte> byteVectorSpecies256 = ByteVector.SPECIES_256;
        var sum = ShortVector.zero(shortSpecies512);

        // Lookup table implementation
        for (int i = 0; i < codebookCount; i++) {
            ShortVector shuffle512 = (ShortVector) ByteVector.fromArray(byteVectorSpecies256, shuffles.get(),
                            i * byteVectorSpecies256.length())
                    .convertShape(VectorOperators.ZERO_EXTEND_B2S, shortSpecies512, 0);
            var partialsVec = lookUpPartialSums(shuffle512, quantizedPartialsShorts, i);
            sum = sum.lanewise(VectorOperators.SUADD, partialsVec);
        }
        ShortVector quantizedResultsLeftRaw = (ShortVector) sum.reinterpretShape(ShortVector.SPECIES_256, 0);
        ShortVector quantizedResultsRightRaw = (ShortVector) sum.reinterpretShape(ShortVector.SPECIES_256, 1);
        FloatVector resultsLeft = dequantize(quantizedResultsLeftRaw, delta, minDistance).add(CONSTANT1F);
        FloatVector resultsRight = dequantize(quantizedResultsRightRaw, delta, minDistance).add(CONSTANT1F);

        switch (vsf) {
            case DOT_PRODUCT -> {
                resultsLeft = resultsLeft.div(CONSTANT2F);
                resultsRight = resultsRight.div(CONSTANT2F);
                resultsLeft.intoArray(results.get(), 0);
                resultsRight.intoArray(results.get(), 16);
            }
            case EUCLIDEAN -> {
                resultsLeft = CONSTANT1F.div(resultsLeft);
                resultsRight = CONSTANT1F.div(resultsRight);
                resultsLeft.intoArray(results.get(), 0);
                resultsRight.intoArray(results.get(), 16);
            }
        }
    }

    public static void bulkShuffleQuantizedSimilarityCosine(ArrayByteSequence shuffles,
            int codebookCount,
            short[] quantizedPartialSumsShorts, float sumDelta, float minDistance,
            short[] quantizedPartialSquaredMagnitudesShorts,float magnitudeDelta,
            float minMagnitude, float queryMagnitudeSquared, ArrayVectorFloat results) {
        VectorSpecies<Short> shortSpecies512 = ShortVector.SPECIES_512;
        VectorSpecies<Byte> byteVectorSpecies256 = ByteVector.SPECIES_256;
        var sum = ShortVector.zero(shortSpecies512);
        var magnitude = ShortVector.zero(shortSpecies512);

        // Lookup table implementation
        for (int i = 0; i < codebookCount; i++) {
            ShortVector shuffle512 = (ShortVector) ByteVector.fromArray(byteVectorSpecies256, shuffles.get(),
                            i * byteVectorSpecies256.length())
                    .convertShape(VectorOperators.ZERO_EXTEND_B2S, shortSpecies512, 0);
            var partialsVec = lookUpPartialSums(shuffle512, quantizedPartialSumsShorts, i);
            sum = sum.lanewise(VectorOperators.SUADD, partialsVec);

            var partialsMag = lookUpPartialSums(shuffle512, quantizedPartialSquaredMagnitudesShorts, i);
            magnitude = magnitude.lanewise(VectorOperators.SUADD, partialsMag);
        }

        ShortVector quantizedResultsLeftRaw = (ShortVector) sum.reinterpretShape(ShortVector.SPECIES_256, 0);
        ShortVector quantizedResultsRightRaw = (ShortVector) sum.reinterpretShape(ShortVector.SPECIES_256, 1);
        FloatVector resultsSumLeft = dequantize(quantizedResultsLeftRaw, sumDelta, minDistance);
        FloatVector resultsSumRight = dequantize(quantizedResultsRightRaw, sumDelta, minDistance);

        quantizedResultsLeftRaw = (ShortVector) magnitude.reinterpretShape(ShortVector.SPECIES_256, 0);
        quantizedResultsRightRaw = (ShortVector) magnitude.reinterpretShape(ShortVector.SPECIES_256, 1);
        FloatVector resultsMagLeft = dequantize(quantizedResultsLeftRaw, magnitudeDelta, minMagnitude);
        FloatVector resultsMagRight = dequantize(quantizedResultsRightRaw, magnitudeDelta, minMagnitude);

        FloatVector queryMagnitudeSquaredVec = FloatVector.broadcast(FloatVector.SPECIES_512, queryMagnitudeSquared);
        resultsMagLeft = resultsMagLeft.mul(queryMagnitudeSquaredVec).sqrt();
        resultsMagRight = resultsMagRight.mul(queryMagnitudeSquaredVec).sqrt();

        resultsSumLeft = resultsSumLeft.div(resultsMagLeft).add(CONSTANT1F).div(CONSTANT2F);
        resultsSumRight = resultsSumRight.div(resultsMagRight).add(CONSTANT1F).div(CONSTANT2F);
        resultsSumLeft.intoArray(results.get(), 0);
        resultsSumRight.intoArray(results.get(), 16);
    }

    private static FloatVector dequantize(ShortVector quantizedVec, float delta, float base)
    {
        IntVector quantizedVecWidened = quantizedVec.convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0).reinterpretAsInts();
        //Use convert shape as reinterpretAsFloats does not perform numerical conversion, but interprets binary form as float.
        FloatVector floatVec = quantizedVecWidened.convert(VectorOperators.I2F, 0).reinterpretAsFloats();
        //Broadcast deltaVect and baseVec
        FloatVector deltaVec = FloatVector.broadcast(FloatVector.SPECIES_512, delta);
        FloatVector baseVec = FloatVector.broadcast(FloatVector.SPECIES_512, base);
        //Compute FMA
        return floatVec.fma(deltaVec, baseVec);
    }

    private static ShortVector lookUpPartialSums(ShortVector shuffle512, short[] quantizedPartialsShorts, int i)
    {
        VectorSpecies<Short> shortVectorSpecies512 = ShortVector.SPECIES_512;
        int baseOffset = (i * 256);
        ShortVector partialsVecA = ShortVector.fromArray(shortVectorSpecies512, quantizedPartialsShorts, baseOffset);
        ShortVector partialsVecB = ShortVector.fromArray(shortVectorSpecies512, quantizedPartialsShorts, baseOffset + 32);
        ShortVector partialsVecC = ShortVector.fromArray(shortVectorSpecies512, quantizedPartialsShorts, baseOffset + 64);
        ShortVector partialsVecD = ShortVector.fromArray(shortVectorSpecies512, quantizedPartialsShorts, baseOffset + 96);
        ShortVector partialsVecE = ShortVector.fromArray(shortVectorSpecies512, quantizedPartialsShorts, baseOffset + 128);
        ShortVector partialsVecF = ShortVector.fromArray(shortVectorSpecies512, quantizedPartialsShorts, baseOffset + 160);
        ShortVector partialsVecG = ShortVector.fromArray(shortVectorSpecies512, quantizedPartialsShorts, baseOffset + 192);
        ShortVector partialsVecH = ShortVector.fromArray(shortVectorSpecies512, quantizedPartialsShorts, baseOffset + 224);

        ShortVector partialsVecAB = shuffle512.selectFrom(partialsVecA, partialsVecB);
        ShortVector partialsVecCD = shuffle512.selectFrom(partialsVecC, partialsVecD);
        ShortVector partialsVecEF = shuffle512.selectFrom(partialsVecE, partialsVecF);
        ShortVector partialsVecGH = shuffle512.selectFrom(partialsVecG, partialsVecH);

        ShortVector maskSevenBitVector = ShortVector.broadcast(ShortVector.SPECIES_512, 0x0040);
        ShortVector maskEightBitVector = ShortVector.broadcast(ShortVector.SPECIES_512, 0x0080);
        VectorMask<Short> maskSeven = shuffle512.and(maskSevenBitVector).compare(VectorOperators.NE, ShortVector.zero(ShortVector.SPECIES_512));
        VectorMask<Short> maskEight = shuffle512.and(maskEightBitVector).compare(VectorOperators.NE, ShortVector.zero(ShortVector.SPECIES_512));

        ShortVector partialsVecABCD = partialsVecAB.blend(partialsVecCD, maskSeven);
        ShortVector partialsVecEFGH = partialsVecEF.blend(partialsVecGH, maskSeven);

        return partialsVecABCD.blend(partialsVecEFGH, maskEight);
    }

    //-----------------------------------------------------------------------------------------
    // edgeLoadingSimilarity quantized similarity implementation using Java Vector API end here
    //-------------------------------------------------------------------------------- --------
}
