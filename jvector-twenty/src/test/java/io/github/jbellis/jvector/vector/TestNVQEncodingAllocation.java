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
import java.util.Random;
import io.github.jbellis.jvector.util.MathUtil;
import io.github.jbellis.jvector.vector.types.*;
import jdk.incubator.vector.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
class TestNVQEncodingAllocation {
 @Test void scalarEncodingPreservesDestinationSlicesAndEmptyVectors() {
  var current=new PanamaVectorUtilSupport();var old=new Reference();
  for(int length=0;length<FloatVector.SPECIES_PREFERRED.length();length++) {
   var vector=new ArrayVectorFloat(new float[length]);
   var expected=new ArrayByteSequence(length);
   old.nvqQuantize8bit(vector,1,0,-1,1,expected);
   var parent=new ArrayByteSequence(length+6);
   for(int i=0;i<parent.length();i++)parent.set(i,(byte)91);
   current.nvqQuantize8bit(vector,1,0,-1,1,parent.slice(3,length));
   for(int i=0;i<parent.length();i++)
    assertEquals(i>=3&&i<3+length?expected.get(i-3):(byte)91,parent.get(i));
  }
 }
 @Test void preservesFrozenEncodingAndNumericalResults() {
  var current=new PanamaVectorUtilSupport();var old=new Reference();var random=new Random(9117);
  for(int dim:new int[]{0,1,7,8,9,15,16,17,31,32,33,127,128,129,256,768,1536}) {
   float[] data=new float[dim];for(int i=0;i<dim;i++)data[i]=random.nextFloat()*2-1;
   var vector=new ArrayVectorFloat(data);
   for(float alpha:new float[]{.1f,.5f,1f,2f,5f})for(float midpoint:new float[]{-.25f,0f,.25f,.5f}) {
    var a=new ArrayByteSequence(new byte[dim]);var b=new ArrayByteSequence(new byte[dim]);
    current.nvqQuantize8bit(vector,alpha,midpoint,-1,1,a);old.nvqQuantize8bit(vector,alpha,midpoint,-1,1,b);
    for(int i=0;i<dim;i++)assertEquals(b.get(i),a.get(i),"dim="+dim+" i="+i);
    assertReduction(old.nvqUniformLoss(vector,-1,1,8),current.nvqUniformLoss(vector,-1,1,8),"uniform loss dim="+dim);
    assertReduction(old.nvqSquareL2Distance8bit(vector,b,alpha,midpoint,-1,1),current.nvqSquareL2Distance8bit(vector,b,alpha,midpoint,-1,1),"L2 dim="+dim);
    float[] expected=old.nvqCosine8bit(vector,b,alpha,midpoint,-1,1,vector),actual=current.nvqCosine8bit(vector,b,alpha,midpoint,-1,1,vector);
    for(int n=0;n<2;n++)assertReduction(expected[n],actual[n],"cosine component="+n+" dim="+dim);
    // Floating reductions may use different addition orders as the JVM warms up.
    for(int bits:new int[]{1,2,4,8}) {
     float expectedLoss=old.nvqLoss(vector,alpha,midpoint,-1,1,bits);
     float actualLoss=current.nvqLoss(vector,alpha,midpoint,-1,1,bits);
     assertReduction(expectedLoss,actualLoss,
             "dim="+dim+" alpha="+alpha+" midpoint="+midpoint+" bits="+bits);
    }
   }
  }
 }
 // Match the existing scalar-oracle bounds in TestNVQLoss and TestNVQScoring.
 // ADD reductions may reorder additions; encoded bytes are still checked exactly.
 private static void assertReduction(float expected,float actual,String message) {
  assertEquals(expected,actual,Math.max(1e-6,Math.abs(expected)*2e-6),message);
 }
 static class Reference extends PanamaVectorUtilSupport {
    public float nvqLoss(VectorFloat<?> vector, float alpha, float x0, float minValue, float maxValue, int nBits) {
        int constant = (1 << nBits) - 1;
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / constant;
        var invLogisticScale = 1 / logisticScale;

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var arr = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            var recArr = logisticNQT(arr, scaledAlpha, scaledX0);
            recArr = recArr.sub(logisticBias).mul(invLogisticScale);
            IntVector rounded = (IntVector) recArr.add(const05f)
                    .convertShape(VectorOperators.F2I, IntVector.SPECIES_PREFERRED, 0);
            recArr = (FloatVector) rounded.convertShape(VectorOperators.I2F, FloatVector.SPECIES_PREFERRED, 0);
            recArr = recArr.fma(logisticScale, logisticBias);
            recArr = logitNQT(recArr, invScaledAlpha, scaledX0);

            var diff = arr.sub(recArr);
            squaredSumVec = diff.fma(diff, squaredSumVec);
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value, recValue;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value = vector.get(i);

            recValue = logisticNQT(value, scaledAlpha, scaledX0);
            recValue = (recValue - logisticBias) * invLogisticScale;
            recValue = Math.round(recValue);
            recValue = Math.fma(logisticScale, recValue, logisticBias);
            recValue = logitNQT(recValue, invScaledAlpha, scaledX0);

            squaredSum += MathUtil.square(value - recValue);
        }

        return squaredSum;
    }
    public float nvqUniformLoss(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
        float constant = (1 << nBits) - 1;
        float delta = maxValue - minValue;

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var arr = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            var recArr = arr.sub(minValue).mul(constant / delta);
            recArr = recArr.add(const05f)
                    .convert(VectorOperators.F2I, 0)
                    .reinterpretAsInts()
                    .convert(VectorOperators.I2F, 0)
                    .reinterpretAsFloats();
            recArr = recArr.fma(delta / constant, minValue);

            var diff = arr.sub(recArr);
            squaredSumVec = diff.fma(diff, squaredSumVec);
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value, recValue;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value = vector.get(i);

            recValue = (value - minValue) / delta;
            recValue = Math.round(constant * recValue) / constant;
            recValue = recValue * delta + minValue;

            squaredSum += MathUtil.square(value - recValue);
        }

        return squaredSum;
    }
    public float nvqSquareL2Distance8bit(VectorFloat<?> vector, ByteSequence<?> quantizedVector,
                                         float alpha, float x0, float minValue, float maxValue) {
        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(quantizedVector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = fromByteSequence(ByteVector.SPECIES_PREFERRED, quantizedVector, i);

            for (int j = 0; j < 4; j++) {
                var v1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i + floatStep * j);
                var v2 = nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j);

                var diff = v1.sub(v2);
                squaredSumVec = diff.fma(diff, squaredSumVec);
            }
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2, diff;
        for (int i = vectorizedLength; i < quantizedVector.length(); i++) {
            value2 = Byte.toUnsignedInt(quantizedVector.get(i));
            value2 = Math.fma(logisticScale, value2, logisticBias);
            value2 = logitNQT(value2, invScaledAlpha, scaledX0);
            diff = vector.get(i) - value2;
            squaredSum += MathUtil.square(diff);
        }

        return squaredSum;
    }
    public float[] nvqCosine8bit(VectorFloat<?> vector,
                                 ByteSequence<?> quantizedVector, float alpha, float x0, float minValue, float maxValue,
                                 VectorFloat<?> centroid) {
        if (vector.length() != centroid.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;

        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(vector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = fromByteSequence(ByteVector.SPECIES_PREFERRED, quantizedVector, i);

            for (int j = 0; j < 4; j++) {
                var va = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i + floatStep * j);
                var vb = nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j);

                var vCentroid = fromVectorFloat(FloatVector.SPECIES_PREFERRED, centroid, i + floatStep * j);
                vb = vb.add(vCentroid);

                vsum = va.fma(vb, vsum);
                vbMagnitude = vb.fma(vb, vbMagnitude);
            }
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value2 = Byte.toUnsignedInt(quantizedVector.get(i));
            value2 = Math.fma(logisticScale, value2, logisticBias);
            value2 = logitNQT(value2, invScaledAlpha, scaledX0) + centroid.get(i);
            sum = Math.fma(vector.get(i), value2, sum);
            bMagnitude = Math.fma(value2, value2, bMagnitude);
        }

        // TODO can we avoid returning a new array?
        return new float[]{sum, bMagnitude};
    }
    FloatVector logisticNQT(FloatVector vector, float alpha, float x0) {
        FloatVector temp = vector.fma(alpha, -alpha * x0);
        VectorMask<Float> isPositive = temp.test(VectorOperators.IS_NEGATIVE).not();
        IntVector p = (IntVector) temp.add(1, isPositive)
                .convertShape(VectorOperators.F2I, IntVector.SPECIES_PREFERRED, 0);
        FloatVector e = (FloatVector) p.convertShape(VectorOperators.I2F, FloatVector.SPECIES_PREFERRED, 0);
        IntVector m = temp.sub(e).fma(0.5f, 1).reinterpretAsInts();

        temp = m.add(p.lanewise(VectorOperators.LSHL, 23)).reinterpretAsFloats();  // temp = m * 2^p
        return temp.div(temp.add(1));
    }
    public void nvqQuantize8bit(VectorFloat<?> vector, float alpha, float x0, float minValue, float maxValue, ByteSequence<?> destination) {
        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        final var mask = ByteVector.SPECIES_PREFERRED.indexInRange(0, FloatVector.SPECIES_PREFERRED.length());

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var invLogisticScale = 255 / (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias);

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var arr = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            arr = logisticNQT(arr, scaledAlpha, scaledX0);
            arr = arr.sub(logisticBias).mul(invLogisticScale);
            var bytes = arr.add(const05f)
                    .convertShape(VectorOperators.F2B, ByteVector.SPECIES_PREFERRED, 0)
                    .reinterpretAsBytes();

            intoByteSequence(bytes, destination, i, mask);
        }

        // Process the tail
        for (int d = vectorizedLength; d < vector.length(); d++) {
            // Ensure the quantized value is within the 0 to constant range
            float value = vector.get(d);
            value = logisticNQT(value, scaledAlpha, scaledX0);
            value = (value - logisticBias) * invLogisticScale;
            int quantizedValue = Math.round(value);
            destination.set(d, (byte) quantizedValue);
        }
    }
 }
}
