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

unsigned int combineBytes(int i, unsigned int shuffle, const char* quantizedPartials) {
    // This is a 16-bit value stored in two bytes, so we need to move in multiples of two and then combine them.
    unsigned int lowByte = quantizedPartials[i * 512 + shuffle];
    unsigned int highByte = quantizedPartials[i * 512 + shuffle + 1];
    return (highByte << 8) | lowByte;
}

unsigned int computeSingleShuffle(int i, int j, const unsigned char* shuffles, int nNeighbors) {
    // This points to a 16-bit value stored in two bytes, so we need to move in multiples of two.
    unsigned int temp = shuffles[i * nNeighbors + j];
    return temp * 2;
}