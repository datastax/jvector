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

import io.github.jbellis.jvector.disk.MemorySegmentReader;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.*;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

/** Reader compatibility, byte-exact output, slice boundaries and direct read dispatch. */
public class ByteSequenceReadTest {
 @TempDir Path directory;
 @Test void wholeArrayReadsUseDirectReader() throws Exception {
  byte[] input=new byte[16384];new Random(314159).nextBytes(input);
  Path file=directory.resolve("bytes");Files.write(file,input);
  var provider=new MemorySegmentVectorProvider();
  try(var supplier=new MemorySegmentReader.Supplier(file);var r=new CountingReader(supplier.get())) {
   for(int length:new int[]{0,1,7,16,17,256,1536}) {
    r.seek(19);
    var created=provider.readByteSequence(r,length);
    for(int i=0;i<length;i++)assertEquals(input[19+i],created.get(i));
    assertEquals(19+length,r.getPosition());
    var destination=provider.createByteSequence(length);
    r.seek(29);provider.readByteSequence(r,destination);
    for(int i=0;i<length;i++)assertEquals(input[29+i],destination.get(i));
    assertEquals(29+length,r.getPosition());
   }
   assertEquals(14,r.arrayReads);
   assertEquals(0,r.bufferReads,"Whole-array storage must not create ByteBuffer wrappers");
  }
 }
 @Test void nestedSlicesPreserveSurroundingBytes() throws Exception {
  byte[] input=new byte[8192];new Random(271828).nextBytes(input);
  Path file=directory.resolve("slice");Files.write(file,input);
  var provider=new MemorySegmentVectorProvider();
  try(var supplier=new MemorySegmentReader.Supplier(file);var r=new CountingReader(supplier.get())) {
   for(int length:new int[]{0,1,7,17,256,1536}) {
    byte[] data=new byte[length+11];Arrays.fill(data,(byte)99);
    var parent=provider.createByteSequence(data);
    var slice=parent.slice(2,length+6).slice(3,length);
    r.seek(23);provider.readByteSequence(r,slice);
    for(int i=0;i<data.length;i++)assertEquals(i>=5&&i<5+length?input[23+i-5]:(byte)99,data[i]);
    assertEquals(23+length,r.getPosition());
   }
   assertEquals(0,r.arrayReads);
   assertEquals(6,r.bufferReads);
  }
 }
 @Test void failedReadDoesNotAdvanceReader() throws Exception {
  Path file=directory.resolve("short");Files.write(file,new byte[]{1,2,3,4});
  var provider=new MemorySegmentVectorProvider();
  try(var supplier=new MemorySegmentReader.Supplier(file);var r=supplier.get()) {
   r.seek(3);
   assertThrows(IndexOutOfBoundsException.class,()->provider.readByteSequence(r,2));
   assertEquals(3,r.getPosition());
   byte[] data={9,9};var destination=provider.createByteSequence(data);
   assertThrows(IndexOutOfBoundsException.class,()->provider.readByteSequence(r,destination));
   assertEquals(3,r.getPosition());assertArrayEquals(new byte[]{9,9},data);
  }
 }
 static class CountingReader implements RandomAccessReader {
  final RandomAccessReader r;int arrayReads,bufferReads;
  CountingReader(RandomAccessReader r){this.r=r;}
  public void seek(long p)throws IOException{r.seek(p);}
  public long getPosition()throws IOException{return r.getPosition();}
  public int readInt()throws IOException{return r.readInt();}
  public float readFloat()throws IOException{return r.readFloat();}
  public long readLong()throws IOException{return r.readLong();}
  public void readFully(byte[] b)throws IOException{arrayReads++;r.readFully(b);}
  public void readFully(ByteBuffer b)throws IOException{bufferReads++;r.readFully(b);}
  public void readFully(long[] b)throws IOException{r.readFully(b);}
  public void read(int[] b,int o,int n)throws IOException{r.read(b,o,n);}
  public void read(float[] b,int o,int n)throws IOException{r.read(b,o,n);}
  public long length()throws IOException{return r.length();}
  public void close()throws IOException{r.close();}
 }
}
