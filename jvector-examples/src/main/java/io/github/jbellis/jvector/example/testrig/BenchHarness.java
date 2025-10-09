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

package io.github.jbellis.jvector.example.testrig;


import io.nosqlbench.nbdatatools.api.concurrent.ProgressIndicator;
import io.nosqlbench.vectordata.discovery.TestDataView;
import io.nosqlbench.vectordata.downloader.DatasetEntry;
import io.nosqlbench.vectordata.spec.datasets.types.BaseVectors;

import java.util.Arrays;
import java.util.concurrent.*;

public class BenchHarness implements Runnable {

  private final DatasetEntry datasetEntry;
  private final String profile;
  private final int concurrency;
  private final ExecutorService virtualThreadExecutor;
  private final Semaphore semaphore;

  public BenchHarness(
      io.nosqlbench.vectordata.downloader.DatasetEntry datasetEntry,
      String profile
  )
  {
    this(datasetEntry, profile, 1);
  }

  public BenchHarness(
      io.nosqlbench.vectordata.downloader.DatasetEntry datasetEntry,
      String profile,
      int concurrency
  )
  {
    this.datasetEntry = datasetEntry;
    this.profile = profile;
    this.concurrency = concurrency;
    this.virtualThreadExecutor = Executors.newCachedThreadPool();
    this.semaphore = new Semaphore(concurrency);
  }

  @Override
  public void run() {
    TestDataView testDataView = datasetEntry.select().profile(profile);
    smokeTestDataLoad(testDataView);
  }

  private void smokeTestDataLoad(TestDataView testDataView) {
      BaseVectors bv = testDataView.getBaseVectors().orElseThrow();

      System.out.println("Prebuffering...");
      CompletableFuture<Void> prebuffer = bv.prebuffer();
      if (prebuffer instanceof ProgressIndicator<?>) {
          ((ProgressIndicator<?>)prebuffer).monitorProgress(1000);
      }
      prebuffer.join();
      System.out.println("Prebuffered");

      float[] v1 = bv.get(1);
      System.out.println(Arrays.toString(v1));

    float[] vend = bv.get(bv.getCount() - 1);
    System.out.println(Arrays.toString(vend));

    /// Create tasks for processing vectors concurrently
    CompletableFuture<?>[] futures = new CompletableFuture[100];

    for (int i = 0; i < 100; i++) {
      final int index = i;
      futures[i] = CompletableFuture.runAsync(() -> {
        try {
          semaphore.acquire();
      try {
            /// This will be a stepping through the space of vectors
            int idx = (int) ((float)index / 100 * bv.getCount());
            float[] v = bv.get(idx);
            System.out.println(Arrays.toString(v));
          } finally {
            semaphore.release();
          }
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException(e);
        }
      }, virtualThreadExecutor);
    }

    /// Wait for all tasks to complete
    CompletableFuture.allOf(futures).join();

    /// Shutdown the executor
    virtualThreadExecutor.shutdown();
    try {
      if (!virtualThreadExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
        virtualThreadExecutor.shutdownNow();
      }
    } catch (InterruptedException e) {
      virtualThreadExecutor.shutdownNow();
      Thread.currentThread().interrupt();
    }
  }


}
