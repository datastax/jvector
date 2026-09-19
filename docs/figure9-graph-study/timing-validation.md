## Timing validation

The predeclared rule retimed any case/JVM with a sample CV above 10%, using both overquery settings and at least 10 seconds per sample (three samples). Four cached-index JVMs were retimed. All recall and visited-count values matched the originals exactly. No reconstruction was performed. These longer observations supply QPS in the reported tables; original observations remain in results.csv and in the original columns of reported-results.csv.

| Case | Overquery | Original QPS | Longer QPS | Original CV | Longer CV |
|---|---:|---:|---:|---:|---:|
| ada002-1M-b4-pq-r2 | 1× | 1,821 | 2,214 | 30.7% | 2.7% |
| ada002-1M-b4-pq-r2 | 2× | 1,588 | 1,697 | 14.3% | 0.6% |
| cap-1M-b2-ash-r2 | 1× | 2,258 | 2,672 | 23.8% | 6.6% |
| cap-1M-b2-ash-r2 | 2× | 2,053 | 2,181 | 2.7% | 1.4% |
| cap-1M-b4-pq-r2 | 1× | 1,997 | 2,463 | 18.8% | 5.7% |
| cap-1M-b4-pq-r2 | 2× | 1,628 | 1,937 | 15.1% | 0.9% |
| cohere-english-v3-1M-b4-ash-r1 | 1× | 2,915 | 3,660 | 18.9% | 3.3% |
| cohere-english-v3-1M-b4-ash-r1 | 2× | 2,319 | 2,653 | 2.8% | 0.8% |
