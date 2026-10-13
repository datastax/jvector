from pathlib import Path
import csv,statistics,json,subprocess
p=Path(__file__).resolve().parent.parent
from data import load_rows
rows=load_rows()
assert len(rows)==96,len(rows)
for r in rows:
 for k in ['bits','rep','threads','overquery','qps','recall','build_s','provider_s']:
  r[k]=float(r[k]) if r[k] else 0.
by={}
for r in rows:
 key=(r['dataset'],r['bits'],r['mode'],r['rep'],r['overquery'])
 by.setdefault(key,{})[r['threads']]=r
for key,pair in by.items():
 assert set(pair)=={1,48},key
 assert pair[1]['recall']==pair[48]['recall'],key
 assert pair[1]['visited']==pair[48]['visited'],key
for r in rows:assert r['qps']>0 and 0<=r['recall']<=1
subprocess.run(['python3',str(Path(__file__).with_name('tables.py'))],check=True)
subprocess.run(['python3',str(Path(__file__).with_name('plot.py'))],check=True)
def group(ds,b,m,o):return [r for r in rows if r['dataset']==ds and r['bits']==b and r['mode']==m and r['threads']==1 and r['overquery']==o]
def avg(rr,k):return statistics.mean(r[k] for r in rr)
dslist=['ada002-1M','cap-1M','cohere-english-v3-1M']
rec_wins=qps_wins=build_wins=0
for ds in dslist:
 for b in (2,4):
  a,q=group(ds,b,'ASH',1),group(ds,b,'PQ',1)
  build_wins+=avg(a,'build_s')+avg(a,'provider_s')<avg(q,'build_s')+avg(q,'provider_s')
  for o in (1,2):
   a,q=group(ds,b,'ASH',o),group(ds,b,'PQ',o)
   assert len(a)==len(q)==2
   rec_wins+=avg(a,'recall')>avg(q,'recall')
   qps_wins+=avg(a,'qps')>avg(q,'qps')
text='# Production ASH versus latest-main PQ: Figure 9 graph study\n\n'
text+=f'Completed 24 fresh builds and 24 cached-index concurrency passes. ASH has higher mean recall at **{rec_wins}/12** matched query operating points, higher mean single-thread QPS at **{qps_wins}/12**, and faster graph construction plus scorer setup in **{build_wins}/6** configurations. These counts summarize the measured points; they do not imply a universal winner at matched recall.\n\n'
text+='The new symmetric scorer is used for encoded-node comparisons during construction. Production ASH queries remain asymmetric and include raw-query projection. Neither branch includes the separate duplicate-ID fix.\n\n'
text+='## Recommendation\n\nKeep the corrected symmetric scorer for construction and asymmetric scoring for raw queries. At these measured settings, ASH improves recall in every comparison and builds faster, but it has not universally beaten PQ on query throughput: Ada and CAP still have a throughput gap. Cohere wins both query metrics, while its original construction outlier remains unexplained. Prioritize measured score-kernel and ThreadLocal overhead before changing graph pruning policy; keep deduplication experiments separate.\n\n'
text+='## Plots\n\n![Single-thread graph QPS and recall](graph-qps-recall.png)\n\n![Construction phases](graph-construction.png)\n\n![Cached-index concurrency](graph-concurrency.png)\n\n'
text+='## Published IVF reference\n\n![Published Figure 9 panels, unchanged](published-ivf-reference.png)\n\nThe reference panels retain the paper’s original colors (red ASH, blue PQ); the new graph plots use blue ASH and red PQ. No IVF benchmark was rerun. These are architectural comparisons, with material differences: graph ASH uses C=1 and 20D training, while paper IVF uses C=32 and 10D; graph PQ uses JVector 8-bit ADC while paper PQ uses Faiss 4-bit FastScan; the graph runs use NVQ reranking. Increasing graph overquery also enlarges its search candidate set, so its gain is not solely a reranking ablation.\n\n'
text+='## Integration and validation\n\n[Branch audit](branch-audit.md) records the completed work integrated into ash-dev. The production baseline is unchanged from the original frozen ASH JAR; the byte-write fix was present throughout. Scalar and SIMD regression executions each passed 33 tests.\n\n'
for extra in ['cohere-investigation.md','timing-validation.md']:
 if (p/extra).exists():text+=(p/extra).read_text()+'\n\n'
text+=(p/'tables.md').read_text()+'\n\n'+(p/'methodology.md').read_text()
text+='\n## Reusable artifacts\n\n[Reported observations](reported-results.csv) explicitly identify longer cached-query retimings and preserve original QPS/CV columns. [Original observations](results.csv) include each build repetition, both query concurrencies, timing variation, recall, visited counts and phase times. Every retained index has its own configuration and compressor cache. The remote query-only helper is `/mnt/raid10/jvector-bench/ted_willke/figure9-graph/retest.py`. For example:\n\n```sh\npython3 /mnt/raid10/jvector-bench/ted_willke/figure9-graph/retest.py ada002-1M-b2-ash-r1 --threads 1 --overquery 1.0,2.0\n```\n\nAll 48 serial/concurrent operating-point checks compare complete ranked results and visited counts for every supplied query at each index/overquery setting. Separate reported recall and visited metrics agree exactly between serial and concurrent passes. The concurrency harness is committed as `638edc37`.\n'
(p/'report.md').write_text(text)
print(dict(recall_wins=rec_wins,qps_wins=qps_wins,build_wins=build_wins))
