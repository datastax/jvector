from pathlib import Path
import csv,statistics
p=Path(__file__).resolve().parent.parent
from data import load_rows
rows=load_rows()
for r in rows:
 for k in ['bits','rep','threads','overquery','qps','recall','visited','build_s','train_s','encode_s','provider_s','nvq_s']:
  if k in r:r[k]=float(r[k]) if r[k] else 0.
def sel(ds,b,m,t,o):return [r for r in rows if r['dataset']==ds and r['bits']==b and r['mode']==m and r['threads']==t and r['overquery']==o]
def mean(rr,key):return statistics.mean(x[key] for x in rr)
dslist=['ada002-1M','cap-1M','cohere-english-v3-1M']
text='# Final graph results\n\nMeans of two independent builds per configuration. ASH/PQ values are shown in that order.\n\n'
text+='## Single-threaded queries\n\n| Dataset | ASH bits / PQ matching budget | Overquery | Recall@10 (%): ASH / PQ | QPS: ASH / PQ | ASH/PQ QPS | Recall difference (pp) |\n|---|---:|---:|---:|---:|---:|---:|\n'
for ds in dslist:
 for b in (2,4):
  for o in (1,2):
   aa,pp=sel(ds,b,'ASH',1,o),sel(ds,b,'PQ',1,o)
   if len(aa)!=2 or len(pp)!=2:continue
   ar,pr=mean(aa,'recall')*100,mean(pp,'recall')*100
   aq,pq=mean(aa,'qps'),mean(pp,'qps')
   text+=f'| {ds} | {b} | {o}× | {ar:.3f} / {pr:.3f} | {aq:,.0f} / {pq:,.0f} | {aq/pq:.2f}× | {ar-pr:+.3f} |\n'
text+='\n## Construction\n\nGraph column includes scorer-provider setup. Phase subtotal also includes compressor training, base encoding and NVQ compressor setup; it excludes dataset loading, JVM startup and uninstrumented writer initialization.\n\n| Dataset | Bits/budget | Graph + provider seconds: ASH / PQ | Training seconds: ASH / PQ | Encoding seconds: ASH / PQ | Measured phase subtotal seconds: ASH / PQ |\n|---|---:|---:|---:|---:|---:|\n'
for ds in dslist:
 for b in (2,4):
  rr=[sel(ds,b,m,1,1) for m in ['ASH','PQ']]
  if any(len(x)!=2 for x in rr):continue
  v=[]
  for x in rr:
   graph=mean(x,'build_s')+mean(x,'provider_s');train=mean(x,'train_s');enc=mean(x,'encode_s');total=graph+train+enc+mean(x,'nvq_s');v.append((graph,train,enc,total))
  vals=[' / '.join(f'{v[i][j]:.2f}' for i in (0,1)) for j in range(4)]
  text+=f'| {ds} | {b} | '+' | '.join(vals)+' |\n'
text+='\n## Same cached indexes, 48 query workers\n\n| Dataset | Bits/budget | Overquery | Concurrent QPS: ASH / PQ | Speedup over one worker: ASH / PQ |\n|---|---:|---:|---:|---:|\n'
for ds in dslist:
 for b in (2,4):
  for o in (1,2):
   aa,pp=sel(ds,b,'ASH',48,o),sel(ds,b,'PQ',48,o)
   if len(aa)!=2 or len(pp)!=2:continue
   aq,pq=mean(aa,'qps'),mean(pp,'qps');asq=mean(sel(ds,b,'ASH',1,o),'qps');psq=mean(sel(ds,b,'PQ',1,o),'qps')
   text+=f'| {ds} | {b} | {o}× | {aq:,.0f} / {pq:,.0f} | {aq/asq:.2f}× / {pq/psq:.2f}× |\n'
text+='\n## Between-build ranges\n\nRanges of two observations, not confidence intervals.\n\n| Dataset | Bits/budget | Method | Overquery | Single-thread QPS range | Recall@10 (%) range | Graph + setup seconds range |\n|---|---:|---|---:|---:|---:|---:|\n'
for ds in dslist:
 for b in (2,4):
  for m in ['ASH','PQ']:
   for o in (1,2):
    rr=sel(ds,b,m,1,o)
    if len(rr)!=2:continue
    q=[x['qps'] for x in rr];r=[x['recall']*100 for x in rr];g=[x['build_s']+x['provider_s'] for x in rr]
    text+=f'| {ds} | {b} | {m} | {o}× | {min(q):,.0f}–{max(q):,.0f} | {min(r):.3f}–{max(r):.3f} | {min(g):.2f}–{max(g):.2f} |\n'
(p/'tables.md').write_text(text)
print('Table rows available',len(rows))
