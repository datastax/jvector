"""Preserve primary observations and explicitly overlay predeclared longer retimings."""
from pathlib import Path
import csv

def load_rows():
 p=Path(__file__).resolve().parent.parent
 rows=list(csv.DictReader((p/'results.csv').open()))
 for x in rows:x['throughput_source']='primary-3s';x['original_qps']=x['qps'];x['original_qps_cv']=x['qps_cv']
 if (p/'quality-results.csv').exists():
  for q in csv.DictReader((p/'quality-results.csv').open()):
   x=next(x for x in rows if x['name']==q['name'] and x['threads']==q['threads'] and float(x['overquery'])==float(q['overquery']))
   assert float(x['recall'])==float(q['recall']) and float(x['visited'])==float(q['visited'])
   x.update(qps=q['qps'],qps_cv=q['qps_cv'],throughput_source='cached-10s: '+q['log'])
 if len(rows)==96:
  with (p/'reported-results.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return rows
