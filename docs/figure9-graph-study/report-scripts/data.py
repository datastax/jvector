# Copyright DataStax, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
