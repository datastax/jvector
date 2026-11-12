#!/usr/bin/env python3
#
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
#
"""Exact top-100 inner-product neighbours of every query, written as .ivecs (JVector's ground-truth format).
Streams the raw float32 base file in chunks, so it needs only a chunk in memory.
usage: ground_truth.py base.f32 query.f32 dim out.ivecs [k=100] [rows=all]"""
import sys, time
import numpy as np

base_path, query_path, dim, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
k = int(sys.argv[5]) if len(sys.argv) > 5 else 100
base = np.memmap(base_path, dtype=np.float32, mode='r').reshape(-1, dim)
if len(sys.argv) > 6:
    base = base[:int(sys.argv[6])]
queries = np.fromfile(query_path, dtype=np.float32).reshape(-1, dim)
nq, n = len(queries), len(base)
best_s = np.full((nq, k), -np.inf, dtype=np.float32)
best_i = np.zeros((nq, k), dtype=np.int64)
CHUNK, QB = 200_000, 2_000
t0 = time.time()
for start in range(0, n, CHUNK):
    chunk = np.ascontiguousarray(base[start:start + CHUNK])
    for q0 in range(0, nq, QB):
        s = queries[q0:q0 + QB] @ chunk.T                       # (QB, chunk)
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        cand_s = np.concatenate([best_s[q0:q0 + QB], np.take_along_axis(s, top, 1)], axis=1)
        cand_i = np.concatenate([best_i[q0:q0 + QB], top + start], axis=1)
        keep = np.argpartition(-cand_s, k - 1, axis=1)[:, :k]
        best_s[q0:q0 + QB] = np.take_along_axis(cand_s, keep, 1)
        best_i[q0:q0 + QB] = np.take_along_axis(cand_i, keep, 1)
    print(f'{min(start + CHUNK, n)} / {n} rows, {time.time() - t0:.0f} s', flush=True)
order = np.argsort(-best_s, axis=1)
gt = np.take_along_axis(best_i, order, 1).astype(np.int32)
with open(out_path, 'wb') as f:
    for row in gt:
        np.array([k], dtype=np.int32).tofile(f)
        row.tofile(f)
print('wrote', out_path, gt.shape)
