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
"""Converts the downloaded parquet files to raw float32: files 0000 .. N-1 into base.f32, the first 10,000 rows of
file 0100 into query.f32. usage: wikipedia_convert.py <dir> <N>"""
import os
import sys

import numpy as np
import pyarrow.parquet as pq

root, files = sys.argv[1], int(sys.argv[2])
DIM, QUERIES = 1024, 10_000


def rows(name):
    col = pq.read_table(os.path.join(root, 'parquet', name), columns=['emb']).column('emb').combine_chunks()
    a = np.asarray(col.values, dtype=np.float32).reshape(-1, DIM)
    assert a.shape[0] == len(col)
    return a


with open(os.path.join(root, 'base.f32'), 'wb') as out:
    for f in range(files):
        a = rows(f'{f:04d}.parquet')
        a.tofile(out)
        print(f'{f:04d}: {a.shape[0]:,} rows, {out.tell() // (4 * DIM):,} in total', flush=True)
q = rows('0100.parquet')[:QUERIES]
q.tofile(os.path.join(root, 'query.f32'))
print(f'queries: {q.shape[0]:,} rows', flush=True)
