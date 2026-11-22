#!/usr/bin/env python3
"""Run the ASH versus PQ study on this machine, one JVM per measurement.

For each dataset and algorithm, one JVM builds the index on every physical core and
saves it; then fresh JVMs query it with 1 and with 8 physical cores, one overquery per
JVM. Each JVM is launched by run-ash-pq-benchmark.sh. Results are BenchYAML's own
output: its CSV under each case directory's logging/ folder and the tables it prints.
See docs/ash-pq-benchmark.md.
"""
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
from pathlib import Path
import argparse
import os
import re
import subprocess
import sys

DATASETS = ("ada002", "cohere", "dprgemma", "openai1536", "openai3072")
REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "jvector-examples/yaml-configs/index-parameters/studies/ash-vs-pq"
LAUNCHER = REPO / "jvector-examples/run-ash-pq-benchmark.sh"


def cpu_topology():
    """Online CPUs as (cpu, physical core key, NUMA node), in CPU order."""
    rows = []
    for line in subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE,ONLINE"], text=True).splitlines():
        if line.startswith("#"):
            continue
        cpu, core, socket, node, online = (field or "0" for field in line.split(","))
        if online != "N":
            rows.append((int(cpu), (socket, core), int(node)))
    return rows


def cpu_list(cpus):
    """Format CPU numbers as a numactl list, for example 0-7,48."""
    ranges, cpus = [], sorted(cpus)
    start = end = cpus[0]
    for cpu in cpus[1:] + [None]:
        if cpu is not None and cpu == end + 1:
            end = cpu
            continue
        ranges.append(str(start) if start == end else f"{start}-{end}")
        if cpu is not None:
            start = end = cpu
    return ",".join(ranges)


def query_execution(config):
    """The search.queryExecution the index YAML selects (direct when absent)."""
    match = re.search(r"^\s+queryExecution:\s*(direct|queued)\b", config.read_text(), re.M)
    return match.group(1) if match else "direct"


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--output", type=Path, required=True,
                    help="directory for the case directories (fast local storage)")
parser.add_argument("--data", type=Path, required=True,
                    help="directory where datasets are downloaded once and reused")
parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
parser.add_argument("--algorithms", nargs="+", choices=("ash", "pq"), default=["ash", "pq"])
parser.add_argument("--build-smt", nargs="*", choices=("ash", "pq"), default=["ash"],
                    help="algorithms built on both hardware threads of each core "
                         "(default: ash; each algorithm is built at its fastest setting)")
parser.add_argument("--no-ash-adaptive", action="store_true",
                    help="search ASH without adaptive stopping (on by default)")
parser.add_argument("--overqueries", nargs="+", default=["1.0", "2.0"],
                    help="overquery values; each is measured in its own JVM")
parser.add_argument("--query-cores", nargs="+", type=int, default=[1, 8],
                    help="physical-core budgets for the query JVMs")
parser.add_argument("--repetitions", type=int, default=1,
                    help="query JVMs per measurement (construction runs once per case)")
parser.add_argument("--dry-run", action="store_true", help="print the JVM launches without running them")
args = parser.parse_args()
if args.repetitions < 1 or min(args.query_cores) < 1:
    parser.error("Repetitions and core budgets must be positive")

# Construction: every physical core (and its sibling thread with SMT builds), memory
# interleaved across all NUMA nodes. Queries: the first physical cores of the node that
# holds CPU 0, with memory bound to that node.
topology = cpu_topology()
first_threads = {}
for cpu, core, node in topology:
    first_threads.setdefault(core, (cpu, node))
build_cores = len(first_threads)
has_smt = len(topology) == 2 * build_cores
if args.build_smt and not has_smt:
    print("This machine does not have two hardware threads on every core; building without SMT.", file=sys.stderr)
nodes = sorted({node for _, _, node in topology})
query_node = first_threads[next(iter(first_threads))][1]
query_cpus = [cpu for cpu, node in first_threads.values() if node == query_node]
if max(args.query_cores) > len(query_cpus):
    parser.error(f"NUMA node {query_node} has only {len(query_cpus)} physical cores")
output = args.output.resolve()
data = args.data.resolve()


def run(config, mode, cores, cpus, case, smt=False, adaptive=False, overquery=None):
    environment = dict(os.environ, JVECTOR_BENCH_DATA_DIR=str(data),
                       JVECTOR_BENCH_BUILD_SMT="1" if smt else "0",
                       JVECTOR_BENCH_ADAPTIVE="1" if adaptive else "0",
                       JVECTOR_BENCH_NUMA_POLICY=("interleave:" + ",".join(map(str, nodes)))
                       if mode == "construction" else f"bind:{query_node}")
    environment.pop("JVECTOR_BENCH_OVERQUERY", None)
    if overquery is not None:
        environment["JVECTOR_BENCH_OVERQUERY"] = overquery
    command = ["bash", str(LAUNCHER), config.name, mode, str(cores), cpus, str(case)]
    print(f"\n=== {config.stem} {mode} cores={cores}"
          + (f" overquery={overquery}" if overquery else "") + f" ({case})", flush=True)
    if args.dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, env=environment, check=True)


for dataset in args.datasets:
    for algorithm in args.algorithms:
        config = CONFIGS / f"{dataset}-2bit-{algorithm}.yml"
        case = output / config.stem
        smt = algorithm in args.build_smt and has_smt
        adaptive = algorithm == "ash" and not args.no_ash_adaptive
        # A case directory that already holds a saved index was constructed by an earlier
        # run; reuse it, so an interrupted study can be restarted with the same command.
        if (case / "index_cache").is_dir() and any((case / "index_cache").iterdir()):
            print(f"\n=== {config.stem} construction: reusing the index in {case}")
        else:
            cpus = [cpu for cpu, _, _ in topology] if smt else [cpu for cpu, _ in first_threads.values()]
            run(config, "construction", build_cores, cpu_list(cpus), case, smt=smt)
        mode = query_execution(config)
        for overquery in args.overqueries:
            for _ in range(args.repetitions):
                for cores in args.query_cores:
                    run(config, mode, cores, cpu_list(query_cpus[:cores]), case,
                        adaptive=adaptive, overquery=overquery)
print(f"\nDone. BenchYAML's CSV files are under {output}/*/logging/.")
