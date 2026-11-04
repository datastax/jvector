#!/usr/bin/env python3
"""Orchestrate the checked-in standalone BenchYAML recipe; no query loop or performance measurement code."""
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
import collections
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import sys
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--build-cores", type=int, default=48)
parser.add_argument("--build-cpus", default="0-47")
parser.add_argument("--build-nodes", default="0,1")
parser.add_argument("--one-core-cpu", default="0")
parser.add_argument("--eight-core-cpus", default="0-7")
parser.add_argument("--query-node", default="0")
parser.add_argument("--build-repetitions", type=int, default=5)
parser.add_argument("--query-repetitions", type=int, default=3)
parser.add_argument("--dataset-set", choices=("1m", "large"), default="1m")
parser.add_argument("--plan-only", action="store_true")
args = parser.parse_args()
if min(args.build_cores, args.build_repetitions, args.query_repetitions) <= 0:
    parser.error("Core and repetition counts must be positive")
repo = Path(__file__).resolve().parent.parent
helper = repo / "jvector-examples/run-ash-pq-benchmark.sh"
source_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
jars = list((repo / "jvector-examples/target").glob("jvector-examples-*-jar-with-dependencies.jar"))
if len(jars) != 1:
    parser.error("Build exactly one standalone examples assembly before starting")
jar_sha = hashlib.sha256(jars[0].read_bytes()).hexdigest()
output = args.output.resolve()
data = args.data.resolve()
if not data.is_dir():
    parser.error("Dataset directory does not exist")
jobs = []
prefixes = ("openai3072", "mpnet", "cohere", "ada002") if args.dataset_set == "1m" else ("mpnet10m", "cohere10m", "cap6m")
for group, (prefix, bits) in enumerate((p, b) for p in prefixes for b in (2, 4)):
    for repetition in range(1, args.build_repetitions + 1):
        algorithms = ("pq", "ash") if (group + repetition) % 2 else ("ash", "pq")
        for algorithm in algorithms:
            jobs.append(dict(prefix=prefix, bits=bits, algorithm=algorithm, mode="construction",
                             cores=args.build_cores, cpus=args.build_cpus, repetition=repetition,
                             numa="interleave:" + args.build_nodes))
    for repetition in range(1, args.query_repetitions + 1):
        algorithms = ("pq", "ash") if (group + repetition) % 2 else ("ash", "pq")
        modes = ("direct", "queued") if repetition % 2 else ("queued", "direct")
        for cores, cpus in ((1, args.one_core_cpu), (8, args.eight_core_cpus)):
            for mode in modes:
                for algorithm in algorithms:
                    jobs.append(dict(prefix=prefix, bits=bits, algorithm=algorithm, mode=mode,
                                     cores=cores, cpus=cpus, repetition=repetition,
                                     numa="bind:" + args.query_node))
for job in jobs:
    job["id"] = "{prefix}-{bits}bit-{algorithm}-{mode}-c{cores}-r{repetition}".format(**job)
    case_repetition = job["repetition"] if job["mode"] == "construction" else 1
    job["case"] = str(output / "cases" / f"{job['prefix']}-{job['bits']}bit-{job['algorithm']}-r{case_repetition}")
    job["config"] = f"{job['prefix']}-{job['bits']}bit-{job['algorithm']}.yml"
watchdog_minutes = {"construction": 180 if args.dataset_set == "large" else 30, "query": 30}
plan = dict(source_ref=source_ref, jar_sha256=jar_sha, data=str(data), dataset_set=args.dataset_set, watchdog_minutes=watchdog_minutes, jobs=jobs,
            construction_jobs=sum(j["mode"] == "construction" for j in jobs),
            query_jobs=sum(j["mode"] != "construction" for j in jobs))
if args.plan_only:
    print(json.dumps(plan, indent=2))
    raise SystemExit(0)
output.mkdir(parents=True, exist_ok=True)


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


if (output / "plan.json").exists():
    if json.loads((output / "plan.json").read_text()) != plan:
        raise RuntimeError("Existing campaign has different source/artifact/settings; use a new output directory")
else:
    save(output / "plan.json", plan)
results_file = output / "results.json"
results = json.loads(results_file.read_text()) if results_file.exists() else []
complete_ids = {result["job"]["id"] for result in results}
process = None


def stop(signum, frame):
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGTERM)  # The checked-in shell forwards it to its JVM.
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    save(output / "status.json", dict(state="stopped", completed=len(results), planned=len(jobs)))
    raise SystemExit(128 + signum)


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)


def report_failure(kind, error, traceback):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    save(output / "status.json", dict(state="failed", error=str(error),
        current=globals().get("job"), completed=len(results), planned=len(jobs)))
    sys.__excepthook__(kind, error, traceback)


sys.excepthook = report_failure


def cpu_set(specification):
    result = set()
    for item in specification.split(","):
        limits = item.split("-")
        first, last = int(limits[0]), int(limits[-1])
        result.update(range(first, last + 1))
    return result


def wait_idle():
    """Read-only check between JVMs; never gates compilation inside a benchmark."""
    while True:
        conflicts = []
        for directory in Path("/proc").iterdir():
            if not directory.name.isdigit() or int(directory.name) == os.getpid():
                continue
            try:
                command = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if "/JetBrains/" not in command and re.search(
                        r"BenchYAML|DistancesASH|DistancesPQ|org\.junit\.runner|surefirebooter|(?:^|/)cc1plus(?: |$)", command):
                    conflicts.append(int(directory.name))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        if not conflicts:
            return
        save(output / "status.json", dict(state="waiting_for_other_work", pids=conflicts,
                                           completed=len(results), planned=len(jobs)))
        time.sleep(5)


def summarize():
    groups = collections.defaultdict(list)
    for result in results:
        job = result["job"]
        for row in result["rows"]:
            if job["mode"] == "construction" and float(row["overquery"]) != 1:
                continue  # Two overquery rows describe one construction invocation.
            key = (job["prefix"], job["bits"], job["algorithm"], job["mode"], job["cores"], row["overquery"])
            groups[key].append(row)
    records = []
    for key, rows in sorted(groups.items()):
        record = dict(zip(("dataset", "bits", "algorithm", "mode", "cores", "overquery"), key))
        record["jvms"] = len(rows)
        metrics = {}
        for name in rows[0]:
            if not name.startswith(("construction.", "search.")):
                continue
            if not all(row.get(name) for row in rows):
                continue
            values = [float(row[name]) for row in rows]
            if any(marker in name for marker in ("p999", "p95", "p99")):
                metrics[name] = dict(median_of_jvm_values=statistics.median(values),
                                     minimum=min(values), maximum=max(values))
            else:
                metrics[name] = dict(mean=statistics.mean(values),
                    sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                    minimum=min(values), maximum=max(values))
        record["metrics"] = metrics
        records.append(record)
    save(output / "summary.json", records)


for job in jobs:
    if job["id"] in complete_ids:
        continue
    wait_idle()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == source_ref, "Source changed during the campaign"
    save(output / "status.json", dict(state="running", current=job,
                                       completed=len(results), planned=len(jobs)))
    case = Path(job["case"])
    before = set((case / "runs").glob("*")) if (case / "runs").exists() else set()
    argv = ["bash", str(helper), job["config"], job["mode"], str(job["cores"]), job["cpus"], str(case)]
    env = os.environ.copy()
    env.update(JVECTOR_BENCH_DATA_DIR=str(data), JVECTOR_BENCH_NUMA_POLICY=job["numa"],
               JVECTOR_BENCH_CATALOG="catalog-large.yml" if args.dataset_set == "large" else "catalog-1m.yml")
    log_path = output / (job["id"] + ".log")
    print("START " + job["id"], flush=True)
    libraries = set()
    run_directory = None
    with log_path.open("w") as log:
        process = subprocess.Popen(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while process.poll() is None:
            limit_minutes = watchdog_minutes["construction" if job["mode"] == "construction" else "query"]
            if time.monotonic() - started > limit_minutes * 60:
                process.terminate()
                process.wait(timeout=15)
                raise RuntimeError(f"BenchYAML exceeded the {limit_minutes}-minute process watchdog; inspect {log_path}")
            if run_directory is None:
                created = set((case / "runs").glob("*")) - before
                if len(created) == 1:
                    run_directory = created.pop()
            if run_directory is not None and (run_directory / "jvm.pid").exists():
                pid = (run_directory / "jvm.pid").read_text().strip()
                try:
                    proc = Path("/proc") / pid
                    status = (proc / "status").read_text()
                    if "\nName:\tjava\n" in "\n" + status:
                        mask = next(line.split(":", 1)[1].strip() for line in status.splitlines()
                                    if line.startswith("Cpus_allowed_list:"))
                        assert cpu_set(mask) == cpu_set(job["cpus"]), (mask, job["cpus"])
                        libraries.update(line.split(maxsplit=5)[-1] for line in (proc / "maps").read_text().splitlines()
                                         if any(token in line for token in ("libjvector", "libdnnl", "libmkl", "libopenblas", "libnetlib")))
                        with (run_directory / "placement.jsonl").open("a") as placement:
                            placement.write(json.dumps(dict(cpu_mask=mask, libraries=sorted(libraries))) + "\n")
                except (FileNotFoundError, ProcessLookupError):
                    pass
            time.sleep(1)
    if process.returncode:
        save(output / "status.json", dict(state="failed", job=job, log=str(log_path), exit_code=process.returncode))
        raise RuntimeError("Standalone BenchYAML failed: " + str(log_path))
    created = set((case / "runs").glob("*")) - before
    assert len(created) == 1
    run_directory = created.pop()
    assert (run_directory / "source-ref.txt").read_text().strip() == source_ref
    assert (run_directory / "jar.sha256").read_text().split()[0] == jar_sha, "Artifact changed during the campaign"
    # A new invocation contributes exactly one CSV, identified by its run mode
    # and the source reference recorded by the normal reporting system.
    csvs = sorted((case / "logging").rglob("experiments.csv"), key=lambda path: path.stat().st_mtime_ns)
    assert csvs
    rows = list(csv.DictReader(csvs[-1].open()))
    assert len(rows) == 2 and {float(row["overquery"]) for row in rows} == {1, 2}
    assert all("FUSED_" + job["algorithm"].upper() in row["feature_set"] and "NVQ_VECTORS" in row["feature_set"] for row in rows)
    assert any("libjvector-" in library for library in libraries), "Native provider was not observed"
    if job["mode"] == "construction":
        assert all(row.get("construction.index_build_time_s") for row in rows)
    else:
        assert all(row.get("search.throughput.avg_qps") and row.get("search.latency.mean_ms") for row in rows)
        reference = next(result for result in results if result["job"]["prefix"] == job["prefix"]
                         and result["job"]["bits"] == job["bits"] and result["job"]["algorithm"] == job["algorithm"]
                         and result["job"]["mode"] == "construction" and result["job"]["repetition"] == 1)
        actual_recall = [float(row["search.accuracy.recall_at_10"]) for row in rows]
        reference_recall = [float(row["search.accuracy.recall_at_10"]) for row in reference["rows"]]
        if job["algorithm"] == "ash" and job["mode"] == "queued":
            # Batch GEMM may differ from singleton GEMV by FP32 reduction rounding.
            # Keep the actual values; an improvement is not a regression. Stop on
            # any decline for investigation rather than silently applying a tolerance.
            assert all(actual >= expected for actual, expected in zip(actual_recall, reference_recall)), "Execution-policy recall regression"
        else:
            assert actual_recall == reference_recall, "Execution-policy recall regression"
    result = dict(job=job, rows=rows, csv=str(csvs[-1]), artifacts=str(run_directory), libraries=sorted(libraries))
    results.append(result)
    complete_ids.add(job["id"])
    save(results_file, results)
    summarize()
    if job["mode"] == "construction" and job["repetition"] != 1:
        # Only these extra graphs have no future use. Keep first-build indexes,
        # raw CSVs, metadata and console logs; remove no unrelated workspace.
        assert case.parent == output / "cases"
        for name in ("index_cache", "compressor_cache"):
            shutil.rmtree(case / name)
    print("PASS " + job["id"], flush=True)
save(output / "complete.json", dict(source_ref=source_ref, jar_sha256=jar_sha,
    construction_jobs=plan["construction_jobs"], query_jobs=plan["query_jobs"], completed=len(results)))
save(output / "status.json", dict(state="complete", completed=len(results), planned=len(jobs)))
print("COMPLETE", flush=True)
