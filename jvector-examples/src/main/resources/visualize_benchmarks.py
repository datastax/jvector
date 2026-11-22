#!/usr/bin/env python3
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

import argparse
import csv
import os
import re
import subprocess
import textwrap
import urllib.request
import platform
import psutil
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# Define metrics where higher values are better and lower values are better

HIGHER_IS_BETTER = ["QPS", "Recall@10"]
LOWER_IS_BETTER = ["Mean Latency", "Index Build Time", "Average Nodes Visited",
                   "Build Heap Used (MB)", "Build Off-Heap (MB)",
                   "Search Heap Used (MB)", "Search Off-Heap (MB)"]

# Walk up from this script's location to find the repo root (contains .github/)
def _find_repo_root(start: str) -> str:
    d = start
    while True:
        if os.path.isdir(os.path.join(d, '.github')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return start  # fallback: couldn't find repo root
        d = parent

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _find_repo_root(_SCRIPT_DIR)
_YAML_CONFIG_DIR = os.path.join(_REPO_ROOT, 'jvector-examples', 'yaml-configs', 'index-parameters')
_WORKFLOW_PATH = os.path.join(_REPO_ROOT, '.github', 'workflows', 'run-bench.yml')


# ---------------------------------------------------------------------------
# Footer helpers
# ---------------------------------------------------------------------------

def get_instance_type() -> str:
    """
    Query the cloud Instance Metadata Service (IMDS) for the instance type.
    Tries AWS IMDSv2 first, then GCP.  Returns 'Unknown (not on cloud)' when
    neither endpoint is reachable within 1 second.
    """
    # AWS IMDSv2: obtain a short-lived token, then fetch instance-type
    try:
        token_req = urllib.request.Request(
            'http://169.254.169.254/latest/api/token',
            method='PUT',
            headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'},
        )
        token = urllib.request.urlopen(token_req, timeout=1).read().decode()
        inst_req = urllib.request.Request(
            'http://169.254.169.254/latest/meta-data/instance-type',
            headers={'X-aws-ec2-metadata-token': token},
        )
        return urllib.request.urlopen(inst_req, timeout=1).read().decode()
    except Exception:
        pass

    # GCP IMDS
    try:
        req = urllib.request.Request(
            'http://169.254.169.254/computeMetadata/v1/instance/machine-type',
            headers={'Metadata-Flavor': 'Google'},
        )
        machine_type = urllib.request.urlopen(req, timeout=1).read().decode()
        # GCP returns "projects/12345/machineTypes/n1-standard-4" — take the last segment
        return machine_type.split('/')[-1]
    except Exception:
        pass

    return 'Unknown (not on cloud)'


def get_jdk_version() -> Tuple[str, int]:
    """
    Return (short version string, major version number) for the active JDK.
    Uses 'java -version'; falls back to ('Unknown', 0) on any error.
    """
    try:
        result = subprocess.run(
            ['java', '-version'],
            capture_output=True,
            text=True,
        )
        # java -version writes to stderr
        first_line = (result.stderr or result.stdout).split('\n')[0].strip()
        # Extract the quoted version string, e.g. "24.0.2"
        match = re.search(r'"(\d+)(?:[._](\d+))?', first_line)
        if match:
            major = int(match.group(1))
            # Produce a compact label like "OpenJDK 24.0.2"
            label_match = re.match(r'(\S+)\s+version\s+"([^"]+)"', first_line, re.IGNORECASE)
            short = f"{label_match.group(1)} {label_match.group(2)}" if label_match else first_line
            return short, major
        return first_line, 0
    except Exception:
        return 'Unknown', 0


def _half_mem_gb() -> int:
    """
    Replicate the workflow's HALF_MEM_GB calculation:
        TOTAL_MEM_GB=$(free -g | awk '/^Mem:/ {print $2}')   # floor to integer GiB
        HALF_MEM_GB=$((TOTAL_MEM_GB / 2))                    # integer division
        if [[ "$HALF_MEM_GB" -lt 1 ]]; then HALF_MEM_GB=1   # minimum 1
    """
    total_gib = int(psutil.virtual_memory().total / (1024 ** 3))
    if total_gib <= 0:
        total_gib = 16  # matches the workflow's fallback default
    return max(1, total_gib // 2)


def get_jvm_flags(jdk_major: int, workflow_path: str = _WORKFLOW_PATH) -> str:
    """
    Parse the non-pull-request Java invocation inside the 'Run benchmark' step
    of run-bench.yml and return the flags that apply for *jdk_major*.

    Locates the step by name so the function is resilient to line-number changes.
    Within that step, it finds the java invocation that is NOT under a
    pull_request conditional and collects its continuation lines.

    Handles GitHub Actions conditional expressions of the form:
        ${{ matrix.jdk >= N && 'flag1 flag2' || '' }}
    and also extracts unconditional JVM flags (-XX:, -D, -X prefixes).
    ${HALF_MEM_GB} is resolved to the actual value computed from system memory,
    matching the workflow's calculation exactly.
    """
    flags: List[str] = []
    try:
        with open(workflow_path, 'r') as fh:
            lines = fh.readlines()

        # Find the start of the 'Run benchmark' step
        step_start = None
        for i, line in enumerate(lines):
            if re.search(r'^\s*-\s*name:\s*Run benchmark\s*$', line, re.IGNORECASE):
                step_start = i
                break
        if step_start is None:
            return 'standard JVM defaults'

        # Find the end of the step (next top-level list item at the same indent)
        step_indent = len(lines[step_start]) - len(lines[step_start].lstrip())
        step_end = len(lines)
        for i in range(step_start + 1, len(lines)):
            stripped = lines[i].lstrip()
            if stripped.startswith('- ') and (len(lines[i]) - len(stripped)) <= step_indent:
                step_end = i
                break

        step_lines = lines[step_start:step_end]

        # Find the non-pull-request java invocation: skip the if/pull_request block,
        # then locate the first 'java ' line after the else.
        in_pr_block = False
        java_start = None
        for i, line in enumerate(step_lines):
            if re.search(r'github\.event_name.*pull_request', line):
                in_pr_block = True
            if in_pr_block and re.search(r'^\s*else\b', line):
                in_pr_block = False
                continue
            if not in_pr_block and re.match(r'\s+java\s', line):
                java_start = i
                break

        if java_start is None:
            return 'standard JVM defaults'

        # Collect the java command and its continuation lines (ending with \)
        java_lines = []
        for line in step_lines[java_start:]:
            java_lines.append(line)
            if not line.rstrip().endswith('\\'):
                break

        for line in java_lines:
            # Evaluate conditional flag blocks
            for m in re.finditer(
                r'\$\{\{[^}]*matrix\.jdk\s*>=\s*(\d+)\s*&&\s*\'([^\']+)\'', line
            ):
                if jdk_major >= int(m.group(1)):
                    flags.append(m.group(2).strip())

            # Remove all GitHub Actions expressions, then shell variable refs.
            # HALF_MEM_GB is resolved to the actual value before other variables
            # are blanked so that -Xmx renders as e.g. -Xmx32g rather than -Xmxg.
            clean = re.sub(r'\$\{\{[^}]*\}\}', '', line)
            clean = clean.replace('${HALF_MEM_GB}', str(_half_mem_gb()))
            clean = re.sub(r'\$\{[^}]+\}', '', clean)

            # Extract static JVM flags (-XX:, -D, -X prefixes)
            for flag in re.findall(r'(?<![/\w])(-(?:XX:[+\-]?|D|X[a-z]+)[^\s\\]+)', clean):
                if flag not in flags:
                    flags.append(flag)

    except Exception:
        pass

    return ' '.join(flags) if flags else 'standard JVM defaults'


def get_dataset_config_map(
    datasets: set,
    config_dir: str = _YAML_CONFIG_DIR,
) -> Dict[str, str]:
    """
    Map each dataset name to its YAML configuration filename.
    Uses '<dataset>.yml' when that file exists, otherwise 'default.yml'.
    """
    mapping: Dict[str, str] = {}
    for dataset in sorted(datasets):
        candidate = os.path.join(config_dir, f'{dataset}.yml')
        mapping[dataset] = f'{dataset}.yml' if os.path.exists(candidate) else 'default.yml'
    return mapping


def build_footer_text(
    instance_type: str,
    jdk_version: str,
    jvm_flags: str,
    dataset_config_map: Dict[str, str],
    max_line_width: int = 120,
) -> str:
    """
    Produce the two-line footer string to embed in each chart.

    Line 1 — environment: instance type, JDK, JVM flags
    Line 2 — dataset configs grouped by config file for compactness
    """
    line1 = f"Instance: {instance_type}  |  JDK: {jdk_version}  |  Flags: {jvm_flags}"

    # Group datasets by config file: default.yml (ds1, ds2) | cap-1M.yml (cap-1M)
    config_to_datasets: Dict[str, List[str]] = defaultdict(list)
    for dataset, cfg in dataset_config_map.items():
        config_to_datasets[cfg].append(dataset)
    parts = [
        f"{cfg} ({', '.join(sorted(ds))})"
        for cfg, ds in sorted(config_to_datasets.items())
    ]
    line2 = "Dataset configs: " + " | ".join(parts)

    # Wrap long lines so they don't overflow the figure
    line1 = textwrap.fill(line1, width=max_line_width)
    line2 = textwrap.fill(line2, width=max_line_width)

    return f"{line1}\n{line2}"


def _add_footer(footer_text: str, bottom_margin: float = 0.10) -> None:
    """
    Attach *footer_text* to the current figure as a bottom annotation and
    adjust the layout so the plot area does not overlap the text.
    """
    plt.tight_layout(rect=[0, bottom_margin, 1, 1])
    plt.figtext(
        0.5, 0.01,
        footer_text,
        ha='center', va='bottom',
        fontsize=7,
        bbox={'facecolor': 'white', 'alpha': 0.85, 'pad': 4, 'edgecolor': 'lightgrey'},
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class BenchmarkData:

    def __init__(self):
        # Structure: {release: {dataset: {metric: value}}}
        self.data = defaultdict(lambda: defaultdict(dict))
        self.releases = []  # maintains insertion order
        self.datasets = set()
        self.metrics = set()

    def add_file(self, file_path: str, release: str):
        """Add benchmark data from a CSV file for a specific release."""
        if release not in self.releases:
            self.releases.append(release)

        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dataset = row['dataset']
                    self.datasets.add(dataset)

                    for metric, value in row.items():
                        if metric != 'dataset':
                            try:
                                value = float(value)
                                self.metrics.add(metric)
                                self.data[release][dataset][metric] = value
                            except ValueError:
                                pass
        except Exception as e:
            print(f"Error loading benchmark results from {file_path}: {e}")
            return False

        return True

    def get_metric_data_for_dataset(self, metric: str, dataset: str) -> Tuple[List[str], List[float]]:
        """Get the values of a specific metric for a dataset across all releases."""
        releases = []
        values = []

        for release in self.releases:
            if dataset in self.data[release] and metric in self.data[release][dataset]:
                releases.append(release)
                values.append(self.data[release][dataset][metric])

        return releases, values

    def detect_changes(self, threshold_percent: float = 5.0) -> List[Dict[str, Any]]:
        """Detect significant changes in metrics between consecutive releases."""
        changes = []

        for dataset in self.datasets:
            for metric in self.metrics:
                releases, values = self.get_metric_data_for_dataset(metric, dataset)

                for i in range(1, len(releases)):
                    current_value = values[i]
                    previous_value = values[i - 1]
                    current_release = releases[i]
                    previous_release = releases[i - 1]

                    if previous_value == 0:
                        continue

                    change_pct = ((current_value - previous_value) / previous_value) * 100

                    if metric in HIGHER_IS_BETTER:
                        status = "improvement" if change_pct > 0 else "regression"
                    elif metric in LOWER_IS_BETTER:
                        status = "improvement" if change_pct < 0 else "regression"
                    else:
                        status = "unknown"

                    if abs(change_pct) >= threshold_percent:
                        changes.append({
                            "dataset": dataset,
                            "metric": metric,
                            "current_release": current_release,
                            "previous_release": previous_release,
                            "current_value": current_value,
                            "previous_value": previous_value,
                            "change_pct": change_pct,
                            "status": status,
                        })

        return changes


# ---------------------------------------------------------------------------
# Plot / report generation
# ---------------------------------------------------------------------------

def extract_release_from_filename(filename: str) -> str:
    """Extract a release label from a CSV filename."""
    match = re.search(r'(?:v|release[-_])?(\d+\.\d+(?:\.\d+)?)', filename)
    if match:
        return match.group(1)
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]


def generate_plots(benchmark_data: BenchmarkData, output_dir: str):
    """Generate per-metric plots and a combined comparison chart, each with a footer."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Collect footer information once ---
    instance_type = get_instance_type()
    jdk_version, jdk_major = get_jdk_version()
    jvm_flags = get_jvm_flags(jdk_major)
    dataset_config_map = get_dataset_config_map(benchmark_data.datasets)
    footer_text = build_footer_text(instance_type, jdk_version, jvm_flags, dataset_config_map)

    # --- Per-metric charts ---
    for metric in benchmark_data.metrics:
        plt.figure(figsize=(10, 6))

        for dataset in benchmark_data.datasets:
            releases, values = benchmark_data.get_metric_data_for_dataset(metric, dataset)
            if releases and values:
                if metric == "QPS":
                    std_releases, std_values = benchmark_data.get_metric_data_for_dataset("QPS StdDev", dataset)
                    if std_releases and std_values:
                        std_map = {r: v for r, v in zip(std_releases, std_values)}
                        yerr = [std_map.get(r, 0.0) for r in releases]
                        plt.errorbar(releases, values, yerr=yerr, marker='o', capsize=4, label=dataset)
                    else:
                        plt.plot(releases, values, marker='o', label=dataset)
                else:
                    plt.plot(releases, values, marker='o', label=dataset)

        plt.title(f"{metric} Over Time")
        plt.xlabel("Release")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        _add_footer(footer_text)

        safe_metric_name = metric.replace('@', '_at_').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f"{safe_metric_name}.png"))
        plt.close()

    # --- Combined aggregated comparison chart ---
    plt.figure(figsize=(12, 8))

    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
    memory_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    processor = platform.processor() or platform.machine()
    system_info = f"CPU: {processor}, Cores: {cpu_count}, Memory: {memory_gb} GB"

    metrics_list = list(benchmark_data.metrics)
    for idx, metric in enumerate(metrics_list):
        plt.subplot(len(metrics_list), 1, idx + 1)

        for dataset in benchmark_data.datasets:
            releases, values = benchmark_data.get_metric_data_for_dataset(metric, dataset)
            if releases and values and values[0] != 0:
                aggregated = [v / values[0] for v in values]
                plt.plot(releases, aggregated, marker='o', label=dataset)

        plt.title(f"Aggregated {metric}")
        plt.grid(True, linestyle='--', alpha=0.7)
        if idx == len(metrics_list) - 1:
            plt.xlabel("Release")
        plt.ylabel("Relative Change")
        plt.xticks(rotation=45)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Footer for the combined chart includes both system info and the benchmark footer
    combined_footer = f"{system_info}\n{footer_text}"
    _add_footer(combined_footer, bottom_margin=0.12)

    plt.savefig(os.path.join(output_dir, "aggregated_comparison.png"))
    plt.close()


def generate_report(benchmark_data: BenchmarkData, changes: List[Dict[str, Any]], output_file: str):
    """Generate a markdown report summarizing the benchmark results and changes."""
    with open(output_file, 'w') as f:
        f.write("# Benchmark Performance Report\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Total releases analyzed: {len(benchmark_data.releases)}\n")
        f.write(f"- Datasets: {', '.join(benchmark_data.datasets)}\n")
        f.write(f"- Metrics: {', '.join(benchmark_data.metrics)}\n\n")

        improvements = [c for c in changes if c['status'] == 'improvement']
        regressions = [c for c in changes if c['status'] == 'regression']

        f.write("## Significant Changes\n\n")
        f.write(f"- Total improvements: {len(improvements)}\n")
        f.write(f"- Total regressions: {len(regressions)}\n\n")

        if improvements:
            f.write("### Improvements\n\n")
            f.write("| Dataset | Metric | Releases | Previous | Current | Change |\n")
            f.write("|---------|--------|----------|----------|---------|-------|\n")
            for change in improvements:
                f.write(
                    f"| {change['dataset']} | {change['metric']} | "
                    f"{change['previous_release']} → {change['current_release']} | "
                    f"{change['previous_value']:.4f} | {change['current_value']:.4f} | "
                    f"{change['change_pct']:+.2f}% |\n"
                )
            f.write("\n")

        if regressions:
            f.write("### Regressions\n\n")
            f.write("| Dataset | Metric | Releases | Previous | Current | Change |\n")
            f.write("|---------|--------|----------|----------|---------|-------|\n")
            for change in regressions:
                f.write(
                    f"| {change['dataset']} | {change['metric']} | "
                    f"{change['previous_release']} → {change['current_release']} | "
                    f"{change['previous_value']:.4f} | {change['current_value']:.4f} | "
                    f"{change['change_pct']:+.2f}% |\n"
                )
            f.write("\n")

        f.write("## Latest Results\n\n")
        latest_release = benchmark_data.releases[-1] if benchmark_data.releases else "N/A"
        f.write(f"### Release: {latest_release}\n\n")
        f.write("| Dataset | QPS | Mean Latency | Recall@10 |\n")
        f.write("|---------|-----|-------------|----------|\n")
        for dataset in benchmark_data.datasets:
            if latest_release in benchmark_data.data and dataset in benchmark_data.data[latest_release]:
                qps = benchmark_data.data[latest_release][dataset].get("QPS", "N/A")
                latency = benchmark_data.data[latest_release][dataset].get("Mean Latency", "N/A")
                recall = benchmark_data.data[latest_release][dataset].get("Recall@10", "N/A")
                if isinstance(qps, float):
                    qps = f"{qps:.2f}"
                if isinstance(latency, float):
                    latency = f"{latency:.4f}"
                if isinstance(recall, float):
                    recall = f"{recall:.4f}"
                f.write(f"| {dataset} | {qps} | {latency} | {recall} |\n")
        f.write("\n")

        f.write("## Performance Graphs\n\n")
        for metric in benchmark_data.metrics:
            safe_metric_name = metric.replace('@', '_at_').replace(' ', '_')
            f.write(f"![{metric}]({safe_metric_name}.png)\n\n")
        f.write("![Aggregated Comparison](aggregated_comparison.png)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare and visualize benchmark results across multiple releases"
    )
    parser.add_argument("files", nargs="+", help="Paths to CSV benchmark result files")
    parser.add_argument("--output-dir", default="benchmark_reports",
                        help="Directory to save plots and reports")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Threshold percentage for detecting significant changes (default: 5.0)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    benchmark_data = BenchmarkData()
    for file_path in args.files:
        release = extract_release_from_filename(file_path)
        print(f"Loading {file_path} as release {release}...")
        benchmark_data.add_file(file_path, release)

    changes = benchmark_data.detect_changes(args.threshold)

    generate_plots(benchmark_data, args.output_dir)

    report_path = os.path.join(args.output_dir, "benchmark_report.md")
    generate_report(benchmark_data, changes, report_path)

    print(f"Benchmark analysis complete. Report saved to {report_path}")
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
