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
import sys
import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# Define metrics where higher values are better and lower values are better

HIGHER_IS_BETTER = ["QPS", "Recall@10"]
LOWER_IS_BETTER = ["Mean Latency"]


class BenchmarkData:

    def __init__(self):
        # Structure: {release: {dataset: {metric: value}}}
        self.data = defaultdict(lambda: defaultdict(dict))
        self.releases = []  # This will maintain the order of releases as they are added
        self.datasets = set()
        self.metrics = set()

    def add_file(self, file_path: str, release: str):
        """
        Add benchmark data from a CSV file for a specific release
        """
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
                                # Convert to float if possible
                                value = float(value)
                                self.metrics.add(metric)
                                self.data[release][dataset][metric] = value
                            except ValueError:
                                # Skip non-numeric values
                                pass
        except Exception as e:
            print(f"Error loading benchmark results from {file_path}: {e}")
            return False

        return True

    def get_metric_data_for_dataset(self, metric: str, dataset: str) -> Tuple[List[str], List[float]]:
        """
        Get the values of a specific metric for a dataset across all releases
        """
        releases = []
        values = []

        for release in self.releases:
            if dataset in self.data[release] and metric in self.data[release][dataset]:
                releases.append(release)
                values.append(self.data[release][dataset][metric])

        return releases, values

    def detect_changes(self, threshold_percent: float = 5.0) -> List[Dict[str, Any]]:
        """
        Detect significant changes in metrics between consecutive releases
        """
        changes = []

        for dataset in self.datasets:
            for metric in self.metrics:
                releases, values = self.get_metric_data_for_dataset(metric, dataset)

                for i in range(1, len(releases)):
                    current_value = values[i]
                    previous_value = values[i-1]
                    current_release = releases[i]
                    previous_release = releases[i-1]

                    if previous_value == 0:
                        continue  # Avoid division by zero

                    change_pct = ((current_value - previous_value) / previous_value) * 100

                    # Determine if this is an improvement or regression
                    if metric in HIGHER_IS_BETTER:
                        status = "improvement" if change_pct > 0 else "regression"
                    elif metric in LOWER_IS_BETTER:
                        status = "improvement" if change_pct < 0 else "regression"
                    else:
                        status = "unknown"

                    # Only include significant changes
                    if abs(change_pct) >= threshold_percent:
                        changes.append({
                            "dataset": dataset,
                            "metric": metric,
                            "current_release": current_release,
                            "previous_release": previous_release,
                            "current_value": current_value,
                            "previous_value": previous_value,
                            "change_pct": change_pct,
                            "status": status
                        })

        return changes


def extract_release_from_filename(filename: str) -> str:
    """
    Extract release number from filename
    """
    # Try to find a version pattern like v1.2.3 or 1.2.3
    match = re.search(r'(?:v|release[-_])?(\d+\.\d+(?:\.\d+)?)', filename)
    if match:
        return match.group(1)

    # If no version pattern, use the filename without extension
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]


def generate_plots(benchmark_data: BenchmarkData, output_dir: str):
    """
    Generate plots for each metric, with one line per dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a plot for each metric
    for metric in benchmark_data.metrics:
        plt.figure(figsize=(10, 6))

        # Plot one line per dataset
        for dataset in benchmark_data.datasets:
            releases, values = benchmark_data.get_metric_data_for_dataset(metric, dataset)
            if releases and values:
                plt.plot(releases, values, marker='o', label=dataset)

        plt.title(f"{metric} Over Time")
        plt.xlabel("Release")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Save the plot
        safe_metric_name = metric.replace('@', '_at_').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f"{safe_metric_name}.png"))
        plt.close()

    # Create a combined plot with normalized values for comparison
    plt.figure(figsize=(12, 8))

    for metric in benchmark_data.metrics:
        plt.subplot(len(benchmark_data.metrics), 1, list(benchmark_data.metrics).index(metric) + 1)

        for dataset in benchmark_data.datasets:
            releases, values = benchmark_data.get_metric_data_for_dataset(metric, dataset)
            if releases and values:
                # Normalize values to the first release
                if values[0] != 0:
                    normalized_values = [v / values[0] for v in values]
                    plt.plot(releases, normalized_values, marker='o', label=f"{dataset}")

        plt.title(f"Normalized {metric}")
        plt.grid(True, linestyle='--', alpha=0.7)
        if metric == list(benchmark_data.metrics)[-1]:
            plt.xlabel("Release")
        plt.ylabel("Relative Change")
        plt.xticks(rotation=45)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "normalized_comparison.png"))
    plt.close()


def generate_report(benchmark_data: BenchmarkData, changes: List[Dict[str, Any]], output_file: str):
    """
    Generate a markdown report summarizing the benchmark results and changes
    """
    with open(output_file, 'w') as f:
        f.write("# Benchmark Performance Report\n\n")

        # Summary of datasets and metrics
        f.write("## Summary\n\n")
        f.write(f"- Total releases analyzed: {len(benchmark_data.releases)}\n")
        f.write(f"- Datasets: {', '.join(benchmark_data.datasets)}\n")
        f.write(f"- Metrics: {', '.join(benchmark_data.metrics)}\n\n")

        # Significant changes
        improvements = [c for c in changes if c['status'] == 'improvement']
        regressions = [c for c in changes if c['status'] == 'regression']

        f.write("## Significant Changes\n\n")
        f.write(f"- Total improvements: {len(improvements)}\n")
        f.write(f"- Total regressions: {len(regressions)}\n\n")

        # List improvements
        if improvements:
            f.write("### Improvements\n\n")
            f.write("| Dataset | Metric | Releases | Previous | Current | Change |\n")
            f.write("|---------|--------|----------|----------|---------|-------|\n")

            for change in improvements:
                f.write(f"| {change['dataset']} | {change['metric']} | " +
                        f"{change['previous_release']} → {change['current_release']} | " +
                        f"{change['previous_value']:.4f} | {change['current_value']:.4f} | " +
                        f"{change['change_pct']:+.2f}% |\n")

            f.write("\n")

        # List regressions
        if regressions:
            f.write("### Regressions\n\n")
            f.write("| Dataset | Metric | Releases | Previous | Current | Change |\n")
            f.write("|---------|--------|----------|----------|---------|-------|\n")

            for change in regressions:
                f.write(f"| {change['dataset']} | {change['metric']} | " +
                        f"{change['previous_release']} → {change['current_release']} | " +
                        f"{change['previous_value']:.4f} | {change['current_value']:.4f} | " +
                        f"{change['change_pct']:+.2f}% |\n")

            f.write("\n")

        # Latest results
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

        # Include links to the plots
        f.write("## Performance Graphs\n\n")
        for metric in benchmark_data.metrics:
            safe_metric_name = metric.replace('@', '_at_').replace(' ', '_')
            f.write(f"![{metric}]({safe_metric_name}.png)\n\n")

        f.write("![Normalized Comparison](normalized_comparison.png)\n")


def main():
    parser = argparse.ArgumentParser(description="Compare and visualize benchmark results across multiple releases")
    parser.add_argument("files", nargs="+", help="Paths to CSV benchmark result files")
    parser.add_argument("--output-dir", default="benchmark_reports", help="Directory to save plots and reports")
    parser.add_argument("--threshold", type=float, default=5.0, 
                        help="Threshold percentage for detecting significant changes (default: 5.0)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load benchmark data
    benchmark_data = BenchmarkData()

    for file_path in args.files:
        release = extract_release_from_filename(file_path)
        print(f"Loading {file_path} as release {release}...")
        benchmark_data.add_file(file_path, release)

    # We no longer sort the releases - we keep them in the order they were provided
    # This comment replaces the previous sorting code

    # Detect significant changes
    changes = benchmark_data.detect_changes(args.threshold)

    # Generate plots
    generate_plots(benchmark_data, args.output_dir)

    # Generate report
    report_path = os.path.join(args.output_dir, "benchmark_report.md")
    generate_report(benchmark_data, changes, report_path)

    print(f"Benchmark analysis complete. Report saved to {report_path}")
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
