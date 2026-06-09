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

"""
Compare two different iterations of the same benchmark comparison on a single graph.
This script takes two directories containing CSV benchmark files and overlays the results
with the same graph structure as visualize_benchmarks.py.
"""

import argparse
import csv
import os
import sys
import re
import platform
import psutil
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# Define metrics where higher values are better and lower values are better
HIGHER_IS_BETTER = ["QPS", "Recall@10"]
LOWER_IS_BETTER = ["Mean Latency", "Index Build Time"]


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
        
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Handle both 'Dataset' and 'dataset' column names
                dataset = row.get('Dataset') or row.get('dataset')
                if not dataset:
                    # Skip rows without a dataset identifier
                    continue
                    
                self.datasets.add(dataset)
                
                # Add each metric
                for key, value in row.items():
                    if key.lower() not in ['dataset'] and value.strip():  # Skip dataset column and empty values
                        try:
                            numeric_value = float(value)
                            self.data[release][dataset][key] = numeric_value
                            self.metrics.add(key)
                        except ValueError:
                            # Skip non-numeric values
                            pass

    def get_metric_data_for_dataset(self, metric: str, dataset: str) -> Tuple[List[str], List[float]]:
        """
        Get metric data for a specific dataset across all releases
        Returns tuple of (releases, values)
        """
        releases = []
        values = []
        
        for release in self.releases:
            if dataset in self.data[release] and metric in self.data[release][dataset]:
                releases.append(release)
                values.append(self.data[release][dataset][metric])
        
        return releases, values


def extract_release_from_filename(filename: str) -> str:
    """
    Extract release number from filename
    """
    # Try to extract version number from filename
    # Look for patterns like v1.2.3, 1.2.3, etc.
    version_match = re.search(r'v?(\d+\.\d+\.\d+)', filename)
    if version_match:
        return version_match.group(1)
    
    # Fallback to filename without extension
    return os.path.splitext(filename)[0]


def load_benchmark_data(directory: str) -> BenchmarkData:
    """
    Load benchmark data from all CSV files in a directory
    """
    benchmark_data = BenchmarkData()
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV files found in {directory}")
        sys.exit(1)
    
    # Custom sort to ensure "main" appears last
    def sort_key(filename):
        release = extract_release_from_filename(filename)
        # If the release is "main", return a tuple that sorts it last
        if release.lower() == "main":
            return (0, release)  # 0 ensures it comes before all 1s
        else:
            return (1, release)  # 1 ensures it comes after main
    
    csv_files.sort(key=sort_key)
    
    for filename in csv_files:
        file_path = os.path.join(directory, filename)
        release = extract_release_from_filename(filename)
        print(f"Loading {filename} as release {release}")
        benchmark_data.add_file(file_path, release)
    
    return benchmark_data


def generate_comparison_plots(benchmark_data1: BenchmarkData, benchmark_data2: BenchmarkData, 
                            output_dir: str, run1_label: str = "Run 1", run2_label: str = "Run 2"):
    """
    Generate comparison plots for each metric, with two lines per dataset (one for each run)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all releases from both runs (maintaining order)
    all_releases = list(benchmark_data1.releases)
    for release in benchmark_data2.releases:
        if release not in all_releases:
            all_releases.append(release)
    
    # Get all datasets from both runs
    all_datasets = benchmark_data1.datasets | benchmark_data2.datasets
    
    # Get all metrics from both runs
    all_metrics = benchmark_data1.metrics | benchmark_data2.metrics
    
    # Define a color cycle for datasets
    import matplotlib.pyplot as plt
    colors = plt.cm.tab10(range(len(all_datasets)))
    dataset_colors = {dataset: colors[i] for i, dataset in enumerate(sorted(all_datasets))}
    
    # Create a plot for each metric
    for metric in sorted(all_metrics):
        plt.figure(figsize=(10, 6))
        
        # Plot lines for each dataset from both runs
        for dataset in sorted(all_datasets):
            color = dataset_colors[dataset]
            
            # Get data for run 1
            releases1, values1 = benchmark_data1.get_metric_data_for_dataset(metric, dataset)
            if releases1 and values1:
                plt.plot(releases1, values1, marker='o', label=f"{dataset} ({run1_label})", 
                        linestyle='-', linewidth=2, color=color)
            
            # Get data for run 2
            releases2, values2 = benchmark_data2.get_metric_data_for_dataset(metric, dataset)
            if releases2 and values2:
                plt.plot(releases2, values2, marker='s', label=f"{dataset} ({run2_label})", 
                        linestyle='--', linewidth=2, color=color, alpha=0.8)
        
        plt.title(f"{metric} Over Time - Comparison")
        plt.xlabel("Release")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        safe_metric_name = metric.replace('@', '_at_').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f"{safe_metric_name}_comparison.png"), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Generated comparison plot for {metric}")


def generate_summary_report(benchmark_data1: BenchmarkData, benchmark_data2: BenchmarkData, 
                          output_file: str, run1_label: str = "Run 1", run2_label: str = "Run 2"):
    """
    Generate a summary report comparing the two runs
    """
    with open(output_file, 'w') as f:
        f.write(f"# Benchmark Comparison Report: {run1_label} vs {run2_label}\n\n")
        
        # Get system information
        cpu_count = psutil.cpu_count(logical=False)
        if cpu_count is None:
            cpu_count = psutil.cpu_count(logical=True)
        memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        processor = platform.processor()
        if not processor:
            processor = platform.machine()
        
        f.write(f"**System Information:**\n")
        f.write(f"- CPU: {processor}\n")
        f.write(f"- Cores: {cpu_count}\n")
        f.write(f"- Memory: {memory_gb} GB\n")
        f.write(f"- Platform: {platform.system()} {platform.release()}\n\n")
        
        f.write(f"**Comparison Summary:**\n")
        f.write(f"- {run1_label}: {len(benchmark_data1.releases)} releases, {len(benchmark_data1.datasets)} datasets\n")
        f.write(f"- {run2_label}: {len(benchmark_data2.releases)} releases, {len(benchmark_data2.datasets)} datasets\n")
        f.write(f"- Metrics compared: {', '.join(sorted(benchmark_data1.metrics | benchmark_data2.metrics))}\n\n")
        
        f.write("## Datasets\n\n")
        all_datasets = sorted(benchmark_data1.datasets | benchmark_data2.datasets)
        for dataset in all_datasets:
            in_run1 = dataset in benchmark_data1.datasets
            in_run2 = dataset in benchmark_data2.datasets
            status = ""
            if in_run1 and in_run2:
                status = "✓ Both runs"
            elif in_run1:
                status = f"✓ {run1_label} only"
            else:
                status = f"✓ {run2_label} only"
            f.write(f"- **{dataset}**: {status}\n")
        
        f.write("\n## Releases\n\n")
        f.write(f"**{run1_label}**: {', '.join(benchmark_data1.releases)}\n\n")
        f.write(f"**{run2_label}**: {', '.join(benchmark_data2.releases)}\n\n")
        
        f.write("Generated plots show the comparison of both runs with:\n")
        f.write(f"- Solid lines with circle markers for {run1_label}\n")
        f.write(f"- Dashed lines with square markers for {run2_label}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare two benchmark iterations and generate overlay plots")
    parser.add_argument('dir1', help="Directory containing CSV files for the first run")
    parser.add_argument('dir2', help="Directory containing CSV files for the second run")
    parser.add_argument('--output-dir', default='benchmark_comparison', 
                       help="Directory to save comparison plots (default: benchmark_comparison)")
    parser.add_argument('--run1-label', default='Run 1', 
                       help="Label for the first run (default: Run 1)")
    parser.add_argument('--run2-label', default='Run 2', 
                       help="Label for the second run (default: Run 2)")
    
    args = parser.parse_args()
    
    print(f"Loading benchmark data from {args.dir1}...")
    benchmark_data1 = load_benchmark_data(args.dir1)
    
    print(f"Loading benchmark data from {args.dir2}...")
    benchmark_data2 = load_benchmark_data(args.dir2)
    
    print(f"Generating comparison plots...")
    generate_comparison_plots(benchmark_data1, benchmark_data2, args.output_dir, 
                            args.run1_label, args.run2_label)
    
    # Generate summary report
    report_file = os.path.join(args.output_dir, "comparison_summary.md")
    generate_summary_report(benchmark_data1, benchmark_data2, report_file, 
                           args.run1_label, args.run2_label)
    
    print(f"Comparison complete! Results saved to {args.output_dir}")
    print(f"Summary report: {report_file}")


if __name__ == "__main__":
    main()
