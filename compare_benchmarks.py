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
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

def load_benchmark_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load benchmark results from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing benchmark results
        
    Returns:
        List of benchmark results
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading benchmark results from {file_path}: {e}")
        sys.exit(1)

def group_results_by_config(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Group benchmark results by configuration and metric.
    
    Args:
        results: List of benchmark results
        
    Returns:
        Dictionary mapping configuration strings to metric values
    """
    grouped_results = {}
    
    for result in results:
        dataset = result.get('dataset', 'unknown')
        params = result.get('parameters', {})
        metrics = result.get('metrics', {})
        
        if not metrics:
            continue
        
        # Create a configuration key that uniquely identifies this benchmark configuration
        config_key = f"{dataset}|"
        config_key += "|".join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Store all metrics for this configuration
        if config_key not in grouped_results:
            grouped_results[config_key] = {}
        
        # Process all metrics instead of just the first one
        for metric_name, metric_value in metrics.items():
            grouped_results[config_key][metric_name] = float(metric_value)
    
    return grouped_results

def compare_results(
    current_results: Dict[str, Dict[str, float]], 
    previous_results: Dict[str, Dict[str, float]]
) -> List[Dict[str, Any]]:
    """
    Compare current benchmark results with previous results.
    
    Args:
        current_results: Current benchmark results grouped by configuration
        previous_results: Previous benchmark results grouped by configuration
        
    Returns:
        List of comparison results
    """
    comparison = []
    
    # Find all unique configurations across both result sets
    all_configs = set(current_results.keys()) | set(previous_results.keys())
    
    for config in all_configs:
        current_metrics = current_results.get(config, {})
        previous_metrics = previous_results.get(config, {})
        
        # Find all unique metrics across both result sets for this configuration
        all_metrics = set(current_metrics.keys()) | set(previous_metrics.keys())
        
        for metric in all_metrics:
            current_value = current_metrics.get(metric)
            previous_value = previous_metrics.get(metric)
            
            if current_value is not None and previous_value is not None:
                # Calculate the change as a percentage
                change_pct = ((current_value - previous_value) / previous_value) * 100
                
                # Determine if this is an improvement or regression
                # For metrics where higher is better (QPS, Recall)
                higher_is_better = metric in ["QPS", "Recall@10", "Recall@100", "MAP@10", "MAP@100"]
                
                # For metrics where lower is better (latency, nodes visited/expanded)
                lower_is_better = metric in ["Mean Latency (ms)", "STD Latency (ms)", "p999 Latency (ms)", 
                                           "Avg Visited", "Avg Expanded", "Avg Expanded Base Layer"]
                
                if higher_is_better:
                    status = "improvement" if change_pct > 0 else "regression"
                elif lower_is_better:
                    status = "improvement" if change_pct < 0 else "regression"
                else:
                    status = "unknown"
                
                # Only mark as improvement/regression if the change is significant (>1%)
                if abs(change_pct) < 1.0:
                    status = "unchanged"
                
                comparison.append({
                    "config": config,
                    "metric": metric,
                    "current_value": current_value,
                    "previous_value": previous_value,
                    "change_pct": change_pct,
                    "status": status
                })
            elif current_value is not None:
                comparison.append({
                    "config": config,
                    "metric": metric,
                    "current_value": current_value,
                    "previous_value": None,
                    "change_pct": None,
                    "status": "new"
                })
            elif previous_value is not None:
                comparison.append({
                    "config": config,
                    "metric": metric,
                    "current_value": None,
                    "previous_value": previous_value,
                    "change_pct": None,
                    "status": "removed"
                })
    
    return comparison

def generate_report(comparison: List[Dict[str, Any]]) -> str:
    """
    Generate a human-readable report from the comparison results.
    
    Args:
        comparison: List of comparison results
        
    Returns:
        String containing the report
    """
    report = "# Benchmark Comparison Report\n\n"
    
    # Group by status for a summary
    status_counts = defaultdict(int)
    for item in comparison:
        status_counts[item["status"]] += 1
    
    report += "## Summary\n\n"
    report += f"- Total metrics compared: {len(comparison)}\n"
    report += f"- Improvements: {status_counts['improvement']}\n"
    report += f"- Regressions: {status_counts['regression']}\n"
    report += f"- Unchanged: {status_counts['unchanged']}\n"
    report += f"- New metrics: {status_counts['new']}\n"
    report += f"- Removed metrics: {status_counts['removed']}\n\n"
    
    # Show regressions first
    if status_counts['regression'] > 0:
        report += "## Regressions\n\n"
        for item in comparison:
            if item["status"] == "regression":
                config_parts = item["config"].split("|")
                dataset = config_parts[0]
                params = "|".join(config_parts[1:])
                
                report += f"### {dataset} - {item['metric']}\n\n"
                report += f"- Configuration: {params}\n"
                report += f"- Current value: {item['current_value']:.4f}\n"
                report += f"- Previous value: {item['previous_value']:.4f}\n"
                report += f"- Change: {item['change_pct']:.2f}%\n\n"
    
    # Then show improvements
    if status_counts['improvement'] > 0:
        report += "## Improvements\n\n"
        for item in comparison:
            if item["status"] == "improvement":
                config_parts = item["config"].split("|")
                dataset = config_parts[0]
                params = "|".join(config_parts[1:])
                
                report += f"### {dataset} - {item['metric']}\n\n"
                report += f"- Configuration: {params}\n"
                report += f"- Current value: {item['current_value']:.4f}\n"
                report += f"- Previous value: {item['previous_value']:.4f}\n"
                report += f"- Change: {item['change_pct']:.2f}%\n\n"
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results between runs")
    parser.add_argument("current", help="Path to the current benchmark results JSON file")
    parser.add_argument("previous", help="Path to the previous benchmark results JSON file")
    parser.add_argument("--output", help="Path to write the comparison report (default: stdout)")
    
    args = parser.parse_args()
    
    # Load benchmark results
    current_results = load_benchmark_results(args.current)
    previous_results = load_benchmark_results(args.previous)
    
    # Group results by configuration
    grouped_current = group_results_by_config(current_results)
    grouped_previous = group_results_by_config(previous_results)
    
    # Compare results
    comparison = compare_results(grouped_current, grouped_previous)
    
    # Generate report
    report = generate_report(comparison)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

if __name__ == "__main__":
    main()