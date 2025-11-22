#!/usr/bin/env python3

"""
NS-3 Simulation Results Analyzer
Advanced analysis and visualization for MPI research simulations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any
import warnings

warnings.filterwarnings('ignore')

class Ns3ResultsAnalyzer:
    """Advanced analyzer for NS-3 simulation results"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.data = {}
        self.metrics = {}

    def load_results(self) -> None:
        """Load all simulation results from directory"""
        print(f"Loading results from {self.results_dir}")

        # Find all CSV files
        csv_files = list(self.results_dir.glob('**/*.csv'))

        for csv_file in csv_files:
            topology_name = csv_file.parent.name
            try:
                df = pd.read_csv(csv_file)
                if topology_name not in self.data:
                    self.data[topology_name] = {}

                # Store dataframe and extract key metrics
                self.data[topology_name][csv_file.stem] = df
                self._extract_metrics(topology_name, csv_file.stem, df)

                print(f"  Loaded {csv_file} with {len(df)} rows")

            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")

    def _extract_metrics(self, topology: str, scenario: str, df: pd.DataFrame) -> None:
        """Extract key performance metrics from dataframe"""
        if topology not in self.metrics:
            self.metrics[topology] = {}

        metrics = {}

        # Basic statistics
        if 'ExecutionTime' in df.columns:
            metrics['avg_execution_time'] = df['ExecutionTime'].mean()
            metrics['std_execution_time'] = df['ExecutionTime'].std()
            metrics['min_execution_time'] = df['ExecutionTime'].min()
            metrics['max_execution_time'] = df['ExecutionTime'].max()

        if 'BandwidthUtilization' in df.columns:
            metrics['avg_bandwidth'] = df['BandwidthUtilization'].mean()
            metrics['max_bandwidth'] = df['BandwidthUtilization'].max()

        if 'MessagesSent' in df.columns:
            metrics['total_messages'] = df['MessagesSent'].sum()
            metrics['avg_messages'] = df['MessagesSent'].mean()

        if 'DataVolume' in df.columns:
            metrics['total_data'] = df['DataVolume'].sum() / (1024 * 1024)  # Convert to MB
            metrics['avg_data'] = df['DataVolume'].mean() / 1024  # Convert to KB

        self.metrics[topology][scenario] = metrics

    def generate_summary_report(self) -> pd.DataFrame:
        """Generate comprehensive summary report"""
        summary_data = []

        for topology, scenarios in self.metrics.items():
            for scenario, metrics in scenarios.items():
                row = {
                    'Topology': topology,
                    'Scenario': scenario,
                    'Avg_Execution_Time_ms': metrics.get('avg_execution_time', 0) * 1000,
                    'Std_Execution_Time_ms': metrics.get('std_execution_time', 0) * 1000,
                    'Avg_Bandwidth_%': metrics.get('avg_bandwidth', 0),
                    'Max_Bandwidth_%': metrics.get('max_bandwidth', 0),
                    'Total_Messages': metrics.get('total_messages', 0),
                    'Total_Data_MB': metrics.get('total_data', 0),
                    'Avg_Data_KB': metrics.get('avg_data', 0)
                }
                summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        return summary_df

    def plot_performance_comparison(self, output_file: str = None) -> None:
        """Create performance comparison plots"""
        if not self.metrics:
            print("No metrics available. Run load_results() first.")
            return

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MPI Collective Operations Performance Comparison', fontsize=16, fontweight='bold')

        # Prepare data for plotting
        comparison_data = []
        for topology, scenarios in self.metrics.items():
            for scenario, metrics in scenarios.items():
                comparison_data.append({
                    'Topology': topology,
                    'Scenario': scenario,
                    'Execution_Time': metrics.get('avg_execution_time', 0) * 1000,
                    'Bandwidth_Utilization': metrics.get('avg_bandwidth', 0),
                    'Message_Count': metrics.get('avg_messages', 0),
                    'Data_Volume_MB': metrics.get('total_data', 0)
                })

        comp_df = pd.DataFrame(comparison_data)

        if comp_df.empty:
            print("No data available for plotting")
            return

        # Plot 1: Execution Time Comparison
        if 'Execution_Time' in comp_df.columns:
            sns.barplot(data=comp_df, x='Topology', y='Execution_Time', ax=axes[0,0])
            axes[0,0].set_title('Average Execution Time (ms)')
            axes[0,0].set_ylabel('Time (ms)')
            axes[0,0].tick_params(axis='x', rotation=45)

        # Plot 2: Bandwidth Utilization
        if 'Bandwidth_Utilization' in comp_df.columns:
            sns.barplot(data=comp_df, x='Topology', y='Bandwidth_Utilization', ax=axes[0,1])
            axes[0,1].set_title('Average Bandwidth Utilization (%)')
            axes[0,1].set_ylabel('Utilization (%)')
            axes[0,1].tick_params(axis='x', rotation=45)

        # Plot 3: Message Count
        if 'Message_Count' in comp_df.columns:
            sns.barplot(data=comp_df, x='Topology', y='Message_Count', ax=axes[1,0])
            axes[1,0].set_title('Average Message Count')
            axes[1,0].set_ylabel('Messages')
            axes[1,0].tick_params(axis='x', rotation=45)

        # Plot 4: Data Volume
        if 'Data_Volume_MB' in comp_df.columns:
            sns.barplot(data=comp_df, x='Topology', y='Data_Volume_MB', ax=axes[1,1])
            axes[1,1].set_title('Total Data Volume (MB)')
            axes[1,1].set_ylabel('Data (MB)')
            axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        else:
            plt.show()

    def plot_scalability_analysis(self, output_file: str = None) -> None:
        """Analyze scalability across different topologies"""
        if not self.data:
            print("No data available. Run load_results() first.")
            return

        # Extract scalability data
        scalability_data = []

        for topology, scenarios in self.data.items():
            for scenario, df in scenarios.items():
                # Try to extract node count from topology name
                node_count = self._extract_node_count(topology, scenario)

                if node_count > 0 and 'ExecutionTime' in df.columns:
                    avg_time = df['ExecutionTime'].mean() * 1000  # Convert to ms
                    scalability_data.append({
                        'Topology': topology,
                        'Nodes': node_count,
                        'Execution_Time_ms': avg_time
                    })

        if not scalability_data:
            print("No scalability data available")
            return

        scal_df = pd.DataFrame(scalability_data)

        # Create scalability plot
        plt.figure(figsize=(10, 6))

        for topology in scal_df['Topology'].unique():
            topology_data = scal_df[scal_df['Topology'] == topology].sort_values('Nodes')
            plt.plot(topology_data['Nodes'], topology_data['Execution_Time_ms'], 
                    marker='o', label=topology, linewidth=2)

        plt.xlabel('Number of Nodes')
        plt.ylabel('Average Execution Time (ms)')
        plt.title('MPI Collective Operations Scalability Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Scalability plot saved to {output_file}")
        else:
            plt.show()

    def _extract_node_count(self, topology: str, scenario: str) -> int:
        """Extract node count from topology and scenario names"""
        # Simple heuristic-based extraction
        import re

        # Look for patterns like "k4", "4x4", "4x4x2"
        patterns = [
            r'k(\d+)',  # fat_tree_k4
            r'(\d+)x(\d+)',  # torus_4x4
            r'(\d+)x(\d+)x(\d+)',  # torus_4x4x2
        ]

        combined = f"{topology}_{scenario}"

        for pattern in patterns:
            matches = re.findall(pattern, combined)
            if matches:
                if isinstance(matches[0], tuple):
                    # Multiple dimensions
                    factors = [int(x) for x in matches[0]]
                    return np.prod(factors)
                else:
                    # Single dimension
                    return int(matches[0])

        return 0

    def generate_detailed_analysis(self, output_dir: str = None) -> None:
        """Generate detailed analysis report with multiple visualizations"""
        if output_dir is None:
            output_dir = self.results_dir / "analysis"

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"Generating detailed analysis in {output_path}")

        # 1. Summary report
        summary_df = self.generate_summary_report()
        summary_file = output_path / "performance_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary report: {summary_file}")

        # 2. Performance comparison plot
        comparison_plot = output_path / "performance_comparison.png"
        self.plot_performance_comparison(str(comparison_plot))

        # 3. Scalability analysis
        scalability_plot = output_path / "scalability_analysis.png"
        self.plot_scalability_analysis(str(scalability_plot))

        # 4. Topology-specific analysis
        self._generate_topology_analysis(output_path)

        # 5. Statistical analysis
        self._generate_statistical_analysis(output_path, summary_df)

        print(f"Detailed analysis completed in {output_path}")

    def _generate_topology_analysis(self, output_path: Path) -> None:
        """Generate topology-specific analysis"""
        for topology, scenarios in self.data.items():
            topology_path = output_path / topology
            topology_path.mkdir(exist_ok=True)

            # Create topology-specific plots
            for scenario, df in scenarios.items():
                if 'ExecutionTime' in df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.hist(df['ExecutionTime'] * 1000, bins=20, alpha=0.7, edgecolor='black')
                    plt.xlabel('Execution Time (ms)')
                    plt.ylabel('Frequency')
                    plt.title(f'Execution Time Distribution - {topology} - {scenario}')
                    plt.grid(True, alpha=0.3)

                    plot_file = topology_path / f"{scenario}_distribution.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()

    def _generate_statistical_analysis(self, output_path: Path, summary_df: pd.DataFrame) -> None:
        """Generate statistical analysis report"""
        stats_file = output_path / "statistical_analysis.txt"

        with open(stats_file, 'w') as f:
            f.write("MPI Research - Statistical Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            # Basic statistics
            f.write("1. Overall Performance Statistics:\n")
            f.write("-" * 30 + "\n")

            if 'Avg_Execution_Time_ms' in summary_df.columns:
                f.write(f"Minimum Execution Time: {summary_df['Avg_Execution_Time_ms'].min():.2f} ms\n")
                f.write(f"Maximum Execution Time: {summary_df['Avg_Execution_Time_ms'].max():.2f} ms\n")
                f.write(f"Average Execution Time: {summary_df['Avg_Execution_Time_ms'].mean():.2f} ms\n")
                f.write(f"Standard Deviation: {summary_df['Avg_Execution_Time_ms'].std():.2f} ms\n\n")

            # Topology ranking
            f.write("2. Topology Performance Ranking:\n")
            f.write("-" * 30 + "\n")

            if 'Avg_Execution_Time_ms' in summary_df.columns:
                ranked = summary_df.sort_values('Avg_Execution_Time_ms')
                for i, (_, row) in enumerate(ranked.iterrows(), 1):
                    f.write(f"{i}. {row['Topology']}: {row['Avg_Execution_Time_ms']:.2f} ms\n")
                f.write("\n")

            # Recommendations
            f.write("3. Recommendations:\n")
            f.write("-" * 30 + "\n")

            # Simple heuristic-based recommendations
            if not summary_df.empty:
                best_topology = summary_df.loc[summary_df['Avg_Execution_Time_ms'].idxmin(), 'Topology']
                best_bw = summary_df.loc[summary_df['Avg_Bandwidth_%'].idxmax(), 'Topology']

                f.write(f"- Best overall performance: {best_topology}\n")
                f.write(f"- Best bandwidth utilization: {best_bw}\n")

                # Contextual recommendations
                if 'fat_tree' in summary_df['Topology'].values:
                    f.write("- Fat Tree: Good for balanced workloads, excellent bisection bandwidth\n")
                if 'torus' in summary_df['Topology'].values:
                    f.write("- Torus: Good for nearest-neighbor communication, cost-effective\n")
                if 'dragonfly' in summary_df['Topology'].values:
                    f.write("- Dragonfly: Excellent for global communication, scalable\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze NS-3 MPI research simulation results')
    parser.add_argument('results_dir', help='Directory containing simulation results')
    parser.add_argument('-o', '--output', help='Output directory for analysis results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--summary', action='store_true', help='Generate summary report')
    parser.add_argument('--detailed', action='store_true', help='Generate detailed analysis')

    args = parser.parse_args()

    # Create analyzer
    analyzer = Ns3ResultsAnalyzer(args.results_dir)

    # Load results
    analyzer.load_results()

    # Generate requested outputs
    if args.summary or args.detailed:
        summary_df = analyzer.generate_summary_report()
        print("\nPerformance Summary:")
        print("=" * 50)
        print(summary_df.to_string(index=False))

        # Save summary
        if args.output:
            summary_file = os.path.join(args.output, "performance_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"\nSummary saved to: {summary_file}")

    if args.plot or args.detailed:
        output_dir = args.output if args.output else args.results_dir
        analyzer.generate_detailed_analysis(output_dir)

    if not any([args.plot, args.summary, args.detailed]):
        # Default: show basic info
        summary_df = analyzer.generate_summary_report()
        if not summary_df.empty:
            print("\nPerformance Summary:")
            print("=" * 50)
            print(summary_df.to_string(index=False))
        else:
            print("No results found or unable to load data")

if __name__ == "__main__":
    main()