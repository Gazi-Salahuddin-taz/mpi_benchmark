#!/usr/bin/env python3

"""
Advanced Performance Plotting for MPI Research
Generates comprehensive performance visualizations from benchmark results
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
from typing import Dict, List, Tuple, Any
import warnings
from scipy import stats
from scipy.optimize import curve_fit
import scienceplots  # For publication-quality plots

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use(['science', 'ieee', 'grid'])

class PerformancePlotter:
    """Advanced performance visualization for MPI research"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.data = {}
        self.metrics = {}
        self.colors = sns.color_palette("husl", 8)
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
    def load_all_results(self) -> None:
        """Load results from all benchmark directories"""
        print("Loading performance results...")
        
        # Find all CSV files in results directory
        csv_files = list(self.results_dir.glob('**/*.csv'))
        
        for csv_file in csv_files:
            try:
                # Extract topology and benchmark type from path
                rel_path = csv_file.relative_to(self.results_dir)
                topology = rel_path.parts[0] if len(rel_path.parts) > 1 else 'unknown'
                
                if topology not in self.data:
                    self.data[topology] = {}
                
                df = pd.read_csv(csv_file)
                benchmark_name = csv_file.stem
                self.data[topology][benchmark_name] = df
                
                print(f"  Loaded {topology}/{benchmark_name}: {len(df)} rows")
                
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")
    
    def plot_performance_comparison(self, output_dir: Path) -> None:
        """Create comprehensive performance comparison plots"""
        print("Generating performance comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MPI Collective Operations Performance Comparison', 
                    fontsize=16, fontweight='bold')
        
        # Prepare comparison data
        comparison_data = []
        for topology, benchmarks in self.data.items():
            for benchmark, df in benchmarks.items():
                if 'ExecutionTime' in df.columns and 'MessageSize' in df.columns:
                    for _, row in df.iterrows():
                        comparison_data.append({
                            'Topology': topology,
                            'Benchmark': benchmark,
                            'MessageSize': row['MessageSize'],
                            'ExecutionTime': row['ExecutionTime'],
                            'BandwidthUtilization': row.get('BandwidthUtilization', 0),
                            'MessagesSent': row.get('MessagesSent', 0)
                        })
        
        if not comparison_data:
            print("No performance data available for plotting")
            return
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Plot 1: Execution Time vs Message Size
        self._plot_execution_time(comp_df, axes[0, 0])
        
        # Plot 2: Bandwidth Utilization
        self._plot_bandwidth_utilization(comp_df, axes[0, 1])
        
        # Plot 3: Performance by Topology
        self._plot_topology_performance(comp_df, axes[1, 0])
        
        # Plot 4: Algorithm Efficiency
        self._plot_algorithm_efficiency(comp_df, axes[1, 1])
        
        plt.tight_layout()
        output_file = output_dir / "performance_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison plot saved: {output_file}")
    
    def _plot_execution_time(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot execution time vs message size"""
        for i, topology in enumerate(df['Topology'].unique()):
            topology_data = df[df['Topology'] == topology]
            
            # Aggregate by message size
            aggregated = topology_data.groupby('MessageSize').agg({
                'ExecutionTime': ['mean', 'std']
            }).reset_index()
            
            aggregated.columns = ['MessageSize', 'MeanTime', 'StdTime']
            
            # Plot with error bars
            ax.errorbar(aggregated['MessageSize'], aggregated['MeanTime'],
                       yerr=aggregated['StdTime'], 
                       label=topology, color=self.colors[i],
                       marker=self.markers[i], markersize=6,
                       capsize=3, capthick=1, elinewidth=1)
        
        ax.set_xlabel('Message Size (bytes)')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Execution Time vs Message Size')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_bandwidth_utilization(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot bandwidth utilization comparison"""
        if 'BandwidthUtilization' not in df.columns:
            ax.text(0.5, 0.5, 'Bandwidth data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Bandwidth Utilization')
            return
        
        # Prepare data for box plot
        plot_data = []
        for topology in df['Topology'].unique():
            topology_data = df[df['Topology'] == topology]
            for utilization in topology_data['BandwidthUtilization']:
                plot_data.append({'Topology': topology, 'BandwidthUtilization': utilization})
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            sns.boxplot(data=plot_df, x='Topology', y='BandwidthUtilization', ax=ax)
            ax.set_title('Bandwidth Utilization Distribution')
            ax.set_ylabel('Bandwidth Utilization (%)')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_topology_performance(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot overall performance by topology"""
        performance_data = []
        
        for topology in df['Topology'].unique():
            topology_data = df[df['Topology'] == topology]
            
            # Calculate performance metrics
            avg_time = topology_data['ExecutionTime'].mean()
            std_time = topology_data['ExecutionTime'].std()
            avg_bw = topology_data.get('BandwidthUtilization', pd.Series([0])).mean()
            
            performance_data.append({
                'Topology': topology,
                'AvgExecutionTime': avg_time,
                'StdExecutionTime': std_time,
                'AvgBandwidth': avg_bw
            })
        
        perf_df = pd.DataFrame(performance_data)
        
        # Create bar plot with error bars
        x_pos = np.arange(len(perf_df))
        bars = ax.bar(x_pos, perf_df['AvgExecutionTime'], 
                     yerr=perf_df['StdExecutionTime'],
                     color=self.colors[:len(perf_df)],
                     alpha=0.7, capsize=5)
        
        ax.set_xlabel('Network Topology')
        ax.set_ylabel('Average Execution Time (ms)')
        ax.set_title('Performance by Network Topology')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(perf_df['Topology'], rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
    
    def _plot_algorithm_efficiency(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot algorithm efficiency analysis"""
        # Calculate efficiency metrics
        efficiency_data = []
        
        for topology in df['Topology'].unique():
            topology_data = df[df['Topology'] == topology]
            
            if len(topology_data) < 2:
                continue
            
            # Calculate coefficient of variation (lower is better)
            cv = topology_data['ExecutionTime'].std() / topology_data['ExecutionTime'].mean()
            
            # Calculate efficiency (inverse of execution time)
            max_time = topology_data['ExecutionTime'].max()
            min_time = topology_data['ExecutionTime'].min()
            efficiency_range = (max_time - min_time) / max_time
            
            efficiency_data.append({
                'Topology': topology,
                'CoefficientOfVariation': cv,
                'EfficiencyRange': efficiency_range,
                'AverageTime': topology_data['ExecutionTime'].mean()
            })
        
        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data)
            
            # Create scatter plot
            scatter = ax.scatter(eff_df['CoefficientOfVariation'], 
                               eff_df['AverageTime'],
                               c=eff_df['EfficiencyRange'], 
                               s=100, alpha=0.7,
                               cmap='viridis')
            
            ax.set_xlabel('Coefficient of Variation')
            ax.set_ylabel('Average Execution Time (ms)')
            ax.set_title('Algorithm Efficiency Analysis')
            ax.grid(True, alpha=0.3)
            
            # Add topology labels
            for i, row in eff_df.iterrows():
                ax.annotate(row['Topology'], 
                           (row['CoefficientOfVariation'], row['AverageTime']),
                           xytext=(5, 5), textcoords='offset points')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Efficiency Range')
    
    def plot_scalability_analysis(self, output_dir: Path) -> None:
        """Generate scalability analysis plots"""
        print("Generating scalability analysis plots...")
        
        # Extract scalability data
        scalability_data = self._extract_scalability_data()
        
        if not scalability_data:
            print("No scalability data found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MPI Collective Operations Scalability Analysis', 
                    fontsize=16, fontweight='bold')
        
        scal_df = pd.DataFrame(scalability_data)
        
        # Plot 1: Strong Scaling
        self._plot_strong_scaling(scal_df, axes[0, 0])
        
        # Plot 2: Weak Scaling
        self._plot_weak_scaling(scal_df, axes[0, 1])
        
        # Plot 3: Efficiency Analysis
        self._plot_efficiency_analysis(scal_df, axes[1, 0])
        
        # Plot 4: Scalability Models
        self._plot_scalability_models(scal_df, axes[1, 1])
        
        plt.tight_layout()
        output_file = output_dir / "scalability_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Scalability analysis plot saved: {output_file}")
    
    def _extract_scalability_data(self) -> List[Dict]:
        """Extract scalability data from results"""
        scalability_data = []
        
        for topology, benchmarks in self.data.items():
            for benchmark, df in benchmarks.items():
                # Look for scalability-related columns
                if any(col in df.columns for col in ['Processes', 'Nodes', 'Scale']):
                    for _, row in df.iterrows():
                        data_point = {'Topology': topology, 'Benchmark': benchmark}
                        
                        # Extract scalability metrics
                        if 'Processes' in df.columns:
                            data_point['Processes'] = row['Processes']
                        if 'Nodes' in df.columns:
                            data_point['Nodes'] = row['Nodes']
                        if 'ExecutionTime' in df.columns:
                            data_point['ExecutionTime'] = row['ExecutionTime']
                        if 'Efficiency' in df.columns:
                            data_point['Efficiency'] = row['Efficiency']
                        if 'Speedup' in df.columns:
                            data_point['Speedup'] = row['Speedup']
                        
                        scalability_data.append(data_point)
        
        return scalability_data
    
    def _plot_strong_scaling(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot strong scaling analysis"""
        if 'Processes' not in df.columns or 'ExecutionTime' not in df.columns:
            ax.text(0.5, 0.5, 'Strong scaling data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Strong Scaling Analysis')
            return
        
        for i, topology in enumerate(df['Topology'].unique()):
            topology_data = df[df['Topology'] == topology]
            
            # Group by process count
            grouped = topology_data.groupby('Processes').agg({
                'ExecutionTime': 'mean'
            }).reset_index()
            
            if len(grouped) > 1:
                ax.plot(grouped['Processes'], grouped['ExecutionTime'],
                       label=topology, color=self.colors[i],
                       marker=self.markers[i], linewidth=2)
        
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Strong Scaling: Execution Time vs Processes')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_weak_scaling(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot weak scaling analysis"""
        if 'Processes' not in df.columns or 'Efficiency' not in df.columns:
            ax.text(0.5, 0.5, 'Weak scaling data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weak Scaling Analysis')
            return
        
        for i, topology in enumerate(df['Topology'].unique()):
            topology_data = df[df['Topology'] == topology]
            
            # Group by process count
            grouped = topology_data.groupby('Processes').agg({
                'Efficiency': 'mean'
            }).reset_index()
            
            if len(grouped) > 1:
                ax.plot(grouped['Processes'], grouped['Efficiency'],
                       label=topology, color=self.colors[i],
                       marker=self.markers[i], linewidth=2)
        
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Efficiency (%)')
        ax.set_title('Weak Scaling: Efficiency vs Processes')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add ideal efficiency line
        max_processes = df['Processes'].max()
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Ideal')
    
    def _plot_efficiency_analysis(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot efficiency analysis"""
        if 'Processes' not in df.columns or 'Efficiency' not in df.columns:
            ax.text(0.5, 0.5, 'Efficiency data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Efficiency Analysis')
            return
        
        # Calculate efficiency drop rates
        efficiency_data = []
        
        for topology in df['Topology'].unique():
            topology_data = df[df['Topology'] == topology].sort_values('Processes')
            
            if len(topology_data) > 1:
                first_eff = topology_data['Efficiency'].iloc[0]
                last_eff = topology_data['Efficiency'].iloc[-1]
                efficiency_drop = first_eff - last_eff
                
                efficiency_data.append({
                    'Topology': topology,
                    'EfficiencyDrop': efficiency_drop,
                    'AverageEfficiency': topology_data['Efficiency'].mean()
                })
        
        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data)
            
            # Create bar plot
            x_pos = np.arange(len(eff_df))
            bars = ax.bar(x_pos, eff_df['EfficiencyDrop'], 
                         color=self.colors[:len(eff_df)], alpha=0.7)
            
            ax.set_xlabel('Network Topology')
            ax.set_ylabel('Efficiency Drop (%)')
            ax.set_title('Efficiency Loss with Scaling')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(eff_df['Topology'], rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom')
    
    def _plot_scalability_models(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot scalability model fitting"""
        if 'Processes' not in df.columns or 'ExecutionTime' not in df.columns:
            ax.text(0.5, 0.5, 'Scalability modeling data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Scalability Models')
            return
        
        # Fit Amdahl's law model
        for i, topology in enumerate(df['Topology'].unique()):
            topology_data = df[df['Topology'] == topology].sort_values('Processes')
            
            if len(topology_data) > 2:
                processes = topology_data['Processes'].values
                execution_times = topology_data['ExecutionTime'].values
                
                try:
                    # Fit simple scalability model
                    def model(p, a, b):
                        return a + b/p
                    
                    popt, _ = curve_fit(model, processes, execution_times,
                                      p0=[execution_times[0], execution_times[0]])
                    
                    # Plot fitted model
                    p_fit = np.linspace(processes.min(), processes.max(), 100)
                    t_fit = model(p_fit, *popt)
                    
                    ax.plot(p_fit, t_fit, color=self.colors[i], 
                           linestyle='--', alpha=0.7,
                           label=f'{topology} (model)')
                    
                    # Plot actual data
                    ax.scatter(processes, execution_times, 
                              color=self.colors[i], marker=self.markers[i],
                              label=topology, s=50)
                    
                except Exception as e:
                    print(f"Model fitting failed for {topology}: {e}")
        
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Scalability Model Fitting')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def generate_interactive_dashboard(self, output_dir: Path) -> None:
        """Generate interactive HTML dashboard"""
        try:
            import plotly.express as px
            import plotly.subplots as sp
            import plotly.graph_objects as go
            from plotly.offline import plot
            
            print("Generating interactive dashboard...")
            
            # Prepare data for dashboard
            dashboard_data = []
            for topology, benchmarks in self.data.items():
                for benchmark, df in benchmarks.items():
                    for _, row in df.iterrows():
                        data_point = {'Topology': topology, 'Benchmark': benchmark}
                        
                        # Copy all relevant columns
                        for col in df.columns:
                            if col in ['ExecutionTime', 'MessageSize', 'Processes', 
                                      'BandwidthUtilization', 'Efficiency', 'Speedup']:
                                data_point[col] = row[col]
                        
                        dashboard_data.append(data_point)
            
            if not dashboard_data:
                print("No data available for dashboard")
                return
            
            dash_df = pd.DataFrame(dashboard_data)
            
            # Create interactive plots
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=('Execution Time vs Message Size', 
                              'Bandwidth Utilization',
                              'Scalability Analysis', 
                              'Performance by Topology'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Plot 1: Execution Time
            if 'MessageSize' in dash_df.columns and 'ExecutionTime' in dash_df.columns:
                for topology in dash_df['Topology'].unique():
                    topology_data = dash_df[dash_df['Topology'] == topology]
                    fig.add_trace(
                        go.Scatter(x=topology_data['MessageSize'], 
                                 y=topology_data['ExecutionTime'],
                                 mode='markers', name=topology,
                                 hovertemplate='<b>%{x}</b><br>Time: %{y:.3f} ms'),
                        row=1, col=1
                    )
            
            # Plot 2: Bandwidth Utilization
            if 'BandwidthUtilization' in dash_df.columns:
                for topology in dash_df['Topology'].unique():
                    topology_data = dash_df[dash_df['Topology'] == topology]
                    fig.add_trace(
                        go.Box(y=topology_data['BandwidthUtilization'], 
                              name=topology),
                        row=1, col=2
                    )
            
            # Update layout
            fig.update_layout(height=800, title_text="MPI Performance Dashboard",
                            showlegend=True)
            
            # Save interactive dashboard
            dashboard_file = output_dir / "performance_dashboard.html"
            plot(fig, filename=str(dashboard_file), auto_open=False)
            
            print(f"Interactive dashboard saved: {dashboard_file}")
            
        except ImportError:
            print("Plotly not available, skipping interactive dashboard")
    
    def generate_comprehensive_report(self, output_dir: Path) -> None:
        """Generate comprehensive performance report"""
        print("Generating comprehensive performance report...")
        
        # Create all plots
        self.plot_performance_comparison(output_dir)
        self.plot_scalability_analysis(output_dir)
        self.generate_interactive_dashboard(output_dir)
        
        # Generate summary statistics
        self._generate_summary_statistics(output_dir)
        
        print(f"Comprehensive report generated in: {output_dir}")

    def _generate_summary_statistics(self, output_dir: Path) -> None:
        """Generate summary statistics file"""
        stats_data = []
        
        for topology, benchmarks in self.data.items():
            for benchmark, df in benchmarks.items():
                if 'ExecutionTime' in df.columns:
                    stats = {
                        'Topology': topology,
                        'Benchmark': benchmark,
                        'Samples': len(df),
                        'MeanTime': df['ExecutionTime'].mean(),
                        'StdTime': df['ExecutionTime'].std(),
                        'MinTime': df['ExecutionTime'].min(),
                        'MaxTime': df['ExecutionTime'].max(),
                        'CV': df['ExecutionTime'].std() / df['ExecutionTime'].mean()
                    }
                    
                    if 'BandwidthUtilization' in df.columns:
                        stats['AvgBandwidth'] = df['BandwidthUtilization'].mean()
                    
                    stats_data.append(stats)
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            stats_file = output_dir / "performance_statistics.csv"
            stats_df.to_csv(stats_file, index=False)
            print(f"Performance statistics saved: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate performance plots from MPI research results')
    parser.add_argument('results_dir', help='Directory containing benchmark results')
    parser.add_argument('-o', '--output', help='Output directory for plots', default='plots')
    parser.add_argument('--interactive', action='store_true', help='Generate interactive dashboard')
    parser.add_argument('--scalability', action='store_true', help='Generate scalability analysis')
    parser.add_argument('--comprehensive', action='store_true', help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Create plotter and load data
    plotter = PerformancePlotter(args.results_dir)
    plotter.load_all_results()
    
    # Generate requested plots
    if args.comprehensive:
        plotter.generate_comprehensive_report(output_dir)
    else:
        if not args.scalability:
            plotter.plot_performance_comparison(output_dir)
        if args.scalability or args.comprehensive:
            plotter.plot_scalability_analysis(output_dir)
        if args.interactive:
            plotter.generate_interactive_dashboard(output_dir)
    
    print("Performance plotting completed successfully")

if __name__ == "__main__":
    main()