#!/usr/bin/env python3

"""
Advanced Scalability Analysis for MPI Research
Analyzes strong and weak scaling properties of MPI collective operations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import curve_fit
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use(['science', 'ieee', 'grid'])

class ScalabilityAnalyzer:
    """Advanced scalability analysis for MPI collective operations"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.scaling_data = {}
        self.models = {}
        
    def load_scaling_data(self) -> None:
        """Load scalability results from benchmark data"""
        print("Loading scalability data...")
        
        csv_files = list(self.results_dir.glob('**/*.csv'))
        
        for csv_file in csv_files:
            try:
                # Check if this is a scalability benchmark
                if any(keyword in csv_file.stem.lower() for keyword in 
                      ['scalability', 'scaling', 'weak', 'strong', 'process']):
                    
                    df = pd.read_csv(csv_file)
                    
                    # Extract topology from path
                    topology = csv_file.parent.name
                    if topology not in self.scaling_data:
                        self.scaling_data[topology] = {}
                    
                    benchmark_name = csv_file.stem
                    self.scaling_data[topology][benchmark_name] = df
                    
                    print(f"  Loaded {topology}/{benchmark_name}: {len(df)} data points")
                    
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")
    
    def analyze_strong_scaling(self, output_dir: Path) -> None:
        """Perform comprehensive strong scaling analysis"""
        print("Performing strong scaling analysis...")
        
        strong_scaling_data = self._extract_strong_scaling_data()
        
        if not strong_scaling_data:
            print("No strong scaling data found")
            return
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strong Scaling Analysis of MPI Collective Operations', 
                    fontsize=16, fontweight='bold')
        
        # Convert to DataFrame
        ss_df = pd.DataFrame(strong_scaling_data)
        
        # Plot 1: Execution Time vs Processes
        self._plot_strong_scaling_time(ss_df, axes[0, 0])
        
        # Plot 2: Speedup Analysis
        self._plot_speedup_analysis(ss_df, axes[0, 1])
        
        # Plot 3: Efficiency Analysis
        self._plot_efficiency_analysis(ss_df, axes[1, 0])
        
        # Plot 4: Scaling Models
        self._plot_scaling_models(ss_df, axes[1, 1])
        
        plt.tight_layout()
        output_file = output_dir / "strong_scaling_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate scaling models and metrics
        scaling_metrics = self._calculate_scaling_metrics(ss_df)
        self._save_scaling_metrics(scaling_metrics, output_dir)
        
        print(f"Strong scaling analysis saved: {output_file}")
    
    def _extract_strong_scaling_data(self) -> List[Dict]:
        """Extract strong scaling data from loaded results"""
        strong_scaling_data = []
        
        for topology, benchmarks in self.scaling_data.items():
            for benchmark, df in benchmarks.items():
                # Look for strong scaling data (fixed problem size, varying processes)
                if 'Processes' in df.columns and 'ExecutionTime' in df.columns:
                    
                    # Find base case (smallest number of processes)
                    base_processes = df['Processes'].min()
                    base_data = df[df['Processes'] == base_processes]
                    
                    if len(base_data) > 0:
                        base_time = base_data['ExecutionTime'].mean()
                        
                        for _, row in df.iterrows():
                            processes = row['Processes']
                            execution_time = row['ExecutionTime']
                            
                            # Calculate speedup and efficiency
                            speedup = base_time / execution_time if execution_time > 0 else 0
                            efficiency = (speedup / processes) * 100 if processes > 0 else 0
                            
                            strong_scaling_data.append({
                                'Topology': topology,
                                'Benchmark': benchmark,
                                'Processes': processes,
                                'ExecutionTime': execution_time,
                                'Speedup': speedup,
                                'Efficiency': efficiency,
                                'BaseProcesses': base_processes,
                                'BaseTime': base_time
                            })
        
        return strong_scaling_data
    
    def _plot_strong_scaling_time(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot execution time vs number of processes"""
        colors = sns.color_palette("husl", len(df['Topology'].unique()))
        
        for i, topology in enumerate(df['Topology'].unique()):
            topology_data = df[df['Topology'] == topology]
            
            # Group by process count
            grouped = topology_data.groupby('Processes').agg({
                'ExecutionTime': ['mean', 'std']
            }).reset_index()
            
            grouped.columns = ['Processes', 'MeanTime', 'StdTime']
            
            # Plot with error bars
            ax.errorbar(grouped['Processes'], grouped['MeanTime'],
                       yerr=grouped['StdTime'], label=topology,
                       color=colors[i], marker='o', markersize=6,
                       capsize=3, linewidth=2)
        
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Strong Scaling: Execution Time')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_speedup_analysis(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot speedup analysis"""
        colors = sns.color_palette("husl", len(df['Topology'].unique()))
        
        for i, topology in enumerate(df['Topology'].unique()):
            topology_data = df[df['Topology'] == topology]
            
            # Group by process count
            grouped = topology_data.groupby('Processes').agg({
                'Speedup': 'mean'
            }).reset_index()
            
            # Plot actual speedup
            ax.plot(grouped['Processes'], grouped['Speedup'],
                   label=topology, color=colors[i], marker='s',
                   markersize=6, linewidth=2)
        
        # Plot ideal speedup
        max_processes = df['Processes'].max()
        ideal_x = np.linspace(1, max_processes, 100)
        ideal_y = ideal_x
        ax.plot(ideal_x, ideal_y, 'k--', label='Ideal', alpha=0.7)
        
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Speedup')
        ax.set_title('Strong Scaling: Speedup Analysis')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_efficiency_analysis(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot parallel efficiency"""
        colors = sns.color_palette("husl", len(df['Topology'].unique()))
        
        for i, topology in enumerate(df['Topology'].unique()):
            topology_data = df[df['Topology'] == topology]
            
            # Group by process count
            grouped = topology_data.groupby('Processes').agg({
                'Efficiency': 'mean'
            }).reset_index()
            
            ax.plot(grouped['Processes'], grouped['Efficiency'],
                   label=topology, color=colors[i], marker='^',
                   markersize=6, linewidth=2)
        
        # Plot ideal efficiency
        max_processes = df['Processes'].max()
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Ideal')
        
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Parallel Efficiency (%)')
        ax.set_title('Strong Scaling: Efficiency Analysis')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_scaling_models(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot scalability model fitting"""
        colors = sns.color_palette("husl", len(df['Topology'].unique()))
        
        for i, topology in enumerate(df['Topology'].unique()):
            topology_data = df[df['Topology'] == topology].sort_values('Processes')
            
            if len(topology_data) > 2:
                processes = topology_data['Processes'].values
                speedup = topology_data['Speedup'].values
                
                try:
                    # Fit Amdahl's law: S(p) = p / (1 + α(p - 1))
                    def amdahl_law(p, alpha):
                        return p / (1 + alpha * (p - 1))
                    
                    popt, pcov = curve_fit(amdahl_law, processes, speedup,
                                         bounds=(0, 1))
                    
                    alpha = popt[0]
                    
                    # Plot fitted model
                    p_fit = np.linspace(processes.min(), processes.max(), 100)
                    s_fit = amdahl_law(p_fit, alpha)
                    
                    ax.plot(p_fit, s_fit, color=colors[i], linestyle='--',
                           alpha=0.7, label=f'{topology} (α={alpha:.3f})')
                    
                    # Plot actual data
                    ax.scatter(processes, speedup, color=colors[i], 
                              marker='o', s=50)
                    
                except Exception as e:
                    print(f"Amdahl model fitting failed for {topology}: {e}")
        
        # Plot ideal scaling
        max_processes = df['Processes'].max()
        ideal_p = np.linspace(1, max_processes, 100)
        ideal_s = ideal_p
        ax.plot(ideal_p, ideal_s, 'k-', label='Ideal', alpha=0.5)
        
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Speedup')
        ax.set_title('Scaling Model Fitting (Amdahl\'s Law)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _calculate_scaling_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive scaling metrics"""
        scaling_metrics = {}
        
        for topology in df['Topology'].unique():
            topology_data = df[df['Topology'] == topology].sort_values('Processes')
            
            if len(topology_data) > 1:
                metrics = {
                    'topology': topology,
                    'process_range': {
                        'min': topology_data['Processes'].min(),
                        'max': topology_data['Processes'].max()
                    },
                    'efficiency_at_scale': {
                        'max_processes': topology_data['Processes'].max(),
                        'efficiency': topology_data[topology_data['Processes'] == 
                                                  topology_data['Processes'].max()]['Efficiency'].values[0]
                    },
                    'efficiency_drop': {
                        'min_eff': topology_data['Efficiency'].min(),
                        'max_eff': topology_data['Efficiency'].max(),
                        'drop': topology_data['Efficiency'].max() - topology_data['Efficiency'].min()
                    }
                }
                
                # Fit Amdahl's law to estimate serial fraction
                try:
                    processes = topology_data['Processes'].values
                    speedup = topology_data['Speedup'].values
                    
                    def amdahl_law(p, alpha):
                        return p / (1 + alpha * (p - 1))
                    
                    popt, _ = curve_fit(amdahl_law, processes, speedup, bounds=(0, 1))
                    metrics['amdahl_analysis'] = {
                        'serial_fraction': popt[0],
                        'parallel_fraction': 1 - popt[0],
                        'max_speedup': 1 / popt[0] if popt[0] > 0 else float('inf')
                    }
                except:
                    metrics['amdahl_analysis'] = {'error': 'Model fitting failed'}
                
                scaling_metrics[topology] = metrics
        
        return scaling_metrics
    
    def _save_scaling_metrics(self, scaling_metrics: Dict, output_dir: Path) -> None:
        """Save scaling metrics to JSON file"""
        metrics_file = output_dir / "scaling_metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(scaling_metrics, f, indent=2)
        
        print(f"Scaling metrics saved: {metrics_file}")
        
        # Also create a summary CSV
        summary_data = []
        for topology, metrics in scaling_metrics.items():
            summary_data.append({
                'Topology': topology,
                'MinProcesses': metrics['process_range']['min'],
                'MaxProcesses': metrics['process_range']['max'],
                'EfficiencyAtMaxScale': metrics['efficiency_at_scale']['efficiency'],
                'EfficiencyDrop': metrics['efficiency_drop']['drop'],
                'SerialFraction': metrics.get('amdahl_analysis', {}).get('serial_fraction', 0),
                'MaxTheoreticalSpeedup': metrics.get('amdahl_analysis', {}).get('max_speedup', 0)
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = output_dir / "scaling_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"Scaling summary saved: {summary_file}")
    
    def analyze_weak_scaling(self, output_dir: Path) -> None:
        """Perform weak scaling analysis"""
        print("Performing weak scaling analysis...")
        
        weak_scaling_data = self._extract_weak_scaling_data()
        
        if not weak_scaling_data:
            print("No weak scaling data found")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Weak Scaling Analysis of MPI Collective Operations', 
                    fontsize=16, fontweight='bold')
        
        ws_df = pd.DataFrame(weak_scaling_data)
        
        # Plot 1: Execution Time vs Problem Size per Process
        self._plot_weak_scaling_time(ws_df, axes[0])
        
        # Plot 2: Weak Scaling Efficiency
        self._plot_weak_scaling_efficiency(ws_df, axes[1])
        
        plt.tight_layout()
        output_file = output_dir / "weak_scaling_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Weak scaling analysis saved: {output_file}")
    
    def _extract_weak_scaling_data(self) -> List[Dict]:
        """Extract weak scaling data (fixed work per process)"""
        weak_scaling_data = []
        
        for topology, benchmarks in self.scaling_data.items():
            for benchmark, df in benchmarks.items():
                # Look for weak scaling data (varying problem size with processes)
                if 'Processes' in df.columns and 'ProblemSize' in df.columns:
                    
                    for _, row in df.iterrows():
                        weak_scaling_data.append({
                            'Topology': topology,
                            'Benchmark': benchmark,
                            'Processes': row['Processes'],
                            'ProblemSize': row['ProblemSize'],
                            'ExecutionTime': row.get('ExecutionTime', 0),
                            'Efficiency': row.get('Efficiency', 0)
                        })
        
        return weak_scaling_data
    
    def _plot_weak_scaling_time(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot execution time for weak scaling"""
        colors = sns.color_palette("husl", len(df['Topology'].unique()))
        
        for i, topology in enumerate(df['Topology'].unique()):
            topology_data = df[df['Topology'] == topology]
            
            # Group by processes
            grouped = topology_data.groupby('Processes').agg({
                'ExecutionTime': 'mean',
                'ProblemSize': 'mean'
            }).reset_index()
            
            ax.plot(grouped['Processes'], grouped['ExecutionTime'],
                   label=topology, color=colors[i], marker='o',
                   markersize=6, linewidth=2)
        
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Weak Scaling: Execution Time')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_weak_scaling_efficiency(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot weak scaling efficiency"""
        colors = sns.color_palette("husl", len(df['Topology'].unique()))
        
        for i, topology in enumerate(df['Topology'].unique()):
            topology_data = df[df['Topology'] == topology]
            
            # Group by processes
            grouped = topology_data.groupby('Processes').agg({
                'Efficiency': 'mean'
            }).reset_index()
            
            ax.plot(grouped['Processes'], grouped['Efficiency'],
                   label=topology, color=colors[i], marker='s',
                   markersize=6, linewidth=2)
        
        # Ideal efficiency line
        max_processes = df['Processes'].max()
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Ideal')
        
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Parallel Efficiency (%)')
        ax.set_title('Weak Scaling: Efficiency')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def generate_scalability_report(self, output_dir: Path) -> None:
        """Generate comprehensive scalability report"""
        print("Generating comprehensive scalability report...")
        
        # Perform all analyses
        self.analyze_strong_scaling(output_dir)
        self.analyze_weak_scaling(output_dir)
        
        # Generate summary insights
        self._generate_scalability_insights(output_dir)
        
        print(f"Comprehensive scalability report generated in: {output_dir}")
    
    def _generate_scalability_insights(self, output_dir: Path) -> None:
        """Generate scalability insights and recommendations"""
        insights = {
            "analysis_summary": {},
            "topology_comparison": {},
            "recommendations": []
        }
        
        # Load scaling metrics if available
        metrics_file = output_dir / "scaling_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                scaling_metrics = json.load(f)
            
            insights["analysis_summary"] = scaling_metrics
            
            # Generate recommendations
            best_efficiency = 0
            best_topology = None
            
            for topology, metrics in scaling_metrics.items():
                efficiency = metrics['efficiency_at_scale']['efficiency']
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_topology = topology
            
            if best_topology:
                insights["recommendations"].append(
                    f"Best scaling topology: {best_topology} "
                    f"(efficiency: {best_efficiency:.1f}% at max scale)"
                )
            
            # Additional insights
            for topology, metrics in scaling_metrics.items():
                serial_frac = metrics.get('amdahl_analysis', {}).get('serial_fraction', 0)
                if serial_frac > 0.1:
                    insights["recommendations"].append(
                        f"High serial fraction ({serial_frac:.3f}) in {topology} "
                        f"limits scalability to {1/serial_frac:.1f}x"
                    )
        
        # Save insights
        insights_file = output_dir / "scalability_insights.json"
        with open(insights_file, 'w') as f:
            json.dump(insights, f, indent=2)
        
        print(f"Scalability insights saved: {insights_file}")

def main():
    parser = argparse.ArgumentParser(description='Perform scalability analysis of MPI collective operations')
    parser.add_argument('results_dir', help='Directory containing benchmark results')
    parser.add_argument('-o', '--output', help='Output directory for analysis', default='scalability_analysis')
    parser.add_argument('--strong', action='store_true', help='Perform strong scaling analysis')
    parser.add_argument('--weak', action='store_true', help='Perform weak scaling analysis')
    parser.add_argument('--comprehensive', action='store_true', help='Perform comprehensive analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Create analyzer and load data
    analyzer = ScalabilityAnalyzer(args.results_dir)
    analyzer.load_scaling_data()
    
    # Perform requested analyses
    if args.comprehensive:
        analyzer.generate_scalability_report(output_dir)
    else:
        if args.strong or not args.weak:
            analyzer.analyze_strong_scaling(output_dir)
        if args.weak:
            analyzer.analyze_weak_scaling(output_dir)
    
    print("Scalability analysis completed successfully")

if __name__ == "__main__":
    main()