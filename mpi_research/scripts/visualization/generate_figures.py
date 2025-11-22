#!/usr/bin/env python3

"""
Advanced Figure Generation for MPI Research
Creates publication-quality figures from analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use(['science', 'ieee', 'grid', 'std-colors'])

class FigureGenerator:
    """Generates publication-quality figures for MPI research"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.data = {}
        self.colors = sns.color_palette("colorblind")
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
    def load_analysis_results(self) -> None:
        """Load analysis results for figure generation"""
        print("Loading analysis results...")
        
        # Load performance statistics
        stats_files = list(self.results_dir.glob('**/performance_statistics.csv'))
        for stats_file in stats_files:
            try:
                df = pd.read_csv(stats_file)
                topology = stats_file.parent.name
                self.data[topology] = df
                print(f"  Loaded {topology}: {len(df)} data points")
            except Exception as e:
                print(f"  Error loading {stats_file}: {e}")
        
        # Load scaling metrics
        scaling_files = list(self.results_dir.glob('**/scaling_metrics.json'))
        for scaling_file in scaling_files:
            try:
                with open(scaling_file, 'r') as f:
                    scaling_data = json.load(f)
                # Store scaling data separately
                if 'scaling' not in self.data:
                    self.data['scaling'] = {}
                self.data['scaling'].update(scaling_data)
            except Exception as e:
                print(f"  Error loading {scaling_file}: {e}")
    
    def generate_publication_figures(self, output_dir: Path) -> None:
        """Generate all publication-quality figures"""
        print("Generating publication figures...")
        
        # 1. Performance comparison figure
        self._create_performance_comparison_figure(output_dir)
        
        # 2. Scalability analysis figure
        self._create_scalability_figure(output_dir)
        
        # 3. Topology efficiency figure
        self._create_efficiency_figure(output_dir)
        
        # 4. Statistical significance figure
        self._create_statistical_figure(output_dir)
        
        # 5. Comprehensive summary figure
        self._create_summary_figure(output_dir)
        
        print("Publication figures generated successfully")
    
    def _create_performance_comparison_figure(self, output_dir: Path) -> None:
        """Create performance comparison figure"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Prepare data
        plot_data = []
        for topology, df in self.data.items():
            if isinstance(df, pd.DataFrame):
                for _, row in df.iterrows():
                    plot_data.append({
                        'Topology': topology,
                        'Benchmark': row.get('Benchmark', 'unknown'),
                        'ExecutionTime': row.get('MeanTime', 0),
                        'Bandwidth': row.get('AvgBandwidth', 0),
                        'Messages': row.get('AvgMessages', 0)
                    })
        
        if not plot_data:
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        # Plot 1: Execution time comparison
        self._plot_execution_time_comparison(plot_df, ax1)
        
        # Plot 2: Bandwidth utilization
        self._plot_bandwidth_comparison(plot_df, ax2)
        
        # Plot 3: Performance distribution
        self._plot_performance_distribution(plot_df, ax3)
        
        # Plot 4: Message efficiency
        self._plot_message_efficiency(plot_df, ax4)
        
        plt.tight_layout()
        output_file = output_dir / "performance_comparison.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Performance comparison figure saved: {output_file}")
    
    def _plot_execution_time_comparison(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot execution time comparison"""
        # Calculate average execution time per topology
        avg_times = df.groupby('Topology')['ExecutionTime'].mean().sort_values()
        
        bars = ax.bar(range(len(avg_times)), avg_times.values, 
                     color=self.colors[:len(avg_times)], alpha=0.8)
        
        ax.set_xlabel('Network Topology')
        ax.set_ylabel('Average Execution Time (ms)')
        ax.set_title('Performance Comparison Across Topologies')
        ax.set_xticks(range(len(avg_times)))
        ax.set_xticklabels(avg_times.index, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_bandwidth_comparison(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot bandwidth utilization comparison"""
        if 'Bandwidth' not in df.columns:
            ax.text(0.5, 0.5, 'Bandwidth data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Bandwidth Utilization')
            return
        
        # Create box plot of bandwidth utilization
        plot_data = []
        for topology in df['Topology'].unique():
            topology_data = df[df['Topology'] == topology]
            for bw in topology_data['Bandwidth']:
                plot_data.append({'Topology': topology, 'Bandwidth': bw})
        
        if plot_data:
            bw_df = pd.DataFrame(plot_data)
            sns.boxplot(data=bw_df, x='Topology', y='Bandwidth', ax=ax)
            ax.set_title('Bandwidth Utilization Distribution')
            ax.set_ylabel('Bandwidth Utilization (%)')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_performance_distribution(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot performance distribution"""
        for i, topology in enumerate(df['Topology'].unique()):
            topology_data = df[df['Topology'] == topology]['ExecutionTime']
            
            # Create histogram with density curve
            n, bins, patches = ax.hist(topology_data, bins=20, alpha=0.6, 
                                      density=True, label=topology,
                                      color=self.colors[i])
        
        ax.set_xlabel('Execution Time (ms)')
        ax.set_ylabel('Density')
        ax.set_title('Performance Distribution')
        ax.legend()
    
    def _plot_message_efficiency(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot message efficiency analysis"""
        if 'Messages' not in df.columns:
            ax.text(0.5, 0.5, 'Message data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Message Efficiency')
            return
        
        # Calculate messages per time unit (efficiency metric)
        efficiency_data = []
        for topology in df['Topology'].unique():
            topology_data = df[df['Topology'] == topology]
            avg_messages = topology_data['Messages'].mean()
            avg_time = topology_data['ExecutionTime'].mean()
            
            if avg_time > 0:
                efficiency = avg_messages / avg_time
                efficiency_data.append({
                    'Topology': topology,
                    'Efficiency': efficiency
                })
        
        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data)
            eff_df = eff_df.sort_values('Efficiency', ascending=False)
            
            bars = ax.bar(range(len(eff_df)), eff_df['Efficiency'],
                         color=self.colors[:len(eff_df)], alpha=0.8)
            
            ax.set_xlabel('Network Topology')
            ax.set_ylabel('Messages per Millisecond')
            ax.set_title('Message Processing Efficiency')
            ax.set_xticks(range(len(eff_df)))
            ax.set_xticklabels(eff_df['Topology'], rotation=45, ha='right')
    
    def _create_scalability_figure(self, output_dir: Path) -> None:
        """Create scalability analysis figure"""
        if 'scaling' not in self.data:
            print("No scaling data available for figure generation")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        scaling_data = self.data['scaling']
        
        # Plot 1: Strong scaling efficiency
        self._plot_strong_scaling_efficiency(scaling_data, ax1)
        
        # Plot 2: Weak scaling performance
        self._plot_weak_scaling_performance(scaling_data, ax2)
        
        # Plot 3: Efficiency drop analysis
        self._plot_efficiency_drop(scaling_data, ax3)
        
        # Plot 4: Amdahl's law fitting
        self._plot_amdahl_analysis(scaling_data, ax4)
        
        plt.tight_layout()
        output_file = output_dir / "scalability_analysis.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Scalability analysis figure saved: {output_file}")
    
    def _plot_strong_scaling_efficiency(self, scaling_data: Dict, ax: plt.Axes) -> None:
        """Plot strong scaling efficiency"""
        # Extract efficiency data
        efficiency_data = []
        for topology, metrics in scaling_data.items():
            if 'efficiency_at_scale' in metrics:
                efficiency = metrics['efficiency_at_scale']['efficiency']
                processes = metrics['efficiency_at_scale']['max_processes']
                efficiency_data.append({
                    'Topology': topology,
                    'Processes': processes,
                    'Efficiency': efficiency
                })
        
        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data)
            
            # Create bar plot
            bars = ax.bar(range(len(eff_df)), eff_df['Efficiency'],
                         color=self.colors[:len(eff_df)], alpha=0.8)
            
            ax.set_xlabel('Network Topology')
            ax.set_ylabel('Parallel Efficiency (%)')
            ax.set_title('Strong Scaling Efficiency at Maximum Scale')
            ax.set_xticks(range(len(eff_df)))
            ax.set_xticklabels(eff_df['Topology'], rotation=45, ha='right')
            
            # Add ideal efficiency line
            ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, 
                      label='Ideal Efficiency')
            ax.legend()
    
    def _plot_weak_scaling_performance(self, scaling_data: Dict, ax: plt.Axes) -> None:
        """Plot weak scaling performance"""
        # This would require specific weak scaling data
        # For now, create a placeholder
        ax.text(0.5, 0.5, 'Weak scaling analysis\nwould go here', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Weak Scaling Performance')
        ax.set_xlabel('Problem Size per Process')
        ax.set_ylabel('Execution Time')
    
    def _plot_efficiency_drop(self, scaling_data: Dict, ax: plt.Axes) -> None:
        """Plot efficiency drop analysis"""
        efficiency_drop_data = []
        for topology, metrics in scaling_data.items():
            if 'efficiency_drop' in metrics:
                drop = metrics['efficiency_drop']['drop']
                efficiency_drop_data.append({
                    'Topology': topology,
                    'EfficiencyDrop': drop
                })
        
        if efficiency_drop_data:
            drop_df = pd.DataFrame(efficiency_drop_data)
            drop_df = drop_df.sort_values('EfficiencyDrop')
            
            bars = ax.bar(range(len(drop_df)), drop_df['EfficiencyDrop'],
                         color=self.colors[:len(drop_df)], alpha=0.8)
            
            ax.set_xlabel('Network Topology')
            ax.set_ylabel('Efficiency Drop (%)')
            ax.set_title('Efficiency Loss with Scaling')
            ax.set_xticks(range(len(drop_df)))
            ax.set_xticklabels(drop_df['Topology'], rotation=45, ha='right')
    
    def _plot_amdahl_analysis(self, scaling_data: Dict, ax: plt.Axes) -> None:
        """Plot Amdahl's law analysis"""
        amdahl_data = []
        for topology, metrics in scaling_data.items():
            amdahl_analysis = metrics.get('amdahl_analysis', {})
            if 'serial_fraction' in amdahl_analysis:
                serial_frac = amdahl_analysis['serial_fraction']
                max_speedup = amdahl_analysis.get('max_speedup', 0)
                
                amdahl_data.append({
                    'Topology': topology,
                    'SerialFraction': serial_frac,
                    'MaxSpeedup': max_speedup
                })
        
        if amdahl_data:
            amdahl_df = pd.DataFrame(amdahl_data)
            
            # Create scatter plot
            scatter = ax.scatter(amdahl_df['SerialFraction'], 
                               amdahl_df['MaxSpeedup'],
                               s=100, alpha=0.7,
                               c=range(len(amdahl_df)), 
                               cmap='viridis')
            
            ax.set_xlabel('Serial Fraction')
            ax.set_ylabel('Maximum Theoretical Speedup')
            ax.set_title("Amdahl's Law Analysis")
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Add topology labels
            for i, row in amdahl_df.iterrows():
                ax.annotate(row['Topology'], 
                           (row['SerialFraction'], row['MaxSpeedup']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
    
    def _create_efficiency_figure(self, output_dir: Path) -> None:
        """Create topology efficiency figure"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Prepare efficiency data
        efficiency_metrics = self._calculate_efficiency_metrics()
        
        if efficiency_metrics:
            eff_df = pd.DataFrame(efficiency_metrics)
            
            # Plot 1: Overall efficiency ranking
            self._plot_efficiency_ranking(eff_df, ax1)
            
            # Plot 2: Efficiency vs performance trade-off
            self._plot_efficiency_tradeoff(eff_df, ax2)
        
        plt.tight_layout()
        output_file = output_dir / "topology_efficiency.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Topology efficiency figure saved: {output_file}")
    
    def _calculate_efficiency_metrics(self) -> List[Dict]:
        """Calculate comprehensive efficiency metrics"""
        efficiency_data = []
        
        for topology, df in self.data.items():
            if isinstance(df, pd.DataFrame):
                # Calculate various efficiency metrics
                avg_time = df['ExecutionTime'].mean() if 'ExecutionTime' in df.columns else 0
                avg_bw = df['Bandwidth'].mean() if 'Bandwidth' in df.columns else 0
                cv = df['ExecutionTime'].std() / df['ExecutionTime'].mean() if 'ExecutionTime' in df.columns else 0
                
                efficiency_score = (avg_bw / 100) * (1 / (1 + cv)) if avg_time > 0 else 0
                
                efficiency_data.append({
                    'Topology': topology,
                    'AverageTime': avg_time,
                    'AverageBandwidth': avg_bw,
                    'CoefficientOfVariation': cv,
                    'EfficiencyScore': efficiency_score
                })
        
        return efficiency_data
    
    def _plot_efficiency_ranking(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot efficiency ranking"""
        df_sorted = df.sort_values('EfficiencyScore', ascending=False)
        
        bars = ax.bar(range(len(df_sorted)), df_sorted['EfficiencyScore'],
                     color=self.colors[:len(df_sorted)], alpha=0.8)
        
        ax.set_xlabel('Network Topology')
        ax.set_ylabel('Efficiency Score')
        ax.set_title('Topology Efficiency Ranking')
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted['Topology'], rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_efficiency_tradeoff(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot efficiency vs performance trade-off"""
        scatter = ax.scatter(df['AverageTime'], df['EfficiencyScore'],
                           s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
        
        ax.set_xlabel('Average Execution Time (ms)')
        ax.set_ylabel('Efficiency Score')
        ax.set_title('Performance vs Efficiency Trade-off')
        ax.grid(True, alpha=0.3)
        
        # Add topology labels
        for i, row in df.iterrows():
            ax.annotate(row['Topology'], 
                       (row['AverageTime'], row['EfficiencyScore']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8)
    
    def _create_statistical_figure(self, output_dir: Path) -> None:
        """Create statistical significance figure"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Load statistical results if available
        stats_files = list(self.results_dir.glob('**/statistical_analysis_results.json'))
        if stats_files:
            try:
                with open(stats_files[0], 'r') as f:
                    stats_data = json.load(f)
                
                # Plot statistical significance
                self._plot_statistical_significance(stats_data, ax1)
                
                # Plot effect sizes
                self._plot_effect_sizes(stats_data, ax2)
                
            except Exception as e:
                print(f"Error loading statistical data: {e}")
                ax1.text(0.5, 0.5, 'Statistical data\nnot available', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax2.text(0.5, 0.5, 'Statistical data\nnot available', 
                        ha='center', va='center', transform=ax2.transAxes)
        else:
            ax1.text(0.5, 0.5, 'Run statistical analysis first', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'Run statistical analysis first', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        ax1.set_title('Statistical Significance')
        ax2.set_title('Effect Sizes')
        
        plt.tight_layout()
        output_file = output_dir / "statistical_analysis.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Statistical analysis figure saved: {output_file}")
    
    def _plot_statistical_significance(self, stats_data: Dict, ax: plt.Axes) -> None:
        """Plot statistical significance results"""
        significance_data = stats_data.get('significance_tests', {})
        
        significant_comparisons = []
        for comparison, benchmarks in significance_data.items():
            significant_count = 0
            total_count = 0
            
            for benchmark, tests in benchmarks.items():
                if tests.get('t_test', {}).get('significant', False):
                    significant_count += 1
                total_count += 1
            
            if total_count > 0:
                significant_comparisons.append({
                    'Comparison': comparison,
                    'SignificantRatio': significant_count / total_count
                })
        
        if significant_comparisons:
            sig_df = pd.DataFrame(significant_comparisons)
            sig_df = sig_df.sort_values('SignificantRatio', ascending=False)
            
            bars = ax.bar(range(len(sig_df)), sig_df['SignificantRatio'],
                         color=self.colors[:len(sig_df)], alpha=0.8)
            
            ax.set_xlabel('Topology Comparison')
            ax.set_ylabel('Proportion of Significant Differences')
            ax.set_xticks(range(len(sig_df)))
            ax.set_xticklabels(sig_df['Comparison'], rotation=45, ha='right')
    
    def _plot_effect_sizes(self, stats_data: Dict, ax: plt.Axes) -> None:
        """Plot effect sizes from statistical analysis"""
        significance_data = stats_data.get('significance_tests', {})
        
        effect_sizes = []
        for comparison, benchmarks in significance_data.items():
            for benchmark, tests in benchmarks.items():
                effect_size = tests.get('effect_size', {}).get('cohens_d', 0)
                if effect_size != 0:
                    effect_sizes.append({
                        'Comparison': comparison,
                        'Benchmark': benchmark,
                        'EffectSize': effect_size
                    })
        
        if effect_sizes:
            effect_df = pd.DataFrame(effect_sizes)
            
            # Create box plot of effect sizes by comparison
            plot_data = []
            for comparison in effect_df['Comparison'].unique():
                comp_data = effect_df[effect_df['Comparison'] == comparison]
                for effect_size in comp_data['EffectSize']:
                    plot_data.append({
                        'Comparison': comparison,
                        'EffectSize': effect_size
                    })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                sns.boxplot(data=plot_df, x='Comparison', y='EffectSize', ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_ylabel("Cohen's d Effect Size")
    
    def _create_summary_figure(self, output_dir: Path) -> None:
        """Create comprehensive summary figure"""
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid specification
        gs = fig.add_gridspec(3, 4)
        
        ax1 = fig.add_subplot(gs[0, :2])  # Performance summary
        ax2 = fig.add_subplot(gs[0, 2:])  # Efficiency summary
        ax3 = fig.add_subplot(gs[1, :2])  # Scalability summary
        ax4 = fig.add_subplot(gs[1, 2:])  # Statistical summary
        ax5 = fig.add_subplot(gs[2, :])   # Recommendations
        
        # Plot 1: Performance summary
        self._plot_performance_summary(ax1)
        
        # Plot 2: Efficiency summary
        self._plot_efficiency_summary(ax2)
        
        # Plot 3: Scalability summary
        self._plot_scalability_summary(ax3)
        
        # Plot 4: Statistical summary
        self._plot_statistical_summary(ax4)
        
        # Plot 5: Recommendations
        self._plot_recommendations(ax5)
        
        plt.tight_layout()
        output_file = output_dir / "comprehensive_summary.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Comprehensive summary figure saved: {output_file}")
    
    def _plot_performance_summary(self, ax: plt.Axes) -> None:
        """Plot performance summary"""
        # Calculate performance metrics
        performance_data = []
        for topology, df in self.data.items():
            if isinstance(df, pd.DataFrame):
                avg_time = df['ExecutionTime'].mean() if 'ExecutionTime' in df.columns else 0
                performance_data.append({
                    'Topology': topology,
                    'Performance': 1 / avg_time if avg_time > 0 else 0  # Performance as inverse of time
                })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            perf_df = perf_df.sort_values('Performance', ascending=False)
            
            bars = ax.bar(range(len(perf_df)), perf_df['Performance'],
                         color=self.colors[:len(perf_df)], alpha=0.8)
            
            ax.set_xlabel('Network Topology')
            ax.set_ylabel('Performance (1/time)')
            ax.set_title('Overall Performance Ranking')
            ax.set_xticks(range(len(perf_df)))
            ax.set_xticklabels(perf_df['Topology'], rotation=45, ha='right')
    
    def _plot_efficiency_summary(self, ax: plt.Axes) -> None:
        """Plot efficiency summary"""
        efficiency_data = self._calculate_efficiency_metrics()
        
        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data)
            eff_df = eff_df.sort_values('EfficiencyScore', ascending=False)
            
            # Create radar chart placeholder (simplified)
            categories = ['Performance', 'Efficiency', 'Scalability', 'Stability']
            values = [0.7, 0.8, 0.6, 0.9]  # Placeholder values
            
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title('Efficiency Profile')
            ax.grid(True)
    
    def _plot_scalability_summary(self, ax: plt.Axes) -> None:
        """Plot scalability summary"""
        if 'scaling' in self.data:
            scaling_data = self.data['scaling']
            
            scalability_scores = []
            for topology, metrics in scaling_data.items():
                efficiency = metrics.get('efficiency_at_scale', {}).get('efficiency', 0)
                scalability_scores.append({
                    'Topology': topology,
                    'ScalabilityScore': efficiency / 100  # Normalize to 0-1
                })
            
            if scalability_scores:
                scal_df = pd.DataFrame(scalability_scores)
                scal_df = scal_df.sort_values('ScalabilityScore', ascending=False)
                
                bars = ax.bar(range(len(scal_df)), scal_df['ScalabilityScore'],
                             color=self.colors[:len(scal_df)], alpha=0.8)
                
                ax.set_xlabel('Network Topology')
                ax.set_ylabel('Scalability Score')
                ax.set_title('Scalability Ranking')
                ax.set_xticks(range(len(scal_df)))
                ax.set_xticklabels(scal_df['Topology'], rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'Scalability data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_statistical_summary(self, ax: plt.Axes) -> None:
        """Plot statistical summary"""
        ax.text(0.5, 0.5, 'Statistical significance\nsummary would go here', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Statistical Significance Summary')
        ax.grid(False)
    
    def _plot_recommendations(self, ax: plt.Axes) -> None:
        """Plot recommendations summary"""
        # Generate recommendations based on analysis
        recommendations = self._generate_recommendations()
        
        ax.axis('off')
        ax.set_title('Key Recommendations', fontsize=14, fontweight='bold')
        
        # Display recommendations as text
        y_position = 0.9
        for i, rec in enumerate(recommendations[:5]):  # Show top 5 recommendations
            ax.text(0.05, y_position, f"{i+1}. {rec}", 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top')
            y_position -= 0.15
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Basic recommendations based on available data
        if self.data:
            # Find best performing topology
            best_topology = None
            best_performance = float('inf')
            
            for topology, df in self.data.items():
                if isinstance(df, pd.DataFrame) and 'ExecutionTime' in df.columns:
                    avg_time = df['ExecutionTime'].mean()
                    if avg_time < best_performance:
                        best_performance = avg_time
                        best_topology = topology
            
            if best_topology:
                recommendations.append(f"Use {best_topology} topology for best overall performance")
            
            # Check for scalability data
            if 'scaling' in self.data:
                scaling_data = self.data['scaling']
                best_scaling = None
                best_efficiency = 0
                
                for topology, metrics in scaling_data.items():
                    efficiency = metrics.get('efficiency_at_scale', {}).get('efficiency', 0)
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_scaling = topology
                
                if best_scaling and best_scaling != best_topology:
                    recommendations.append(f"Use {best_scaling} for better scalability at large scales")
        
        # Add general recommendations
        recommendations.extend([
            "Consider hybrid approaches for mixed workloads",
            "Optimize message sizes based on network characteristics",
            "Use topology-aware algorithms for large-scale deployments",
            "Monitor network utilization to avoid congestion",
            "Validate performance on target hardware configuration"
        ])
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description='Generate publication-quality figures from MPI research results')
    parser.add_argument('results_dir', help='Directory containing analysis results')
    parser.add_argument('-o', '--output', help='Output directory for figures', default='figures')
    parser.add_argument('--all', action='store_true', help='Generate all figures')
    parser.add_argument('--performance', action='store_true', help='Generate performance figures')
    parser.add_argument('--scalability', action='store_true', help='Generate scalability figures')
    parser.add_argument('--efficiency', action='store_true', help='Generate efficiency figures')
    parser.add_argument('--statistical', action='store_true', help='Generate statistical figures')
    parser.add_argument('--summary', action='store_true', help='Generate summary figure')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Create figure generator
    generator = FigureGenerator(args.results_dir)
    generator.load_analysis_results()
    
    # Generate requested figures
    if args.all or not any([args.performance, args.scalability, args.efficiency, args.statistical, args.summary]):
        generator.generate_publication_figures(output_dir)
    else:
        if args.performance:
            generator._create_performance_comparison_figure(output_dir)
        if args.scalability:
            generator._create_scalability_figure(output_dir)
        if args.efficiency:
            generator._create_efficiency_figure(output_dir)
        if args.statistical:
            generator._create_statistical_figure(output_dir)
        if args.summary:
            generator._create_summary_figure(output_dir)
    
    print("Figure generation completed successfully")

if __name__ == "__main__":
    main()