#!/usr/bin/env python3

"""
Advanced Comparison Visualization for MPI Research
Creates detailed comparison plots between different configurations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use(['science', 'ieee', 'grid'])

class ComparisonVisualizer:
    """Creates detailed comparison visualizations for MPI research"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.data = {}
        self.colors = sns.color_palette("Set2")
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
    def load_comparison_data(self) -> None:
        """Load data for comparison visualization"""
        print("Loading comparison data...")
        
        # Load all CSV files
        csv_files = list(self.results_dir.glob('**/*.csv'))
        
        for csv_file in csv_files:
            try:
                # Extract comparison context from path
                rel_path = csv_file.relative_to(self.results_dir)
                
                if len(rel_path.parts) >= 2:
                    category = rel_path.parts[0]  # e.g., 'fat_tree', 'torus_2d'
                    benchmark = rel_path.stem
                    
                    if category not in self.data:
                        self.data[category] = {}
                    
                    df = pd.read_csv(csv_file)
                    self.data[category][benchmark] = df
                    
                    print(f"  Loaded {category}/{benchmark}: {len(df)} rows")
                    
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")
    
    def create_comprehensive_comparisons(self, output_dir: Path) -> None:
        """Create comprehensive comparison visualizations"""
        print("Creating comprehensive comparisons...")
        
        # 1. Cross-topology performance comparison
        self._create_topology_comparison(output_dir)
        
        # 2. Algorithm efficiency comparison
        self._create_algorithm_comparison(output_dir)
        
        # 3. Scaling behavior comparison
        self._create_scaling_comparison(output_dir)
        
        # 4. Statistical significance comparison
        self._create_statistical_comparison(output_dir)
        
        # 5. Performance profile comparison
        self._create_performance_profile(output_dir)
        
        print("Comprehensive comparisons created successfully")
    
    def _create_topology_comparison(self, output_dir: Path) -> None:
        """Create cross-topology performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Topology Performance Comparison', 
                    fontsize=16, fontweight='bold')
        
        # Prepare comparison data
        comparison_data = []
        for topology, benchmarks in self.data.items():
            for benchmark, df in benchmarks.items():
                if 'ExecutionTime' in df.columns:
                    for _, row in df.iterrows():
                        comparison_data.append({
                            'Topology': topology,
                            'Benchmark': benchmark,
                            'ExecutionTime': row['ExecutionTime'],
                            'MessageSize': row.get('MessageSize', 0),
                            'Processes': row.get('Processes', 1)
                        })
        
        if not comparison_data:
            return
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Plot 1: Performance distribution by topology
        self._plot_topology_distribution(comp_df, axes[0, 0])
        
        # Plot 2: Performance vs message size
        self._plot_message_size_comparison(comp_df, axes[0, 1])
        
        # Plot 3: Performance ranking
        self._plot_performance_ranking(comp_df, axes[1, 0])
        
        # Plot 4: Performance variability
        self._plot_performance_variability(comp_df, axes[1, 1])
        
        plt.tight_layout()
        output_file = output_dir / "topology_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Topology comparison saved: {output_file}")
    
    def _plot_topology_distribution(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot performance distribution by topology"""
        # Create violin plot
        sns.violinplot(data=df, x='Topology', y='ExecutionTime', ax=ax)
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Performance Distribution by Topology')
        ax.tick_params(axis='x', rotation=45)
        
        # Add mean markers
        means = df.groupby('Topology')['ExecutionTime'].mean()
        for i, (topology, mean_val) in enumerate(means.items()):
            ax.scatter(i, mean_val, color='red', s=50, zorder=3, label='Mean' if i == 0 else "")
    
    def _plot_message_size_comparison(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot performance vs message size comparison"""
        if 'MessageSize' not in df.columns:
            ax.text(0.5, 0.5, 'Message size data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance vs Message Size')
            return
        
        for i, topology in enumerate(df['Topology'].unique()):
            topology_data = df[df['Topology'] == topology]
            
            # Aggregate by message size
            aggregated = topology_data.groupby('MessageSize').agg({
                'ExecutionTime': ['mean', 'std']
            }).reset_index()
            
            aggregated.columns = ['MessageSize', 'MeanTime', 'StdTime']
            
            # Plot with error bars
            ax.errorbar(aggregated['MessageSize'], aggregated['MeanTime'],
                       yerr=aggregated['StdTime'], label=topology,
                       color=self.colors[i], marker=self.markers[i],
                       markersize=6, capsize=3, linewidth=2)
        
        ax.set_xlabel('Message Size (bytes)')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Performance vs Message Size')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_ranking(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot performance ranking across topologies"""
        # Calculate average performance per topology
        avg_performance = df.groupby('Topology')['ExecutionTime'].mean().sort_values()
        
        # Create horizontal bar chart
        y_pos = np.arange(len(avg_performance))
        bars = ax.barh(y_pos, avg_performance.values, 
                      color=self.colors[:len(avg_performance)], alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(avg_performance.index)
        ax.set_xlabel('Average Execution Time (ms)')
        ax.set_title('Performance Ranking')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.2f}', ha='left', va='center', fontsize=9)
    
    def _plot_performance_variability(self, df: pd.DataFrame, ax