#!/usr/bin/env python3

"""
Advanced Statistical Analysis for MPI Research
Performs rigorous statistical analysis of performance results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson, ttest_ind, mannwhitneyu
import warnings
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use(['science', 'ieee', 'grid'])

class StatisticalAnalyzer:
    """Advanced statistical analysis for MPI performance data"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.data = {}
        self.analysis_results = {}
        
    def load_all_data(self) -> None:
        """Load all performance data for statistical analysis"""
        print("Loading data for statistical analysis...")
        
        csv_files = list(self.results_dir.glob('**/*.csv'))
        
        for csv_file in csv_files:
            try:
                # Extract topology and benchmark info
                rel_path = csv_file.relative_to(self.results_dir)
                topology = rel_path.parts[0] if len(rel_path.parts) > 1 else 'unknown'
                
                if topology not in self.data:
                    self.data[topology] = {}
                
                df = pd.read_csv(csv_file)
                benchmark_name = csv_file.stem
                self.data[topology][benchmark_name] = df
                
                print(f"  Loaded {topology}/{benchmark_name}: {len(df)} samples")
                
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")
    
    def perform_comprehensive_analysis(self, output_dir: Path) -> None:
        """Perform comprehensive statistical analysis"""
        print("Performing comprehensive statistical analysis...")
        
        # 1. Normality tests
        normality_results = self._perform_normality_tests()
        
        # 2. Descriptive statistics
        descriptive_stats = self._calculate_descriptive_statistics()
        
        # 3. Statistical significance tests
        significance_results = self._perform_significance_tests()
        
        # 4. Correlation analysis
        correlation_results = self._perform_correlation_analysis()
        
        # 5. Outlier detection
        outlier_results = self._detect_outliers()
        
        # 6. Statistical power analysis
        power_analysis = self._perform_power_analysis()
        
        # Store all results
        self.analysis_results = {
            'normality_tests': normality_results,
            'descriptive_statistics': descriptive_stats,
            'significance_tests': significance_results,
            'correlation_analysis': correlation_results,
            'outlier_analysis': outlier_results,
            'power_analysis': power_analysis
        }
        
        # Generate visualizations
        self._generate_statistical_plots(output_dir)
        
        # Save results
        self._save_analysis_results(output_dir)
        
        print("Comprehensive statistical analysis completed")
    
    def _perform_normality_tests(self) -> Dict:
        """Perform multiple normality tests on performance data"""
        print("  Performing normality tests...")
        
        normality_results = {}
        
        for topology, benchmarks in self.data.items():
            normality_results[topology] = {}
            
            for benchmark, df in benchmarks.items():
                if 'ExecutionTime' in df.columns:
                    times = df['ExecutionTime'].values
                    
                    # Remove any zeros or negative values
                    times = times[times > 0]
                    
                    if len(times) >= 3:  # Need at least 3 samples for normality tests
                        test_results = {}
                        
                        # Shapiro-Wilk test (good for small samples)
                        shapiro_stat, shapiro_p = shapiro(times)
                        test_results['shapiro_wilk'] = {
                            'statistic': shapiro_stat,
                            'p_value': shapiro_p,
                            'normal': shapiro_p > 0.05
                        }
                        
                        # D'Agostino's normality test
                        dagostino_stat, dagostino_p = normaltest(times)
                        test_results['dagostino'] = {
                            'statistic': dagostino_stat,
                            'p_value': dagostino_p,
                            'normal': dagostino_p > 0.05
                        }
                        
                        # Anderson-Darling test
                        anderson_result = anderson(times)
                        test_results['anderson_darling'] = {
                            'statistic': anderson_result.statistic,
                            'critical_values': anderson_result.critical_values.tolist(),
                            'significance_level': anderson_result.significance_level.tolist()
                        }
                        
                        normality_results[topology][benchmark] = test_results
        
        return normality_results
    
    def _calculate_descriptive_statistics(self) -> Dict:
        """Calculate comprehensive descriptive statistics"""
        print("  Calculating descriptive statistics...")
        
        descriptive_stats = {}
        
        for topology, benchmarks in self.data.items():
            descriptive_stats[topology] = {}
            
            for benchmark, df in benchmarks.items():
                if 'ExecutionTime' in df.columns:
                    times = df['ExecutionTime'].values
                    times = times[times > 0]  # Remove invalid values
                    
                    if len(times) > 0:
                        stats_dict = {
                            'sample_size': len(times),
                            'mean': np.mean(times),
                            'median': np.median(times),
                            'std_dev': np.std(times),
                            'variance': np.var(times),
                            'min': np.min(times),
                            'max': np.max(times),
                            'range': np.ptp(times),
                            'q1': np.percentile(times, 25),
                            'q3': np.percentile(times, 75),
                            'iqr': np.percentile(times, 75) - np.percentile(times, 25),
                            'skewness': stats.skew(times),
                            'kurtosis': stats.kurtosis(times),
                            'coefficient_of_variation': np.std(times) / np.mean(times) if np.mean(times) > 0 else 0
                        }
                        
                        descriptive_stats[topology][benchmark] = stats_dict
        
        return descriptive_stats
    
    def _perform_significance_tests(self) -> Dict:
        """Perform statistical significance tests between topologies"""
        print("  Performing significance tests...")
        
        significance_results = {}
        
        # Get all topology pairs
        topologies = list(self.data.keys())
        
        for i, topology1 in enumerate(topologies):
            for topology2 in topologies[i+1:]:
                comparison_key = f"{topology1}_vs_{topology2}"
                significance_results[comparison_key] = {}
                
                # Find common benchmarks
                common_benchmarks = set(self.data[topology1].keys()) & set(self.data[topology2].keys())
                
                for benchmark in common_benchmarks:
                    df1 = self.data[topology1][benchmark]
                    df2 = self.data[topology2][benchmark]
                    
                    if 'ExecutionTime' in df1.columns and 'ExecutionTime' in df2.columns:
                        times1 = df1['ExecutionTime'].values
                        times2 = df2['ExecutionTime'].values
                        
                        # Remove invalid values
                        times1 = times1[times1 > 0]
                        times2 = times2[times2 > 0]
                        
                        if len(times1) >= 2 and len(times2) >= 2:
                            test_results = {}
                            
                            # Student's t-test (parametric)
                            t_stat, t_p = ttest_ind(times1, times2, equal_var=False)
                            test_results['t_test'] = {
                                'statistic': t_stat,
                                'p_value': t_p,
                                'significant': t_p < 0.05
                            }
                            
                            # Mann-Whitney U test (non-parametric)
                            u_stat, u_p = mannwhitneyu(times1, times2)
                            test_results['mann_whitney'] = {
                                'statistic': u_stat,
                                'p_value': u_p,
                                'significant': u_p < 0.05
                            }
                            
                            # Effect size (Cohen's d)
                            pooled_std = np.sqrt((np.std(times1)**2 + np.std(times2)**2) / 2)
                            cohens_d = (np.mean(times1) - np.mean(times2)) / pooled_std if pooled_std > 0 else 0
                            test_results['effect_size'] = {
                                'cohens_d': cohens_d,
                                'magnitude': self._interpret_effect_size(cohens_d)
                            }
                            
                            significance_results[comparison_key][benchmark] = test_results
        
        return significance_results
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if abs(d) < 0.2:
            return 'negligible'
        elif abs(d) < 0.5:
            return 'small'
        elif abs(d) < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _perform_correlation_analysis(self) -> Dict:
        """Perform correlation analysis between performance metrics"""
        print("  Performing correlation analysis...")
        
        correlation_results = {}
        
        for topology, benchmarks in self.data.items():
            correlation_results[topology] = {}
            
            for benchmark, df in benchmarks.items():
                # Check for multiple numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) >= 2:
                    # Calculate correlation matrix
                    corr_matrix = df[numeric_cols].corr(method='pearson')
                    
                    # Also calculate Spearman correlation for non-linear relationships
                    spearman_matrix = df[numeric_cols].corr(method='spearman')
                    
                    correlation_results[topology][benchmark] = {
                        'pearson_correlation': corr_matrix.to_dict(),
                        'spearman_correlation': spearman_matrix.to_dict()
                    }
        
        return correlation_results
    
    def _detect_outliers(self) -> Dict:
        """Detect outliers in performance data"""
        print("  Detecting outliers...")
        
        outlier_results = {}
        
        for topology, benchmarks in self.data.items():
            outlier_results[topology] = {}
            
            for benchmark, df in benchmarks.items():
                if 'ExecutionTime' in df.columns:
                    times = df['ExecutionTime'].values
                    times = times[times > 0]
                    
                    if len(times) >= 4:  # Need reasonable sample size
                        outlier_methods = {}
                        
                        # IQR method
                        Q1 = np.percentile(times, 25)
                        Q3 = np.percentile(times, 75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        iqr_outliers = times[(times < lower_bound) | (times > upper_bound)]
                        outlier_methods['iqr'] = {
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'outlier_count': len(iqr_outliers),
                            'outlier_percentage': (len(iqr_outliers) / len(times)) * 100
                        }
                        
                        # Z-score method (for normally distributed data)
                        z_scores = np.abs(stats.zscore(times))
                        z_outliers = times[z_scores > 3]
                        outlier_methods['z_score'] = {
                            'outlier_count': len(z_outliers),
                            'outlier_percentage': (len(z_outliers) / len(times)) * 100
                        }
                        
                        outlier_results[topology][benchmark] = outlier_methods
        
        return outlier_results
    
    def _perform_power_analysis(self) -> Dict:
        """Perform statistical power analysis"""
        print("  Performing power analysis...")
        
        power_results = {}
        
        # Analyze power for detecting performance differences
        for topology, benchmarks in self.data.items():
            power_results[topology] = {}
            
            for benchmark, df in benchmarks.items():
                if 'ExecutionTime' in df.columns:
                    times = df['ExecutionTime'].values
                    times = times[times > 0]
                    
                    if len(times) >= 2:
                        # Calculate effect sizes we can detect with current sample size
                        power_analyzer = TTestIndPower()
                        sample_size = len(times)
                        
                        # Power for different effect sizes
                        effect_sizes = [0.2, 0.5, 0.8]  # small, medium, large
                        power_levels = {}
                        
                        for effect_size in effect_sizes:
                            power = power_analyzer.solve_power(
                                effect_size=effect_size,
                                nobs1=sample_size,
                                alpha=0.05,
                                ratio=1.0
                            )
                            power_levels[f'effect_size_{effect_size}'] = power
                        
                        # Required sample size for 80% power
                        required_samples = {}
                        for effect_size in effect_sizes:
                            n = power_analyzer.solve_power(
                                effect_size=effect_size,
                                power=0.8,
                                alpha=0.05,
                                ratio=1.0
                            )
                            required_samples[f'effect_size_{effect_size}'] = n
                        
                        power_results[topology][benchmark] = {
                            'current_sample_size': sample_size,
                            'detection_power': power_levels,
                            'required_sample_size': required_samples
                        }
        
        return power_results
    
    def _generate_statistical_plots(self, output_dir: Path) -> None:
        """Generate statistical analysis plots"""
        print("  Generating statistical plots...")
        
        # 1. Distribution plots
        self._plot_distributions(output_dir)
        
        # 2. Box plots for comparison
        self._plot_comparison_boxplots(output_dir)
        
        # 3. Correlation heatmaps
        self._plot_correlation_heatmaps(output_dir)
        
        # 4. Statistical power plots
        self._plot_power_analysis(output_dir)
    
    def _plot_distributions(self, output_dir: Path) -> None:
        """Plot distribution of performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Distribution Analysis of MPI Performance', 
                    fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        plot_data = []
        for topology, benchmarks in self.data.items():
            for benchmark, df in benchmarks.items():
                if 'ExecutionTime' in df.columns:
                    for time in df['ExecutionTime']:
                        if time > 0:
                            plot_data.append({
                                'Topology': topology,
                                'Benchmark': benchmark,
                                'ExecutionTime': time
                            })
        
        if not plot_data:
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        # Plot 1: Histograms by topology
        for i, topology in enumerate(plot_df['Topology'].unique()):
            topology_data = plot_df[plot_df['Topology'] == topology]['ExecutionTime']
            axes[0, 0].hist(topology_data, bins=20, alpha=0.6, 
                           label=topology, density=True)
        axes[0, 0].set_xlabel('Execution Time (ms)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Performance Distribution by Topology')
        axes[0, 0].legend()
        
        # Plot 2: Q-Q plots for normality check
        for i, topology in enumerate(plot_df['Topology'].unique()[:4]):  # Limit to 4 for clarity
            topology_data = plot_df[plot_df['Topology'] == topology]['ExecutionTime']
            stats.probplot(topology_data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot for Normality Check')
        
        # Plot 3: Violin plots
        sns.violinplot(data=plot_df, x='Topology', y='ExecutionTime', ax=axes[1, 0])
        axes[1, 0].set_title('Performance Distribution (Violin Plot)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Cumulative distribution
        for topology in plot_df['Topology'].unique():
            topology_data = plot_df[plot_df['Topology'] == topology]['ExecutionTime']
            sorted_data = np.sort(topology_data)
            yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
            axes[1, 1].plot(sorted_data, yvals, label=topology)
        axes[1, 1].set_xlabel('Execution Time (ms)')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution Function')
        axes[1, 1].legend()
        
        plt.tight_layout()
        output_file = output_dir / "statistical_distributions.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comparison_boxplots(self, output_dir: Path) -> None:
        """Create box plots for statistical comparison"""
        # Prepare data
        plot_data = []
        for topology, benchmarks in self.data.items():
            for benchmark, df in benchmarks.items():
                if 'ExecutionTime' in df.columns:
                    for time in df['ExecutionTime']:
                        if time > 0:
                            plot_data.append({
                                'Topology': topology,
                                'Benchmark': benchmark,
                                'ExecutionTime': time
                            })
        
        if not plot_data:
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create box plots
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=plot_df, x='Topology', y='ExecutionTime')
        plt.title('Performance Comparison Across Topologies')
        plt.xlabel('Network Topology')
        plt.ylabel('Execution Time (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_file = output_dir / "statistical_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmaps(self, output_dir: Path) -> None:
        """Create correlation heatmaps"""
        for topology, benchmarks in self.data.items():
            for benchmark, df in benchmarks.items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) >= 2:
                    # Calculate correlations
                    corr_matrix = df[numeric_cols].corr()
                    
                    # Create heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                               center=0, square=True)
                    plt.title(f'Correlation Matrix: {topology} - {benchmark}')
                    plt.tight_layout()
                    
                    output_file = output_dir / f"correlation_{topology}_{benchmark}.png"
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
    
    def _plot_power_analysis(self, output_dir: Path) -> None:
        """Plot statistical power analysis results"""
        if not self.analysis_results.get('power_analysis'):
            return
        
        power_data = []
        for topology, benchmarks in self.analysis_results['power_analysis'].items():
            for benchmark, analysis in benchmarks.items():
                for effect_size, power in analysis['detection_power'].items():
                    power_data.append({
                        'Topology': topology,
                        'Benchmark': benchmark,
                        'EffectSize': effect_size,
                        'Power': power
                    })
        
        if not power_data:
            return
        
        power_df = pd.DataFrame(power_data)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=power_df, x='Topology', y='Power', hue='EffectSize')
        plt.title('Statistical Power Analysis')
        plt.xlabel('Network Topology')
        plt.ylabel('Detection Power')
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Desired Power (0.8)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_file = output_dir / "power_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_analysis_results(self, output_dir: Path) -> None:
        """Save all statistical analysis results"""
        # Save JSON results
        results_file = output_dir / "statistical_analysis_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_types(self.analysis_results), f, indent=2)
        
        print(f"Statistical analysis results saved: {results_file}")
        
        # Generate summary report
        self._generate_summary_report(output_dir)
    
    def _generate_summary_report(self, output_dir: Path) -> None:
        """Generate human-readable summary report"""
        summary_lines = []
        summary_lines.append("STATISTICAL ANALYSIS SUMMARY REPORT")
        summary_lines.append("=" * 50)
        summary_lines.append("")
        
        # Normality summary
        summary_lines.append("NORMALITY ASSESSMENT:")
        summary_lines.append("-" * 30)
        for topology, benchmarks in self.analysis_results.get('normality_tests', {}).items():
            for benchmark, tests in benchmarks.items():
                shapiro_normal = tests.get('shapiro_wilk', {}).get('normal', False)
                summary_lines.append(f"{topology}/{benchmark}: {'Normal' if shapiro_normal else 'Non-normal'}")
        summary_lines.append("")
        
        # Significance summary
        summary_lines.append("STATISTICAL SIGNIFICANCE:")
        summary_lines.append("-" * 30)
        for comparison, benchmarks in self.analysis_results.get('significance_tests', {}).items():
            significant_count = 0
            total_count = 0
            for benchmark, tests in benchmarks.items():
                if tests.get('t_test', {}).get('significant', False):
                    significant_count += 1
                total_count += 1
            
            if total_count > 0:
                summary_lines.append(f"{comparison}: {significant_count}/{total_count} benchmarks show significant differences")
        summary_lines.append("")
        
        # Outlier summary
        summary_lines.append("OUTLIER ANALYSIS:")
        summary_lines.append("-" * 30)
        for topology, benchmarks in self.analysis_results.get('outlier_analysis', {}).items():
            for benchmark, methods in benchmarks.items():
                outlier_pct = methods.get('iqr', {}).get('outlier_percentage', 0)
                summary_lines.append(f"{topology}/{benchmark}: {outlier_pct:.1f}% outliers")
        summary_lines.append("")
        
        # Power analysis summary
        summary_lines.append("STATISTICAL POWER:")
        summary_lines.append("-" * 30)
        for topology, benchmarks in self.analysis_results.get('power_analysis', {}).items():
            for benchmark, analysis in benchmarks.items():
                power = analysis.get('detection_power', {}).get('effect_size_0.5', 0)
                summary_lines.append(f"{topology}/{benchmark}: Power = {power:.3f} for medium effects")
        
        # Write summary report
        summary_file = output_dir / "statistical_summary.txt"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"Statistical summary saved: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Perform statistical analysis of MPI performance results')
    parser.add_argument('results_dir', help='Directory containing benchmark results')
    parser.add_argument('-o', '--output', help='Output directory for analysis', default='statistical_analysis')
    parser.add_argument('--comprehensive', action='store_true', help='Perform comprehensive analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Create analyzer and load data
    analyzer = StatisticalAnalyzer(args.results_dir)
    analyzer.load_all_data()
    
    # Perform analysis
    if args.comprehensive:
        analyzer.perform_comprehensive_analysis(output_dir)
    else:
        # Perform basic analysis
        analyzer.perform_comprehensive_analysis(output_dir)
    
    print("Statistical analysis completed successfully")

if __name__ == "__main__":
    main()