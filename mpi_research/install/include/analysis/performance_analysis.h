#ifndef PERFORMANCE_ANALYSIS_H
#define PERFORMANCE_ANALYSIS_H

#include <mpi.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <fstream>
#include "../core/collective_optimizer.h"
#include "../utils/performance_measurement.h"
//#include "statistical_analyzer.h"

namespace TopologyAwareResearch {

    class ScalabilityAnalyzer {
    private:
        MPI_Comm comm_;
        std::map<int, PerformanceMetrics> scalability_data_;
        int max_processes_;
        int step_size_;

    public:
        ScalabilityAnalyzer(MPI_Comm comm);
        ~ScalabilityAnalyzer();

        void analyze_strong_scaling(int base_processes, int max_processes,
            void* buffer, int count, MPI_Datatype datatype,
            int root, int iterations = 5);

        void analyze_weak_scaling(int base_processes, int max_processes,
            int base_count, MPI_Datatype datatype,
            int root, int iterations = 5);

        void analyze_isoefficiency(int base_processes, int max_processes,
            MPI_Datatype datatype, int root,
            int iterations = 5);

        // Analysis results
        double calculate_strong_scaling_efficiency(int processes) const;
        double calculate_weak_scaling_efficiency(int processes) const;
        double calculate_parallel_efficiency() const;
        std::map<int, double> get_scaling_factors() const;

        // Reporting
        void generate_scalability_report(const std::string& filename) const;
        void plot_scalability_curves(const std::string& filename) const;

    private:
        PerformanceMetrics measure_scaling_operation(void* buffer, int count,
            MPI_Datatype datatype, int root,
            int iterations);
        void create_sub_communicator(int processes, MPI_Comm* sub_comm);
    };

    class BottleneckDetector {
    private:
        MPI_Comm comm_;
        NetworkCharacteristics network_config_;
        double detection_threshold_;

    public:
        BottleneckDetector(MPI_Comm comm, const NetworkCharacteristics& config);
        ~BottleneckDetector();

        // Bottleneck detection methods
        std::vector<int> detect_communication_bottlenecks(const PerformanceMetrics& metrics);
        std::vector<int> detect_load_imbalance(const std::vector<PerformanceMetrics>& per_process_metrics);
        std::vector<std::pair<int, int>> detect_contention_points();

        // Network analysis
        double calculate_network_saturation() const;
        std::vector<int> identify_congested_links() const;
        double estimate_bottleneck_severity(int bottleneck_type) const;

        // Optimization suggestions
        std::vector<std::string> generate_optimization_suggestions(const std::vector<int>& bottlenecks);
        std::string recommend_algorithm(const std::vector<int>& bottlenecks, int message_size) const;

        void set_detection_threshold(double threshold) { detection_threshold_ = threshold; }

    private:
        double calculate_communication_imbalance(const std::vector<PerformanceMetrics>& metrics) const;
        double calculate_computation_imbalance(const std::vector<PerformanceMetrics>& metrics) const;
        bool is_network_contention_detected() const;
    };

    class EnergyEfficiencyAnalyzer {
    private:
        MPI_Comm comm_;
        EnergyMonitor energy_monitor_;
        double power_measurement_interval_;

    public:
        EnergyEfficiencyAnalyzer(MPI_Comm comm);
        ~EnergyEfficiencyAnalyzer();

        // Energy analysis
        double measure_energy_consumption(void* buffer, int count,
            MPI_Datatype datatype, int root,
            int iterations = 10);

        PerformanceMetrics analyze_energy_efficiency(void* buffer, int count,
            MPI_Datatype datatype, int root,
            int iterations = 10);

        // Efficiency metrics
        double calculate_energy_delay_product(const PerformanceMetrics& metrics) const;
        double calculate_power_efficiency(const PerformanceMetrics& metrics) const;
        double calculate_computational_efficiency(const PerformanceMetrics& metrics) const;

        // Comparative analysis
        std::map<std::string, double> compare_energy_efficiency(
            const std::map<std::string, PerformanceMetrics>& algorithm_metrics);

        void set_power_measurement_interval(double interval) {
            power_measurement_interval_ = interval;
        }

    private:
        double estimate_dynamic_power(const PerformanceMetrics& metrics) const;
        double estimate_static_power(double execution_time) const;
    };

    class StatisticalSignificanceTester {
    private:
        double confidence_level_;
        int min_sample_size_;

    public:
        StatisticalSignificanceTester();
        ~StatisticalSignificanceTester();

        // Statistical tests
        bool is_performance_difference_significant(const PerformanceMetrics& metrics1,
            const PerformanceMetrics& metrics2,
            double alpha = 0.05) const;

        double calculate_p_value(const std::vector<double>& sample1,
            const std::vector<double>& sample2) const;

        bool is_improvement_significant(const std::vector<PerformanceMetrics>& baseline,
            const std::vector<PerformanceMetrics>& optimized,
            double improvement_threshold = 0.05) const;

        // Confidence analysis
        double calculate_confidence_interval_width(const std::vector<double>& samples) const;
        int calculate_required_sample_size(double desired_margin, double estimated_stddev) const;

        void set_confidence_level(double level) { confidence_level_ = level; }
        void set_min_sample_size(int size) { min_sample_size_ = size; }

    private:
        double perform_t_test(const std::vector<double>& sample1,
            const std::vector<double>& sample2) const;
        double perform_mann_whitney_test(const std::vector<double>& sample1,
            const std::vector<double>& sample2) const;
    };

    class MultiObjectiveAnalyzer {
    private:
        std::vector<ParetoSolution> pareto_front_;
        std::vector<double> objective_weights_;
        bool normalized_objectives_;

    public:
        MultiObjectiveAnalyzer();
        ~MultiObjectiveAnalyzer();

        // Pareto analysis
        void update_pareto_front(const std::vector<ParetoSolution>& solutions);
        std::vector<ParetoSolution> get_pareto_optimal_solutions() const;
        std::vector<ParetoSolution> get_non_dominated_solutions() const;

        // Multi-objective metrics
        double calculate_hypervolume(const std::vector<double>& reference_point) const;
        double calculate_spacing_metric() const;
        double calculate_diversity_metric() const;
        std::vector<double> calculate_objective_ranges() const;

        // Solution selection
        ParetoSolution select_best_solution(const std::vector<double>& weights) const;
        ParetoSolution select_compromise_solution() const;
        std::vector<ParetoSolution> get_knee_points() const;

        void set_objective_weights(const std::vector<double>& weights) {
            objective_weights_ = weights;
        }
        void enable_normalization(bool enable) { normalized_objectives_ = enable; }

    private:
        void normalize_objectives(std::vector<ParetoSolution>& solutions) const;
        double calculate_crowding_distance(const std::vector<ParetoSolution>& solutions,
            int objective_index) const;
        bool dominates(const ParetoSolution& sol1, const ParetoSolution& sol2) const;
    };

    class ComparativeAnalysis {
    private:
        MPI_Comm comm_;
        std::map<std::string, std::vector<PerformanceMetrics>> algorithm_results_;
        StatisticalSignificanceTester significance_tester_;

    public:
        ComparativeAnalysis(MPI_Comm comm);
        ~ComparativeAnalysis();

        // Comparison methods
        void add_algorithm_results(const std::string& algorithm_name,
            const std::vector<PerformanceMetrics>& results);

        void compare_algorithms(const std::vector<std::string>& algorithm_names) const;
        std::string identify_best_algorithm(int message_size,
            OptimizationObjective objective) const;

        // Statistical comparison
        std::map<std::string, double> calculate_improvement_percentages(
            const std::string& baseline_algorithm) const;

        bool is_algorithm_significantly_better(const std::string& algo1,
            const std::string& algo2,
            double confidence_level = 0.95) const;

        // Reporting
        void generate_comparison_report(const std::string& filename) const;
        void generate_latex_table(const std::string& filename) const;

    private:
        PerformanceMetrics calculate_average_metrics(const std::vector<PerformanceMetrics>& metrics) const;
        double calculate_geometric_mean(const std::vector<double>& values) const;
    };

} // namespace TopologyAwareResearch

#endif