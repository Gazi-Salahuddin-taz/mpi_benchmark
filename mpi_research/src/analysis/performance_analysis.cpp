#include "performance_analysis.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

namespace TopologyAwareResearch {

    ScalabilityAnalyzer::ScalabilityAnalyzer(MPI_Comm comm)
        : comm_(comm), max_processes_(0), step_size_(2) {
    }

    ScalabilityAnalyzer::~ScalabilityAnalyzer() {}

    void ScalabilityAnalyzer::analyze_strong_scaling(int base_processes, int max_processes,
        void* buffer, int count, MPI_Datatype datatype,
        int root, int iterations) {
        max_processes_ = max_processes;

        int world_size;
        MPI_Comm_size(comm_, &world_size);

        for (int processes = base_processes; processes <= std::min(max_processes, world_size);
            processes *= step_size_) {

            MPI_Comm sub_comm;
            create_sub_communicator(processes, &sub_comm);

            if (sub_comm != MPI_COMM_NULL) {
                PerformanceMetrics metrics = measure_scaling_operation(buffer, count, datatype,
                    root, iterations);
                scalability_data_[processes] = metrics;

                MPI_Comm_free(&sub_comm);
            }
        }
    }

    void ScalabilityAnalyzer::create_sub_communicator(int processes, MPI_Comm* sub_comm) {
        int world_size, world_rank;
        MPI_Comm_size(comm_, &world_size);
        MPI_Comm_rank(comm_, &world_rank);

        if (processes > world_size) {
            *sub_comm = MPI_COMM_NULL;
            return;
        }

        // Create a sub-communicator with the first 'processes' ranks
        MPI_Group world_group, sub_group;
        MPI_Comm_group(comm_, &world_group);

        std::vector<int> ranks(processes);
        for (int i = 0; i < processes; ++i) {
            ranks[i] = i;
        }

        MPI_Group_incl(world_group, processes, ranks.data(), &sub_group);
        MPI_Comm_create(comm_, sub_group, sub_comm);

        MPI_Group_free(&world_group);
        MPI_Group_free(&sub_group);
    }

    PerformanceMetrics ScalabilityAnalyzer::measure_scaling_operation(void* buffer, int count,
        MPI_Datatype datatype, int root,
        int iterations) {
        PerformanceMetrics avg_metrics;
        std::vector<double> execution_times;

        for (int i = 0; i < iterations; ++i) {
            MPI_Barrier(comm_);

            auto start_time = MPI_Wtime();
            MPI_Bcast(buffer, count, datatype, root, comm_);
            auto end_time = MPI_Wtime();

            execution_times.push_back(end_time - start_time);
        }

        // Calculate statistics
        StatisticalAnalyzer analyzer;
        for (double time : execution_times) {
            analyzer.add_sample(time);
        }

        avg_metrics.execution_time = analyzer.calculate_mean();
        avg_metrics.standard_deviation = analyzer.calculate_standard_deviation();
        avg_metrics.confidence_interval = analyzer.calculate_confidence_interval();

        return avg_metrics;
    }

    double ScalabilityAnalyzer::calculate_strong_scaling_efficiency(int processes) const {
        auto it = scalability_data_.find(processes);
        if (it == scalability_data_.end() || processes <= 0) return 0.0;

        // Find base performance (assuming processes=1 is the base)
        auto base_it = scalability_data_.find(1);
        if (base_it == scalability_data_.end()) {
            base_it = scalability_data_.begin();
        }

        double base_time = base_it->second.execution_time;
        double scaled_time = it->second.execution_time;

        if (scaled_time == 0.0) return 0.0;

        double ideal_time = base_time / processes;
        return (ideal_time / scaled_time) * 100.0; // Percentage efficiency
    }

    void ScalabilityAnalyzer::generate_scalability_report(const std::string& filename) const {
        int world_rank;
        MPI_Comm_rank(comm_, &world_rank);

        if (world_rank != 0) return;

        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        file << "=== Scalability Analysis Report ===" << std::endl;
        file << "Processes,ExecutionTime(ms),Efficiency(%),Speedup,StdDev,ConfidenceInterval" << std::endl;

        for (const auto& [processes, metrics] : scalability_data_) {
            double efficiency = calculate_strong_scaling_efficiency(processes);
            double speedup = (scalability_data_.begin()->second.execution_time / metrics.execution_time);

            file << processes << ","
                << metrics.execution_time * 1000 << ","
                << efficiency << ","
                << speedup << ","
                << metrics.standard_deviation * 1000 << ","
                << metrics.confidence_interval * 1000 << std::endl;
        }

        file.close();
    }

    // BottleneckDetector implementation
    BottleneckDetector::BottleneckDetector(MPI_Comm comm, const NetworkCharacteristics& config)
        : comm_(comm), network_config_(config), detection_threshold_(0.1) {
    }

    BottleneckDetector::~BottleneckDetector() {}

    std::vector<int> BottleneckDetector::detect_communication_bottlenecks(const PerformanceMetrics& metrics) {
        std::vector<int> bottlenecks;

        // Analyze communication patterns
        if (metrics.bandwidth_utilization > 90.0) {
            bottlenecks.push_back(1); // Network saturation
        }

        if (metrics.load_imbalance > detection_threshold_) {
            bottlenecks.push_back(2); // Load imbalance
        }

        if (metrics.communication_time > metrics.execution_time * 0.8) {
            bottlenecks.push_back(3); // Communication bound
        }

        if (metrics.messages_sent > network_config_.total_processes * 10) {
            bottlenecks.push_back(4); // Excessive messaging
        }

        return bottlenecks;
    }

    std::vector<int> BottleneckDetector::detect_load_imbalance(const std::vector<PerformanceMetrics>& per_process_metrics) {
        std::vector<int> imbalanced_processes;

        if (per_process_metrics.empty()) return imbalanced_processes;

        // Calculate average execution time
        double total_time = 0.0;
        for (const auto& metrics : per_process_metrics) {
            total_time += metrics.execution_time;
        }
        double avg_time = total_time / per_process_metrics.size();

        // Identify processes significantly above average
        for (size_t i = 0; i < per_process_metrics.size(); ++i) {
            if (per_process_metrics[i].execution_time > avg_time * (1.0 + detection_threshold_)) {
                imbalanced_processes.push_back(i);
            }
        }

        return imbalanced_processes;
    }

    std::vector<std::string> BottleneckDetector::generate_optimization_suggestions(const std::vector<int>& bottlenecks) {
        std::vector<std::string> suggestions;

        for (int bottleneck : bottlenecks) {
            switch (bottleneck) {
            case 1: // Network saturation
                suggestions.push_back("Consider using topology-aware algorithms to reduce network congestion");
                suggestions.push_back("Implement message aggregation to reduce small messages");
                suggestions.push_back("Use pipeline techniques for large messages");
                break;

            case 2: // Load imbalance
                suggestions.push_back("Implement dynamic load balancing");
                suggestions.push_back("Consider work stealing for uneven workloads");
                suggestions.push_back("Use hierarchical approaches to balance communication");
                break;

            case 3: // Communication bound
                suggestions.push_back("Overlap computation and communication");
                suggestions.push_back("Use non-blocking collectives");
                suggestions.push_back("Optimize message sizes and segmentation");
                break;

            case 4: // Excessive messaging
                suggestions.push_back("Combine multiple small messages into larger ones");
                suggestions.push_back("Use broadcast trees instead of point-to-point");
                suggestions.push_back("Implement message coalescing");
                break;

            default:
                suggestions.push_back("Review algorithm selection and implementation");
                break;
            }
        }

        return suggestions;
    }

    // EnergyEfficiencyAnalyzer implementation
    EnergyEfficiencyAnalyzer::EnergyEfficiencyAnalyzer(MPI_Comm comm)
        : comm_(comm), power_measurement_interval_(0.1) {
    }

    EnergyEfficiencyAnalyzer::~EnergyEfficiencyAnalyzer() {}

    double EnergyEfficiencyAnalyzer::measure_energy_consumption(void* buffer, int count,
        MPI_Datatype datatype, int root,
        int iterations) {
        double total_energy = 0.0;

        for (int i = 0; i < iterations; ++i) {
            energy_monitor_.start_measurement();

            MPI_Barrier(comm_);
            auto start_time = MPI_Wtime();

            MPI_Bcast(buffer, count, datatype, root, comm_);

            auto end_time = MPI_Wtime();
            double energy = energy_monitor_.stop_measurement();

            total_energy += energy;
        }

        return total_energy / iterations;
    }

    double EnergyEfficiencyAnalyzer::calculate_energy_delay_product(const PerformanceMetrics& metrics) const {
        return metrics.execution_time * metrics.energy_consumption;
    }

    double EnergyEfficiencyAnalyzer::calculate_power_efficiency(const PerformanceMetrics& metrics) const {
        if (metrics.energy_consumption == 0.0) return 0.0;

        int type_size;
        MPI_Type_size(MPI_DOUBLE, &type_size); // Assuming double for calculation
        double data_processed = metrics.data_volume / (1024.0 * 1024.0); // MB

        return data_processed / metrics.energy_consumption; // MB/J
    }

    // MultiObjectiveAnalyzer implementation
    MultiObjectiveAnalyzer::MultiObjectiveAnalyzer()
        : normalized_objectives_(true) {
        objective_weights_ = { 0.33, 0.33, 0.34 }; // Default equal weights
    }

    MultiObjectiveAnalyzer::~MultiObjectiveAnalyzer() {}

    void MultiObjectiveAnalyzer::update_pareto_front(const std::vector<ParetoSolution>& solutions) {
        // Add new solutions to Pareto front
        pareto_front_.insert(pareto_front_.end(), solutions.begin(), solutions.end());

        // Remove dominated solutions
        std::vector<ParetoSolution> non_dominated;

        for (size_t i = 0; i < pareto_front_.size(); ++i) {
            bool dominated = false;

            for (size_t j = 0; j < pareto_front_.size(); ++j) {
                if (i != j && dominates(pareto_front_[j], pareto_front_[i])) {
                    dominated = true;
                    break;
                }
            }

            if (!dominated) {
                non_dominated.push_back(pareto_front_[i]);
            }
        }

        pareto_front_ = non_dominated;
    }

    std::vector<ParetoSolution> MultiObjectiveAnalyzer::get_pareto_optimal_solutions() const {
        return pareto_front_;
    }

    bool MultiObjectiveAnalyzer::dominates(const ParetoSolution& sol1, const ParetoSolution& sol2) const {
        // Check if sol1 dominates sol2
        bool at_least_one_better = false;

        for (size_t i = 0; i < sol1.objectives.size(); ++i) {
            // Assuming minimization for all objectives
            if (sol1.objectives[i] > sol2.objectives[i]) {
                return false; // sol1 is worse in at least one objective
            }
            if (sol1.objectives[i] < sol2.objectives[i]) {
                at_least_one_better = true;
            }
        }

        return at_least_one_better;
    }

    ParetoSolution MultiObjectiveAnalyzer::select_best_solution(const std::vector<double>& weights) const {
        if (pareto_front_.empty()) {
            return ParetoSolution();
        }

        ParetoSolution best_solution = pareto_front_[0];
        double best_score = -1e9;

        for (const auto& solution : pareto_front_) {
            double weighted_sum = 0.0;
            for (size_t i = 0; i < solution.objectives.size(); ++i) {
                double weight = (i < weights.size()) ? weights[i] : 1.0 / solution.objectives.size();
                weighted_sum += solution.objectives[i] * weight;
            }

            if (weighted_sum > best_score) {
                best_score = weighted_sum;
                best_solution = solution;
            }
        }

        return best_solution;
    }

    // ComparativeAnalysis implementation
    ComparativeAnalysis::ComparativeAnalysis(MPI_Comm comm)
        : comm_(comm) {
    }

    ComparativeAnalysis::~ComparativeAnalysis() {}

    void ComparativeAnalysis::add_algorithm_results(const std::string& algorithm_name,
        const std::vector<PerformanceMetrics>& results) {
        algorithm_results_[algorithm_name] = results;
    }

    void ComparativeAnalysis::compare_algorithms(const std::vector<std::string>& algorithm_names) const {
        int world_rank;
        MPI_Comm_rank(comm_, &world_rank);

        if (world_rank != 0) return;

        std::cout << "=== Algorithm Comparison ===" << std::endl;

        for (const auto& algo_name : algorithm_names) {
            auto it = algorithm_results_.find(algo_name);
            if (it == algorithm_results_.end()) continue;

            PerformanceMetrics avg_metrics = calculate_average_metrics(it->second);

            std::cout << "Algorithm: " << algo_name << std::endl;
            std::cout << "  Execution Time: " << avg_metrics.execution_time * 1000 << " ms" << std::endl;
            std::cout << "  Bandwidth Utilization: " << avg_metrics.bandwidth_utilization << " %" << std::endl;
            std::cout << "  Energy Consumption: " << avg_metrics.energy_consumption << " J" << std::endl;
            std::cout << "  Messages Sent: " << avg_metrics.messages_sent << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        }
    }

    std::string ComparativeAnalysis::identify_best_algorithm(int message_size,
        OptimizationObjective objective) const {
        if (algorithm_results_.empty()) return "";

        std::string best_algorithm;
        double best_score = -1e9;

        for (const auto& [algo_name, metrics_list] : algorithm_results_) {
            PerformanceMetrics avg_metrics = calculate_average_metrics(metrics_list);
            double score = 0.0;

            switch (objective) {
            case OptimizationObjective::MINIMIZE_LATENCY:
                score = -avg_metrics.execution_time;
                break;
            case OptimizationObjective::MAXIMIZE_BANDWIDTH:
                score = avg_metrics.bandwidth_utilization;
                break;
            case OptimizationObjective::MINIMIZE_ENERGY:
                score = -avg_metrics.energy_consumption;
                break;
            case OptimizationObjective::BALANCED_OPTIMIZATION:
                score = (-avg_metrics.execution_time * 0.4 +
                    avg_metrics.bandwidth_utilization * 0.4 +
                    -avg_metrics.energy_consumption * 0.2);
                break;
            default:
                score = -avg_metrics.execution_time;
                break;
            }

            if (score > best_score) {
                best_score = score;
                best_algorithm = algo_name;
            }
        }

        return best_algorithm;
    }

    PerformanceMetrics ComparativeAnalysis::calculate_average_metrics(const std::vector<PerformanceMetrics>& metrics) const {
        PerformanceMetrics avg;
        int count = metrics.size();

        if (count == 0) return avg;

        for (const auto& m : metrics) {
            avg.execution_time += m.execution_time;
            avg.communication_time += m.communication_time;
            avg.bandwidth_utilization += m.bandwidth_utilization;
            avg.energy_consumption += m.energy_consumption;
            avg.messages_sent += m.messages_sent;
            avg.data_volume += m.data_volume;
        }

        avg.execution_time /= count;
        avg.communication_time /= count;
        avg.bandwidth_utilization /= count;
        avg.energy_consumption /= count;
        avg.messages_sent /= count;
        avg.data_volume /= count;

        return avg;
    }

} // namespace TopologyAwareResearch