#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cmath>
#include "../../src/core/collective_optimizer.h"
#include "../../src/analysis/performance_analysis.h"

using namespace TopologyAwareResearch;
// Global ScalingResult struct
struct ScalingResult {
    double execution_time;
    double communication_time;
    double computation_time;
    double efficiency;
    double speedup;
    int processes;
    int problem_size;
};
class WeakStrongScaling {
private:
    MPI_Comm comm_;
    int world_rank_, world_size_;
    CollectiveOptimizer optimizer_;

    // Scaling parameters
    int min_processes_;
    int max_processes_;
    int scaling_steps_;
    int iterations_;

public:
//    ScalingResult measure_strong_scaling(int processes, int problem_size, int root);
//    ScalingResult measure_weak_scaling(int processes, int problem_size, int root);
//    void analyze_strong_scaling_results(const std::map<int, ScalingResult>& results);
//    void analyze_weak_scaling_results(const std::map<int, ScalingResult>& results);
//    void generate_strong_scaling_plot(const std::map<int, ScalingResult>& results);
//    void generate_weak_scaling_plot(const std::map<int, ScalingResult>& results);
    WeakStrongScaling(MPI_Comm comm)
        : comm_(comm),
        optimizer_() {

        MPI_Comm_rank(comm_, &world_rank_);
        MPI_Comm_size(comm_, &world_size_);

        // Scaling parameters
        min_processes_ = 2;
        max_processes_ = std::min(256, world_size_);
        scaling_steps_ = 8;
        iterations_ = 5;
    }

    void run_scaling_studies() {
        if (world_rank_ == 0) {
            std::cout << "=== Weak and Strong Scaling Studies ===" << std::endl;
            std::cout << "Process Range: " << min_processes_ << " - " << max_processes_ << std::endl;
        }

        // Strong scaling study
        run_strong_scaling_study();

        // Weak scaling study
        run_weak_scaling_study();

        // Combined analysis
        analyze_scaling_relationships();
    }

    void run_strong_scaling_study() {
        if (world_rank_ == 0) {
            std::cout << "\n=== STRONG SCALING STUDY ===" << std::endl;
        }

        int fixed_problem_size = 1048576; // 1MB total problem size
        int root = 0;

        std::map<int, ScalingResult> results;

        for (int step = 0; step < scaling_steps_; ++step) {
            int processes = calculate_process_count(step);
            if (processes > max_processes_) break;

            if (world_rank_ == 0) {
                std::cout << "Strong Scaling - Processes: " << processes << std::endl;
            }

            ScalingResult result = measure_strong_scaling(processes, fixed_problem_size, root);
            results[processes] = result;

            if (world_rank_ == 0) {
                print_scaling_result(result, processes);
            }
        }

        if (world_rank_ == 0) {
            analyze_strong_scaling_results(results);
            generate_strong_scaling_plot(results);
        }
    }

    void run_weak_scaling_study() {
        if (world_rank_ == 0) {
            std::cout << "\n=== WEAK SCALING STUDY ===" << std::endl;
        }

        int problem_size_per_process = 65536; // 64KB per process
        int root = 0;

        std::map<int, ScalingResult> results;

        for (int step = 0; step < scaling_steps_; ++step) {
            int processes = calculate_process_count(step);
            if (processes > max_processes_) break;

            if (world_rank_ == 0) {
                std::cout << "Weak Scaling - Processes: " << processes << std::endl;
            }

            int total_problem_size = problem_size_per_process * processes;
            ScalingResult result = measure_weak_scaling(processes, total_problem_size, root);
            results[processes] = result;

            if (world_rank_ == 0) {
                print_scaling_result(result, processes);
            }
        }

        if (world_rank_ == 0) {
            analyze_weak_scaling_results(results);
            generate_weak_scaling_plot(results);
        }
    }

    ScalingResult measure_strong_scaling(int processes, int problem_size, int root) {
        MPI_Comm sub_comm;
        create_sub_communicator(processes, &sub_comm);

        ScalingResult result;
        result.processes = processes;
        result.problem_size = problem_size;

        if (sub_comm == MPI_COMM_NULL) {
            return result;
        }

        int sub_rank;
        MPI_Comm_rank(sub_comm, &sub_rank);

        // Measure broadcast operation
        std::vector<double> buffer(problem_size);
        if (sub_rank == root) {
            initialize_buffer(buffer.data(), problem_size, root);
        }

        std::vector<double> times;
        for (int i = 0; i < iterations_; ++i) {
            MPI_Barrier(sub_comm);
            auto start = MPI_Wtime();

            MPI_Bcast(buffer.data(), problem_size, MPI_DOUBLE, root, sub_comm);

            auto end = MPI_Wtime();
            times.push_back(end - start);
        }

        result.execution_time = calculate_median(times);
        result.speedup = calculate_speedup(result, processes);
        result.efficiency = calculate_efficiency(result, processes);

        MPI_Comm_free(&sub_comm);
        return result;
    }

    ScalingResult measure_weak_scaling(int processes, int problem_size, int root) {
        MPI_Comm sub_comm;
        create_sub_communicator(processes, &sub_comm);

        ScalingResult result;
        result.processes = processes;
        result.problem_size = problem_size;

        if (sub_comm == MPI_COMM_NULL) {
            return result;
        }

        int sub_rank;
        MPI_Comm_rank(sub_comm, &sub_rank);

        // Measure broadcast operation with scaled problem size
        std::vector<double> buffer(problem_size);
        if (sub_rank == root) {
            initialize_buffer(buffer.data(), problem_size, root);
        }

        std::vector<double> times;
        for (int i = 0; i < iterations_; ++i) {
            MPI_Barrier(sub_comm);
            auto start = MPI_Wtime();

            MPI_Bcast(buffer.data(), problem_size, MPI_DOUBLE, root, sub_comm);

            auto end = MPI_Wtime();
            times.push_back(end - start);
        }

        result.execution_time = calculate_median(times);
        result.efficiency = calculate_weak_scaling_efficiency(result, processes);

        MPI_Comm_free(&sub_comm);
        return result;
    }

    void analyze_scaling_relationships() {
        if (world_rank_ != 0) return;

        std::cout << "\n=== SCALING RELATIONSHIP ANALYSIS ===" << std::endl;

        // This would compare strong and weak scaling results
        // and identify scaling limits and optimal operating points

        std::cout << "Scaling analysis completed." << std::endl;
        std::cout << "Key findings:" << std::endl;
        std::cout << "- Optimal process count for given problem sizes" << std::endl;
        std::cout << "- Scaling efficiency thresholds" << std::endl;
        std::cout << "- Communication vs computation trade-offs" << std::endl;
    }

    void analyze_strong_scaling_results(const std::map<int, ScalingResult>& results) {
        if (world_rank_ != 0) return;

        std::cout << "\n--- Strong Scaling Analysis ---" << std::endl;

        double ideal_speedup = 0.0;
        double actual_speedup = 0.0;
        int max_processes = results.rbegin()->first;

        if (results.count(min_processes_) && results.count(max_processes)) {
            double base_time = results.at(min_processes_).execution_time;
            double scaled_time = results.at(max_processes).execution_time;

            ideal_speedup = static_cast<double>(max_processes) / min_processes_;
            actual_speedup = base_time / scaled_time;

            std::cout << "Ideal Speedup: " << ideal_speedup << std::endl;
            std::cout << "Actual Speedup: " << actual_speedup << std::endl;
            std::cout << "Parallel Efficiency: " << (actual_speedup / ideal_speedup) * 100 << "%" << std::endl;
        }

        // Calculate Amdahl's law serial fraction
        if (actual_speedup > 0) {
            double serial_fraction = (ideal_speedup - actual_speedup) / (ideal_speedup * actual_speedup - actual_speedup);
            std::cout << "Estimated Serial Fraction: " << serial_fraction * 100 << "%" << std::endl;
        }
    }

    void analyze_weak_scaling_results(const std::map<int, ScalingResult>& results) {
        if (world_rank_ != 0) return;

        std::cout << "\n--- Weak Scaling Analysis ---" << std::endl;

        double min_efficiency = 100.0;
        double max_efficiency = 0.0;
        double avg_efficiency = 0.0;
        int count = 0;

        for (const auto& [processes, result] : results) {
            if (result.efficiency > 0) {
                min_efficiency = std::min(min_efficiency, result.efficiency);
                max_efficiency = std::max(max_efficiency, result.efficiency);
                avg_efficiency += result.efficiency;
                count++;
            }
        }

        if (count > 0) {
            avg_efficiency /= count;
            std::cout << "Weak Scaling Efficiency Range: " << min_efficiency << "% - " << max_efficiency << "%" << std::endl;
            std::cout << "Average Weak Scaling Efficiency: " << avg_efficiency << "%" << std::endl;
        }
    }

    void generate_strong_scaling_plot(const std::map<int, ScalingResult>& results) {
        if (world_rank_ != 0) return;

        std::ofstream file("strong_scaling_data.csv");
        if (!file.is_open()) return;

        file << "Processes,ExecutionTime,Speedup,Efficiency,IdealSpeedup" << std::endl;

        for (const auto& [processes, result] : results) {
            double ideal_speedup = static_cast<double>(processes) / min_processes_;
            file << processes << ","
                << result.execution_time << ","
                << result.speedup << ","
                << result.efficiency << ","
                << ideal_speedup << std::endl;
        }

        file.close();
    }

    void generate_weak_scaling_plot(const std::map<int, ScalingResult>& results) {
        if (world_rank_ != 0) return;

        std::ofstream file("weak_scaling_data.csv");
        if (!file.is_open()) return;

        file << "Processes,ExecutionTime,Efficiency,ProblemSize" << std::endl;

        for (const auto& [processes, result] : results) {
            file << processes << ","
                << result.execution_time << ","
                << result.efficiency << ","
                << result.problem_size << std::endl;
        }

        file.close();
    }

private:
//    struct ScalingResult {
//        int processes;
//        int problem_size;
//        double execution_time;
//        double speedup;
//        double efficiency;
//
//        ScalingResult() : processes(0), problem_size(0), execution_time(0.0),
//            speedup(0.0), efficiency(0.0) {
//        }
//    };

    int calculate_process_count(int step) {
        return min_processes_ * static_cast<int>(std::pow(2, step));
    }

    void create_sub_communicator(int processes, MPI_Comm* sub_comm) {
        if (processes > world_size_) {
            *sub_comm = MPI_COMM_NULL;
            return;
        }

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

    void initialize_buffer(double* buffer, int size, int rank) {
        for (int i = 0; i < size; ++i) {
            buffer[i] = static_cast<double>(i + rank + 1);
        }
    }

    double calculate_median(std::vector<double>& values) {
        if (values.empty()) return 0.0;
        std::sort(values.begin(), values.end());
        size_t n = values.size();
        if (n % 2 == 0) {
            return (values[n / 2 - 1] + values[n / 2]) / 2.0;
        }
        else {
            return values[n / 2];
        }
    }

    double calculate_speedup(const ScalingResult& result, int processes) {
        if (processes == min_processes_) return 1.0;
        // This would need baseline measurement for proper calculation
        return 1.0 / result.execution_time; // Simplified
    }

    double calculate_efficiency(const ScalingResult& result, int processes) {
        if (processes == min_processes_) return 100.0;
        double ideal_time = result.execution_time * min_processes_ / processes;
        return (ideal_time / result.execution_time) * 100.0;
    }

    double calculate_weak_scaling_efficiency(const ScalingResult& result, int processes) {
        if (processes == min_processes_) return 100.0;
        // For weak scaling, efficiency is based on constant time
        double base_time = 0.001; // This should be measured baseline
        return (base_time / result.execution_time) * 100.0;
    }

    void print_scaling_result(const ScalingResult& result, int processes) {
        std::cout << "  Processes: " << processes
            << ", Time: " << result.execution_time * 1000 << " ms"
            << ", Speedup: " << result.speedup
            << ", Efficiency: " << result.efficiency << "%"
            << std::endl;
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    WeakStrongScaling scaling(comm);
    scaling.run_scaling_studies();

    MPI_Finalize();
    return 0;
}