#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <chrono>
#include <fstream>
#include "../../src/core/collective_optimizer.h"
#include "../../src/algorithms/topology_aware_broadcast.h"
#include "../../src/utils/performance_measurement.h"

using namespace TopologyAwareResearch;

class CollectiveBenchmark {
private:
    MPI_Comm comm_;
    int world_rank_, world_size_;
    CollectiveOptimizer optimizer_;
    PerformanceProfiler profiler_;

    // Benchmark parameters
    std::vector<int> message_sizes_;
    int iterations_;
    int warmup_iterations_;

public:
    CollectiveBenchmark(MPI_Comm comm)
        : comm_(comm),
        optimizer_(),
        profiler_(comm) {

        MPI_Comm_rank(comm_, &world_rank_);
        MPI_Comm_size(comm_, &world_size_);

        // Initialize benchmark parameters
        message_sizes_ = { 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144 };
        iterations_ = 50;
        warmup_iterations_ = 5;
    }

    void run_all_collectives_benchmark() {
        if (world_rank_ == 0) {
            std::cout << "=== Collective Operations Comprehensive Benchmark ===" << std::endl;
            std::cout << "Testing: Broadcast, Allreduce, Reduce, Allgather, Barrier" << std::endl;
        }

        std::map<std::string, std::map<int, PerformanceMetrics>> results;

        // Benchmark each collective operation
        results["Broadcast"] = benchmark_broadcast();
        results["Allreduce"] = benchmark_allreduce();
        results["Reduce"] = benchmark_reduce();
        results["Allgather"] = benchmark_allgather();
        results["Barrier"] = benchmark_barrier();

        // Analyze and report results
        if (world_rank_ == 0) {
            generate_collective_report(results);
            analyze_collective_patterns(results);
        }
    }

    std::map<int, PerformanceMetrics> benchmark_broadcast() {
        if (world_rank_ == 0) {
            std::cout << "\n--- Benchmarking Broadcast ---" << std::endl;
        }

        std::map<int, PerformanceMetrics> results;
        int root = 0;

        for (int size : message_sizes_) {
            std::vector<double> send_buffer(size);
            initialize_buffer(send_buffer.data(), size, world_rank_);

            PerformanceMetrics metrics;
            std::vector<double> execution_times;

            // Warmup
            for (int i = 0; i < warmup_iterations_; ++i) {
                MPI_Bcast(send_buffer.data(), size, MPI_DOUBLE, root, comm_);
            }

            // Measurement
            for (int i = 0; i < iterations_; ++i) {
                MPI_Barrier(comm_);
                auto start = MPI_Wtime();

                MPI_Bcast(send_buffer.data(), size, MPI_DOUBLE, root, comm_);

                auto end = MPI_Wtime();
                execution_times.push_back(end - start);
            }

            metrics = calculate_metrics(execution_times, size);
            results[size] = metrics;

            if (world_rank_ == 0) {
                std::cout << "  Size " << size << ": " << metrics.execution_time * 1000 << " ms" << std::endl;
            }
        }

        return results;
    }

    std::map<int, PerformanceMetrics> benchmark_allreduce() {
        if (world_rank_ == 0) {
            std::cout << "\n--- Benchmarking Allreduce ---" << std::endl;
        }

        std::map<int, PerformanceMetrics> results;

        for (int size : message_sizes_) {
            std::vector<double> send_buffer(size);
            std::vector<double> recv_buffer(size);
            initialize_buffer(send_buffer.data(), size, world_rank_);

            PerformanceMetrics metrics;
            std::vector<double> execution_times;

            // Warmup
            for (int i = 0; i < warmup_iterations_; ++i) {
                MPI_Allreduce(send_buffer.data(), recv_buffer.data(), size, MPI_DOUBLE, MPI_SUM, comm_);
            }

            // Measurement
            for (int i = 0; i < iterations_; ++i) {
                MPI_Barrier(comm_);
                auto start = MPI_Wtime();

                MPI_Allreduce(send_buffer.data(), recv_buffer.data(), size, MPI_DOUBLE, MPI_SUM, comm_);

                auto end = MPI_Wtime();
                execution_times.push_back(end - start);

                // Verify result occasionally
                if (i == 0) {
                    verify_allreduce_result(recv_buffer.data(), size);
                }
            }

            metrics = calculate_metrics(execution_times, size);
            results[size] = metrics;

            if (world_rank_ == 0) {
                std::cout << "  Size " << size << ": " << metrics.execution_time * 1000 << " ms" << std::endl;
            }
        }

        return results;
    }

    std::map<int, PerformanceMetrics> benchmark_reduce() {
        if (world_rank_ == 0) {
            std::cout << "\n--- Benchmarking Reduce ---" << std::endl;
        }

        std::map<int, PerformanceMetrics> results;
        int root = 0;

        for (int size : message_sizes_) {
            std::vector<double> send_buffer(size);
            std::vector<double> recv_buffer(size);
            initialize_buffer(send_buffer.data(), size, world_rank_);

            PerformanceMetrics metrics;
            std::vector<double> execution_times;

            // Warmup
            for (int i = 0; i < warmup_iterations_; ++i) {
                if (world_rank_ == root) {
                    MPI_Reduce(MPI_IN_PLACE, recv_buffer.data(), size, MPI_DOUBLE, MPI_SUM, root, comm_);
                }
                else {
                    MPI_Reduce(send_buffer.data(), nullptr, size, MPI_DOUBLE, MPI_SUM, root, comm_);
                }
            }

            // Measurement
            for (int i = 0; i < iterations_; ++i) {
                MPI_Barrier(comm_);
                auto start = MPI_Wtime();

                if (world_rank_ == root) {
                    MPI_Reduce(MPI_IN_PLACE, recv_buffer.data(), size, MPI_DOUBLE, MPI_SUM, root, comm_);
                }
                else {
                    MPI_Reduce(send_buffer.data(), nullptr, size, MPI_DOUBLE, MPI_SUM, root, comm_);
                }

                auto end = MPI_Wtime();
                execution_times.push_back(end - start);
            }

            metrics = calculate_metrics(execution_times, size);
            results[size] = metrics;

            if (world_rank_ == 0) {
                std::cout << "  Size " << size << ": " << metrics.execution_time * 1000 << " ms" << std::endl;
            }
        }

        return results;
    }

    std::map<int, PerformanceMetrics> benchmark_allgather() {
        if (world_rank_ == 0) {
            std::cout << "\n--- Benchmarking Allgather ---" << std::endl;
        }

        std::map<int, PerformanceMetrics> results;

        for (int size : message_sizes_) {
            std::vector<double> send_buffer(size);
            std::vector<double> recv_buffer(size * world_size_);
            initialize_buffer(send_buffer.data(), size, world_rank_);

            PerformanceMetrics metrics;
            std::vector<double> execution_times;

            // Warmup
            for (int i = 0; i < warmup_iterations_; ++i) {
                MPI_Allgather(send_buffer.data(), size, MPI_DOUBLE,
                    recv_buffer.data(), size, MPI_DOUBLE, comm_);
            }

            // Measurement
            for (int i = 0; i < iterations_; ++i) {
                MPI_Barrier(comm_);
                auto start = MPI_Wtime();

                MPI_Allgather(send_buffer.data(), size, MPI_DOUBLE,
                    recv_buffer.data(), size, MPI_DOUBLE, comm_);

                auto end = MPI_Wtime();
                execution_times.push_back(end - start);

                // Verify result occasionally
                if (i == 0) {
                    verify_allgather_result(recv_buffer.data(), size);
                }
            }

            metrics = calculate_metrics(execution_times, size);
            results[size] = metrics;

            if (world_rank_ == 0) {
                std::cout << "  Size " << size << ": " << metrics.execution_time * 1000 << " ms" << std::endl;
            }
        }

        return results;
    }

    std::map<int, PerformanceMetrics> benchmark_barrier() {
        if (world_rank_ == 0) {
            std::cout << "\n--- Benchmarking Barrier ---" << std::endl;
        }

        std::map<int, PerformanceMetrics> results;

        // Barrier doesn't have message size, so we use a dummy size
        int dummy_size = 1;

        PerformanceMetrics metrics;
        std::vector<double> execution_times;

        // Warmup
        for (int i = 0; i < warmup_iterations_; ++i) {
            MPI_Barrier(comm_);
        }

        // Measurement
        for (int i = 0; i < iterations_; ++i) {
            auto start = MPI_Wtime();

            MPI_Barrier(comm_);

            auto end = MPI_Wtime();
            execution_times.push_back(end - start);
        }

        metrics = calculate_metrics(execution_times, dummy_size);
        results[dummy_size] = metrics;

        if (world_rank_ == 0) {
            std::cout << "  Barrier: " << metrics.execution_time * 1000 << " ms" << std::endl;
        }

        return results;
    }

    void generate_collective_report(const std::map<std::string, std::map<int, PerformanceMetrics>>& results) {
        std::ofstream report("collective_benchmark_report.csv");
        if (!report.is_open()) {
            std::cerr << "Failed to open report file" << std::endl;
            return;
        }

        // Write CSV header
        report << "Collective,MessageSize,ExecutionTime,DataVolume,BandwidthUtilization" << std::endl;

        for (const auto& [collective, size_metrics] : results) {
            for (const auto& [size, metrics] : size_metrics) {
                report << collective << ","
                    << size << ","
                    << metrics.execution_time * 1000 << ","
                    << metrics.data_volume << ","
                    << metrics.bandwidth_utilization
                    << std::endl;
            }
        }

        report.close();
        std::cout << "\n=== Collective Benchmark Report Generated ===" << std::endl;
    }

    void analyze_collective_patterns(const std::map<std::string, std::map<int, PerformanceMetrics>>& results) {
        std::cout << "\n=== Collective Operations Analysis ===" << std::endl;

        // Analyze performance patterns
        for (int size : {1, 1024, 65536}) {
            if (results.at("Broadcast").count(size) &&
                results.at("Allreduce").count(size)) {

                double broadcast_time = results.at("Broadcast").at(size).execution_time;
                double allreduce_time = results.at("Allreduce").at(size).execution_time;
                double reduce_time = results.at("Reduce").at(size).execution_time;
                double allgather_time = results.at("Allgather").at(size).execution_time;

                std::cout << "Message Size: " << size << " elements" << std::endl;
                std::cout << "  Broadcast/Allreduce ratio: " << broadcast_time / allreduce_time << std::endl;
                std::cout << "  Reduce/Allreduce ratio: " << reduce_time / allreduce_time << std::endl;
                std::cout << "  Allgather/Allreduce ratio: " << allgather_time / allreduce_time << std::endl;
            }
        }
    }

private:
    void initialize_buffer(double* buffer, int size, int rank) {
        for (int i = 0; i < size; ++i) {
            buffer[i] = static_cast<double>(i + rank + 1);
        }
    }

    void verify_allreduce_result(double* result, int size) {
        bool correct = true;
        double expected_sum = 0.0;

        // Calculate expected sum
        for (int r = 0; r < world_size_; ++r) {
            for (int i = 0; i < size; ++i) {
                expected_sum += static_cast<double>(i + r + 1);
            }
        }

        for (int i = 0; i < size; ++i) {
            if (std::abs(result[i] - expected_sum) > 1e-9) {
                correct = false;
                break;
            }
        }

        if (!correct && world_rank_ == 0) {
            std::cerr << "WARNING: Allreduce result verification failed" << std::endl;
        }
    }

    void verify_allgather_result(double* result, int size) {
        bool correct = true;

        for (int r = 0; r < world_size_; ++r) {
            for (int i = 0; i < size; ++i) {
                double expected = static_cast<double>(i + r + 1);
                double actual = result[r * size + i];

                if (std::abs(actual - expected) > 1e-9) {
                    correct = false;
                    break;
                }
            }
            if (!correct) break;
        }

        if (!correct && world_rank_ == 0) {
            std::cerr << "WARNING: Allgather result verification failed" << std::endl;
        }
    }

    PerformanceMetrics calculate_metrics(const std::vector<double>& execution_times, int message_size) {
        PerformanceMetrics metrics;

        StatisticalAnalyzer analyzer;
        for (double time : execution_times) {
            analyzer.add_sample(time);
        }

        metrics.execution_time = analyzer.calculate_mean();
        metrics.standard_deviation = analyzer.calculate_standard_deviation();

        int type_size;
        MPI_Type_size(MPI_DOUBLE, &type_size);
        metrics.data_volume = message_size * type_size * world_size_;
        metrics.bandwidth_utilization = (metrics.data_volume / metrics.execution_time) / 1e9 * 100;

        return metrics;
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    CollectiveBenchmark benchmark(comm);
    benchmark.run_all_collectives_benchmark();

    MPI_Finalize();
    return 0;
}