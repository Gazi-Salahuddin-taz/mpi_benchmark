#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cstring>
#include <limits>
#include <cmath>
#include "../../src/core/collective_optimizer.h"
#include "../../src/utils/performance_measurement.h"

using namespace TopologyAwareResearch;

class ReduceBenchmark {
private:
    MPI_Comm comm_;
    int world_rank_, world_size_;
    CollectiveOptimizer optimizer_;
    PerformanceProfiler profiler_;

    std::vector<int> message_sizes_;
    std::vector<MPI_Op> operations_;
    int iterations_;
    int warmup_iterations_;

public:
    ReduceBenchmark(MPI_Comm comm)
        : comm_(comm),
        optimizer_(),
        profiler_(comm) {

        MPI_Comm_rank(comm_, &world_rank_);
        MPI_Comm_size(comm_, &world_size_);

        // Benchmark parameters
        message_sizes_ = { 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536 };
        operations_ = { MPI_SUM, MPI_MAX, MPI_MIN, MPI_PROD };
        iterations_ = 50;
        warmup_iterations_ = 5;
    }

    void run_comprehensive_reduce_benchmark() {
        if (world_rank_ == 0) {
            std::cout << "=== Comprehensive Reduce Operation Benchmark ===" << std::endl;
            std::cout << "Testing different operations, message sizes, and root positions" << std::endl;
        }

        // Test different root positions
        std::vector<int> roots = { 0, world_size_ / 4, world_size_ / 2, world_size_ - 1 };

        for (int root : roots) {
            if (world_rank_ == 0) {
                std::cout << "\n--- Testing with Root: " << root << " ---" << std::endl;
            }
            benchmark_root_performance(root);
        }

        // Compare different reduction algorithms
        if (world_rank_ == 0) {
            std::cout << "\n--- Algorithm Comparison ---" << std::endl;
        }
        compare_reduce_algorithms();
    }

    void benchmark_root_performance(int root) {
        std::map<MPI_Op, std::map<int, PerformanceMetrics>> results;

        for (MPI_Op op : operations_) {
            std::string op_name = get_operation_name(op);

            if (world_rank_ == 0) {
                std::cout << "Operation: " << op_name << std::endl;
            }

            for (int size : message_sizes_) {
                PerformanceMetrics metrics = benchmark_reduce_operation(size, op, root);
                results[op][size] = metrics;

                if (world_rank_ == 0) {
                    std::cout << "  Size " << size << ": "
                        << metrics.execution_time * 1000 << " ms" << std::endl;
                }
            }
        }

        if (world_rank_ == 0) {
            analyze_operation_performance(root, results);
        }
    }

    PerformanceMetrics benchmark_reduce_operation(int message_size, MPI_Op op, int root) {
        std::vector<double> send_buffer(message_size);
        std::vector<double> recv_buffer(message_size);
        initialize_buffer(send_buffer.data(), message_size, world_rank_);

        PerformanceMetrics metrics;
        std::vector<double> execution_times;

        // Warmup
        for (int i = 0; i < warmup_iterations_; ++i) {
            if (world_rank_ == root) {
                MPI_Reduce(MPI_IN_PLACE, recv_buffer.data(), message_size,
                    MPI_DOUBLE, op, root, comm_);
            }
            else {
                MPI_Reduce(send_buffer.data(), nullptr, message_size,
                    MPI_DOUBLE, op, root, comm_);
            }
        }

        // Measurement
        for (int i = 0; i < iterations_; ++i) {
            MPI_Barrier(comm_);
            auto start = MPI_Wtime();

            if (world_rank_ == root) {
                MPI_Reduce(MPI_IN_PLACE, recv_buffer.data(), message_size,
                    MPI_DOUBLE, op, root, comm_);
            }
            else {
                MPI_Reduce(send_buffer.data(), nullptr, message_size,
                    MPI_DOUBLE, op, root, comm_);
            }

            auto end = MPI_Wtime();
            execution_times.push_back(end - start);

            // Verify result on first iteration
            if (i == 0 && world_rank_ == root) {
                verify_reduce_result(recv_buffer.data(), message_size, op);
            }
        }

        metrics = calculate_metrics(execution_times, message_size);
        return metrics;
    }

    void compare_reduce_algorithms() {
        if (world_rank_ == 0) {
            std::cout << "Comparing different reduce algorithms..." << std::endl;
        }

        int test_size = 4096;
        MPI_Op test_op = MPI_SUM;
        int root = 0;

        // Benchmark different approaches
        auto native_metrics = benchmark_native_reduce(test_size, test_op, root);
        auto optimized_metrics = benchmark_optimized_reduce(test_size, test_op, root);
        auto binomial_metrics = benchmark_binomial_reduce(test_size, test_op, root);

        if (world_rank_ == 0) {
            std::cout << "Native MPI_Reduce: " << native_metrics.execution_time * 1000 << " ms" << std::endl;
            std::cout << "Optimized Reduce: " << optimized_metrics.execution_time * 1000 << " ms" << std::endl;
            std::cout << "Binomial Tree Reduce: " << binomial_metrics.execution_time * 1000 << " ms" << std::endl;

            double improvement = (native_metrics.execution_time - optimized_metrics.execution_time)
                / native_metrics.execution_time * 100;
            std::cout << "Optimization Improvement: " << improvement << "%" << std::endl;
        }
    }

    PerformanceMetrics benchmark_native_reduce(int message_size, MPI_Op op, int root) {
        std::vector<double> send_buffer(message_size);
        std::vector<double> recv_buffer(message_size);
        initialize_buffer(send_buffer.data(), message_size, world_rank_);

        PerformanceMetrics metrics;
        std::vector<double> execution_times;

        for (int i = 0; i < iterations_; ++i) {
            MPI_Barrier(comm_);
            auto start = MPI_Wtime();

            if (world_rank_ == root) {
                MPI_Reduce(MPI_IN_PLACE, recv_buffer.data(), message_size,
                    MPI_DOUBLE, op, root, comm_);
            }
            else {
                MPI_Reduce(send_buffer.data(), nullptr, message_size,
                    MPI_DOUBLE, op, root, comm_);
            }

            auto end = MPI_Wtime();
            execution_times.push_back(end - start);
        }

        metrics = calculate_metrics(execution_times, message_size);
        return metrics;
    }

    PerformanceMetrics benchmark_optimized_reduce(int message_size, MPI_Op op, int root) {
        std::vector<double> send_buffer(message_size);
        std::vector<double> recv_buffer(message_size);
        initialize_buffer(send_buffer.data(), message_size, world_rank_);

        PerformanceMetrics metrics;
        std::vector<double> execution_times;

        for (int i = 0; i < iterations_; ++i) {
            MPI_Barrier(comm_);
            auto start = MPI_Wtime();

            // Use the optimized reduce from CollectiveOptimizer
            metrics = optimizer_.optimize_reduce(send_buffer.data(), recv_buffer.data(),
                message_size, MPI_DOUBLE, op, root, comm_);

            auto end = MPI_Wtime();
            execution_times.push_back(end - start);
        }

        metrics.execution_time = calculate_mean(execution_times);
        return metrics;
    }

    PerformanceMetrics benchmark_binomial_reduce(int message_size, MPI_Op op, int root) {
        // Custom binomial tree reduce implementation
        std::vector<double> send_buffer(message_size);
        std::vector<double> recv_buffer(message_size);
        initialize_buffer(send_buffer.data(), message_size, world_rank_);

        PerformanceMetrics metrics;
        std::vector<double> execution_times;

        for (int i = 0; i < iterations_; ++i) {
            MPI_Barrier(comm_);
            auto start = MPI_Wtime();

            binomial_tree_reduce(send_buffer.data(), recv_buffer.data(),
                message_size, MPI_DOUBLE, op, root, comm_);

            auto end = MPI_Wtime();
            execution_times.push_back(end - start);
        }

        metrics.execution_time = calculate_mean(execution_times);
        return metrics;
    }

    void binomial_tree_reduce(const void* sendbuf, void* recvbuf, int count,
        MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        if (size == 1) {
            if (sendbuf != MPI_IN_PLACE) {
                MPI_Type_size(datatype, nullptr); // Get type size if needed
                memcpy(recvbuf, sendbuf, count * sizeof(double)); // Simplified
            }
            return;
        }

        // Temporary buffer for intermediate results
        std::vector<double> temp_buffer(count);

        int relative_rank = (rank - root + size) % size;
        int mask = 1;

        while (mask < size) {
            if (relative_rank & mask) {
                int source = (rank - mask + size) % size;
                MPI_Recv(temp_buffer.data(), count, datatype, source, 0, comm, MPI_STATUS_IGNORE);

                // Perform reduction
                if (op == MPI_SUM) {
                    const double* send_data = (rank == root) ? static_cast<const double*>(recvbuf)
                        : static_cast<const double*>(sendbuf);
                    for (int i = 0; i < count; ++i) {
                        static_cast<double*>(recvbuf)[i] = send_data[i] + temp_buffer[i];
                    }
                }
                // Add other operations as needed

                break;
            }
            mask <<= 1;
        }

        mask >>= 1;
        while (mask > 0) {
            if ((relative_rank + mask) < size) {
                int dest = (rank + mask) % size;
                MPI_Send(recvbuf, count, datatype, dest, 0, comm);
            }
            mask >>= 1;
        }
    }

    void analyze_operation_performance(int root,
        const std::map<MPI_Op, std::map<int, PerformanceMetrics>>& results) {
        std::cout << "\n--- Performance Analysis for Root " << root << " ---" << std::endl;

        for (int size : {1, 1024, 65536}) {
            if (results.at(MPI_SUM).count(size)) {
                std::cout << "Message Size: " << size << std::endl;

                for (MPI_Op op : operations_) {
                    if (results.at(op).count(size)) {
                        double time = results.at(op).at(size).execution_time;
                        std::cout << "  " << get_operation_name(op) << ": "
                            << time * 1000 << " ms" << std::endl;
                    }
                }
            }
        }
    }

    void verify_reduce_result(double* result, int size, MPI_Op op) {
        bool correct = true;

        // Convert MPI_Op to comparable values
        if (op == MPI_SUM) {
            double expected_sum = 0.0;
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
        }
        else if (op == MPI_MAX) {
            double expected_max = 0.0;
            for (int r = 0; r < world_size_; ++r) {
                for (int i = 0; i < size; ++i) {
                    double value = static_cast<double>(i + r + 1);
                    if (value > expected_max) expected_max = value;
                }
            }

            for (int i = 0; i < size; ++i) {
                if (std::abs(result[i] - expected_max) > 1e-9) {
                    correct = false;
                    break;
                }
            }
        }
        else if (op == MPI_MIN) {
            double expected_min = std::numeric_limits<double>::max();
            for (int r = 0; r < world_size_; ++r) {
                for (int i = 0; i < size; ++i) {
                    double value = static_cast<double>(i + r + 1);
                    if (value < expected_min) expected_min = value;
                }
            }

            for (int i = 0; i < size; ++i) {
                if (std::abs(result[i] - expected_min) > 1e-9) {
                    correct = false;
                    break;
                }
            }
        }
        else if (op == MPI_PROD) {
            double expected_prod = 1.0;
            for (int r = 0; r < world_size_; ++r) {
                for (int i = 0; i < size; ++i) {
                    expected_prod *= static_cast<double>(i + r + 1);
                }
            }

            for (int i = 0; i < size; ++i) {
                if (std::abs(result[i] - expected_prod) > 1e-9) {
                    correct = false;
                    break;
                }
            }
        }
        // Add other operations as needed

        if (!correct) {
            std::cerr << "WARNING: Reduce result verification failed for operation "
                      << get_operation_name(op) << std::endl;
        }
    }

    std::string get_operation_name(MPI_Op op) {
        if (op == MPI_SUM) return "MPI_SUM";
        if (op == MPI_MAX) return "MPI_MAX";
        if (op == MPI_MIN) return "MPI_MIN";
        if (op == MPI_PROD) return "MPI_PROD";
        return "UNKNOWN";
    }

private:
    void initialize_buffer(double* buffer, int size, int rank) {
        for (int i = 0; i < size; ++i) {
            buffer[i] = static_cast<double>(i + rank + 1);
        }
    }

    PerformanceMetrics calculate_metrics(const std::vector<double>& execution_times, int message_size) {
        PerformanceMetrics metrics;
        metrics.execution_time = calculate_mean(execution_times);

        int type_size;
        MPI_Type_size(MPI_DOUBLE, &type_size);
        metrics.data_volume = message_size * type_size * world_size_;

        return metrics;
    }

    double calculate_mean(const std::vector<double>& values) {
        if (values.empty()) return 0.0;
        double sum = 0.0;
        for (double v : values) sum += v;
        return sum / values.size();
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    ReduceBenchmark benchmark(comm);
    benchmark.run_comprehensive_reduce_benchmark();

    MPI_Finalize();
    return 0;
}