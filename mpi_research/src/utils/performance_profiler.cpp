#include "performance_measurement.h"
#include <iostream>
#include <fstream>

namespace TopologyAwareResearch {

PerformanceProfiler::PerformanceProfiler(MPI_Comm comm) : comm_(comm), detailed_profiling_(false) {
    MPI_Comm_rank(comm_, &world_rank_);
    MPI_Comm_size(comm_, &world_size_);
}

PerformanceProfiler::~PerformanceProfiler() {}

void PerformanceProfiler::start_profiling(const std::string& operation_name) {
    auto start_time = MPI_Wtime();
    start_times_[operation_name] = start_time;
}

void PerformanceProfiler::stop_profiling(const std::string& operation_name) {
    auto end_time = MPI_Wtime();
    if (start_times_.find(operation_name) != start_times_.end()) {
        double duration = end_time - start_times_[operation_name];
        if (profile_data_.find(operation_name) == profile_data_.end()) {
            profile_data_[operation_name] = PerformanceMetrics();
        }
        profile_data_[operation_name].execution_time = duration;
    }
}

void PerformanceProfiler::record_metrics(const std::string& operation_name, const PerformanceMetrics& metrics) {
    profile_data_[operation_name] = metrics;
}

void PerformanceProfiler::generate_profile_report(const std::string& filename) const {
    if (world_rank_ != 0) return;

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening profile report file: " << filename << std::endl;
        return;
    }

    file << "Performance Profile Report\n";
    file << "==========================\n\n";

    for (const auto& entry : profile_data_) {
        file << "Operation: " << entry.first << "\n";
        file << "Execution Time: " << entry.second.execution_time << " seconds\n";
        file << "Communication Time: " << entry.second.communication_time << " seconds\n";
        file << "Messages Sent: " << entry.second.messages_sent << "\n";
        file << "Data Volume: " << entry.second.data_volume << " bytes\n";
        file << "Bandwidth Utilization: " << entry.second.bandwidth_utilization << "%\n\n";
    }

    file.close();
}

void PerformanceProfiler::compare_operations() const {
    if (world_rank_ != 0) return;

    std::cout << "=== Operation Comparison ===" << std::endl;
    for (const auto& entry : profile_data_) {
        std::cout << "Operation: " << entry.first
                  << ", Time: " << entry.second.execution_time << " s"
                  << ", Messages: " << entry.second.messages_sent
                  << ", Data: " << entry.second.data_volume << " bytes" << std::endl;
    }
    std::cout << "===========================" << std::endl;
}

PerformanceMetrics PerformanceProfiler::get_operation_metrics(const std::string& operation_name) const {
    auto it = profile_data_.find(operation_name);
    if (it != profile_data_.end()) {
        return it->second;
    }
    return PerformanceMetrics();
}

void PerformanceProfiler::profile_collective_operation(const std::string& operation_name,
    std::function<void()> operation,
    int iterations) {

    PerformanceMetrics metrics;
    double total_time = 0.0;

    for (int i = 0; i < iterations; ++i) {
        MPI_Barrier(comm_);
        double start = MPI_Wtime();
        operation();
        double end = MPI_Wtime();
        total_time += (end - start);
    }

    metrics.execution_time = total_time / iterations;
    record_metrics(operation_name, metrics);
}

void PerformanceProfiler::profile_point_to_point(int src, int dst, int message_size, int iterations) {
    int world_rank;
    MPI_Comm_rank(comm_, &world_rank);

    std::vector<char> buffer(message_size);
    double total_time = 0.0;

    for (int i = 0; i < iterations; ++i) {
        MPI_Barrier(comm_);
        double start = MPI_Wtime();

        if (world_rank == src) {
            MPI_Send(buffer.data(), message_size, MPI_BYTE, dst, 0, comm_);
        } else if (world_rank == dst) {
            MPI_Recv(buffer.data(), message_size, MPI_BYTE, src, 0, comm_, MPI_STATUS_IGNORE);
        }

        double end = MPI_Wtime();
        if (world_rank == src || world_rank == dst) {
            total_time += (end - start);
        }
    }

    PerformanceMetrics metrics;
    if (world_rank == src || world_rank == dst) {
        metrics.execution_time = total_time / iterations;
    }
    metrics.messages_sent = (world_rank == src || world_rank == dst) ? iterations : 0;
    metrics.data_volume = (world_rank == src || world_rank == dst) ? message_size * iterations : 0;

    std::string operation_name = "Point-to-Point " + std::to_string(src) + "->" + std::to_string(dst);
    record_metrics(operation_name, metrics);
}

void PerformanceProfiler::gather_profile_data() {
    // Implementation for gathering data from all processes
}

PerformanceMetrics PerformanceProfiler::calculate_statistics(const std::vector<PerformanceMetrics>& samples) const {
    PerformanceMetrics stats;
    if (samples.empty()) return stats;

    // Calculate average execution time
    double total_time = 0.0;
    for (const auto& sample : samples) {
        total_time += sample.execution_time;
    }
    stats.execution_time = total_time / samples.size();

    return stats;
}

} // namespace TopologyAwareResearch