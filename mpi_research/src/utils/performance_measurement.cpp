#include "performance_measurement.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>

namespace TopologyAwareResearch {

    PerformanceTimer::PerformanceTimer() {
        reset();
    }

    PerformanceTimer::~PerformanceTimer() {}

    void PerformanceTimer::start(const std::string& section_name) {
        current_section_ = section_name;
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void PerformanceTimer::stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);

        timings_[current_section_] += duration.count() / 1000.0; // Convert to milliseconds
        current_section_.clear();
    }

    void PerformanceTimer::reset() {
        timings_.clear();
        current_section_.clear();
    }

    double PerformanceTimer::get_elapsed_time(const std::string& section_name) const {
        auto it = timings_.find(section_name);
        if (it != timings_.end()) {
            return it->second;
        }
        return 0.0;
    }

    std::map<std::string, double> PerformanceTimer::get_all_timings() const {
        return timings_;
    }

    void PerformanceTimer::print_timings() const {
        std::cout << "=== Performance Timings ===" << std::endl;
        for (const auto& timing : timings_) {
            std::cout << timing.first << ": " << timing.second << " ms" << std::endl;
        }
        std::cout << "===========================" << std::endl;
    }

    double PerformanceTimer::get_current_time() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
    }

    void PerformanceTimer::busy_wait_us(int microseconds) {
        auto start = std::chrono::high_resolution_clock::now();
        auto end = start + std::chrono::microseconds(microseconds);

        while (std::chrono::high_resolution_clock::now() < end) {
            // Busy wait
            std::this_thread::yield();
        }
    }

    // BandwidthMeasurer implementation
    BandwidthMeasurer::BandwidthMeasurer(MPI_Comm comm)
        : comm_(comm), iterations_(10), warmup_iterations_(5) {
    }

    BandwidthMeasurer::~BandwidthMeasurer() {}

    double BandwidthMeasurer::measure_point_to_point_bandwidth(int src_rank, int dst_rank,
        int message_size, int iterations) {
        int world_rank;
        MPI_Comm_rank(comm_, &world_rank);

        if (world_rank != src_rank && world_rank != dst_rank) {
            return 0.0;
        }

        std::vector<char> buffer(message_size);
        double total_bandwidth = 0.0;

        // Warmup
        warmup(src_rank, dst_rank, message_size);

        for (int i = 0; i < iterations; ++i) {
            MPI_Barrier(comm_);

            auto start_time = MPI_Wtime();

            if (world_rank == src_rank) {
                MPI_Send(buffer.data(), message_size, MPI_BYTE, dst_rank, 0, comm_);
            }
            else if (world_rank == dst_rank) {
                MPI_Recv(buffer.data(), message_size, MPI_BYTE, src_rank, 0, comm_, MPI_STATUS_IGNORE);
            }

            auto end_time = MPI_Wtime();

            if (world_rank == src_rank || world_rank == dst_rank) {
                double elapsed = end_time - start_time;
                double bandwidth = (message_size * 8.0) / (elapsed * 1e6); // Mbps
                total_bandwidth += bandwidth;
            }
        }

        return total_bandwidth / iterations;
    }

    std::vector<std::vector<double>> BandwidthMeasurer::measure_all_to_all_bandwidth(int message_size) {
        int world_size, world_rank;
        MPI_Comm_size(comm_, &world_size);
        MPI_Comm_rank(comm_, &world_rank);

        std::vector<std::vector<double>> bandwidth_matrix(world_size,
            std::vector<double>(world_size, 0.0));

        // Measure bandwidth between all pairs
        for (int src = 0; src < world_size; ++src) {
            for (int dst = src + 1; dst < world_size; ++dst) {
                double bandwidth = measure_point_to_point_bandwidth(src, dst, message_size, iterations_);
                bandwidth_matrix[src][dst] = bandwidth;
                bandwidth_matrix[dst][src] = bandwidth;
            }
        }

        return bandwidth_matrix;
    }

    void BandwidthMeasurer::warmup(int src_rank, int dst_rank, int message_size) {
        int world_rank;
        MPI_Comm_rank(comm_, &world_rank);

        std::vector<char> buffer(message_size);

        for (int i = 0; i < warmup_iterations_; ++i) {
            if (world_rank == src_rank) {
                MPI_Send(buffer.data(), message_size, MPI_BYTE, dst_rank, 0, comm_);
            }
            else if (world_rank == dst_rank) {
                MPI_Recv(buffer.data(), message_size, MPI_BYTE, src_rank, 0, comm_, MPI_STATUS_IGNORE);
            }
        }
    }

    // LatencyMeasurer implementation
    LatencyMeasurer::LatencyMeasurer(MPI_Comm comm)
        : comm_(comm), iterations_(1000), warmup_iterations_(100) {
    }

    LatencyMeasurer::~LatencyMeasurer() {}

    double LatencyMeasurer::measure_point_to_point_latency(int src_rank, int dst_rank, int iterations) {
        int world_rank;
        MPI_Comm_rank(comm_, &world_rank);

        if (world_rank != src_rank && world_rank != dst_rank) {
            return 0.0;
        }

        return ping_pong_latency(src_rank, dst_rank, iterations);
    }

    double LatencyMeasurer::ping_pong_latency(int rank1, int rank2, int iterations) {
        int world_rank;
        MPI_Comm_rank(comm_, &world_rank);

        int buffer = 42;
        double total_latency = 0.0;

        // Warmup
        for (int i = 0; i < warmup_iterations_; ++i) {
            if (world_rank == rank1) {
                MPI_Send(&buffer, 1, MPI_INT, rank2, 0, comm_);
                MPI_Recv(&buffer, 1, MPI_INT, rank2, 0, comm_, MPI_STATUS_IGNORE);
            }
            else if (world_rank == rank2) {
                MPI_Recv(&buffer, 1, MPI_INT, rank1, 0, comm_, MPI_STATUS_IGNORE);
                MPI_Send(&buffer, 1, MPI_INT, rank1, 0, comm_);
            }
        }

        // Actual measurement
        for (int i = 0; i < iterations; ++i) {
            MPI_Barrier(comm_);

            auto start_time = MPI_Wtime();

            if (world_rank == rank1) {
                MPI_Send(&buffer, 1, MPI_INT, rank2, 0, comm_);
                MPI_Recv(&buffer, 1, MPI_INT, rank2, 0, comm_, MPI_STATUS_IGNORE);
            }
            else if (world_rank == rank2) {
                MPI_Recv(&buffer, 1, MPI_INT, rank1, 0, comm_, MPI_STATUS_IGNORE);
                MPI_Send(&buffer, 1, MPI_INT, rank1, 0, comm_);
            }

            auto end_time = MPI_Wtime();

            if (world_rank == rank1 || world_rank == rank2) {
                double round_trip = end_time - start_time;
                total_latency += (round_trip * 1e6) / 2.0; // One-way latency in microseconds
            }
        }

        return total_latency / iterations;
    }

    // StatisticalAnalyzer implementation has been REMOVED to avoid duplicate definitions
    // The implementation is now only in statistical_analyzer.cpp

} // namespace TopologyAwareResearch