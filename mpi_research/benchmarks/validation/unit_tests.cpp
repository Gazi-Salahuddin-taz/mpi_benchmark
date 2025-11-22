#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include "../../src/core/collective_optimizer.h"
#include "../../src/core/topology_detection.h"
#include "../../src/utils/performance_measurement.h"

using namespace TopologyAwareResearch;

class UnitTests {
private:
    MPI_Comm comm_;
    int world_rank_, world_size_;

public:
    UnitTests(MPI_Comm comm) : comm_(comm) {
        MPI_Comm_rank(comm_, &world_rank_);
        MPI_Comm_size(comm_, &world_size_);
    }

    bool run_all_unit_tests() {
        if (world_rank_ == 0) {
            std::cout << "=== Running Unit Tests ===" << std::endl;
        }

        bool all_passed = true;

        // Test utility functions
        all_passed &= test_performance_timer();
        all_passed &= test_statistical_analyzer();

        // Test topology detection
        all_passed &= test_topology_detection();

        // Test network characteristics
        all_passed &= test_network_characteristics();

        // Test performance metrics
        all_passed &= test_performance_metrics();

        if (world_rank_ == 0) {
            if (all_passed) {
                std::cout << "=== ALL UNIT TESTS PASSED ===" << std::endl;
            }
            else {
                std::cout << "=== SOME UNIT TESTS FAILED ===" << std::endl;
            }
        }

        return all_passed;
    }

    bool test_performance_timer() {
        if (world_rank_ == 0) {
            std::cout << "Testing PerformanceTimer..." << std::endl;
        }

        PerformanceTimer timer;

        // Test basic timing
        timer.start("test_section");
        PerformanceTimer::busy_wait_us(1000); // 1ms delay
        timer.stop();

        double elapsed = timer.get_elapsed_time("test_section");
        bool passed = (elapsed >= 0.9 && elapsed <= 2.0); // Allow some tolerance

        if (world_rank_ == 0) {
            std::cout << "  Basic timing: " << (passed ? "PASSED" : "FAILED")
                << " (" << elapsed << " ms)" << std::endl;
        }

        // Test multiple sections
        timer.start("section1");
        PerformanceTimer::busy_wait_us(500);
        timer.stop();

        timer.start("section2");
        PerformanceTimer::busy_wait_us(500);
        timer.stop();

        double section1_time = timer.get_elapsed_time("section1");
        double section2_time = timer.get_elapsed_time("section2");

        passed &= (section1_time >= 0.4 && section1_time <= 1.0);
        passed &= (section2_time >= 0.4 && section2_time <= 1.0);

        if (world_rank_ == 0) {
            std::cout << "  Multiple sections: " << (passed ? "PASSED" : "FAILED") << std::endl;
        }

        return passed;
    }

    bool test_statistical_analyzer() {
        if (world_rank_ == 0) {
            std::cout << "Testing StatisticalAnalyzer..." << std::endl;
        }

        StatisticalAnalyzer analyzer;

        // Test with known data set
        std::vector<double> test_data = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        for (double value : test_data) {
            analyzer.add_sample(value);
        }

        bool passed = true;

        // Test mean calculation
        double mean = analyzer.calculate_mean();
        passed &= (std::abs(mean - 3.0) < 1e-9);

        // Test median calculation
        double median = analyzer.calculate_median();
        passed &= (std::abs(median - 3.0) < 1e-9);

        // Test standard deviation
        double stddev = analyzer.calculate_standard_deviation();
        double expected_stddev = std::sqrt(2.0); // population stddev of {1,2,3,4,5}
        passed &= (std::abs(stddev - expected_stddev) < 1e-9);

        // Test variance
        double variance = analyzer.calculate_variance();
        passed &= (std::abs(variance - 2.0) < 1e-9);

        // Test min/max
        double min_val = analyzer.calculate_min();
        double max_val = analyzer.calculate_max();
        passed &= (std::abs(min_val - 1.0) < 1e-9);
        passed &= (std::abs(max_val - 5.0) < 1e-9);

        if (world_rank_ == 0) {
            std::cout << "  Basic statistics: " << (passed ? "PASSED" : "FAILED") << std::endl;
            std::cout << "    Mean: " << mean << " (expected: 3.0)" << std::endl;
            std::cout << "    StdDev: " << stddev << " (expected: " << expected_stddev << ")" << std::endl;
        }

        // Test outlier detection
        analyzer.clear_samples();
        std::vector<double> data_with_outliers = { 1.0, 2.0, 3.0, 4.0, 5.0, 100.0 }; // 100 is an outlier
        for (double value : data_with_outliers) {
            analyzer.add_sample(value);
        }

        auto outliers = analyzer.detect_outliers();
        bool outlier_detected = (outliers.size() == 1 && std::abs(outliers[0] - 100.0) < 1e-9);
        passed &= outlier_detected;

        if (world_rank_ == 0) {
            std::cout << "  Outlier detection: " << (outlier_detected ? "PASSED" : "FAILED") << std::endl;
        }

        return passed;
    }

    bool test_topology_detection() {
        if (world_rank_ == 0) {
            std::cout << "Testing Topology Detection..." << std::endl;
        }

        TopologyDetector detector;
        NetworkTopologyInfo topology = detector.detect_topology(comm_);

        bool passed = true;

        // Basic validation of topology information
        passed &= (topology.total_processes == world_size_);
        passed &= (topology.total_nodes > 0);
        passed &= (topology.processes_per_node > 0);
        passed &= (!topology.topology_type.empty());

        if (world_rank_ == 0) {
            std::cout << "  Basic topology info: " << (passed ? "PASSED" : "FAILED") << std::endl;
            std::cout << "    Processes: " << topology.total_processes << std::endl;
            std::cout << "    Nodes: " << topology.total_nodes << std::endl;
            std::cout << "    Processes per node: " << topology.processes_per_node << std::endl;
            std::cout << "    Topology type: " << topology.topology_type << std::endl;
        }

        // Validate node mapping
        passed &= (topology.node_mapping.size() == static_cast<size_t>(world_size_));
        for (int i = 0; i < world_size_; ++i) {
            int node_id = topology.node_mapping[i];
            passed &= (node_id >= 0 && node_id < topology.total_nodes);
        }

        if (world_rank_ == 0) {
            std::cout << "  Node mapping: " << (passed ? "PASSED" : "FAILED") << std::endl;
        }

        // Validate node processes
        passed &= (topology.node_processes.size() == static_cast<size_t>(topology.total_nodes));
        int total_mapped_processes = 0;
        for (const auto& processes : topology.node_processes) {
            total_mapped_processes += processes.size();
        }
        passed &= (total_mapped_processes == world_size_);

        if (world_rank_ == 0) {
            std::cout << "  Node processes: " << (passed ? "PASSED" : "FAILED") << std::endl;
        }

        return passed;
    }

    bool test_network_characteristics() {
        if (world_rank_ == 0) {
            std::cout << "Testing Network Characteristics..." << std::endl;
        }

        NetworkCharacteristics config;
        config.total_nodes = 4;
        config.processes_per_node = 2;
        config.total_processes = 8;
        config.topology = NetworkTopology::FAT_TREE;
        config.inter_node_bandwidth = 25.0;
        config.intra_node_bandwidth = 100.0;
        config.inter_node_latency = 2.0;
        config.intra_node_latency = 0.1;

        bool passed = true;

        // Test basic characteristics
        passed &= (config.total_nodes == 4);
        passed &= (config.processes_per_node == 2);
        passed &= (config.total_processes == 8);
        passed &= (config.topology == NetworkTopology::FAT_TREE);
        passed &= (config.inter_node_bandwidth > 0);
        passed &= (config.intra_node_bandwidth > 0);

        if (world_rank_ == 0) {
            std::cout << "  Basic characteristics: " << (passed ? "PASSED" : "FAILED") << std::endl;
        }

        // Test communication cost matrix
        int world_size = 8;
        config.communication_costs.resize(world_size, std::vector<double>(world_size, 1.0));

        // Initialize cost matrix (simplified)
        for (int i = 0; i < world_size; ++i) {
            for (int j = 0; j < world_size; ++j) {
                int node_i = i / config.processes_per_node;
                int node_j = j / config.processes_per_node;

                if (node_i == node_j) {
                    config.communication_costs[i][j] = 0.1; // Intra-node
                }
                else {
                    config.communication_costs[i][j] = 1.0; // Inter-node
                }
            }
        }

        // Validate cost matrix
        passed &= (config.communication_costs.size() == static_cast<size_t>(world_size));
        for (int i = 0; i < world_size; ++i) {
            passed &= (config.communication_costs[i].size() == static_cast<size_t>(world_size));
            passed &= (config.communication_costs[i][i] == 0.0); // Self-communication should be 0
        }

        if (world_rank_ == 0) {
            std::cout << "  Communication costs: " << (passed ? "PASSED" : "FAILED") << std::endl;
        }

        return passed;
    }

    bool test_performance_metrics() {
        if (world_rank_ == 0) {
            std::cout << "Testing Performance Metrics..." << std::endl;
        }

        PerformanceMetrics metrics;
        metrics.execution_time = 1.5;
        metrics.communication_time = 1.0;
        metrics.computation_time = 0.5;
        metrics.bandwidth_utilization = 85.5;
        metrics.energy_consumption = 0.25;
        metrics.messages_sent = 31;
        metrics.data_volume = 31744;
        metrics.scalability_factor = 0.92;
        metrics.load_imbalance = 0.05;

        bool passed = true;

        // Test metric values
        passed &= (metrics.execution_time == 1.5);
        passed &= (metrics.communication_time == 1.0);
        passed &= (metrics.computation_time == 0.5);
        passed &= (metrics.bandwidth_utilization == 85.5);
        passed &= (metrics.energy_consumption == 0.25);
        passed &= (metrics.messages_sent == 31);
        passed &= (metrics.data_volume == 31744);

        // Test derived calculations
        double communication_ratio = metrics.communication_time / metrics.execution_time;
        passed &= (std::abs(communication_ratio - 0.666) < 0.1);

        if (world_rank_ == 0) {
            std::cout << "  Basic metrics: " << (passed ? "PASSED" : "FAILED") << std::endl;
            std::cout << "    Execution time: " << metrics.execution_time << std::endl;
            std::cout << "    Communication ratio: " << communication_ratio << std::endl;
            std::cout << "    Bandwidth utilization: " << metrics.bandwidth_utilization << "%" << std::endl;
        }

        // Test objective scores
        metrics.objective_scores["latency"] = 0.8;
        metrics.objective_scores["bandwidth"] = 0.9;
        metrics.objective_scores["energy"] = 0.7;

        passed &= (metrics.objective_scores.size() == 3);
        passed &= (metrics.objective_scores["latency"] == 0.8);
        passed &= (metrics.objective_scores["bandwidth"] == 0.9);
        passed &= (metrics.objective_scores["energy"] == 0.7);

        if (world_rank_ == 0) {
            std::cout << "  Objective scores: " << (passed ? "PASSED" : "FAILED") << std::endl;
        }

        return passed;
    }
};

// Test runner with MPI-aware reporting
class TestRunner {
private:
    MPI_Comm comm_;
    int world_rank_, world_size_;

public:
    TestRunner(MPI_Comm comm) : comm_(comm) {
        MPI_Comm_rank(comm_, &world_rank_);
        MPI_Comm_size(comm_, &world_size_);
    }

    void run_tests_with_reporting() {
        UnitTests tests(comm_);

        // Run tests on all processes
        bool local_passed = tests.run_all_unit_tests();

        // Gather results
        int local_result = local_passed ? 1 : 0;
        int global_result;
        MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN, comm_);

        if (world_rank_ == 0) {
            std::cout << "\n=== GLOBAL TEST SUMMARY ===" << std::endl;
            if (global_result == 1) {
                std::cout << "ALL PROCESSES: ALL TESTS PASSED" << std::endl;
            }
            else {
                std::cout << "SOME PROCESSES: TESTS FAILED" << std::endl;

                // Identify which processes failed
                std::vector<int> all_results(world_size_);
                MPI_Gather(&local_result, 1, MPI_INT,
                    all_results.data(), 1, MPI_INT, 0, comm_);

                std::cout << "Failed processes: ";
                for (int i = 0; i < world_size_; ++i) {
                    if (all_results[i] == 0) {
                        std::cout << i << " ";
                    }
                }
                std::cout << std::endl;
            }
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    TestRunner runner(comm);
    runner.run_tests_with_reporting();

    MPI_Finalize();
    return 0;
}