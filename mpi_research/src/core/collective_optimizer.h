#ifndef COLLECTIVE_OPTIMIZER_H
#define COLLECTIVE_OPTIMIZER_H

#include <mpi.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <functional>

namespace TopologyAwareResearch {

    // Advanced optimization objectives
    enum class OptimizationObjective {
        MINIMIZE_LATENCY,
        MAXIMIZE_BANDWIDTH,
        MINIMIZE_ENERGY,
        BALANCED_OPTIMIZATION,
        ADAPTIVE_OPTIMIZATION
    };

    // Network topology types
    enum class NetworkTopology {
        FAT_TREE,
        TORUS_2D,
        TORUS_3D,
        DRAGONFLY,
        MULTI_CORE,
        CUSTOM,
        UNKNOWN
    };

    // Algorithm types with advanced variants
    enum class AlgorithmType {
        // Basic algorithms
        BINOMIAL_TREE,
        PIPELINE_RING,
        RING_ALLREDUCE,

        // Advanced topology-aware
        TOPOLOGY_AWARE_BROADCAST,
        HIERARCHICAL_BROADCAST,
        MULTI_LEVEL_REDUCE,
        ADAPTIVE_ALLREDUCE,

        // Graph-based
        SHORTEST_PATH_TREE,
        MINIMUM_SPANNING_TREE,
        STEINER_TREE,

        // Native for comparison
        NATIVE_MPI
    };

    // Advanced performance metrics
    struct PerformanceMetrics {
        double execution_time;
        double communication_time;
        double computation_time;
        double bandwidth_utilization;  // Percentage
        double network_efficiency;
        double energy_consumption;     // Joules (estimated)
        int messages_sent;
        int bytes_transferred;  // Add this
        int bytes_processed;    // Add this if needed
        double data_volume;           // Bytes
        double scalability_factor;
        double load_imbalance;

        // Multi-objective scores
        std::map<std::string, double> objective_scores;

        // Communication pattern
        std::vector<int> communication_sequence;
        std::vector<std::pair<int, int>> communication_edges;

        // Statistical information
        double confidence_interval;
        double standard_deviation;

        PerformanceMetrics() :
        execution_time(0.0),
        computation_time(0.0),
        communication_time(0.0),
        bandwidth_utilization(0.0),
        energy_consumption(0.0),
        messages_sent(0),
        bytes_transferred(0),
        bytes_processed(0) {}
    };

    // Advanced network characteristics
    struct NetworkCharacteristics {
        NetworkTopology topology;
        int total_nodes;
        int processes_per_node;
        int total_processes;

        // Performance characteristics
        double inter_node_bandwidth;  // GB/s
        double intra_node_bandwidth;  // GB/s
        double inter_node_latency;    // microseconds
        double intra_node_latency;    // microseconds

        // Topology-specific parameters
//        union {
//            struct { int k; } fat_tree;           // k-ary fat tree
//            struct { int x, y, z; } torus;        // 3D torus dimensions
//            struct { int groups, routers; } dragonfly; // Dragonfly parameters
//        } topology_params;
        union {
            struct {
                int x, y, z;  // For torus
            } torus;
            struct {
                int levels;    // For fat tree
            } fat_tree;
            struct {
                int groups;
                int routers_per_group;
                int nodes_per_router;
                int global_links;
            } dragonfly;
            struct {
                int branching_factor;
            } tree;
        } topology_params;
        // Process mapping
        std::vector<int> node_mapping;  // process_id -> node_id
        std::vector<std::vector<int>> node_processes;  // node_id -> process_ids

        // Communication cost matrix (simplified)
        std::vector<std::vector<double>> communication_costs;

        NetworkCharacteristics() : topology(NetworkTopology::UNKNOWN),
            total_nodes(0), processes_per_node(0),
            total_processes(0), inter_node_bandwidth(0.0),
            intra_node_bandwidth(0.0), inter_node_latency(0.0),
            intra_node_latency(0.0) {
            topology_params.torus.x = 0;
            topology_params.torus.y = 0;
            topology_params.torus.z = 0;
            topology_params.fat_tree.levels = 0;
            topology_params.dragonfly.groups = 0;
            topology_params.dragonfly.routers_per_group = 0;
            topology_params.dragonfly.nodes_per_router = 0;
            topology_params.dragonfly.global_links = 0;
            topology_params.tree.branching_factor = 0;
        }
    };

    // Multi-objective optimization result
    struct ParetoSolution {
        std::vector<double> objectives;
        std::vector<int> communication_sequence;
        AlgorithmType algorithm;
        double dominance_rank;
        double crowding_distance;

        bool operator<(const ParetoSolution& other) const {
            return dominance_rank < other.dominance_rank ||
                (dominance_rank == other.dominance_rank &&
                    crowding_distance > other.crowding_distance);
        }
    };

    // ILP solution for communication synthesis
    struct ILPSolution {
        std::vector<int> variable_values;
        double objective_value;
        bool feasible;
        int solve_time_ms;
        std::string status;

        ILPSolution() : objective_value(0.0), feasible(false), solve_time_ms(0) {}
    };
    struct DragonflyParams {
        int routers_per_group;
        int nodes_per_router;
        // ... other parameters
    };
    class CollectiveOptimizer {
    private:
        NetworkCharacteristics network_config_;
        OptimizationObjective current_objective_;
        std::map<AlgorithmType, PerformanceMetrics> performance_history_;
        std::vector<ParetoSolution> pareto_front_;

        // Configuration flags
        bool topology_aware_enabled_;
        bool use_ilp_optimization_;
        bool use_multi_objective_;
        bool adaptive_optimization_;

        // Optimization parameters
        int pipeline_depth_;
        int segmentation_factor_;
        double energy_weight_;
        double latency_weight_;
        double bandwidth_weight_;

        // Advanced components
        class ILPOptimizer* ilp_optimizer_;
        class GraphOptimizer* graph_optimizer_;
        class TopologyDetector* topology_detector_;

    public:
        CollectiveOptimizer();
        ~CollectiveOptimizer();

        // Main optimization interface
        PerformanceMetrics optimize_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        PerformanceMetrics optimize_allreduce(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Op op, MPI_Comm comm);

        PerformanceMetrics optimize_allgather(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Comm comm);

        PerformanceMetrics optimize_reduce(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Op op, int root, MPI_Comm comm);

        PerformanceMetrics optimize_barrier(MPI_Comm comm);

        PerformanceMetrics binomial_tree_broadcast(void* buffer, int count,
                                             MPI_Datatype datatype, int root,
                                             MPI_Comm comm);

        // Multi-objective optimization
        std::vector<ParetoSolution> multi_objective_optimization(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm,
            const std::vector<double>& weights);

        std::vector<ParetoSolution> get_pareto_front() const { return pareto_front_; }

        // Advanced configuration
        void set_optimization_objective(OptimizationObjective objective);
        void set_multi_objective_weights(double latency_weight, double bandwidth_weight, double energy_weight);
        void enable_adaptive_optimization(bool enable);
        void set_ilp_timeout(int timeout_ms);
        void set_topology_characteristics(const NetworkCharacteristics& config);

        // Analysis and reporting
        void generate_performance_report(const std::string& filename) const;
        void compare_algorithms() const;
        PerformanceMetrics get_best_performance(AlgorithmType algo) const;
        NetworkCharacteristics get_network_characteristics() const { return network_config_; }

        // Statistical analysis
        void perform_statistical_analysis(MPI_Comm comm);
        double calculate_confidence_interval(const std::vector<double>& samples, double confidence = 0.95) const;
        bool dominates(const ParetoSolution& a, const ParetoSolution& b);
    private:
        // Core algorithm implementations
//        PerformanceMetrics binomial_tree_broadcast(void* buffer, int count,
//            MPI_Datatype datatype, int root,
//            MPI_Comm comm);

        PerformanceMetrics pipeline_ring_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        PerformanceMetrics topology_aware_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        PerformanceMetrics hierarchical_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        PerformanceMetrics ring_allreduce(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Op op, MPI_Comm comm);

        PerformanceMetrics adaptive_allreduce(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Op op, MPI_Comm comm);

        // Topology-specific optimizations
        PerformanceMetrics fat_tree_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        PerformanceMetrics torus_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        PerformanceMetrics dragonfly_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        // Utility methods
        NetworkCharacteristics detect_topology(MPI_Comm comm);
        AlgorithmType select_optimal_algorithm(int message_size, MPI_Comm comm);
        void update_performance_history(AlgorithmType algo, const PerformanceMetrics& metrics);
        double estimate_communication_cost(int src, int dst, int message_size) const;

        // Multi-objective optimization
        std::vector<double> evaluate_objectives(const PerformanceMetrics& metrics) const;
        void update_pareto_front(const ParetoSolution& solution);
        void non_dominated_sort(std::vector<ParetoSolution>& solutions);
        void calculate_crowding_distance(std::vector<ParetoSolution>& solutions);

        // ILP and graph theory
        ILPSolution solve_ilp_problem(int root, int world_size, int message_size);
        std::vector<int> synthesize_communication_sequence(int root, int world_size);

        // Energy estimation
        double estimate_energy_consumption(const PerformanceMetrics& metrics) const;

        // Load balancing
        double calculate_load_imbalance(const std::vector<double>& execution_times) const;
    };

    // Advanced utility functions
    namespace OptimizationUtils {
        double calculate_bandwidth_efficiency(const PerformanceMetrics& metrics);
        double calculate_scalability_factor(int base_processes, double base_time,
            int scaled_processes, double scaled_time);
        bool is_statistically_significant(const PerformanceMetrics& algo1,
            const PerformanceMetrics& algo2,
            double confidence_level = 0.95);
        std::vector<int> generate_binomial_tree_sequence(int root, int world_size);
        std::vector<int> generate_ring_sequence(int root, int world_size);

        // Graph algorithms
        std::vector<std::pair<int, int>> minimum_spanning_tree(const std::vector<std::vector<double>>& cost_matrix);
        std::vector<int> shortest_path_tree(int root, const std::vector<std::vector<double>>& cost_matrix);
        double graph_diameter(const std::vector<std::vector<double>>& cost_matrix);
    }

} // namespace TopologyAwareResearch

#endif