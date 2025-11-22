#ifndef TOPOLOGY_AWARE_BROADCAST_H
#define TOPOLOGY_AWARE_BROADCAST_H

#include <mpi.h>
#include <vector>
#include <memory>
#include <map>
#include "../core/collective_optimizer.h"

namespace TopologyAwareResearch {
    MPI_Comm create_rack_communicator(MPI_Comm comm);
    int calculate_optimal_segment_size(int total_count, int processes,
                                      const NetworkCharacteristics& network);
    class TopologyAwareBroadcast {
    private:
        NetworkCharacteristics network_config_;
        bool use_optimized_paths_;
        int pipeline_depth_;

        // Communication trees
        std::map<int, std::vector<int>> broadcast_trees_;
        std::map<int, std::vector<std::pair<int, int>>> communication_schedules_;

    public:
        TopologyAwareBroadcast(const NetworkCharacteristics& config);
        ~TopologyAwareBroadcast();

        // Main broadcast interface
        PerformanceMetrics broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        // Topology-specific implementations
        PerformanceMetrics fat_tree_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        PerformanceMetrics torus_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        PerformanceMetrics dragonfly_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        PerformanceMetrics multi_core_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        // Advanced broadcast variants
        PerformanceMetrics pipeline_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        PerformanceMetrics binomial_tree_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        PerformanceMetrics k_ary_tree_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm, int k = 4);

        PerformanceMetrics hierarchical_broadcast(void* buffer, int count,
                                            MPI_Datatype datatype, int root,
                                            MPI_Comm comm);
        // Optimization methods
        void build_optimal_broadcast_tree(int root, MPI_Comm comm);
        std::vector<int> get_broadcast_sequence(int root, MPI_Comm comm);
        void optimize_communication_schedule(int root, MPI_Comm comm);

    private:
        // Tree construction algorithms
        std::vector<int> construct_binomial_tree(int root, int world_size);
        std::vector<int> construct_k_ary_tree(int root, int world_size, int k);
        std::vector<int> construct_shortest_path_tree(int root, const std::vector<std::vector<double>>& costs);
        std::vector<int> construct_minimum_spanning_tree(int root, const std::vector<std::vector<double>>& costs);

        // Utility methods
        double estimate_communication_cost(int src, int dst, int message_size) const;
        bool is_same_node(int rank1, int rank2) const;
        int get_node_id(int rank) const;
        std::vector<int> get_ranks_on_same_node(int rank) const;

        // Performance optimization
        void segment_message(void* buffer, int count, MPI_Datatype datatype,
            std::vector<void*>& segments,
            std::vector<int>& segment_sizes);
        void reassemble_message(void* buffer, int count, MPI_Datatype datatype,
            const std::vector<void*>& segments,
            const std::vector<int>& segment_sizes);
    };

    class HierarchicalAllreduce {
    private:
        NetworkCharacteristics network_config_;
        int segment_size_;
        bool use_pipeline_;

    public:
        HierarchicalAllreduce(const NetworkCharacteristics& config);
        ~HierarchicalAllreduce();

        PerformanceMetrics allreduce(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Op op, MPI_Comm comm);

        // Hierarchical implementations
        PerformanceMetrics two_level_allreduce(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Op op, MPI_Comm comm);

        PerformanceMetrics three_level_allreduce(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Op op, MPI_Comm comm);

        // Ring allreduce variants
        PerformanceMetrics ring_allreduce(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Op op, MPI_Comm comm);

        PerformanceMetrics segmented_ring_allreduce(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Op op, MPI_Comm comm);

        MPI_Comm create_rack_communicator(MPI_Comm comm);

    private:
        // Utility methods
        MPI_Comm create_node_communicator(MPI_Comm comm);
        MPI_Comm create_inter_node_communicator(MPI_Comm comm);
        void perform_local_reduction(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Op op, MPI_Comm comm);
    };

    class AdaptiveCollective {
    private:
        NetworkCharacteristics network_config_;
        std::map<std::pair<AlgorithmType, int>, PerformanceMetrics> performance_cache_;
        int adaptation_threshold_;
        bool dynamic_adaptation_;

    public:
        AdaptiveCollective(const NetworkCharacteristics& config);
        ~AdaptiveCollective();

        // Adaptive collective operations
        PerformanceMetrics adaptive_broadcast(void* buffer, int count,
            MPI_Datatype datatype, int root,
            MPI_Comm comm);

        PerformanceMetrics adaptive_allreduce(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Op op, MPI_Comm comm);

        PerformanceMetrics adaptive_allgather(const void* sendbuf, void* recvbuf,
            int count, MPI_Datatype datatype,
            MPI_Comm comm);

        // Adaptation logic
        AlgorithmType select_algorithm(int operation_type, int message_size,
            MPI_Comm comm);
        void update_performance_model(AlgorithmType algo, const PerformanceMetrics& metrics);
        bool should_adapt(AlgorithmType current_algo, AlgorithmType proposed_algo,
            int message_size) const;

    private:
        // Performance prediction
        double predict_execution_time(AlgorithmType algo, int message_size, MPI_Comm comm);
        double predict_bandwidth_utilization(AlgorithmType algo, int message_size, MPI_Comm comm);
        double predict_energy_consumption(AlgorithmType algo, int message_size, MPI_Comm comm);

        // Decision making
        AlgorithmType evaluate_candidates(int operation_type, int message_size,
            const std::vector<AlgorithmType>& candidates,
            MPI_Comm comm);
        std::vector<AlgorithmType> get_candidate_algorithms(int operation_type,
            int message_size) const;
    };

} // namespace TopologyAwareResearch

#endif