#ifndef TOPOLOGY_DETECTION_H
#define TOPOLOGY_DETECTION_H

#include <mpi.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>

namespace TopologyAwareResearch {

    struct NodeInfo {
        std::string hostname;
        int rank;
        int physical_core;
        int numa_node;
        std::vector<int> neighbor_ranks;
        double communication_cost; // Average cost to other nodes
    };

    struct NetworkTopologyInfo {
        std::string topology_type;
        int total_nodes;
        int processes_per_node;
        int total_processes;

        // Performance characteristics
        double inter_node_bandwidth;  // GB/s
        double intra_node_bandwidth;  // GB/s
        double inter_node_latency;    // microseconds
        double intra_node_latency;    // microseconds

        // Topology-specific parameters
        struct FatTreeParams {
            int k; // k-ary fat tree
            int levels;
            std::vector<int> nodes_per_level;
        } fat_tree;

        struct TorusParams {
            int dim_x, dim_y, dim_z;
            bool wrap_around;
        } torus;

        struct DragonflyParams {
            int groups;
            int routers_per_group;
            int nodes_per_router;
        } dragonfly;

        // Process mapping and connectivity
        std::vector<int> node_mapping;  // process_id -> node_id
        std::vector<std::vector<int>> node_processes;  // node_id -> process_ids
        std::vector<std::vector<double>> communication_costs; // process_id x process_id
        std::vector<NodeInfo> node_infos;

        // Analysis results
        double network_diameter;
        double average_shortest_path;
        double clustering_coefficient;
        std::vector<int> central_nodes;

        NetworkTopologyInfo() : total_nodes(0), processes_per_node(0), total_processes(0),
            inter_node_bandwidth(0.0), intra_node_bandwidth(0.0),
            inter_node_latency(0.0), intra_node_latency(0.0),
            network_diameter(0.0), average_shortest_path(0.0),
            clustering_coefficient(0.0) {
            fat_tree.k = 0;
            fat_tree.levels = 0;
            torus.dim_x = torus.dim_y = torus.dim_z = 0;
            torus.wrap_around = false;
            dragonfly.groups = dragonfly.routers_per_group = dragonfly.nodes_per_router = 0;
        }
    };

    class TopologyDetector {
    public:
        TopologyDetector();
        ~TopologyDetector();

        // Main detection interface
        NetworkTopologyInfo detect_topology(MPI_Comm comm);
        NetworkTopologyInfo detect_topology_with_benchmarks(MPI_Comm comm, int benchmark_iterations = 10);

        // Specific topology detection
        NetworkTopologyInfo detect_fat_tree(MPI_Comm comm);
        NetworkTopologyInfo detect_torus(MPI_Comm comm);
        NetworkTopologyInfo detect_dragonfly(MPI_Comm comm);
        NetworkTopologyInfo detect_multi_core(MPI_Comm comm);

        // Advanced analysis
        void analyze_network_characteristics(MPI_Comm comm, NetworkTopologyInfo& topology);
        void build_communication_cost_matrix(MPI_Comm comm, NetworkTopologyInfo& topology);
        void identify_central_nodes(NetworkTopologyInfo& topology);

        // Benchmarking
        double benchmark_latency(int src_rank, int dst_rank, MPI_Comm comm, int iterations = 1000);
        double benchmark_bandwidth(int src_rank, int dst_rank, MPI_Comm comm, int message_size = 1048576);
        void perform_comprehensive_benchmark(MPI_Comm comm, NetworkTopologyInfo& topology);

        // Utility methods
        static std::string get_hostname();
        static int get_physical_core_id();
        static int get_numa_node();
        static std::vector<int> get_neighbor_cores();

    private:
        std::vector<NodeInfo> gather_node_information(MPI_Comm comm);
        NetworkTopologyInfo analyze_node_distribution(const std::vector<NodeInfo>& nodes);
        void infer_topology_parameters(NetworkTopologyInfo& topology);
        void calculate_network_metrics(NetworkTopologyInfo& topology);

        // Graph algorithms for network analysis
        std::vector<std::vector<double>> floyd_warshall(const std::vector<std::vector<double>>& cost_matrix);
        double calculate_clustering_coefficient(const std::vector<std::vector<double>>& cost_matrix);
        std::vector<int> find_graph_centers(const std::vector<std::vector<double>>& distance_matrix);

        // Benchmark helpers
        void latency_benchmark(MPI_Comm comm, NetworkTopologyInfo& topology);
        void bandwidth_benchmark(MPI_Comm comm, NetworkTopologyInfo& topology);
    };

} // namespace TopologyAwareResearch

#endif