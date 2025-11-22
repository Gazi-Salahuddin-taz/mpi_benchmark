#include "topology_detection.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <random>

#ifdef __linux__
#include <unistd.h>
#include <sys/sysinfo.h>
#include <sched.h>
#endif

namespace TopologyAwareResearch {

    TopologyDetector::TopologyDetector() {
        // Constructor implementation
    }

    TopologyDetector::~TopologyDetector() {
        // Destructor implementation
    }

    NetworkTopologyInfo TopologyDetector::detect_topology(MPI_Comm comm) {
        return detect_topology_with_benchmarks(comm, 5); // Default with 5 iterations
    }

    NetworkTopologyInfo TopologyDetector::detect_topology_with_benchmarks(MPI_Comm comm, int benchmark_iterations) {
        NetworkTopologyInfo topology;

        int world_rank, world_size;
        MPI_Comm_rank(comm, &world_rank);
        MPI_Comm_size(comm, &world_size);

        topology.total_processes = world_size;

        // Phase 1: Gather basic node information
        std::vector<NodeInfo> node_infos = gather_node_information(comm);

        if (world_rank == 0) {
            // Phase 2: Analyze node distribution to infer topology
            topology = analyze_node_distribution(node_infos);
            topology.node_infos = node_infos;

            // Phase 3: Infer topology-specific parameters
            infer_topology_parameters(topology);

            // Phase 4: Perform network benchmarks
            perform_comprehensive_benchmark(comm, topology);

            // Phase 5: Calculate network metrics
            calculate_network_metrics(topology);

            std::cout << "=== Network Topology Detection Results ===" << std::endl;
            std::cout << "Topology Type: " << topology.topology_type << std::endl;
            std::cout << "Total Nodes: " << topology.total_nodes << std::endl;
            std::cout << "Processes per Node: " << topology.processes_per_node << std::endl;
            std::cout << "Network Diameter: " << topology.network_diameter << std::endl;
            std::cout << "Average Shortest Path: " << topology.average_shortest_path << std::endl;
            std::cout << "Clustering Coefficient: " << topology.clustering_coefficient << std::endl;
            std::cout << "Inter-node Bandwidth: " << topology.inter_node_bandwidth << " GB/s" << std::endl;
            std::cout << "Inter-node Latency: " << topology.inter_node_latency << " us" << std::endl;
        }

        // Broadcast topology information to all processes
        MPI_Bcast(&topology.total_nodes, 1, MPI_INT, 0, comm);
        MPI_Bcast(&topology.processes_per_node, 1, MPI_INT, 0, comm);

        // Broadcast topology type as string
        int type_length = topology.topology_type.length();
        MPI_Bcast(&type_length, 1, MPI_INT, 0, comm);

        if (world_rank != 0) {
            topology.topology_type.resize(type_length);
        }
        MPI_Bcast(&topology.topology_type[0], type_length, MPI_CHAR, 0, comm);

        return topology;
    }

    std::vector<NodeInfo> TopologyDetector::gather_node_information(MPI_Comm comm) {
        int world_rank, world_size;
        MPI_Comm_rank(comm, &world_rank);
        MPI_Comm_size(comm, &world_size);

        // Create local node info
        NodeInfo local_info;
        local_info.hostname = get_hostname();
        local_info.rank = world_rank;
        local_info.physical_core = get_physical_core_id();
        local_info.numa_node = get_numa_node();
        local_info.neighbor_ranks = get_neighbor_cores();

        // Gather all node information to rank 0
        std::vector<NodeInfo> all_infos;

        if (world_rank == 0) {
            all_infos.resize(world_size);
        }

        // First, gather hostnames to determine node mapping
        std::vector<char> hostname_buffer;
        int hostname_length = local_info.hostname.length() + 1; // Include null terminator

        if (world_rank == 0) {
            hostname_buffer.resize(world_size * MPI_MAX_PROCESSOR_NAME);
        }

        MPI_Gather(local_info.hostname.c_str(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
            hostname_buffer.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, comm);

        // For simplicity, this is a simplified implementation
        // In practice, you would use proper serialization for complex structures

        if (world_rank == 0) {
            // Build node mapping from hostnames
            std::map<std::string, int> node_ids;
            std::vector<std::string> hostnames;

            for (int i = 0; i < world_size; ++i) {
                std::string hostname(hostname_buffer.data() + i * MPI_MAX_PROCESSOR_NAME);
                hostnames.push_back(hostname);

                if (node_ids.find(hostname) == node_ids.end()) {
                    node_ids[hostname] = node_ids.size();
                }
            }

            // Create basic node info structures
            for (int i = 0; i < world_size; ++i) {
                all_infos[i].hostname = hostnames[i];
                all_infos[i].rank = i;
                all_infos[i].physical_core = i % std::thread::hardware_concurrency(); // Simplified
                all_infos[i].numa_node = 0; // Simplified
            }
        }

        return all_infos;
    }

    NetworkTopologyInfo TopologyDetector::analyze_node_distribution(const std::vector<NodeInfo>& nodes) {
        NetworkTopologyInfo topology;

        if (nodes.empty()) {
            topology.topology_type = "UNKNOWN";
            return topology;
        }

        // Count unique hostnames to determine nodes
        std::vector<std::string> hostnames;
        for (const auto& node : nodes) {
            hostnames.push_back(node.hostname);
        }

        std::sort(hostnames.begin(), hostnames.end());
        auto last = std::unique(hostnames.begin(), hostnames.end());
        topology.total_nodes = std::distance(hostnames.begin(), last);
        topology.processes_per_node = nodes.size() / topology.total_nodes;

        // Build node mapping and node_processes
        topology.node_mapping.resize(nodes.size());
        topology.node_processes.resize(topology.total_nodes);

        std::map<std::string, int> node_ids;
        for (int i = 0; i < nodes.size(); ++i) {
            const std::string& hostname = nodes[i].hostname;
            if (node_ids.find(hostname) == node_ids.end()) {
                node_ids[hostname] = node_ids.size();
            }
            int node_id = node_ids[hostname];
            topology.node_mapping[i] = node_id;
            topology.node_processes[node_id].push_back(i);
        }

        // Infer topology type based on node count and distribution patterns
        if (topology.total_nodes == 1) {
            topology.topology_type = "MULTI_CORE";
            topology.intra_node_bandwidth = 100.0;
            topology.intra_node_latency = 0.1;
        }
        else if (topology.total_nodes <= 4) {
            topology.topology_type = "SMALL_CLUSTER";
            topology.inter_node_bandwidth = 10.0;
            topology.inter_node_latency = 5.0;
        }
        else if (topology.total_nodes <= 16) {
            topology.topology_type = "FAT_TREE";
            topology.inter_node_bandwidth = 25.0;
            topology.inter_node_latency = 2.0;
        }
        else if (topology.total_nodes <= 64) {
            topology.topology_type = "TORUS";
            topology.inter_node_bandwidth = 100.0;
            topology.inter_node_latency = 1.0;
        }
        else {
            topology.topology_type = "DRAGONFLY";
            topology.inter_node_bandwidth = 200.0;
            topology.inter_node_latency = 0.5;
        }

        return topology;
    }

    void TopologyDetector::infer_topology_parameters(NetworkTopologyInfo& topology) {
        if (topology.topology_type == "FAT_TREE") {
            // Infer k-ary fat tree parameters
            topology.fat_tree.k = 4; // Default
            topology.fat_tree.levels = 3;
            topology.fat_tree.nodes_per_level = { topology.total_nodes / 4, topology.total_nodes / 2, topology.total_nodes };

        }
        else if (topology.topology_type == "TORUS") {
            // Infer torus dimensions
            int total_nodes = topology.total_nodes;

            // Try to find nice 2D or 3D decomposition
            int dim_x = 1, dim_y = 1, dim_z = 1;

            for (int x = std::sqrt(total_nodes); x >= 1; --x) {
                if (total_nodes % x == 0) {
                    dim_x = x;
                    dim_y = total_nodes / x;
                    break;
                }
            }

            // If 2D decomposition is poor, try 3D
            if (dim_x * dim_y != total_nodes || std::abs(dim_x - dim_y) > 4) {
                for (int x = std::cbrt(total_nodes); x >= 1; --x) {
                    if (total_nodes % x == 0) {
                        int remaining = total_nodes / x;
                        for (int y = std::sqrt(remaining); y >= 1; --y) {
                            if (remaining % y == 0) {
                                dim_x = x;
                                dim_y = y;
                                dim_z = remaining / y;
                                break;
                            }
                        }
                        if (dim_x * dim_y * dim_z == total_nodes) break;
                    }
                }
            }

            topology.torus.dim_x = dim_x;
            topology.torus.dim_y = dim_y;
            topology.torus.dim_z = (dim_x * dim_y * dim_z == total_nodes) ? dim_z : 1;
            topology.torus.wrap_around = true;

        }
        else if (topology.topology_type == "DRAGONFLY") {
            // Infer dragonfly parameters
            topology.dragonfly.groups = 8;
            topology.dragonfly.routers_per_group = topology.total_nodes / 8;
            topology.dragonfly.nodes_per_router = topology.processes_per_node;
        }
    }

    void TopologyDetector::perform_comprehensive_benchmark(MPI_Comm comm, NetworkTopologyInfo& topology) {
        latency_benchmark(comm, topology);
        bandwidth_benchmark(comm, topology);
        build_communication_cost_matrix(comm, topology);
    }

    void TopologyDetector::latency_benchmark(MPI_Comm comm, NetworkTopologyInfo& topology) {
        int world_rank, world_size;
        MPI_Comm_rank(comm, &world_rank);
        MPI_Comm_size(comm, &world_size);

        if (world_rank == 0) {
            std::cout << "Performing latency benchmark..." << std::endl;
        }

        // Measure latency between different node pairs
        std::vector<double> inter_node_latencies;
        std::vector<double> intra_node_latencies;

        for (int i = 0; i < std::min(10, world_size); ++i) {
            for (int j = i + 1; j < std::min(10, world_size); ++j) {
                if (world_rank == i || world_rank == j) {
                    double latency = benchmark_latency(i, j, comm, 1000);

                    if (world_rank == 0) {
                        if (topology.node_mapping[i] == topology.node_mapping[j]) {
                            intra_node_latencies.push_back(latency);
                        }
                        else {
                            inter_node_latencies.push_back(latency);
                        }
                    }
                }
            }
        }

        if (world_rank == 0) {
            if (!inter_node_latencies.empty()) {
                topology.inter_node_latency = std::accumulate(inter_node_latencies.begin(),
                    inter_node_latencies.end(), 0.0) / inter_node_latencies.size();
            }
            if (!intra_node_latencies.empty()) {
                topology.intra_node_latency = std::accumulate(intra_node_latencies.begin(),
                    intra_node_latencies.end(), 0.0) / intra_node_latencies.size();
            }
        }
    }

    void TopologyDetector::bandwidth_benchmark(MPI_Comm comm, NetworkTopologyInfo& topology) {
        int world_rank, world_size;
        MPI_Comm_rank(comm, &world_rank);
        MPI_Comm_size(comm, &world_size);

        if (world_rank == 0) {
            std::cout << "Performing bandwidth benchmark..." << std::endl;
        }

        // Similar structure to latency benchmark but with larger messages
        // Implementation omitted for brevity
    }

    void TopologyDetector::build_communication_cost_matrix(MPI_Comm comm, NetworkTopologyInfo& topology) {
        int world_size = topology.total_processes;
        topology.communication_costs.resize(world_size, std::vector<double>(world_size, 1.0));

        // Build cost matrix based on topology
        for (int i = 0; i < world_size; ++i) {
            for (int j = 0; j < world_size; ++j) {
                if (i == j) {
                    topology.communication_costs[i][j] = 0.0;
                }
                else if (topology.node_mapping[i] == topology.node_mapping[j]) {
                    // Same node - low cost
                    topology.communication_costs[i][j] = 0.1;
                }
                else {
                    // Different nodes - cost based on topology
                    if (topology.topology_type == "FAT_TREE") {
                        // Fat-tree: cost based on common ancestor level
                        int node_i = topology.node_mapping[i];
                        int node_j = topology.node_mapping[j];

                        if (node_i / topology.fat_tree.k == node_j / topology.fat_tree.k) {
                            topology.communication_costs[i][j] = 0.5; // Same pod
                        }
                        else {
                            topology.communication_costs[i][j] = 1.0; // Different pods
                        }
                    }
                    else if (topology.topology_type == "TORUS") {
                        // Torus: cost based on Manhattan distance
                        int node_i = topology.node_mapping[i];
                        int node_j = topology.node_mapping[j];

                        int x_i = node_i % topology.torus.dim_x;
                        int y_i = node_i / topology.torus.dim_x;
                        int x_j = node_j % topology.torus.dim_x;
                        int y_j = node_j / topology.torus.dim_x;

                        int dx = std::abs(x_i - x_j);
                        int dy = std::abs(y_i - y_j);

                        if (topology.torus.wrap_around) {
                            dx = std::min(dx, topology.torus.dim_x - dx);
                            dy = std::min(dy, topology.torus.dim_y - dy);
                        }

                        topology.communication_costs[i][j] = 0.1 * (dx + dy) + 0.1;
                    }
                    else {
                        // Default: uniform cost for inter-node communication
                        topology.communication_costs[i][j] = 1.0;
                    }
                }
            }
        }
    }

    void TopologyDetector::calculate_network_metrics(NetworkTopologyInfo& topology) {
        int world_size = topology.total_processes;

        // Calculate shortest paths using Floyd-Warshall
        auto distance_matrix = floyd_warshall(topology.communication_costs);

        // Calculate network diameter (longest shortest path)
        double diameter = 0.0;
        double total_path_length = 0.0;
        int path_count = 0;

        for (int i = 0; i < world_size; ++i) {
            for (int j = i + 1; j < world_size; ++j) {
                if (distance_matrix[i][j] < 1e9) { // Valid path
                    diameter = std::max(diameter, distance_matrix[i][j]);
                    total_path_length += distance_matrix[i][j];
                    path_count++;
                }
            }
        }

        topology.network_diameter = diameter;
        topology.average_shortest_path = path_count > 0 ? total_path_length / path_count : 0.0;

        // Calculate clustering coefficient
        topology.clustering_coefficient = calculate_clustering_coefficient(topology.communication_costs);

        // Identify central nodes
        topology.central_nodes = find_graph_centers(distance_matrix);
    }

    std::vector<std::vector<double>> TopologyDetector::floyd_warshall(const std::vector<std::vector<double>>& cost_matrix) {
        int n = cost_matrix.size();
        std::vector<std::vector<double>> dist = cost_matrix;

        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }

        return dist;
    }

    double TopologyDetector::calculate_clustering_coefficient(const std::vector<std::vector<double>>& cost_matrix) {
        int n = cost_matrix.size();
        double total_coefficient = 0.0;
        int node_count = 0;

        for (int i = 0; i < n; ++i) {
            // Find neighbors (nodes with cost < 1.0 and != 0)
            std::vector<int> neighbors;
            for (int j = 0; j < n; ++j) {
                if (i != j && cost_matrix[i][j] < 1.0 && cost_matrix[i][j] > 0) {
                    neighbors.push_back(j);
                }
            }

            int k = neighbors.size();
            if (k < 2) continue;

            // Count triangles among neighbors
            int triangles = 0;
            for (int a = 0; a < k; ++a) {
                for (int b = a + 1; b < k; ++b) {
                    if (cost_matrix[neighbors[a]][neighbors[b]] < 1.0 &&
                        cost_matrix[neighbors[a]][neighbors[b]] > 0) {
                        triangles++;
                    }
                }
            }

            double max_triangles = k * (k - 1) / 2.0;
            if (max_triangles > 0) {
                total_coefficient += triangles / max_triangles;
                node_count++;
            }
        }

        return node_count > 0 ? total_coefficient / node_count : 0.0;
    }

    std::vector<int> TopologyDetector::find_graph_centers(const std::vector<std::vector<double>>& distance_matrix) {
        int n = distance_matrix.size();
        std::vector<int> centers;

        if (n == 0) return centers;

        // Find minimum eccentricity (graph center)
        double min_eccentricity = 1e9;

        for (int i = 0; i < n; ++i) {
            double eccentricity = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    eccentricity = std::max(eccentricity, distance_matrix[i][j]);
                }
            }

            if (eccentricity < min_eccentricity) {
                min_eccentricity = eccentricity;
                centers.clear();
                centers.push_back(i);
            }
            else if (eccentricity == min_eccentricity) {
                centers.push_back(i);
            }
        }

        return centers;
    }

    // System information utilities
    std::string TopologyDetector::get_hostname() {
		char hostname[1024];
		hostname[0] = '\0';
		
		if (gethostname(hostname, sizeof(hostname)-1) == 0) {
			return std::string(hostname);
		} else {
			return "unknown";
    }
}

    int TopologyDetector::get_physical_core_id() {
#ifdef __linux__
        return sched_getcpu();
#else
        return 0;
#endif
    }

    int TopologyDetector::get_numa_node() {
        // Simplified implementation
        // In practice, parse /sys/devices/system/node/ or use numactl
        return 0;
    }

    std::vector<int> TopologyDetector::get_neighbor_cores() {
        std::vector<int> neighbors;
        int total_cores = std::thread::hardware_concurrency();

        if (total_cores > 0) {
            int current_core = get_physical_core_id();
            for (int i = 0; i < total_cores; ++i) {
                if (i != current_core) {
                    neighbors.push_back(i);
                }
            }
        }

        return neighbors;
    }

    // Benchmark implementations
    double TopologyDetector::benchmark_latency(int src_rank, int dst_rank, MPI_Comm comm, int iterations) {
        int world_rank;
        MPI_Comm_rank(comm, &world_rank);

        double latency = 0.0;
        int buffer = 42;
        MPI_Status status;

        if (world_rank == src_rank) {
            double total_time = 0.0;

            for (int i = 0; i < iterations; ++i) {
                double start = MPI_Wtime();
                MPI_Send(&buffer, 1, MPI_INT, dst_rank, 0, comm);
                MPI_Recv(&buffer, 1, MPI_INT, dst_rank, 0, comm, &status);
                double end = MPI_Wtime();
                total_time += (end - start);
            }

            latency = (total_time / iterations) * 1e6 / 2.0; // One-way latency in microseconds
        }
        else if (world_rank == dst_rank) {
            for (int i = 0; i < iterations; ++i) {
                MPI_Recv(&buffer, 1, MPI_INT, src_rank, 0, comm, &status);
                MPI_Send(&buffer, 1, MPI_INT, src_rank, 0, comm);
            }
        }

        // Broadcast latency from source to all processes
        MPI_Bcast(&latency, 1, MPI_DOUBLE, src_rank, comm);
        return latency;
    }

} // namespace TopologyAwareResearch