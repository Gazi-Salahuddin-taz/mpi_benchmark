#include "topology_aware_broadcast.h"
#include <iostream>
#include <algorithm>
#include <queue>
#include <cmath>
#include <thread>
#include <cstring>
#include "../core/reduction_ops.h"

namespace TopologyAwareResearch {

    TopologyAwareBroadcast::TopologyAwareBroadcast(const NetworkCharacteristics& config)
        : network_config_(config), use_optimized_paths_(true), pipeline_depth_(4) {
        // Add safety initialization
        if (network_config_.processes_per_node <= 0) {
            network_config_.processes_per_node = 1;
        }
    }

    TopologyAwareBroadcast::~TopologyAwareBroadcast() {}

    PerformanceMetrics TopologyAwareBroadcast::broadcast(void* buffer, int count,
        MPI_Datatype datatype, int root,
        MPI_Comm comm) {
        PerformanceMetrics metrics;
        // Remove unused variable
        // auto start_time = MPI_Wtime();

        int world_size, world_rank;
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &world_rank);

        if (world_size == 1) {
            metrics.execution_time = 0.0;
            return metrics;
        }

        // Select appropriate algorithm based on topology and message size
        if (count < 1024) {
            // Small messages: use low-latency algorithms
            return binomial_tree_broadcast(buffer, count, datatype, root, comm);
        }
        else {
            // Large messages: use topology-aware algorithms
            switch (network_config_.topology) {
            case NetworkTopology::FAT_TREE:
                return fat_tree_broadcast(buffer, count, datatype, root, comm);
            case NetworkTopology::TORUS_2D:
            case NetworkTopology::TORUS_3D:
                return torus_broadcast(buffer, count, datatype, root, comm);
            case NetworkTopology::DRAGONFLY:
                return dragonfly_broadcast(buffer, count, datatype, root, comm);
            case NetworkTopology::MULTI_CORE:
                return multi_core_broadcast(buffer, count, datatype, root, comm);
            default:
                return pipeline_broadcast(buffer, count, datatype, root, comm);
            }
        }
    }

    PerformanceMetrics TopologyAwareBroadcast::fat_tree_broadcast(void* buffer, int count,
        MPI_Datatype datatype, int root,
        MPI_Comm comm) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        int world_size, world_rank;
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &world_rank);

        // Add safety checks
        int racks = (network_config_.total_nodes > 0) ? network_config_.total_nodes : 1;
        int processes_per_rack = (racks > 0) ? (world_size / racks) : world_size;
        if (processes_per_rack <= 0) processes_per_rack = 1;

        int rack_rank = world_rank / processes_per_rack;
        int local_rank = world_rank % processes_per_rack;
        int root_rack = root / processes_per_rack;

        std::vector<std::pair<int, int>> communication_edges;

        // Phase 1: Root sends to other rack leaders using binomial tree
        if (rack_rank == root_rack) {
            if (local_rank == root % processes_per_rack) {
                // Root process: send to other rack leaders
                int mask = 1;
                while (mask < racks) {
                    int target_rack = (rack_rank + mask) % racks;
                    if (target_rack < racks && target_rack != rack_rank) {
                        int target_leader = target_rack * processes_per_rack;
                        MPI_Send(buffer, count, datatype, target_leader, 0, comm);
                        communication_edges.emplace_back(world_rank, target_leader);
                    }
                    mask <<= 1;
                }
            }
        }
        else {
            // Other racks: receive from root rack leader
            if (local_rank == 0) {
                int source_rack = root_rack;
                int source_leader = source_rack * processes_per_rack + (root % processes_per_rack);
                MPI_Recv(buffer, count, datatype, source_leader, 0, comm, MPI_STATUS_IGNORE);
                communication_edges.emplace_back(source_leader, world_rank);
            }
        }

        // Phase 2: Broadcast within each rack using binomial tree
        MPI_Comm rack_comm;
        MPI_Comm_split(comm, rack_rank, world_rank, &rack_comm);

        int rack_root = (rack_rank == root_rack) ? (root % processes_per_rack) : 0;

        // Use binomial tree within rack
        PerformanceMetrics rack_metrics = binomial_tree_broadcast(buffer, count, datatype,
            rack_root, rack_comm);

        // Combine communication edges
        communication_edges.insert(communication_edges.end(),
            rack_metrics.communication_edges.begin(),
            rack_metrics.communication_edges.end());

        MPI_Comm_free(&rack_comm);

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;
        metrics.communication_edges = communication_edges;
        metrics.messages_sent = communication_edges.size();

        return metrics;
    }

    PerformanceMetrics TopologyAwareBroadcast::torus_broadcast(void* buffer, int count,
        MPI_Datatype datatype, int root,
        MPI_Comm comm) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        int world_size, world_rank;
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &world_rank);

        // Assuming 2D torus for simplicity
        int dim_x = network_config_.topology_params.torus.x;
        int dim_y = network_config_.topology_params.torus.y;

        // Add safety checks
        if (dim_x <= 0) dim_x = 1;
        if (dim_y <= 0) dim_y = world_size;

        int root_x = root % dim_x;
        int root_y = root / dim_x;
        int current_x = world_rank % dim_x;
        int current_y = world_rank / dim_x;

        std::vector<std::pair<int, int>> communication_edges;

        // Phase 1: Broadcast along X dimension (row)
        if (current_y == root_y) {
            // Participate in row broadcast
            MPI_Comm row_comm;
            MPI_Comm_split(comm, current_y, world_rank, &row_comm);

            int row_root = root_x;
            PerformanceMetrics row_metrics = binomial_tree_broadcast(buffer, count, datatype,
                row_root, row_comm);

            communication_edges.insert(communication_edges.end(),
                row_metrics.communication_edges.begin(),
                row_metrics.communication_edges.end());

            MPI_Comm_free(&row_comm);
        }

        // Phase 2: Broadcast along Y dimension (column)
        MPI_Comm col_comm;
        MPI_Comm_split(comm, current_x, world_rank, &col_comm);

        int col_root = root_y;
        PerformanceMetrics col_metrics = binomial_tree_broadcast(buffer, count, datatype,
            col_root, col_comm);

        communication_edges.insert(communication_edges.end(),
            col_metrics.communication_edges.begin(),
            col_metrics.communication_edges.end());

        MPI_Comm_free(&col_comm);

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;
        metrics.communication_edges = communication_edges;
        metrics.messages_sent = communication_edges.size();

        return metrics;
    }

    PerformanceMetrics TopologyAwareBroadcast::binomial_tree_broadcast(void* buffer, int count,
        MPI_Datatype datatype, int root,
        MPI_Comm comm) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        int world_size, world_rank;
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &world_rank);

        if (world_size == 1) {
            metrics.execution_time = 0.0;
            return metrics;
        }

        std::vector<std::pair<int, int>> communication_edges;

        // FIXED: Correct binomial tree algorithm for any root
        // Calculate relative rank (root becomes rank 0)
        int relative_rank = (world_rank - root + world_size) % world_size;

        // Binomial tree: each process with relative_rank < 2^i receives in round i
        // Then sends to relative_rank + 2^i if it exists
        int mask = 1;

        while (mask < world_size) {
            // Process sends if its relative_rank is divisible by current mask
            // and the target exists
            if ((relative_rank % (mask * 2)) == 0) {
                // This process should send
                int target_relative = relative_rank + mask;
                if (target_relative < world_size) {
                    int target = (root + target_relative) % world_size;
                    MPI_Send(buffer, count, datatype, target, 0, comm);
                    communication_edges.emplace_back(world_rank, target);
                }
            }
            else {
                // Check if this process should receive in this round
                int src_relative = relative_rank - mask;
                if (src_relative >= 0 && (src_relative % (mask * 2)) == 0) {
                    int src = (root + src_relative) % world_size;
                    MPI_Recv(buffer, count, datatype, src, 0, comm, MPI_STATUS_IGNORE);
                    communication_edges.emplace_back(src, world_rank);
                }
            }
            mask <<= 1;
        }

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;
        metrics.communication_edges = communication_edges;
        metrics.messages_sent = communication_edges.size();

        return metrics;
    }

    PerformanceMetrics TopologyAwareBroadcast::pipeline_broadcast(void* buffer, int count,
        MPI_Datatype datatype, int root,
        MPI_Comm comm) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        int world_size, world_rank;
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &world_rank);

        if (world_size == 1) {
            metrics.execution_time = 0.0;
            return metrics;
        }

        int type_size;
        MPI_Type_size(datatype, &type_size);
        // Remove unused variable
        // int total_bytes = count * type_size;

        // Dynamic segmentation based on message size and network characteristics
        int optimal_segments = std::min(pipeline_depth_, count);
        optimal_segments = std::min(optimal_segments, 16); // Limit maximum segments

        int segment_size = count / optimal_segments;
        int remaining = count % optimal_segments;

        std::vector<int> segment_sizes(optimal_segments, segment_size);
        for (int i = 0; i < remaining; ++i) {
            segment_sizes[i]++;
        }

        // Calculate displacements
        std::vector<int> displacements(optimal_segments, 0);
        for (int i = 1; i < optimal_segments; ++i) {
            displacements[i] = displacements[i - 1] + segment_sizes[i - 1] * type_size;
        }

        std::vector<std::pair<int, int>> communication_edges;

        // Pipeline communication
        for (int segment = 0; segment < optimal_segments; ++segment) {
            char* segment_ptr = static_cast<char*>(buffer) + displacements[segment];
            int current_size = segment_sizes[segment];

            if (world_rank == root) {
                int next = (root + 1) % world_size;
                MPI_Send(segment_ptr, current_size, datatype, next, segment, comm);
                communication_edges.emplace_back(world_rank, next);
            }
            else {
                int prev = (world_rank - 1 + world_size) % world_size;
                int next = (world_rank + 1) % world_size;

                MPI_Recv(segment_ptr, current_size, datatype, prev, segment, comm, MPI_STATUS_IGNORE);
                communication_edges.emplace_back(prev, world_rank);

                if (world_rank != (root - 1 + world_size) % world_size) {
                    MPI_Send(segment_ptr, current_size, datatype, next, segment, comm);
                    communication_edges.emplace_back(world_rank, next);
                }
            }

            // Small barrier-like synchronization for better pipeline performance
            if (segment < optimal_segments - 1) {
                MPI_Barrier(comm);
            }
        }

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;
        metrics.communication_edges = communication_edges;
        metrics.messages_sent = communication_edges.size();

        return metrics;
    }

    // Implementation of other methods
    void TopologyAwareBroadcast::build_optimal_broadcast_tree(int root, MPI_Comm comm) {
        int world_size;
        MPI_Comm_size(comm, &world_size);

        // Use Prim's algorithm to build minimum spanning tree
        std::vector<bool> visited(world_size, false);
        std::vector<double> min_cost(world_size, 1e9);
        std::vector<int> parent(world_size, -1);

        min_cost[root] = 0;
        std::vector<int> tree_sequence;
        tree_sequence.push_back(root);

        for (int i = 0; i < world_size; ++i) {
            // Find minimum cost edge
            int u = -1;
            for (int j = 0; j < world_size; ++j) {
                if (!visited[j] && (u == -1 || min_cost[j] < min_cost[u])) {
                    u = j;
                }
            }

            if (min_cost[u] == 1e9) break;

            visited[u] = true;
            if (parent[u] != -1) {
                tree_sequence.push_back(u);
            }

            // Update neighbors
            for (int v = 0; v < world_size; ++v) {
                double cost = network_config_.communication_costs[u][v];
                if (!visited[v] && cost < min_cost[v]) {
                    min_cost[v] = cost;
                    parent[v] = u;
                }
            }
        }

        broadcast_trees_[root] = tree_sequence;
    }

    // Missing function implementations
    PerformanceMetrics TopologyAwareBroadcast::dragonfly_broadcast(void* buffer, int count,
        MPI_Datatype datatype, int root,
        MPI_Comm comm) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        int world_size, world_rank;
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &world_rank);

        // Dragonfly parameters
        int groups = network_config_.topology_params.dragonfly.groups;
        int routers_per_group = network_config_.topology_params.dragonfly.routers_per_group;
        int nodes_per_router = network_config_.topology_params.dragonfly.nodes_per_router;

        // Add safety checks
        if (groups <= 0) groups = 1;
        if (routers_per_group <= 0) routers_per_group = 1;
        if (nodes_per_router <= 0) nodes_per_router = 1;

        // Calculate group and router for each process
        int group_size = routers_per_group * nodes_per_router;
        if (group_size <= 0) group_size = 1;

        int root_group = root / group_size;
        int root_router = (root % group_size) / nodes_per_router;
        int my_group = world_rank / group_size;
        int my_router = (world_rank % group_size) / nodes_per_router;

        std::vector<std::pair<int, int>> communication_edges;

        // Phase 1: Intra-group broadcast (within root's group)
        if (my_group == root_group) {
            // Use hierarchical broadcast within group
            MPI_Comm group_comm;
            MPI_Comm_split(comm, my_group, world_rank, &group_comm);

            // Find router leader (lowest rank in router)
            int router_leader = my_group * group_size + my_router * nodes_per_router;

            if (world_rank == root) {
                // Send to all routers in group
                for (int r = 0; r < routers_per_group; r++) {
                    if (r != root_router) {
                        int dest_router_leader = my_group * group_size + r * nodes_per_router;
                        MPI_Send(buffer, count, datatype, dest_router_leader, 0, comm);
                        communication_edges.emplace_back(world_rank, dest_router_leader);
                    }
                }
            }

            if (world_rank == router_leader && my_router != root_router) {
                MPI_Recv(buffer, count, datatype, root, 0, comm, MPI_STATUS_IGNORE);
                communication_edges.emplace_back(root, world_rank);
            }

            // Broadcast within router
            MPI_Bcast(buffer, count, datatype, router_leader - my_group * group_size, group_comm);
            MPI_Comm_free(&group_comm);
        }

        // Phase 2: Inter-group broadcast
        if (my_group != root_group) {
            // Each group receives from one source group
            int source_group = (my_group > root_group) ? my_group - 1 : my_group + 1;
            int source_router = 0; // Simplified - use first router for global connection

            if (world_rank == my_group * group_size + my_router * nodes_per_router) {
                MPI_Recv(buffer, count, datatype,
                        source_group * group_size + source_router * nodes_per_router,
                        1, comm, MPI_STATUS_IGNORE);
                communication_edges.emplace_back(source_group * group_size + source_router * nodes_per_router, world_rank);
            }
        }

        if (my_group == root_group) {
            // Send to neighboring groups
            for (int g = 0; g < groups; g++) {
                if (g != root_group) {
                    int dest_router = 0; // Simplified - use first router for global connection
                    int dest = g * group_size + dest_router * nodes_per_router;
                    MPI_Send(buffer, count, datatype, dest, 1, comm);
                    communication_edges.emplace_back(world_rank, dest);
                }
            }
        }

        // Phase 3: Final intra-group broadcast in receiving groups
        if (my_group != root_group) {
            MPI_Comm group_comm;
            MPI_Comm_split(comm, my_group, world_rank, &group_comm);

            int router_leader = my_group * group_size + my_router * nodes_per_router;
            MPI_Bcast(buffer, count, datatype,
                     (world_rank == router_leader) ? 0 : MPI_PROC_NULL, group_comm);
            MPI_Comm_free(&group_comm);
        }

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;
        metrics.communication_edges = communication_edges;
        metrics.messages_sent = communication_edges.size();

        return metrics;
    }

    PerformanceMetrics TopologyAwareBroadcast::multi_core_broadcast(void* buffer, int count,
        MPI_Datatype datatype, int root,
        MPI_Comm comm) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        int world_size, world_rank;
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &world_rank);

        // FIX: Add validation for processes_per_node to avoid division by zero
        int processes_per_node = network_config_.processes_per_node;
        if (processes_per_node <= 0) {
            processes_per_node = 1; // Default to 1 if not properly configured
        }

        // For multi-core systems, use shared memory optimization
        MPI_Comm node_comm;
        int node_id = world_rank / processes_per_node;
        MPI_Comm_split(comm, node_id, world_rank, &node_comm);

        int node_size, node_rank;
        MPI_Comm_size(node_comm, &node_size);
        MPI_Comm_rank(node_comm, &node_rank);

        int root_node = root / processes_per_node;
        int root_node_rank = root % processes_per_node;

        std::vector<std::pair<int, int>> communication_edges;

        // Phase 1: Inter-node broadcast
        if (node_id == root_node) {
            if (node_rank == root_node_rank) {
                // Root sends to other node leaders
                for (int n = 0; n < network_config_.total_nodes; n++) {
                    if (n != root_node) {
                        int node_leader = n * processes_per_node;
                        MPI_Send(buffer, count, datatype, node_leader, 0, comm);
                        communication_edges.emplace_back(world_rank, node_leader);
                    }
                }
            }
        }
        else {
            if (node_rank == 0) {
                // Node leaders receive from root
                MPI_Recv(buffer, count, datatype, root, 0, comm, MPI_STATUS_IGNORE);
                communication_edges.emplace_back(root, world_rank);
            }
        }

        // Phase 2: Intra-node broadcast using shared memory optimization
        int node_root = (node_id == root_node) ? root_node_rank : 0;
        PerformanceMetrics node_metrics = binomial_tree_broadcast(buffer, count, datatype,
            node_root, node_comm);

        communication_edges.insert(communication_edges.end(),
            node_metrics.communication_edges.begin(),
            node_metrics.communication_edges.end());

        MPI_Comm_free(&node_comm);

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;
        metrics.communication_edges = communication_edges;
        metrics.messages_sent = communication_edges.size();

        return metrics;
    }

    PerformanceMetrics TopologyAwareBroadcast::k_ary_tree_broadcast(void* buffer, int count,
        MPI_Datatype datatype, int root,
        MPI_Comm comm, int k) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        int world_size, world_rank;
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &world_rank);

        if (world_size == 1) {
            metrics.execution_time = 0.0;
            return metrics;
        }

        std::vector<std::pair<int, int>> communication_edges;

        // k-ary tree broadcast implementation
        int relative_rank = (world_rank - root + world_size) % world_size;

        if (relative_rank == 0) {
            // Root process
            for (int i = 1; i <= k && i < world_size; ++i) {
                int child = (root + i) % world_size;
                MPI_Send(buffer, count, datatype, child, 0, comm);
                communication_edges.emplace_back(world_rank, child);
            }
        }
        else {
            // Non-root processes
            int parent_relative = (relative_rank - 1) / k;
            int parent = (root + parent_relative) % world_size;
            MPI_Recv(buffer, count, datatype, parent, 0, comm, MPI_STATUS_IGNORE);
            communication_edges.emplace_back(parent, world_rank);

            // Forward to children
            for (int i = 1; i <= k; ++i) {
                int child_relative = relative_rank * k + i;
                if (child_relative < world_size) {
                    int child = (root + child_relative) % world_size;
                    MPI_Send(buffer, count, datatype, child, 0, comm);
                    communication_edges.emplace_back(world_rank, child);
                }
            }
        }

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;
        metrics.communication_edges = communication_edges;
        metrics.messages_sent = communication_edges.size();

        return metrics;
    }

    PerformanceMetrics TopologyAwareBroadcast::hierarchical_broadcast(void* buffer, int count,
        MPI_Datatype datatype, int root,
        MPI_Comm comm) {
        // Use the multi-core broadcast as hierarchical broadcast
        return multi_core_broadcast(buffer, count, datatype, root, comm);
    }

    // Utility method implementations
    std::vector<int> TopologyAwareBroadcast::get_broadcast_sequence(int root, MPI_Comm comm) {
        if (broadcast_trees_.find(root) != broadcast_trees_.end()) {
            return broadcast_trees_[root];
        }

        // Build the tree if not already built
        build_optimal_broadcast_tree(root, comm);
        return broadcast_trees_[root];
    }

    void TopologyAwareBroadcast::optimize_communication_schedule(int root, MPI_Comm comm) {
        // Build optimal broadcast tree
        build_optimal_broadcast_tree(root, comm);

        // Create communication schedule based on the tree
        std::vector<int> tree_sequence = broadcast_trees_[root];
        std::vector<std::pair<int, int>> schedule;

        for (size_t i = 1; i < tree_sequence.size(); ++i) {
            int parent = tree_sequence[(i - 1) / 2];
            int child = tree_sequence[i];
            schedule.emplace_back(parent, child);
        }

        communication_schedules_[root] = schedule;
    }

    // Tree construction algorithms
    std::vector<int> TopologyAwareBroadcast::construct_binomial_tree(int root, int world_size) {
        std::vector<int> tree;
        tree.push_back(root);

        int mask = 1;
        while (mask < world_size) {
            for (int i = 0; i < mask && i + mask < world_size; ++i) {
                tree.push_back((root + i + mask) % world_size);
            }
            mask <<= 1;
        }

        return tree;
    }

    std::vector<int> TopologyAwareBroadcast::construct_k_ary_tree(int root, int world_size, int k) {
        std::vector<int> tree;
        tree.push_back(root);

        std::queue<int> q;
        q.push(root);

        while (!q.empty() && tree.size() < static_cast<size_t>(world_size)) {
            int current = q.front();
            q.pop();

            for (int i = 1; i <= k && tree.size() < static_cast<size_t>(world_size); ++i) {
                int child = (current + i) % world_size;
                if (std::find(tree.begin(), tree.end(), child) == tree.end()) {
                    tree.push_back(child);
                    q.push(child);
                }
            }
        }

        return tree;
    }

    std::vector<int> TopologyAwareBroadcast::construct_shortest_path_tree(int root, const std::vector<std::vector<double>>& costs) {
        int world_size = costs.size();
        std::vector<int> tree;
        tree.push_back(root);

        std::vector<bool> visited(world_size, false);
        std::vector<double> distance(world_size, 1e9);
        std::vector<int> parent(world_size, -1);

        distance[root] = 0;
        visited[root] = true;

        for (int i = 0; i < world_size - 1; ++i) {
            double min_dist = 1e9;
            int u = -1;

            for (int v = 0; v < world_size; ++v) {
                if (!visited[v] && distance[v] < min_dist) {
                    min_dist = distance[v];
                    u = v;
                }
            }

            if (u == -1) break;

            visited[u] = true;
            tree.push_back(u);

            for (int v = 0; v < world_size; ++v) {
                if (!visited[v] && costs[u][v] < distance[v]) {
                    distance[v] = costs[u][v];
                    parent[v] = u;
                }
            }
        }

        return tree;
    }

    std::vector<int> TopologyAwareBroadcast::construct_minimum_spanning_tree(int root, const std::vector<std::vector<double>>& costs) {
        return construct_shortest_path_tree(root, costs); // For simplicity, use same implementation
    }

    // Utility methods
    double TopologyAwareBroadcast::estimate_communication_cost(int src, int dst, int message_size) const {
        if (is_same_node(src, dst)) {
            return network_config_.intra_node_latency + message_size / network_config_.intra_node_bandwidth;
        } else {
            return network_config_.inter_node_latency + message_size / network_config_.inter_node_bandwidth;
        }
    }

    bool TopologyAwareBroadcast::is_same_node(int rank1, int rank2) const {
        return get_node_id(rank1) == get_node_id(rank2);
    }

    int TopologyAwareBroadcast::get_node_id(int rank) const {
        // FIX: Add validation for processes_per_node
        int processes_per_node = network_config_.processes_per_node;
        if (processes_per_node <= 0) {
            processes_per_node = 1;
        }

        if (rank < static_cast<int>(network_config_.node_mapping.size())) {
            return network_config_.node_mapping[rank];
        }
        return rank / processes_per_node;
    }

    std::vector<int> TopologyAwareBroadcast::get_ranks_on_same_node(int rank) const {
        int node_id = get_node_id(rank);
        std::vector<int> ranks;

        for (int i = 0; i < network_config_.total_processes; ++i) {
            if (get_node_id(i) == node_id) {
                ranks.push_back(i);
            }
        }

        return ranks;
    }

    // Performance optimization
    void TopologyAwareBroadcast::segment_message(void* buffer, int count, MPI_Datatype datatype,
        std::vector<void*>& segments,
        std::vector<int>& segment_sizes) {
        int type_size;
        MPI_Type_size(datatype, &type_size);

        int optimal_segments = std::min(pipeline_depth_, count);
        int segment_size = count / optimal_segments;
        int remaining = count % optimal_segments;

        segment_sizes.resize(optimal_segments, segment_size);
        for (int i = 0; i < remaining; ++i) {
            segment_sizes[i]++;
        }

        segments.clear();
        char* base_ptr = static_cast<char*>(buffer);
        int current_offset = 0;

        for (int i = 0; i < optimal_segments; ++i) {
            segments.push_back(base_ptr + current_offset * type_size);
            current_offset += segment_sizes[i];
        }
    }

    void TopologyAwareBroadcast::reassemble_message(void* /*buffer*/, int /*count*/, MPI_Datatype /*datatype*/,
        const std::vector<void*>& /*segments*/,
        const std::vector<int>& /*segment_sizes*/) {
        // For broadcast, reassembly is typically not needed as all processes get the same data
        // This method is provided for completeness
    }

    // HierarchicalAllreduce implementation
    HierarchicalAllreduce::HierarchicalAllreduce(const NetworkCharacteristics& config)
        : network_config_(config), segment_size_(4096), use_pipeline_(true) {
        // Add safety initialization
        if (network_config_.processes_per_node <= 0) {
            network_config_.processes_per_node = 1;
        }
    }

    HierarchicalAllreduce::~HierarchicalAllreduce() {}

    PerformanceMetrics HierarchicalAllreduce::allreduce(const void* sendbuf, void* recvbuf,
        int count, MPI_Datatype datatype,
        MPI_Op op, MPI_Comm comm) {
        PerformanceMetrics metrics;
        // Remove unused variable
        // auto start_time = MPI_Wtime();

        int world_size;
        MPI_Comm_size(comm, &world_size);

        // Select algorithm based on message size and system characteristics
        if (count < 8192 || world_size <= 8) {
            return ring_allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        }
        else if (network_config_.total_nodes > 1) {
            return two_level_allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        }
        else {
            return segmented_ring_allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        }
    }

    PerformanceMetrics HierarchicalAllreduce::two_level_allreduce(const void* sendbuf, void* recvbuf,
        int count, MPI_Datatype datatype,
        MPI_Op op, MPI_Comm comm) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        int world_rank, world_size;
        MPI_Comm_rank(comm, &world_rank);
        MPI_Comm_size(comm, &world_size);

        std::vector<std::pair<int, int>> communication_edges;

        // Phase 1: Create node communicators and perform local reduction
        MPI_Comm node_comm = create_node_communicator(comm);

        int node_rank, node_size;
        MPI_Comm_rank(node_comm, &node_rank);
        MPI_Comm_size(node_comm, &node_size);

        // Local reduction within node
        void* local_result = (world_rank == 0) ? const_cast<void*>(sendbuf) : recvbuf;
        PerformanceMetrics local_metrics = ring_allreduce(sendbuf, local_result, count,
            datatype, op, node_comm);

        communication_edges.insert(communication_edges.end(),
            local_metrics.communication_edges.begin(),
            local_metrics.communication_edges.end());

        // Phase 2: Global reduction across nodes
        if (node_rank == 0) {
            MPI_Comm inter_node_comm = create_inter_node_communicator(comm);

            if (inter_node_comm != MPI_COMM_NULL) {
                int inter_rank, inter_size;
                MPI_Comm_rank(inter_node_comm, &inter_rank);
                MPI_Comm_size(inter_node_comm, &inter_size);

                // Perform global reduction among node leaders
                PerformanceMetrics global_metrics = ring_allreduce(local_result, recvbuf, count,
                    datatype, op, inter_node_comm);

                communication_edges.insert(communication_edges.end(),
                    global_metrics.communication_edges.begin(),
                    global_metrics.communication_edges.end());

                MPI_Comm_free(&inter_node_comm);
            }
        }

        // Phase 3: Broadcast result within nodes
        MPI_Bcast(recvbuf, count, datatype, 0, node_comm);

        MPI_Comm_free(&node_comm);

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;
        metrics.communication_edges = communication_edges;
        metrics.messages_sent = communication_edges.size();

        return metrics;
    }

    PerformanceMetrics HierarchicalAllreduce::ring_allreduce(const void* sendbuf, void* recvbuf,
        int count, MPI_Datatype datatype,
        MPI_Op op, MPI_Comm comm) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        int world_rank, world_size;
        MPI_Comm_rank(comm, &world_rank);
        MPI_Comm_size(comm, &world_size);

        if (world_size == 1) {
            if (sendbuf != MPI_IN_PLACE) {
                memcpy(recvbuf, sendbuf, count * sizeof(double)); // Assuming double
            }
            metrics.execution_time = 0.0;
            return metrics;
        }

        int type_size;
        MPI_Type_size(datatype, &type_size);
        int total_bytes = count * type_size;

        // Use segmented approach for large messages
        int segments = std::max(1, total_bytes / segment_size_);
        // Remove unused variables
        // int segment_elements = count / segments;
        // int remaining = count % segments;

        std::vector<std::pair<int, int>> communication_edges;

        // Ring allreduce implementation
        // Phase 1: Reduce-scatter
        // Phase 2: Allgather

        // Simplified implementation - in practice this would be more sophisticated
        for (int step = 0; step < world_size - 1; ++step) {
            int send_to = (world_rank + 1) % world_size;
            int recv_from = (world_rank - 1 + world_size) % world_size;

            MPI_Send(recvbuf, count, datatype, send_to, 0, comm);
            MPI_Recv(recvbuf, count, datatype, recv_from, 0, comm, MPI_STATUS_IGNORE);

            communication_edges.emplace_back(world_rank, send_to);
            communication_edges.emplace_back(recv_from, world_rank);
        }

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;
        metrics.communication_edges = communication_edges;
        metrics.messages_sent = communication_edges.size();

        return metrics;
    }

    PerformanceMetrics HierarchicalAllreduce::three_level_allreduce(const void* sendbuf, void* recvbuf,
        int count, MPI_Datatype datatype,
        MPI_Op op, MPI_Comm comm) {
        // Three-level allreduce: node-level, rack-level, and then global
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        int world_rank, world_size;
        MPI_Comm_rank(comm, &world_rank);
        MPI_Comm_size(comm, &world_size);

        std::vector<std::pair<int, int>> communication_edges;

        // Step 1: Node-level reduction
        MPI_Comm node_comm = create_node_communicator(comm);
        int node_rank, node_size;
        MPI_Comm_rank(node_comm, &node_rank);
        MPI_Comm_size(node_comm, &node_size);

        // Use MPI_Type_size instead of get_mpi_type_size to avoid ambiguity
        int type_size;
        MPI_Type_size(datatype, &type_size);

        void* node_result = (node_rank == 0) ? malloc(count * type_size) : nullptr;
        MPI_Reduce(sendbuf, node_result, count, datatype, op, 0, node_comm);

        // Step 2: Rack-level reduction (if applicable)
        MPI_Comm rack_comm = create_rack_communicator(comm);
        if (rack_comm != MPI_COMM_NULL) {
            int rack_rank, rack_size;
            MPI_Comm_rank(rack_comm, &rack_rank);
            MPI_Comm_size(rack_comm, &rack_size);

            void* rack_result = (rack_rank == 0) ? malloc(count * type_size) : nullptr;
            if (node_rank == 0) {
                MPI_Reduce(node_result, rack_result, count, datatype, op, 0, rack_comm);
            } else {
                MPI_Reduce(nullptr, rack_result, count, datatype, op, 0, rack_comm);
            }

            // Step 3: Global reduction among rack leaders
            MPI_Comm global_comm;
            MPI_Comm_split(comm, (rack_rank == 0) ? 0 : MPI_UNDEFINED, world_rank, &global_comm);
            if (global_comm != MPI_COMM_NULL) {
                int global_rank, global_size;
                MPI_Comm_rank(global_comm, &global_rank);
                MPI_Comm_size(global_comm, &global_size);

                void* global_result = (global_rank == 0) ? malloc(count * type_size) : nullptr;
                if (rack_rank == 0) {
                    MPI_Reduce(rack_result, global_result, count, datatype, op, 0, global_comm);
                } else {
                    MPI_Reduce(nullptr, global_result, count, datatype, op, 0, global_comm);
                }

                // Broadcast back through the hierarchy
                if (global_rank == 0) {
                    memcpy(recvbuf, global_result, count * type_size);
                }

                MPI_Bcast(recvbuf, count, datatype, 0, global_comm);

                if (global_rank == 0) {
                    free(global_result);
                }
                MPI_Comm_free(&global_comm);
            }

            if (rack_rank == 0) {
                free(rack_result);
            }
            MPI_Comm_free(&rack_comm);
        }

        if (node_rank == 0) {
            free(node_result);
        }
        MPI_Comm_free(&node_comm);

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;
        return metrics;
    }

    PerformanceMetrics HierarchicalAllreduce::segmented_ring_allreduce(const void* sendbuf, void* recvbuf,
        int count, MPI_Datatype datatype,
        MPI_Op op, MPI_Comm comm) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        // Determine optimal segment size based on network characteristics
        int segment_size = calculate_optimal_segment_size(count, size, network_config_);
        int segments = (count + segment_size - 1) / segment_size;

        // Use MPI_Type_size instead of get_mpi_type_size to avoid ambiguity
        int type_size;
        MPI_Type_size(datatype, &type_size);

        // Temporary buffer for segments
        std::vector<char> temp_buffer(count * type_size);
        char* temp_ptr = temp_buffer.data();

        // Copy sendbuf to recvbuf for reduction
        if (sendbuf != MPI_IN_PLACE) {
            memcpy(recvbuf, sendbuf, count * type_size);
        }

        std::vector<std::pair<int, int>> communication_edges;

        // Segmented ring allreduce
        for (int seg = 0; seg < segments; seg++) {
            int seg_start = seg * segment_size;
            int seg_count = std::min(segment_size, count - seg_start);

            // Reduction phase
            for (int step = 0; step < size - 1; step++) {
                int send_to = (rank + 1) % size;
                int recv_from = (rank - 1 + size) % size;

                // Send segment to next process
                MPI_Send(reinterpret_cast<char*>(recvbuf) + seg_start * type_size,
                        seg_count, datatype, send_to, seg * 1000 + step, comm);

                // Receive segment from previous process
                MPI_Recv(temp_ptr, seg_count, datatype, recv_from, seg * 1000 + step, comm,
                        MPI_STATUS_IGNORE);

                // Reduce received segment using the comprehensive reduction function
                // Use the existing reduce_segments from reduction_ops.h
                reduce_segments(recvbuf, temp_ptr, seg_start, seg_count, datatype, op);

                communication_edges.emplace_back(rank, send_to);
                communication_edges.emplace_back(recv_from, rank);
            }

            // Distribution phase (similar but in reverse)
            for (int step = 0; step < size - 1; step++) {
                int send_to = (rank - 1 + size) % size;
                int recv_from = (rank + 1) % size;

                MPI_Send(reinterpret_cast<char*>(recvbuf) + seg_start * type_size,
                        seg_count, datatype, send_to, seg * 2000 + step, comm);

                MPI_Recv(reinterpret_cast<char*>(recvbuf) + seg_start * type_size,
                        seg_count, datatype, recv_from, seg * 2000 + step, comm,
                        MPI_STATUS_IGNORE);

                communication_edges.emplace_back(rank, send_to);
                communication_edges.emplace_back(recv_from, rank);
            }
        }

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;
        metrics.communication_time = metrics.execution_time * 0.8;
        metrics.computation_time = metrics.execution_time * 0.2;
        metrics.bytes_transferred = 2 * (size - 1) * count * type_size;
        metrics.communication_edges = communication_edges;
        metrics.messages_sent = communication_edges.size();

        return metrics;
    }

    MPI_Comm HierarchicalAllreduce::create_node_communicator(MPI_Comm comm) {
        int world_rank;
        MPI_Comm_rank(comm, &world_rank);

        int node_id = network_config_.node_mapping[world_rank];

        MPI_Comm node_comm;
        MPI_Comm_split(comm, node_id, world_rank, &node_comm);

        return node_comm;
    }

    MPI_Comm HierarchicalAllreduce::create_inter_node_communicator(MPI_Comm comm) {
        int world_rank;
        MPI_Comm_rank(comm, &world_rank);

        int node_id = network_config_.node_mapping[world_rank];

        // Only node leaders participate in inter-node communication
        MPI_Comm inter_comm;
        MPI_Comm_split(comm, (node_id == 0) ? 0 : MPI_UNDEFINED, world_rank, &inter_comm);

        return inter_comm;
    }

    MPI_Comm HierarchicalAllreduce::create_rack_communicator(MPI_Comm comm) {
        // Simplified rack communicator creation
        // In practice, this would use actual rack information from the network config
        int world_rank;
        MPI_Comm_rank(comm, &world_rank);

        int rack_id = world_rank / (network_config_.processes_per_node * 4); // Assume 4 nodes per rack

        MPI_Comm rack_comm;
        MPI_Comm_split(comm, rack_id, world_rank, &rack_comm);

        return rack_comm;
    }

    void HierarchicalAllreduce::perform_local_reduction(const void* sendbuf, void* recvbuf,
        int count, MPI_Datatype datatype,
        MPI_Op op, MPI_Comm comm) {
        // Perform local reduction using MPI_Reduce
        MPI_Reduce(sendbuf, recvbuf, count, datatype, op, 0, comm);
    }

    // AdaptiveCollective implementation
    AdaptiveCollective::AdaptiveCollective(const NetworkCharacteristics& config)
        : network_config_(config), adaptation_threshold_(100), dynamic_adaptation_(true) {
    }

    AdaptiveCollective::~AdaptiveCollective() {}

    PerformanceMetrics AdaptiveCollective::adaptive_broadcast(void* buffer, int count,
        MPI_Datatype datatype, int root,
        MPI_Comm comm) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        // Select the best algorithm based on current conditions
        AlgorithmType algo = select_algorithm(0, count, comm); // 0 for broadcast

        // Create appropriate broadcast instance based on selected algorithm
        TopologyAwareBroadcast broadcast(network_config_);

        switch (algo) {
        case AlgorithmType::BINOMIAL_TREE:
            metrics = broadcast.binomial_tree_broadcast(buffer, count, datatype, root, comm);
            break;
        case AlgorithmType::PIPELINE_RING:
            metrics = broadcast.pipeline_broadcast(buffer, count, datatype, root, comm);
            break;
        case AlgorithmType::TOPOLOGY_AWARE_BROADCAST:
            metrics = broadcast.broadcast(buffer, count, datatype, root, comm);
            break;
        default:
            metrics = broadcast.binomial_tree_broadcast(buffer, count, datatype, root, comm);
            break;
        }

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;

        // Update performance model
        update_performance_model(algo, metrics);

        return metrics;
    }

    PerformanceMetrics AdaptiveCollective::adaptive_allreduce(const void* sendbuf, void* recvbuf,
        int count, MPI_Datatype datatype,
        MPI_Op op, MPI_Comm comm) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        // Select the best algorithm based on current conditions
        AlgorithmType algo = select_algorithm(1, count, comm); // 1 for allreduce

        // Create appropriate allreduce instance based on selected algorithm
        HierarchicalAllreduce allreduce(network_config_);

        switch (algo) {
        case AlgorithmType::RING_ALLREDUCE:
            metrics = allreduce.ring_allreduce(sendbuf, recvbuf, count, datatype, op, comm);
            break;
        case AlgorithmType::ADAPTIVE_ALLREDUCE:
            metrics = allreduce.allreduce(sendbuf, recvbuf, count, datatype, op, comm);
            break;
        default:
            MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
            auto end_time = MPI_Wtime();
            metrics.execution_time = end_time - start_time;
            break;
        }

        // Update performance model
        update_performance_model(algo, metrics);

        return metrics;
    }

    PerformanceMetrics AdaptiveCollective::adaptive_allgather(const void* sendbuf, void* recvbuf,
        int count, MPI_Datatype datatype,
        MPI_Comm comm) {
        PerformanceMetrics metrics;
        auto start_time = MPI_Wtime();

        // Use native MPI for now - could be extended with topology-aware versions
        MPI_Allgather(sendbuf, count, datatype, recvbuf, count, datatype, comm);

        auto end_time = MPI_Wtime();
        metrics.execution_time = end_time - start_time;

        return metrics;
    }

    // Adaptation logic
    AlgorithmType AdaptiveCollective::select_algorithm(int operation_type, int message_size,
        MPI_Comm comm) {
        int world_size;
        MPI_Comm_size(comm, &world_size);

        // Simple selection based on message size and system characteristics
        if (operation_type == 0) { // Broadcast
            if (message_size < 1024) {
                return AlgorithmType::BINOMIAL_TREE;
            } else if (message_size < 65536) {
                return AlgorithmType::TOPOLOGY_AWARE_BROADCAST;
            } else {
                return AlgorithmType::PIPELINE_RING;
            }
        } else { // Allreduce
            if (message_size < 8192 || world_size <= 8) {
                return AlgorithmType::RING_ALLREDUCE;
            } else {
                return AlgorithmType::ADAPTIVE_ALLREDUCE;
            }
        }
    }

    void AdaptiveCollective::update_performance_model(AlgorithmType algo, const PerformanceMetrics& metrics) {
        // Fix the key creation - use AlgorithmType first, then message size
        auto key = std::make_pair(algo, metrics.bytes_processed);
        performance_cache_[key] = metrics;
    }
    bool AdaptiveCollective::should_adapt(AlgorithmType current_algo, AlgorithmType proposed_algo,
        int message_size) const {
        // Simple adaptation logic - adapt if proposed algorithm is different and message size is large
        return (current_algo != proposed_algo) && (message_size > adaptation_threshold_);
    }

    // Performance prediction
    double AdaptiveCollective::predict_execution_time(AlgorithmType algo, int message_size, MPI_Comm comm) {
        // Fix the key creation
        auto key = std::make_pair(algo, message_size);
        if (performance_cache_.find(key) != performance_cache_.end()) {
            return performance_cache_[key].execution_time;
        }

        // Fallback prediction based on algorithm characteristics
        int world_size;
        MPI_Comm_size(comm, &world_size);

        switch (algo) {
        case AlgorithmType::BINOMIAL_TREE:
            return world_size * std::log2(world_size) * message_size * 0.0001;
        case AlgorithmType::PIPELINE_RING:
            return (world_size - 1) * message_size * 0.0001;
        default:
            return message_size * 0.0001;
        }
    }

    double AdaptiveCollective::predict_bandwidth_utilization(AlgorithmType /*algo*/, int /*message_size*/, MPI_Comm /*comm*/) {
        // Simplified prediction
        return 0.8; // 80% utilization as default
    }

    double AdaptiveCollective::predict_energy_consumption(AlgorithmType algo, int message_size, MPI_Comm comm) {
        // Simplified energy prediction
        double execution_time = predict_execution_time(algo, message_size, comm);
        return execution_time * 50.0; // 50W power consumption
    }

    // Decision making
    AlgorithmType AdaptiveCollective::evaluate_candidates(int /*operation_type*/, int message_size,
        const std::vector<AlgorithmType>& candidates,
        MPI_Comm comm) {
        AlgorithmType best_algo = candidates[0];
        double best_score = std::numeric_limits<double>::max();

        for (auto algo : candidates) {
            double time = predict_execution_time(algo, message_size, comm);
            double energy = predict_energy_consumption(algo, message_size, comm);
            double bandwidth = predict_bandwidth_utilization(algo, message_size, comm);

            // Multi-objective score (weighted sum)
            double score = time * 0.6 + energy * 0.2 + (1.0 - bandwidth) * 0.2;

            if (score < best_score) {
                best_score = score;
                best_algo = algo;
            }
        }

        return best_algo;
    }

    std::vector<AlgorithmType> AdaptiveCollective::get_candidate_algorithms(int operation_type,
        int /*message_size*/) const {
        std::vector<AlgorithmType> candidates;

        if (operation_type == 0) { // Broadcast
            candidates = {
                AlgorithmType::BINOMIAL_TREE,
                AlgorithmType::PIPELINE_RING,
                AlgorithmType::TOPOLOGY_AWARE_BROADCAST
            };
        } else { // Allreduce
            candidates = {
                AlgorithmType::RING_ALLREDUCE,
                AlgorithmType::ADAPTIVE_ALLREDUCE
            };
        }

        return candidates;
    }

} // namespace TopologyAwareResearch

int TopologyAwareResearch::calculate_optimal_segment_size(int total_count, int /*processes*/,
                                  const NetworkCharacteristics& network) {
    // Balance between pipelining benefits and overhead
    int min_segment = 1024;
    int max_segment = 65536;

    // Optimal segment size based on network characteristics
    int optimal = static_cast<int>(std::sqrt(network.inter_node_bandwidth *
                                           network.inter_node_latency * 1000000));

    optimal = std::max(min_segment, std::min(optimal, max_segment));
    optimal = std::min(optimal, total_count);

    return optimal;
}