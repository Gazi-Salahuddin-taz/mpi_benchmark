#include "graph_optimizer.h"
#include <queue>
#include <set>
#include <algorithm>
#include <cstring>

namespace TopologyAwareResearch {

CommunicationGraph GraphOptimizer::synthesize_communication_pattern(
    int root, int processes, const NetworkTopologyInfo& topology) {

    CommunicationGraph graph(processes);

    switch (topology.detected_topology) {
        case NetworkTopologyInfo::TORUS:
            graph = synthesize_torus_pattern(root, processes, topology);
            break;
        case NetworkTopologyInfo::DRAGONFLY:
            graph = synthesize_dragonfly_pattern(root, processes, topology);
            break;
        case NetworkTopologyInfo::FAT_TREE:
            graph = synthesize_fat_tree_pattern(root, processes, topology);
            break;
        default:
            graph = synthesize_binomial_tree_pattern(root, processes);
            break;
    }

    return graph;
}

CommunicationGraph GraphOptimizer::synthesize_torus_pattern(
    int root, int processes, const NetworkTopologyInfo& topology) {

    CommunicationGraph graph(processes);
    int dim_x = topology.topology_params.torus.x;
    int dim_y = topology.topology_params.torus.y;

    if (dim_x * dim_y != processes) {
        // Fallback to binomial tree if dimensions don't match
        return synthesize_binomial_tree_pattern(root, processes);
    }

    // Convert root to coordinates
    int root_x = root % dim_x;
    int root_y = root / dim_x;

    // Build communication graph using breadth-first search on torus
    std::vector<bool> visited(processes, false);
    std::queue<int> q;
    q.push(root);
    visited[root] = true;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        int curr_x = current % dim_x;
        int curr_y = current / dim_x;

        // Explore neighbors in torus
        int neighbors[4][2] = {
            {(curr_x + 1) % dim_x, curr_y},  // right
            {(curr_x - 1 + dim_x) % dim_x, curr_y},  // left
            {curr_x, (curr_y + 1) % dim_y},  // down
            {curr_x, (curr_y - 1 + dim_y) % dim_y}   // up
        };

        for (auto& neighbor : neighbors) {
            int nx = neighbor[0];
            int ny = neighbor[1];
            int neighbor_rank = ny * dim_x + nx;

            if (!visited[neighbor_rank]) {
                visited[neighbor_rank] = true;
                graph.add_edge(current, neighbor_rank);
                q.push(neighbor_rank);
            }
        }
    }

    return graph;
}

CommunicationGraph GraphOptimizer::synthesize_dragonfly_pattern(
    int root, int processes, const NetworkTopologyInfo& topology) {

    CommunicationGraph graph(processes);
    auto df = topology.topology_params.dragonfly;

    int group_size = df.routers_per_group * df.nodes_per_router;
    int root_group = root / group_size;

    // Phase 1: Build intra-group communication trees
    for (int g = 0; g < df.groups; g++) {
        if (g == root_group) {
            build_intra_group_tree(root, g, topology, graph);
        } else {
            int group_leader = g * group_size; // Use first process as leader
            build_intra_group_tree(group_leader, g, topology, graph);
        }
    }

    // Phase 2: Connect groups through global links
    for (int g = 0; g < df.groups; g++) {
        if (g != root_group) {
            int source_router = 0; // Simplified
            int dest_router = 0; // Simplified

            int source_rank = root_group * group_size + source_router * df.nodes_per_router;
            int dest_rank = g * group_size + dest_router * df.nodes_per_router;

            graph.add_edge(source_rank, dest_rank);
        }
    }

    return graph;
}

CommunicationGraph GraphOptimizer::synthesize_fat_tree_pattern(
    int root, int processes, const NetworkTopologyInfo& topology) {

    CommunicationGraph graph(processes);
    // Implementation for fat tree topology
    // This would create a tree structure based on the fat tree parameters
    int k = topology.topology_params.fat_tree.k;

    // Build a k-ary tree
    std::queue<int> q;
    q.push(root);
    std::vector<bool> visited(processes, false);
    visited[root] = true;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        for (int i = 1; i <= k; i++) {
            int child = (current * k + i) % processes;
            if (child < processes && !visited[child]) {
                visited[child] = true;
                graph.add_edge(current, child);
                q.push(child);
            }
        }
    }

    return graph;
}

CommunicationGraph GraphOptimizer::synthesize_binomial_tree_pattern(
    int root, int processes) {

    CommunicationGraph graph(processes);

    // Build binomial tree
    int mask = 1;
    while (mask < processes) {
        for (int i = 0; i < mask; i++) {
            if (i + mask < processes) {
                int parent = (root + i) % processes;
                int child = (root + i + mask) % processes;
                graph.add_edge(parent, child);
            }
        }
        mask <<= 1;
    }

    return graph;
}

void GraphOptimizer::build_intra_group_tree(int root, int group, 
                                           const NetworkTopologyInfo& topology, 
                                           CommunicationGraph& graph) {
    auto df = topology.topology_params.dragonfly;
    int group_size = df.routers_per_group * df.nodes_per_router;
    int group_start = group * group_size;

    // Build binomial tree within group
    int mask = 1;
    while (mask < group_size) {
        for (int i = 0; i < mask && (i + mask) < group_size; ++i) {
            int src = group_start + i;
            int dest = group_start + i + mask;
            graph.add_edge(src, dest);
        }
        mask <<= 1;
    }
}

} // namespace TopologyAwareResearch