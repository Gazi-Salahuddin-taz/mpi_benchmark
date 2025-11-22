#ifndef GRAPH_OPTIMIZER_H
#define GRAPH_OPTIMIZER_H

#include <vector>
#include <queue>
#include <set>
#include <algorithm>
#include <cstring>
#include "collective_optimizer.h"

namespace TopologyAwareResearch {

// Communication graph for representing broadcast trees
class CommunicationGraph {
private:
    int num_processes;
    std::vector<std::vector<int>> adj_list;

public:
    CommunicationGraph(int n) : num_processes(n), adj_list(n) {}

    void add_edge(int u, int v) {
        adj_list[u].push_back(v);
        // For undirected graphs, we might also add v to u
    }

    const std::vector<int>& get_neighbors(int u) const {
        return adj_list[u];
    }

    int get_num_processes() const {
        return num_processes;
    }

    void clear() {
        for (auto& neighbors : adj_list) {
            neighbors.clear();
        }
    }
};

// Network topology information for graph synthesis
struct NetworkTopologyInfo {
    enum TopologyType {
        TORUS,
        DRAGONFLY,
        FAT_TREE,
        BINOMIAL_TREE,
        UNKNOWN
    } detected_topology;

    union {
        struct { int x, y; } torus;
        struct {
            int groups;
            int routers_per_group;
            int nodes_per_router;
        } dragonfly;
        struct { int k; } fat_tree;
    } topology_params;

    NetworkTopologyInfo() : detected_topology(UNKNOWN) {
        topology_params.torus.x = 0;
        topology_params.torus.y = 0;
    }
};

// Graph optimizer for synthesizing communication patterns
class GraphOptimizer {
public:
    CommunicationGraph synthesize_communication_pattern(int root, int processes,
                                                       const NetworkTopologyInfo& topology);

private:
    CommunicationGraph synthesize_torus_pattern(int root, int processes,
                                               const NetworkTopologyInfo& topology);
    CommunicationGraph synthesize_dragonfly_pattern(int root, int processes,
                                                   const NetworkTopologyInfo& topology);
    CommunicationGraph synthesize_fat_tree_pattern(int root, int processes,
                                                  const NetworkTopologyInfo& topology);
    CommunicationGraph synthesize_binomial_tree_pattern(int root, int processes);

    void build_intra_group_tree(int root, int group,
                               const NetworkTopologyInfo& topology,
                               CommunicationGraph& graph);
};

} // namespace TopologyAwareResearch

#endif // GRAPH_OPTIMIZER_H