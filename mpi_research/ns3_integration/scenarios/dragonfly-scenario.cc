#include "dragonfly-scenario.h"
#include <cmath>
#include <sstream>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("DragonflyScenario");
NS_OBJECT_ENSURE_REGISTERED(DragonflyScenario);

TypeId DragonflyScenario::GetTypeId(void) {
    static TypeId tid = TypeId("ns3::DragonflyScenario")
        .SetParent<MpiResearchScenario>()
        .AddConstructor<DragonflyScenario>()
        .AddAttribute("Groups", "Number of groups in dragonfly",
                     UintegerValue(4),
                     MakeUintegerAccessor(&DragonflyScenario::m_groups),
                     MakeUintegerChecker<uint32_t>())
        .AddAttribute("RoutersPerGroup", "Number of routers per group",
                     UintegerValue(4),
                     MakeUintegerAccessor(&DragonflyScenario::m_routersPerGroup),
                     MakeUintegerChecker<uint32_t>())
        .AddAttribute("NodesPerRouter", "Number of nodes per router",
                     UintegerValue(4),
                     MakeUintegerAccessor(&DragonflyScenario::m_nodesPerRouter),
                     MakeUintegerChecker<uint32_t>());
    return tid;
}

DragonflyScenario::DragonflyScenario(uint32_t groups, uint32_t routersPerGroup, uint32_t nodesPerRouter)
    : m_groups(groups), m_routersPerGroup(routersPerGroup), m_nodesPerRouter(nodesPerRouter) {
    
    NS_LOG_FUNCTION(this << groups << routersPerGroup << nodesPerRouter);
    
    m_totalComputeNodes = groups * routersPerGroup * nodesPerRouter;
    m_totalRouters = groups * routersPerGroup;
    
    SetWorldSize(m_totalComputeNodes);
    SetTopologyType(DRAGONFLY);
}

void DragonflyScenario::CreateTopology() {
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Creating Dragonfly topology with " << m_groups << " groups, " 
                << m_routersPerGroup << " routers/group, " << m_nodesPerRouter << " nodes/router");

    // Create routers and compute nodes
    m_routers.Create(m_totalRouters);
    m_computeNodes.Create(m_totalComputeNodes);
    
    m_allNodes.Add(m_routers);
    m_allNodes.Add(m_computeNodes);

    // Link configuration
    PointToPointHelper p2pLocal;  // Router to node
    p2pLocal.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gbps")));
    p2pLocal.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));

    PointToPointHelper p2pGlobal; // Router to router (inter-group)
    p2pGlobal.SetDeviceAttribute("DataRate", DataRateValue(DataRate("40Gbps")));
    p2pGlobal.SetChannelAttribute("Delay", TimeValue(MicroSeconds(10)));

    // Connect nodes to routers
    for (uint32_t group = 0; group < m_groups; ++group) {
        for (uint32_t router = 0; router < m_routersPerGroup; ++router) {
            uint32_t routerIndex = group * m_routersPerGroup + router;
            Ptr<Node> routerNode = GetNodeSafe(m_routers, routerIndex, "routers");

            for (uint32_t node = 0; node < m_nodesPerRouter; ++node) {
                uint32_t nodeIndex = routerIndex * m_nodesPerRouter + node;
                Ptr<Node> computeNode = GetNodeSafe(m_computeNodes, nodeIndex, "computeNodes");

                NetDeviceContainer link = p2pLocal.Install(routerNode, computeNode);
                m_links.push_back(link);
            }
        }
    }

    // Connect routers within groups (simplified)
    for (uint32_t group = 0; group < m_groups; ++group) {
        for (uint32_t i = 0; i < m_routersPerGroup; ++i) {
            for (uint32_t j = i + 1; j < m_routersPerGroup; ++j) {
                uint32_t router1 = group * m_routersPerGroup + i;
                uint32_t router2 = group * m_routersPerGroup + j;
                
                NetDeviceContainer link = p2pGlobal.Install(
                    m_routers.Get(router1), m_routers.Get(router2)
                );
                m_links.push_back(link);
            }
        }
    }

    // Connect routers between groups (simplified dragonfly)
    for (uint32_t group1 = 0; group1 < m_groups; ++group1) {
        for (uint32_t group2 = group1 + 1; group2 < m_groups; ++group2) {
            // Connect one router from each group
            uint32_t router1 = group1 * m_routersPerGroup;
            uint32_t router2 = group2 * m_routersPerGroup;
            
            NetDeviceContainer link = p2pGlobal.Install(
                m_routers.Get(router1), m_routers.Get(router2)
            );
            m_links.push_back(link);
        }
    }

    NS_LOG_INFO("Dragonfly topology created with " << m_links.size() << " links");
}

void DragonflyScenario::SetupIpAddressing() {
    Ipv4AddressHelper address;
    address.SetBase("10.1.0.0", "255.255.0.0");

    for (uint32_t i = 0; i < m_links.size(); ++i) {
        Ipv4InterfaceContainer interfaces = address.Assign(m_links[i]);
        address.NewNetwork();
    }
}

void DragonflyScenario::SetupRouting() {
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
}

void DragonflyScenario::CollectResults() {
    system(("mkdir -p " + m_resultsDir).c_str());

    std::ofstream topoFile(m_resultsDir + "/dragonfly_topology.txt");
    topoFile << "Dragonfly Topology Analysis\n";
    topoFile << "===========================\n";
    topoFile << "Groups: " << m_groups << "\n";
    topoFile << "Routers per group: " << m_routersPerGroup << "\n";
    topoFile << "Nodes per router: " << m_nodesPerRouter << "\n";
    topoFile << "Total routers: " << m_totalRouters << "\n";
    topoFile << "Total compute nodes: " << m_totalComputeNodes << "\n";
    topoFile << "Total links: " << m_links.size() << "\n";
    topoFile.close();

    WriteCommonResults();
}

} // namespace ns3