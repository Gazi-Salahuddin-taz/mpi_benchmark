#include "fat-tree-scenario.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include <cmath>
#include <sstream>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("FatTreeScenario");
NS_OBJECT_ENSURE_REGISTERED(FatTreeScenario);

TypeId FatTreeScenario::GetTypeId(void) {
    static TypeId tid = TypeId("ns3::FatTreeScenario")
        .SetParent<MpiResearchScenario>()
        .AddConstructor<FatTreeScenario>()
        .AddAttribute("K", "Fat tree k parameter",
                     UintegerValue(4),
                     MakeUintegerAccessor(&FatTreeScenario::m_k),
                     MakeUintegerChecker<uint32_t>());
    return tid;
}

FatTreeScenario::FatTreeScenario(uint32_t k) 
    : m_k(k) {
    
    NS_LOG_FUNCTION(this << k);
    
    if (k < 2 || k % 2 != 0) {
        NS_FATAL_ERROR("k must be an even number >= 2, got: " << k);
    }

    m_pods = k;
    m_cores = (k / 2) * (k / 2);
    m_aggregationPerPod = k / 2;
    m_edgePerPod = k / 2;
    m_computePerEdge = k / 2;
    m_totalComputeNodes = m_pods * m_edgePerPod * m_computePerEdge;
    
    SetWorldSize(m_totalComputeNodes);
    SetTopologyType(FAT_TREE);
}

FatTreeScenario::~FatTreeScenario() {
    NS_LOG_FUNCTION(this);
}

void FatTreeScenario::ConfigureTopology() {
    NS_LOG_FUNCTION(this);
    CreateTopology();
}

void FatTreeScenario::CreateTopology() {
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Creating Fat Tree topology with k=" << m_k 
                << ", total compute nodes: " << m_totalComputeNodes);

    // Create all nodes
    m_computeNodes.Create(m_totalComputeNodes);
    m_edgeSwitches.Create(m_pods * m_edgePerPod);
    m_aggregationSwitches.Create(m_pods * m_aggregationPerPod);
    m_coreSwitches.Create(m_cores);

    // Add to all nodes container
    m_allNodes.Add(m_computeNodes);
    m_allNodes.Add(m_edgeSwitches);
    m_allNodes.Add(m_aggregationSwitches);
    m_allNodes.Add(m_coreSwitches);

    // Configure link properties
    PointToPointHelper p2pComputeToEdge;
    p2pComputeToEdge.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gbps")));
    p2pComputeToEdge.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));
    p2pComputeToEdge.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("1000p"));

    PointToPointHelper p2pEdgeToAgg;
    p2pEdgeToAgg.SetDeviceAttribute("DataRate", DataRateValue(DataRate("40Gbps")));
    p2pEdgeToAgg.SetChannelAttribute("Delay", TimeValue(MicroSeconds(2)));
    p2pEdgeToAgg.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("2000p"));

    PointToPointHelper p2pAggToCore;
    p2pAggToCore.SetDeviceAttribute("DataRate", DataRateValue(DataRate("40Gbps")));
    p2pAggToCore.SetChannelAttribute("Delay", TimeValue(MicroSeconds(5)));
    p2pAggToCore.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("2000p"));

    uint32_t linkCount = 0;

    // Connect compute nodes to edge switches
    NS_LOG_INFO("Connecting " << m_totalComputeNodes << " compute nodes to edge switches");
    for (uint32_t pod = 0; pod < m_pods; ++pod) {
        for (uint32_t edgeIdx = 0; edgeIdx < m_edgePerPod; ++edgeIdx) {
            uint32_t edgeSwitchIndex = pod * m_edgePerPod + edgeIdx;
            Ptr<Node> edgeSwitch = GetNodeSafe(m_edgeSwitches, edgeSwitchIndex, "edgeSwitches");

            for (uint32_t compIdx = 0; compIdx < m_computePerEdge; ++compIdx) {
                uint32_t computeIndex = pod * m_edgePerPod * m_computePerEdge +
                                      edgeIdx * m_computePerEdge + compIdx;
                Ptr<Node> computeNode = GetNodeSafe(m_computeNodes, computeIndex, "computeNodes");

                NetDeviceContainer link = p2pComputeToEdge.Install(computeNode, edgeSwitch);
                m_allDevices.Add(link);
                m_links.push_back(link);
                linkCount++;
            }
        }
    }

    // Connect edge switches to aggregation switches
    NS_LOG_INFO("Connecting edge to aggregation switches");
    for (uint32_t pod = 0; pod < m_pods; ++pod) {
        for (uint32_t edgeIdx = 0; edgeIdx < m_edgePerPod; ++edgeIdx) {
            uint32_t edgeSwitchIndex = pod * m_edgePerPod + edgeIdx;
            Ptr<Node> edgeSwitch = GetNodeSafe(m_edgeSwitches, edgeSwitchIndex, "edgeSwitches");

            for (uint32_t aggIdx = 0; aggIdx < m_aggregationPerPod; ++aggIdx) {
                uint32_t aggSwitchIndex = pod * m_aggregationPerPod + aggIdx;
                Ptr<Node> aggSwitch = GetNodeSafe(m_aggregationSwitches, aggSwitchIndex, "aggregationSwitches");

                NetDeviceContainer link = p2pEdgeToAgg.Install(edgeSwitch, aggSwitch);
                m_allDevices.Add(link);
                m_links.push_back(link);
                linkCount++;
            }
        }
    }

    // Connect aggregation switches to core switches
    NS_LOG_INFO("Connecting aggregation to core switches");
    for (uint32_t pod = 0; pod < m_pods; ++pod) {
        for (uint32_t aggIdx = 0; aggIdx < m_aggregationPerPod; ++aggIdx) {
            uint32_t aggSwitchIndex = pod * m_aggregationPerPod + aggIdx;
            Ptr<Node> aggSwitch = GetNodeSafe(m_aggregationSwitches, aggSwitchIndex, "aggregationSwitches");

            for (uint32_t coreGroup = 0; coreGroup < m_k / 2; ++coreGroup) {
                uint32_t coreIndex = coreGroup * (m_k / 2) + aggIdx;
                if (coreIndex < m_coreSwitches.GetN()) {
                    Ptr<Node> coreSwitch = GetNodeSafe(m_coreSwitches, coreIndex, "coreSwitches");
                    NetDeviceContainer link = p2pAggToCore.Install(aggSwitch, coreSwitch);
                    m_allDevices.Add(link);
                    m_links.push_back(link);
                    linkCount++;
                }
            }
        }
    }

    NS_LOG_INFO("Fat Tree topology created with " << linkCount << " links");
}

void FatTreeScenario::SetupIpAddressing() {
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Setting up IP addressing for " << m_links.size() << " links");

    Ipv4AddressHelper address;
    address.SetBase("10.1.0.0", "255.255.0.0");

    for (uint32_t i = 0; i < m_links.size(); ++i) {
        if (m_links[i].GetN() == 2) {
            Ipv4InterfaceContainer interfaces = address.Assign(m_links[i]);
            address.NewNetwork();
        }
    }

    NS_LOG_INFO("IP addressing completed");
}

void FatTreeScenario::SetupRouting() {
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Setting up routing for Fat Tree");

    // Use global routing for fat tree
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    NS_LOG_INFO("Routing tables populated");
}

void FatTreeScenario::CollectResults() {
    NS_LOG_FUNCTION(this);

    // Create results directory
    system(("mkdir -p " + m_resultsDir).c_str());

    // Write topology-specific results
    std::ofstream topoFile(m_resultsDir + "/fat_tree_topology.txt");
    topoFile << "Fat Tree Topology Analysis\n";
    topoFile << "==========================\n";
    topoFile << "k parameter: " << m_k << "\n";
    topoFile << "Pods: " << m_pods << "\n";
    topoFile << "Core switches: " << m_coreSwitches.GetN() << "\n";
    topoFile << "Aggregation switches: " << m_aggregationSwitches.GetN() << "\n";
    topoFile << "Edge switches: " << m_edgeSwitches.GetN() << "\n";
    topoFile << "Compute nodes: " << m_computeNodes.GetN() << "\n";
    topoFile << "Total links: " << m_links.size() << "\n";
    topoFile << "Bisection bandwidth: " << (m_k * m_k * m_k / 4) << " units\n";
    topoFile.close();

    // Call base class result collection
    WriteCommonResults();

    NS_LOG_INFO("Fat Tree results written to " << m_resultsDir);
}

} // namespace ns3