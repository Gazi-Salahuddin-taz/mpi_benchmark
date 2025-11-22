#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/csma-module.h"
#include "mpi-research-application.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FatTreeScenario");

/**
 * Fat Tree Network Scenario for MPI Research
 * Enhanced version with MPI Research Application integration
 */
class FatTreeScenario {
public:
    FatTreeScenario(uint32_t k);
    ~FatTreeScenario();

    void RunSimulation(Time duration = Seconds(30.0));
    void ConfigureFatTree(uint32_t k);

private:
    void CreateFatTreeTopology(uint32_t k);
    void SetupIpAddressing();
    void SetupRouting();
    void InstallMpiApplications();
    void ScheduleMpiOperations();
    void GenerateBackgroundTraffic();
    void CollectResults();

    // Safe node access with bounds checking
    Ptr<Node> GetNodeSafe(NodeContainer& container, uint32_t index, const std::string& containerName) {
        if (index >= container.GetN()) {
            NS_FATAL_ERROR("Index " << index << " out of bounds for " << containerName
                << " (size: " << container.GetN() << ")");
        }
        return container.Get(index);
    }

    uint32_t m_k;
    NodeContainer m_coreSwitches;
    NodeContainer m_aggregationSwitches;
    NodeContainer m_edgeSwitches;
    NodeContainer m_computeNodes;
    NodeContainer m_allNodes;

    NetDeviceContainer m_allDevices;
    std::vector<NetDeviceContainer> m_links;

    ApplicationContainer m_applications;
    std::vector<Ptr<MpiResearchApplication>> m_mpiApps;

    // Topology parameters
    uint32_t m_pods;
    uint32_t m_cores;
    uint32_t m_aggregationPerPod;
    uint32_t m_edgePerPod;
    uint32_t m_computePerEdge;
    uint32_t m_totalComputeNodes;

    // MPI configuration
    bool m_enableMpiLogging;
    Time m_computationDelay;
    Time m_communicationDelay;
};

FatTreeScenario::FatTreeScenario(uint32_t k)
    : m_k(k),
      m_enableMpiLogging(true),
      m_computationDelay(MilliSeconds(2)),
      m_communicationDelay(MicroSeconds(50)) {

    NS_LOG_FUNCTION(this << k);

    // Validate k parameter
    if (k < 2 || k % 2 != 0) {
        NS_FATAL_ERROR("k must be an even number >= 2, got: " << k);
    }

    m_pods = k;
    m_cores = (k / 2) * (k / 2);
    m_aggregationPerPod = k / 2;
    m_edgePerPod = k / 2;
    m_computePerEdge = k / 2;
    m_totalComputeNodes = m_pods * m_edgePerPod * m_computePerEdge;

    NS_LOG_INFO("FatTree parameters - k: " << k
                << ", pods: " << m_pods
                << ", cores: " << m_cores
                << ", total compute nodes: " << m_totalComputeNodes);
}

FatTreeScenario::~FatTreeScenario() {
    NS_LOG_FUNCTION(this);
    // Clear containers to prevent memory issues
    m_links.clear();
    m_mpiApps.clear();
}

void FatTreeScenario::RunSimulation(Time duration) {
    NS_LOG_FUNCTION(this << duration);

    NS_LOG_INFO("Starting Fat Tree Simulation with k=" << m_k);

    try {
        // Create the fat tree topology
        CreateFatTreeTopology(m_k);

        // Install network stack
        InternetStackHelper stack;
        stack.Install(m_allNodes);

        // Setup IP addressing
        SetupIpAddressing();

        // Setup routing
        SetupRouting();

        // Install MPI applications
        InstallMpiApplications();

        // Schedule MPI operations
        ScheduleMpiOperations();

        // Generate background traffic (optional)
        GenerateBackgroundTraffic();

        // Run simulation
        NS_LOG_INFO("Running simulation for " << duration.GetSeconds() << " seconds");
        Simulator::Stop(duration);
        Simulator::Run();

        // Collect results
        CollectResults();

    } catch (const std::exception& e) {
        NS_FATAL_ERROR("Exception during simulation: " << e.what());
    }

    Simulator::Destroy();
    NS_LOG_INFO("Simulation completed successfully");
}

void FatTreeScenario::CreateFatTreeTopology(uint32_t k) {
    NS_LOG_FUNCTION(this << k);

    NS_LOG_INFO("Creating Fat Tree topology with k=" << k);

    // Create nodes with proper error checking
    NS_LOG_INFO("Creating compute nodes: " << m_totalComputeNodes);
    m_computeNodes.Create(m_totalComputeNodes);

    NS_LOG_INFO("Creating edge switches: " << (m_pods * m_edgePerPod));
    m_edgeSwitches.Create(m_pods * m_edgePerPod);

    NS_LOG_INFO("Creating aggregation switches: " << (m_pods * m_aggregationPerPod));
    m_aggregationSwitches.Create(m_pods * m_aggregationPerPod);

    NS_LOG_INFO("Creating core switches: " << m_cores);
    m_coreSwitches.Create(m_cores);

    // Combine all nodes
    m_allNodes.Add(m_computeNodes);
    m_allNodes.Add(m_edgeSwitches);
    m_allNodes.Add(m_aggregationSwitches);
    m_allNodes.Add(m_coreSwitches);

    NS_LOG_INFO("Created " << m_allNodes.GetN() << " total nodes");

    // Create point-to-point helper with different configurations
    PointToPointHelper p2pComputeToEdge;
    p2pComputeToEdge.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gbps")));
    p2pComputeToEdge.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));

    PointToPointHelper p2pEdgeToAgg;
    p2pEdgeToAgg.SetDeviceAttribute("DataRate", DataRateValue(DataRate("40Gbps")));
    p2pEdgeToAgg.SetChannelAttribute("Delay", TimeValue(MicroSeconds(2)));

    PointToPointHelper p2pAggToCore;
    p2pAggToCore.SetDeviceAttribute("DataRate", DataRateValue(DataRate("40Gbps")));
    p2pAggToCore.SetChannelAttribute("Delay", TimeValue(MicroSeconds(5)));

    uint32_t linkCount = 0;

    // Connect compute nodes to edge switches
    NS_LOG_INFO("Connecting compute nodes to edge switches");
    for (uint32_t pod = 0; pod < m_pods; ++pod) {
        for (uint32_t edgeIdx = 0; edgeIdx < m_edgePerPod; ++edgeIdx) {
            uint32_t edgeSwitchIndex = pod * m_edgePerPod + edgeIdx;
            Ptr<Node> edgeSwitch = GetNodeSafe(m_edgeSwitches, edgeSwitchIndex, "edgeSwitches");

            for (uint32_t compIdx = 0; compIdx < m_computePerEdge; ++compIdx) {
                uint32_t computeIndex = pod * m_edgePerPod * m_computePerEdge +
                                      edgeIdx * m_computePerEdge + compIdx;

                if (computeIndex >= m_computeNodes.GetN()) {
                    NS_FATAL_ERROR("Compute index " << computeIndex << " out of bounds");
                }

                Ptr<Node> computeNode = GetNodeSafe(m_computeNodes, computeIndex, "computeNodes");

                // Connect compute node to edge switch
                NetDeviceContainer link = p2pComputeToEdge.Install(computeNode, edgeSwitch);
                m_allDevices.Add(link);
                m_links.push_back(link);
                linkCount++;

                NS_LOG_DEBUG("Connected compute " << computeIndex << " to edge " << edgeSwitchIndex);
            }
        }
    }

    // Connect edge switches to aggregation switches within pods
    NS_LOG_INFO("Connecting edge to aggregation switches");
    for (uint32_t pod = 0; pod < m_pods; ++pod) {
        for (uint32_t edgeIdx = 0; edgeIdx < m_edgePerPod; ++edgeIdx) {
            uint32_t edgeSwitchIndex = pod * m_edgePerPod + edgeIdx;
            Ptr<Node> edgeSwitch = GetNodeSafe(m_edgeSwitches, edgeSwitchIndex, "edgeSwitches");

            for (uint32_t aggIdx = 0; aggIdx < m_aggregationPerPod; ++aggIdx) {
                uint32_t aggSwitchIndex = pod * m_aggregationPerPod + aggIdx;
                Ptr<Node> aggSwitch = GetNodeSafe(m_aggregationSwitches, aggSwitchIndex, "aggregationSwitches");

                // Connect edge to aggregation switch
                NetDeviceContainer link = p2pEdgeToAgg.Install(edgeSwitch, aggSwitch);
                m_allDevices.Add(link);
                m_links.push_back(link);
                linkCount++;

                NS_LOG_DEBUG("Connected edge " << edgeSwitchIndex << " to aggregation " << aggSwitchIndex);
            }
        }
    }

    // Connect aggregation switches to core switches
    NS_LOG_INFO("Connecting aggregation to core switches");
    for (uint32_t pod = 0; pod < m_pods; ++pod) {
        for (uint32_t aggIdx = 0; aggIdx < m_aggregationPerPod; ++aggIdx) {
            uint32_t aggSwitchIndex = pod * m_aggregationPerPod + aggIdx;
            Ptr<Node> aggSwitch = GetNodeSafe(m_aggregationSwitches, aggSwitchIndex, "aggregationSwitches");

            // Each aggregation switch connects to (k/2) core switches
            for (uint32_t coreGroup = 0; coreGroup < m_k / 2; ++coreGroup) {
                uint32_t coreIndex = coreGroup * (m_k / 2) + aggIdx;

                if (coreIndex >= m_coreSwitches.GetN()) {
                    NS_LOG_ERROR("Core index " << coreIndex << " out of bounds (max: "
                                << (m_coreSwitches.GetN() - 1) << ")");
                    continue;
                }

                Ptr<Node> coreSwitch = GetNodeSafe(m_coreSwitches, coreIndex, "coreSwitches");

                // Connect aggregation to core switch
                NetDeviceContainer link = p2pAggToCore.Install(aggSwitch, coreSwitch);
                m_allDevices.Add(link);
                m_links.push_back(link);
                linkCount++;

                NS_LOG_DEBUG("Connected aggregation " << aggSwitchIndex << " to core " << coreIndex);
            }
        }
    }

    NS_LOG_INFO("Created " << linkCount << " total links");
}

void FatTreeScenario::SetupIpAddressing() {
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Setting up IP addressing for " << m_links.size() << " links");

    Ipv4AddressHelper address;
    address.SetBase("10.1.0.0", "255.255.0.0");

    // Assign IP addresses to all links
    for (uint32_t i = 0; i < m_links.size(); ++i) {
        if (m_links[i].GetN() == 2) {
            Ipv4InterfaceContainer interfaces = address.Assign(m_links[i]);
            address.NewNetwork();

            if (i < 10) {
                NS_LOG_DEBUG("Assigned IPs for link " << i << ": "
                            << interfaces.GetAddress(0) << " <-> " << interfaces.GetAddress(1));
            }
        } else {
            NS_LOG_WARN("Link " << i << " has " << m_links[i].GetN() << " devices (expected 2)");
        }
    }

    NS_LOG_INFO("IP addressing completed");
}

void FatTreeScenario::SetupRouting() {
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Setting up routing");

    // Use global routing for simplicity
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    NS_LOG_INFO("Routing tables populated");
}

void FatTreeScenario::InstallMpiApplications() {
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Installing MPI Research Applications on " << m_computeNodes.GetN() << " compute nodes");

    // Create a map to store IP addresses of compute nodes
    std::map<uint32_t, Ipv4Address> rankToAddress;

    // First pass: install applications and collect addresses
    for (uint32_t i = 0; i < m_computeNodes.GetN(); ++i) {
        Ptr<Node> computeNode = GetNodeSafe(m_computeNodes, i, "computeNodes");

        // Create MPI Research Application
        Ptr<MpiResearchApplication> mpiApp = CreateObject<MpiResearchApplication>();

        // Configure MPI parameters
        mpiApp->SetRank(i);
        mpiApp->SetWorldSize(m_totalComputeNodes);
        mpiApp->SetNetworkTopology(MpiResearchApplication::FAT_TREE);
        mpiApp->EnableDetailedLogging(m_enableMpiLogging);
        mpiApp->SetComputationDelay(m_computationDelay);
        mpiApp->SetCommunicationDelay(m_communicationDelay);

        // Configure node information
        MpiResearchApplication::NetworkNodeInfo nodeInfo;
        nodeInfo.rank = i;

        // Get the IP address of the compute node (first interface)
        Ptr<Ipv4> ipv4 = computeNode->GetObject<Ipv4>();
        if (ipv4 && ipv4->GetNInterfaces() > 1) {
            nodeInfo.ipAddress = ipv4->GetAddress(1, 0).GetLocal();
            rankToAddress[i] = nodeInfo.ipAddress;
        }

        nodeInfo.cpuCapacity = 1000 + (i % 4) * 500; // Varying CPU capacities
        mpiApp->SetNodeInformation(nodeInfo);

        // Install application on compute node
        computeNode->AddApplication(mpiApp);
        mpiApp->SetStartTime(Seconds(0.0));
        mpiApp->SetStopTime(Seconds(25.0));

        m_mpiApps.push_back(mpiApp);
        m_applications.Add(mpiApp);

        NS_LOG_DEBUG("Installed MPI Application on rank " << i);
    }

    // Second pass: establish neighbor relationships
    NS_LOG_INFO("Establishing MPI neighbor relationships");
    for (uint32_t i = 0; i < m_mpiApps.size(); ++i) {
        for (uint32_t j = 0; j < m_mpiApps.size(); ++j) {
            if (i != j) {
                auto it = rankToAddress.find(j);
                if (it != rankToAddress.end()) {
                    m_mpiApps[i]->AddNeighbor(j, it->second);
                }
            }
        }
    }

    NS_LOG_INFO("MPI Applications installed and configured");
}

void FatTreeScenario::ScheduleMpiOperations() {
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Scheduling MPI collective operations");

    if (m_mpiApps.empty()) {
        NS_LOG_ERROR("No MPI applications available for scheduling");
        return;
    }

    // Initial barrier synchronization
    for (uint32_t i = 0; i < m_mpiApps.size(); ++i) {
        Simulator::Schedule(Seconds(1.0), &MpiResearchApplication::SimulateBarrier, m_mpiApps[i]);
    }

    // Broadcast operations with different roots and data sizes
    Simulator::Schedule(Seconds(3.0), &MpiResearchApplication::SimulateBroadcast,
                       m_mpiApps[0], 0, 2048);  // Small broadcast from root 0

    Simulator::Schedule(Seconds(5.0), &MpiResearchApplication::SimulateBroadcast,
                       m_mpiApps[m_mpiApps.size()/2], m_mpiApps.size()/2, 1048576);  // Large broadcast

    // Allreduce operations with different data sizes
    Simulator::Schedule(Seconds(7.0), &MpiResearchApplication::SimulateAllreduce,
                       m_mpiApps[0], 4096);  // Small allreduce

    Simulator::Schedule(Seconds(9.0), &MpiResearchApplication::SimulateAllreduce,
                       m_mpiApps[0], 524288);  // Large allreduce

    // Reduce operation
    Simulator::Schedule(Seconds(11.0), &MpiResearchApplication::SimulateReduce,
                       m_mpiApps[0], 0, 8192);

    // Allgather operation
    Simulator::Schedule(Seconds(13.0), &MpiResearchApplication::SimulateAllgather,
                       m_mpiApps[0], 16384);

    // Topology-aware broadcast (optimized for fat-tree)
    Simulator::Schedule(Seconds(15.0), &MpiResearchApplication::SimulateTopologyAwareBroadcast,
                       m_mpiApps[0], 0, 2097152);

    // Hierarchical allreduce
    Simulator::Schedule(Seconds(17.0), &MpiResearchApplication::SimulateHierarchicalAllreduce,
                       m_mpiApps[0], 1048576);

    // Pipeline broadcast
    Simulator::Schedule(Seconds(19.0), &MpiResearchApplication::SimulatePipelineBroadcast,
                       m_mpiApps[0], 0, 262144);

    // Final barrier
    for (uint32_t i = 0; i < m_mpiApps.size(); ++i) {
        Simulator::Schedule(Seconds(21.0), &MpiResearchApplication::SimulateBarrier, m_mpiApps[i]);
    }

    NS_LOG_INFO("MPI operations scheduled");
}

void FatTreeScenario::GenerateBackgroundTraffic() {
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Generating background traffic");

    // Install packet sink on all compute nodes
    PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                               InetSocketAddress(Ipv4Address::GetAny(), 9));
    ApplicationContainer sinkApps = sinkHelper.Install(m_computeNodes);
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(25.0));
    m_applications.Add(sinkApps);

    // Generate some background traffic to create network load
    for (uint32_t i = 0; i < std::min(uint32_t(3), m_computeNodes.GetN()); ++i) {
        uint32_t destNode = (i + m_computeNodes.GetN()/2) % m_computeNodes.GetN();

        Ptr<Node> srcNode = GetNodeSafe(m_computeNodes, i, "computeNodes");
        Ptr<Node> dstNode = GetNodeSafe(m_computeNodes, destNode, "computeNodes");

        Ptr<Ipv4> ipv4 = dstNode->GetObject<Ipv4>();
        if (ipv4 && ipv4->GetNInterfaces() > 1) {
            Ipv4Address dstAddress = ipv4->GetAddress(1, 0).GetLocal();

            OnOffHelper onOff("ns3::UdpSocketFactory",
                            InetSocketAddress(dstAddress, 9));
            onOff.SetConstantRate(DataRate("500Kbps"));  // Low rate background traffic
            onOff.SetAttribute("PacketSize", UintegerValue(512));

            ApplicationContainer apps = onOff.Install(srcNode);
            apps.Start(Seconds(2.0 + i * 1.0));
            apps.Stop(Seconds(22.0));

            m_applications.Add(apps);

            NS_LOG_DEBUG("Created background traffic from node " << i << " to " << destNode
                        << " address " << dstAddress);
        }
    }

    NS_LOG_INFO("Background traffic generation completed");
}

void FatTreeScenario::CollectResults() {
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Collecting simulation results");

    // Create results directory
    system("mkdir -p results");

    // Write topology information
    std::ofstream topoFile("results/fat_tree_topology.txt");
    topoFile << "Fat Tree Topology Results\n";
    topoFile << "=========================\n";
    topoFile << "k parameter: " << m_k << "\n";
    topoFile << "Total nodes: " << m_allNodes.GetN() << "\n";
    topoFile << "Compute nodes: " << m_computeNodes.GetN() << "\n";
    topoFile << "Edge switches: " << m_edgeSwitches.GetN() << "\n";
    topoFile << "Aggregation switches: " << m_aggregationSwitches.GetN() << "\n";
    topoFile << "Core switches: " << m_coreSwitches.GetN() << "\n";
    topoFile << "Total links: " << m_links.size() << "\n\n";

    // MPI application information
    topoFile << "MPI Configuration:\n";
    topoFile << "World Size: " << m_totalComputeNodes << "\n";
    topoFile << "Topology: FAT_TREE\n";
    topoFile << "Computation Delay: " << m_computationDelay.GetMilliSeconds() << " ms\n";
    topoFile << "Communication Delay: " << m_communicationDelay.GetMicroSeconds() << " μs\n";
    topoFile.close();

    // Write MPI performance results
    std::ofstream mpiFile("results/mpi_performance.csv");
    mpiFile << "rank,total_messages,total_data_sent(bytes),total_comm_time(s),operations_completed\n";

    for (uint32_t i = 0; i < m_mpiApps.size(); ++i) {
        Ptr<MpiResearchApplication> app = m_mpiApps[i];

        std::vector<MpiResearchApplication::MpiPerformanceMetrics> history =
            app->GetOperationHistory();

        mpiFile << i << ","
                << app->m_totalMessagesSent << ","
                << app->m_totalDataSent << ","
                << app->m_totalCommunicationTime.GetSeconds() << ","
                << history.size() << "\n";

        // Write detailed operation history for each rank
        if (i < 5) { // Only write for first 5 ranks to avoid too many files
            std::ofstream rankFile("results/rank_" + std::to_string(i) + "_operations.csv");
            rankFile << "operation_id,execution_time(s),data_volume(bytes)\n";

            for (const auto& metrics : history) {
                rankFile << "op_" << i << "_" << (&metrics - &history[0]) << ","
                         << metrics.executionTime.GetSeconds() << ","
                         << metrics.dataVolume << "\n";
            }
            rankFile.close();
        }
    }
    mpiFile.close();

    // Write summary statistics
    std::ofstream summaryFile("results/simulation_summary.txt");
    summaryFile << "Simulation Summary\n";
    summaryFile << "==================\n";

    uint32_t totalMessages = 0;
    uint64_t totalData = 0;
    double totalCommTime = 0.0;

    for (uint32_t i = 0; i < m_mpiApps.size(); ++i) {
        totalMessages += m_mpiApps[i]->m_totalMessagesSent;
        totalData += m_mpiApps[i]->m_totalDataSent;
        totalCommTime += m_mpiApps[i]->m_totalCommunicationTime.GetSeconds();
    }

    summaryFile << "Total MPI Messages: " << totalMessages << "\n";
    summaryFile << "Total Data Sent: " << totalData << " bytes\n";
    summaryFile << "Total Communication Time: " << totalCommTime << " seconds\n";
    summaryFile << "Average Messages per Rank: " << (double)totalMessages / m_mpiApps.size() << "\n";
    summaryFile << "Average Data per Rank: " << (double)totalData / m_mpiApps.size() << " bytes\n";
    summaryFile.close();

    NS_LOG_INFO("Results written to results/ directory");
}

int main(int argc, char* argv[]) {
    // Enable logging
    LogComponentEnable("FatTreeScenario", LOG_LEVEL_INFO);
    LogComponentEnable("MpiResearchApplication", LOG_LEVEL_INFO);

    uint32_t k = 4; // Default k-ary fat tree
    double duration = 30.0; // Default simulation duration
    bool enableMpiLogging = true;

    CommandLine cmd;
    cmd.AddValue("k", "Fat tree k parameter (even number)", k);
    cmd.AddValue("duration", "Simulation duration in seconds", duration);
    cmd.AddValue("mpi-logging", "Enable MPI detailed logging", enableMpiLogging);
    cmd.Parse(argc, argv);

    // Validate k parameter
    if (k % 2 != 0 || k < 2) {
        std::cerr << "Error: k must be an even number >= 2" << std::endl;
        return 1;
    }

    NS_LOG_INFO("Starting Fat Tree MPI simulation with k=" << k << ", duration=" << duration << "s");

    try {
        FatTreeScenario scenario(k);
        scenario.RunSimulation(Seconds(duration));

        NS_LOG_INFO("Fat Tree MPI simulation completed successfully");
        std::cout << "Simulation completed successfully!" << std::endl;
        std::cout << "Check results/ directory for MPI performance data." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Simulation failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}