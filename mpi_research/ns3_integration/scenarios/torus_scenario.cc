// mpi-research-scenario.cc
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/csma-module.h"
#include "mpi-research-application.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("MpiResearchScenario");

int main(int argc, char *argv[])
{
    // Configuration parameters
    uint32_t worldSize = 8;
    uint32_t simulationTime = 30;
    bool enableLogging = true;
    std::string topologyType = "fat-tree"; // Options: fat-tree, torus-2d, dragonfly, simple

    CommandLine cmd;
    cmd.AddValue("worldSize", "Number of MPI processes", worldSize);
    cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
    cmd.AddValue("enableLogging", "Enable detailed logging", enableLogging);
    cmd.AddValue("topology", "Network topology type", topologyType);
    cmd.Parse(argc, argv);

    // Create MPI nodes
    NodeContainer mpiNodes;
    mpiNodes.Create(worldSize);

    // Configure network topology based on selection
    InternetStackHelper stack;
    NetDeviceContainer devices;
    Ipv4InterfaceContainer interfaces;

    if (topologyType == "fat-tree") {
        NS_LOG_INFO("Configuring Fat-Tree topology");

        // Simplified fat-tree with point-to-point links
        PointToPointHelper p2p;
        p2p.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
        p2p.SetChannelAttribute("Delay", StringValue("0.01ms"));

        // Create core switches
        NodeContainer coreSwitches;
        coreSwitches.Create(2);

        // Install internet stack on all nodes
        stack.Install(mpiNodes);
        stack.Install(coreSwitches);

        // Connect MPI nodes to core switches in fat-tree pattern
        for (uint32_t i = 0; i < worldSize; ++i) {
            NetDeviceContainer link = p2p.Install(mpiNodes.Get(i), coreSwitches.Get(i % 2));
            devices.Add(link);
        }

    } else if (topologyType == "torus-2d") {
        NS_LOG_INFO("Configuring 2D Torus topology");

        PointToPointHelper p2p;
        p2p.SetDeviceAttribute("DataRate", StringValue("5Gbps"));
        p2p.SetChannelAttribute("Delay", StringValue("0.05ms"));

        stack.Install(mpiNodes);

        // Create 2D torus connections (simplified)
        uint32_t gridSize = static_cast<uint32_t>(std::sqrt(worldSize));
        for (uint32_t i = 0; i < worldSize; ++i) {
            for (uint32_t j = i + 1; j < worldSize; ++j) {
                // Connect neighbors in grid (simplified)
                if ((i / gridSize == j / gridSize && std::abs((int)i - (int)j) == 1) ||
                    (i % gridSize == j % gridSize && std::abs((int)(i / gridSize) - (int)(j / gridSize)) == 1)) {
                    NetDeviceContainer link = p2p.Install(mpiNodes.Get(i), mpiNodes.Get(j));
                    devices.Add(link);
                }
            }
        }

    } else {
        // Default: simple star topology
        NS_LOG_INFO("Configuring star topology");

        PointToPointHelper p2p;
        p2p.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
        p2p.SetChannelAttribute("Delay", StringValue("0.1ms"));

        NodeController switchNode;
        switchNode.Create(1);

        stack.Install(mpiNodes);
        stack.Install(switchNode);

        // Connect all MPI nodes to central switch
        for (uint32_t i = 0; i < worldSize; ++i) {
            NetDeviceContainer link = p2p.Install(mpiNodes.Get(i), switchNode.Get(0));
            devices.Add(link);
        }
    }

    // Assign IP addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.0.0", "255.255.0.0");
    interfaces = address.Assign(devices);

    // Create and install MPI Research Applications
    ApplicationContainer mpiApps;
    std::vector<Ptr<MpiResearchApplication>> mpiApplications;

    for (uint32_t i = 0; i < worldSize; ++i) {
        Ptr<MpiResearchApplication> mpiApp = CreateObject<MpiResearchApplication>();

        // Set basic MPI parameters
        mpiApp->SetRank(i);
        mpiApp->SetWorldSize(worldSize);
        mpiApp->EnableDetailedLogging(enableLogging);

        // Set network topology
        if (topologyType == "fat-tree") {
            mpiApp->SetNetworkTopology(MpiResearchApplication::FAT_TREE);
        } else if (topologyType == "torus-2d") {
            mpiApp->SetNetworkTopology(MpiResearchApplication::TORUS_2D);
        } else if (topologyType == "dragonfly") {
            mpiApp->SetNetworkTopology(MpiResearchApplication::DRAGONFLY);
        } else {
            mpiApp->SetNetworkTopology(MpiResearchApplication::UNKNOWN);
        }

        // Configure node information
        MpiResearchApplication::NetworkNodeInfo nodeInfo;
        nodeInfo.rank = i;
        nodeInfo.ipAddress = interfaces.GetAddress(i);
        nodeInfo.cpuCapacity = 1000; // MIPS
        mpiApp->SetNodeInformation(nodeInfo);

        // Set timing parameters
        mpiApp->SetComputationDelay(MilliSeconds(2));
        mpiApp->SetCommunicationDelay(MicroSeconds(50));

        mpiNodes.Get(i)->AddApplication(mpiApp);
        mpiApps.Add(mpiApp);
        mpiApplications.push_back(mpiApp);
    }

    // Build neighbor mappings
    for (uint32_t i = 0; i < worldSize; ++i) {
        for (uint32_t j = 0; j < worldSize; ++j) {
            if (i != j) {
                mpiApplications[i]->AddNeighbor(j, interfaces.GetAddress(j));
            }
        }
    }

    // Schedule MPI operations
    NS_LOG_INFO("Scheduling MPI operations...");

    // Barrier synchronization at start
    for (uint32_t i = 0; i < worldSize; ++i) {
        Simulator::Schedule(Seconds(1.0), &MpiResearchApplication::SimulateBarrier,
                          mpiApplications[i]);
    }

    // Broadcast operations with different roots and data sizes
    Simulator::Schedule(Seconds(2.0), &MpiResearchApplication::SimulateBroadcast,
                       mpiApplications[0], 0, 2048);  // Small broadcast from root 0

    Simulator::Schedule(Seconds(4.0), &MpiResearchApplication::SimulateBroadcast,
                       mpiApplications[worldSize/2], worldSize/2, 1048576);  // Large broadcast

    // Allreduce operations
    Simulator::Schedule(Seconds(6.0), &MpiResearchApplication::SimulateAllreduce,
                       mpiApplications[0], 4096);  // Small allreduce

    Simulator::Schedule(Seconds(8.0), &MpiResearchApplication::SimulateAllreduce,
                       mpiApplications[0], 524288);  // Large allreduce

    // Reduce operation
    Simulator::Schedule(Seconds(10.0), &MpiResearchApplication::SimulateReduce,
                       mpiApplications[0], 0, 8192);

    // Allgather operation
    Simulator::Schedule(Seconds(12.0), &MpiResearchApplication::SimulateAllgather,
                       mpiApplications[0], 16384);

    // Advanced topology-aware operations
    if (topologyType == "fat-tree") {
        Simulator::Schedule(Seconds(14.0), &MpiResearchApplication::SimulateTopologyAwareBroadcast,
                           mpiApplications[0], 0, 2097152);

        Simulator::Schedule(Seconds(16.0), &MpiResearchApplication::SimulateHierarchicalAllreduce,
                           mpiApplications[0], 1048576);
    }

    // Pipeline broadcast
    Simulator::Schedule(Seconds(18.0), &MpiResearchApplication::SimulatePipelineBroadcast,
                       mpiApplications[0], 0, 262144);

    // Final barrier
    for (uint32_t i = 0; i < worldSize; ++i) {
        Simulator::Schedule(Seconds(20.0), &MpiResearchApplication::SimulateBarrier,
                          mpiApplications[i]);
    }

    // Stop applications
    mpiApps.Stop(Seconds(simulationTime));

    // Enable PCAP tracing if needed
    // p2p.EnablePcapAll("mpi-research");

    // Run simulation
    NS_LOG_INFO("Starting simulation...");
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();

    // Collect and display results
    NS_LOG_INFO("Simulation completed. Collecting results...");

    for (uint32_t i = 0; i < worldSize; ++i) {
        Ptr<MpiResearchApplication> app = mpiApplications[i];

        std::vector<MpiResearchApplication::MpiPerformanceMetrics> history =
            app->GetOperationHistory();

        NS_LOG_INFO("=== Rank " << i << " Results ===");
        NS_LOG_INFO("Total operations: " << history.size());
        NS_LOG_INFO("Total messages sent: " << app->m_totalMessagesSent);
        NS_LOG_INFO("Total data sent: " << app->m_totalDataSent << " bytes");
        NS_LOG_INFO("Total communication time: " << app->m_totalCommunicationTime.GetSeconds() << "s");

        if (!history.empty()) {
            MpiResearchApplication::MpiPerformanceMetrics lastOp = app->GetLastOperationMetrics();
            NS_LOG_INFO("Last operation - Execution time: " << lastOp.executionTime.GetSeconds()
                        << "s, Data volume: " << lastOp.dataVolume << " bytes");
        }

        NS_LOG_INFO("");
    }

    Simulator::Destroy();
    NS_LOG_INFO("Simulation finished successfully");

    return 0;
}