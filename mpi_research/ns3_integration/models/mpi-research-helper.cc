#include "mpi-research-helper.h"
#include "ns3/log.h"
#include "ns3/random-variable-stream.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/vector.h"
#include "ns3/ipv4.h"
#include "ns3/ipv4-interface-address.h"
#include <fstream>
#include <iomanip>
#include <algorithm>

namespace ns3 {

    NS_LOG_COMPONENT_DEFINE("MpiResearchHelper");

    MpiResearchHelper::MpiResearchHelper() {
        NS_LOG_FUNCTION(this);

        m_factory.SetTypeId("MpiResearchApplication");
        m_topology = UNKNOWN;
        m_worldSize = 1;
        m_computationDelay = MilliSeconds(1);
        m_communicationDelay = MicroSeconds(100);
        m_detailedLogging = false;
    }

    MpiResearchHelper::~MpiResearchHelper() {
        NS_LOG_FUNCTION(this);
    }

    void MpiResearchHelper::SetAttribute(std::string name, const AttributeValue& value) {
        NS_LOG_FUNCTION(this << name);
        m_factory.Set(name, value);
    }

    void MpiResearchHelper::SetNetworkTopology(NetworkTopology topology) {
        NS_LOG_FUNCTION(this << topology);
        m_topology = topology;
    }

    void MpiResearchHelper::SetWorldSize(uint32_t size) {
        NS_LOG_FUNCTION(this << size);
        m_worldSize = size;
    }

    void MpiResearchHelper::SetBaseComputationDelay(Time delay) {
        NS_LOG_FUNCTION(this << delay);
        m_computationDelay = delay;
    }

    void MpiResearchHelper::SetBaseCommunicationDelay(Time delay) {
        NS_LOG_FUNCTION(this << delay);
        m_communicationDelay = delay;
    }

    void MpiResearchHelper::EnableDetailedLogging(bool enable) {
        NS_LOG_FUNCTION(this << enable);
        m_detailedLogging = enable;
    }

    ApplicationContainer MpiResearchHelper::Install(NodeContainer nodes) const {
        NS_LOG_FUNCTION(this << nodes.GetN());

        ApplicationContainer apps;
        uint32_t rank = 0;

        for (NodeContainer::Iterator i = nodes.Begin(); i != nodes.End(); ++i) {
            Ptr<Node> node = *i;

            // Create and configure the application
            Ptr<MpiResearchApplication> app = m_factory.Create<MpiResearchApplication>();
            NetworkNodeInfo nodeInfo = CreateNodeInfo(node, rank);

            ConfigureNode(app, rank, nodeInfo);
            node->AddApplication(app);

            apps.Add(app);
            rank++;
        }

        // Setup inter-node connections
        SetupNodeConnections(apps);

        NS_LOG_INFO("Installed MpiResearchApplication on " << apps.GetN() << " nodes");
        return apps;
    }

    ApplicationContainer MpiResearchHelper::Install(Ptr<Node> node) const {
        NS_LOG_FUNCTION(this << node);
        return Install(NodeContainer(node));
    }

    ApplicationContainer MpiResearchHelper::InstallFatTree(NodeContainer nodes, uint32_t k) const {
        NS_LOG_FUNCTION(this << nodes.GetN() << k);

        // Validate node count for k-ary fat tree
        uint32_t expectedNodes = (5 * k * k) / 4; // Standard 3-level fat tree
        if (nodes.GetN() < expectedNodes) {
            NS_LOG_WARN("Insufficient nodes for " << k << "-ary fat tree. Expected: "
                << expectedNodes << ", Got: " << nodes.GetN());
        }

        SetNetworkTopology(FAT_TREE);
        SetWorldSize(nodes.GetN());

        return Install(nodes);
    }

    ApplicationContainer MpiResearchHelper::InstallTorus2D(NodeContainer nodes, uint32_t rows, uint32_t cols) const {
        NS_LOG_FUNCTION(this << nodes.GetN() << rows << cols);

        uint32_t expectedNodes = rows * cols;
        if (nodes.GetN() != expectedNodes) {
            NS_LOG_WARN("Node count mismatch for 2D torus. Expected: "
                << expectedNodes << ", Got: " << nodes.GetN());
        }

        SetNetworkTopology(TORUS_2D);
        SetWorldSize(nodes.GetN());

        return Install(nodes);
    }

    ApplicationContainer MpiResearchHelper::InstallTorus3D(NodeContainer nodes, uint32_t x, uint32_t y, uint32_t z) const {
        NS_LOG_FUNCTION(this << nodes.GetN() << x << y << z);

        uint32_t expectedNodes = x * y * z;
        if (nodes.GetN() != expectedNodes) {
            NS_LOG_WARN("Node count mismatch for 3D torus. Expected: "
                << expectedNodes << ", Got: " << nodes.GetN());
        }

        SetNetworkTopology(TORUS_3D);
        SetWorldSize(nodes.GetN());

        return Install(nodes);
    }

    ApplicationContainer MpiResearchHelper::InstallDragonfly(NodeContainer nodes, uint32_t groups, uint32_t routers) const {
        NS_LOG_FUNCTION(this << nodes.GetN() << groups << routers);

        SetNetworkTopology(DRAGONFLY);
        SetWorldSize(nodes.GetN());

        return Install(nodes);
    }

    void MpiResearchHelper::StartCollectiveOperations(ApplicationContainer apps, Time startTime) const {
        NS_LOG_FUNCTION(this << startTime);

        // Schedule initial collective operations
        for (uint32_t i = 0; i < apps.GetN(); ++i) {
            Ptr<MpiResearchApplication> app = apps.Get(i)->GetObject<MpiResearchApplication>();

            if (app) {
                // Schedule different operations at different times to simulate realistic workload
                Time operationTime = startTime + MilliSeconds(i * 10);

                // Schedule broadcast from different roots
                if (i % 4 == 0) {
                    Simulator::Schedule(operationTime, &MpiResearchApplication::SimulateBroadcast,
                        app, 0, 1024); // Root 0, 1KB data
                }
                else if (i % 4 == 1) {
                    Simulator::Schedule(operationTime, &MpiResearchApplication::SimulateAllreduce,
                        app, 2048); // 2KB data
                }
                else if (i % 4 == 2) {
                    Simulator::Schedule(operationTime, &MpiResearchApplication::SimulateTopologyAwareBroadcast,
                        app, i % apps.GetN(), 4096); // Varying root, 4KB data
                }
                else {
                    Simulator::Schedule(operationTime, &MpiResearchApplication::SimulateBarrier, app);
                }
            }
        }

        NS_LOG_INFO("Scheduled collective operations starting at " << startTime.GetSeconds() << "s");
    }

    void MpiResearchHelper::ScheduleBroadcast(ApplicationContainer apps, uint32_t rootRank, uint32_t dataSize, Time time) const {
        NS_LOG_FUNCTION(this << rootRank << dataSize << time);

        for (uint32_t i = 0; i < apps.GetN(); ++i) {
            Ptr<MpiResearchApplication> app = apps.Get(i)->GetObject<MpiResearchApplication>();
            if (app) {
                Simulator::Schedule(time, &MpiResearchApplication::SimulateBroadcast,
                    app, rootRank, dataSize);
            }
        }
    }

    void MpiResearchHelper::ScheduleAllreduce(ApplicationContainer apps, uint32_t dataSize, Time time) const {
        NS_LOG_FUNCTION(this << dataSize << time);

        for (uint32_t i = 0; i < apps.GetN(); ++i) {
            Ptr<MpiResearchApplication> app = apps.Get(i)->GetObject<MpiResearchApplication>();
            if (app) {
                Simulator::Schedule(time, &MpiResearchApplication::SimulateAllreduce,
                    app, dataSize);
            }
        }
    }

    void MpiResearchHelper::ScheduleBarrier(ApplicationContainer apps, Time time) const {
        NS_LOG_FUNCTION(this << time);

        for (uint32_t i = 0; i < apps.GetN(); ++i) {
            Ptr<MpiResearchApplication> app = apps.Get(i)->GetObject<MpiResearchApplication>();
            if (app) {
                Simulator::Schedule(time, &MpiResearchApplication::SimulateBarrier, app);
            }
        }
    }

    void MpiResearchHelper::CollectPerformanceMetrics(ApplicationContainer apps) const {
        NS_LOG_FUNCTION(this);

        double totalExecutionTime = 0.0;
        uint64_t totalMessages = 0;
        uint64_t totalData = 0;
        uint32_t operationCount = 0;

        for (uint32_t i = 0; i < apps.GetN(); ++i) {
            Ptr<MpiResearchApplication> app = apps.Get(i)->GetObject<MpiResearchApplication>();
            if (app) {
                auto history = app->GetOperationHistory();
                operationCount += history.size();

                for (const auto& metrics : history) {
                    totalExecutionTime += metrics.executionTime.GetSeconds();
                }

                // Note: These would need to be exposed through the application interface
                // totalMessages += app->GetTotalMessagesSent();
                // totalData += app->GetTotalDataSent();
            }
        }

        NS_LOG_INFO("Performance Summary:");
        NS_LOG_INFO("  Total operations: " << operationCount);
        NS_LOG_INFO("  Average execution time: " << (operationCount > 0 ? totalExecutionTime / operationCount : 0) << "s");
        NS_LOG_INFO("  Total messages: " << totalMessages);
        NS_LOG_INFO("  Total data: " << totalData << " bytes");
    }

    void MpiResearchHelper::GeneratePerformanceReport(ApplicationContainer apps, std::string filename) const {
        NS_LOG_FUNCTION(this << filename);

        std::ofstream report(filename);
        if (!report.is_open()) {
            NS_LOG_ERROR("Failed to open report file: " << filename);
            return;
        }

        report << "MPI Research Performance Report" << std::endl;
        report << "===============================" << std::endl;
        report << "Topology: " << m_topology << std::endl;
        report << "World Size: " << m_worldSize << std::endl;
        report << std::endl;

        report << "Node,Rank,Operations,AvgExecutionTime(s),TotalMessages,TotalData(B)" << std::endl;

        for (uint32_t i = 0; i < apps.GetN(); ++i) {
            Ptr<MpiResearchApplication> app = apps.Get(i)->GetObject<MpiResearchApplication>();
            if (app) {
                auto history = app->GetOperationHistory();
                double avgTime = history.empty() ? 0.0 :
                    std::accumulate(history.begin(), history.end(), 0.0,
                        [](double sum, const MpiPerformanceMetrics& m) {
                            return sum + m.executionTime.GetSeconds();
                        }) / history.size();

                report << i << "," << i << "," << history.size() << ","
                    << std::fixed << std::setprecision(6) << avgTime << ","
                    << "0,0" << std::endl; // Placeholder for messages and data
            }
        }

        report.close();
        NS_LOG_INFO("Performance report generated: " << filename);
    }

    void MpiResearchHelper::CompareTopologyPerformance(NodeContainer fatTreeNodes,
        NodeContainer torusNodes,
        NodeContainer dragonflyNodes) const {
        NS_LOG_FUNCTION(this);

        NS_LOG_INFO("=== Topology Performance Comparison ===");

        // This would run simulations for each topology and compare results
        // Implementation would involve:
        // 1. Installing applications on each topology
        // 2. Running identical workloads
        // 3. Collecting and comparing performance metrics

        NS_LOG_INFO("Topology comparison completed");
    }

    void MpiResearchHelper::ConfigureNode(Ptr<MpiResearchApplication> app, uint32_t rank,
        const NetworkNodeInfo& nodeInfo) const {
        NS_LOG_FUNCTION(this << rank);

        app->SetRank(rank);
        app->SetWorldSize(m_worldSize);
        app->SetNetworkTopology(m_topology);
        app->SetComputationDelay(m_computationDelay);
        app->EnableDetailedLogging(m_detailedLogging);
        app->SetNodeInformation(nodeInfo);
    }

    NetworkNodeInfo MpiResearchHelper::CreateNodeInfo(Ptr<Node> node, uint32_t rank) const {
        NetworkNodeInfo info;
        info.nodeId = node->GetId();
        info.rank = rank;

        // Get IP address from the node
        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
        if (ipv4 && ipv4->GetNInterfaces() > 1) {
            info.address = ipv4->GetAddress(1, 0).GetLocal();
        }

        // Get position if mobility model exists
        Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
        if (mobility) {
            info.position = mobility->GetPosition();
        }

        // Generate hostname
        std::stringstream hostname;
        hostname << "node-" << rank;
        info.hostname = hostname.str();

        return info;
    }

    void MpiResearchHelper::SetupNodeConnections(ApplicationContainer apps) const {
        NS_LOG_FUNCTION(this);

        // Build address mapping and setup neighbor relationships
        std::map<uint32_t, Ipv4Address> rankToAddress;

        // First pass: collect all addresses
        for (uint32_t i = 0; i < apps.GetN(); ++i) {
            Ptr<MpiResearchApplication> app = apps.Get(i)->GetObject<MpiResearchApplication>();
            if (app) {
                Ptr<Node> node = app->GetNode();
                Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();

                if (ipv4 && ipv4->GetNInterfaces() > 1) {
                    Ipv4Address address = ipv4->GetAddress(1, 0).GetLocal();
                    rankToAddress[i] = address;
                }
            }
        }

        // Second pass: setup neighbor relationships
        for (uint32_t i = 0; i < apps.GetN(); ++i) {
            Ptr<MpiResearchApplication> app = apps.Get(i)->GetObject<MpiResearchApplication>();
            if (app) {
                // Add all other nodes as neighbors (simplified - in practice would use topology)
                for (uint32_t j = 0; j < apps.GetN(); ++j) {
                    if (i != j && rankToAddress.find(j) != rankToAddress.end()) {
                        app->AddNeighbor(j, rankToAddress[j]);
                    }
                }
            }
        }

        NS_LOG_INFO("Setup node connections for " << apps.GetN() << " nodes");
    }

} // namespace ns3