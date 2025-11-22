#ifndef MPI_RESEARCH_HELPER_H
#define MPI_RESEARCH_HELPER_H

#include "ns3/object-factory.h"
#include "ns3/ipv4-address.h"
#include "ns3/node-container.h"
#include "ns3/application-container.h"
#include "ns3/mpi-research-application.h"
#include "ns3/net-device-container.h"
#include "ns3/topology.h"
#include <map>
#include <vector>
#include <string>

namespace ns3 {

    class MpiResearchHelper {
    public:
        MpiResearchHelper();
        ~MpiResearchHelper();

        // Configuration methods
        void SetAttribute(std::string name, const AttributeValue& value);
        void SetNetworkTopology(NetworkTopology topology);
        void SetWorldSize(uint32_t size);
        void SetBaseComputationDelay(Time delay);
        void SetBaseCommunicationDelay(Time delay);
        void EnableDetailedLogging(bool enable);

        // Installation methods
        ApplicationContainer Install(NodeContainer nodes) const;
        ApplicationContainer Install(Ptr<Node> node) const;

        // Topology-specific installation
        ApplicationContainer InstallFatTree(NodeContainer nodes, uint32_t k) const;
        ApplicationContainer InstallTorus2D(NodeContainer nodes, uint32_t rows, uint32_t cols) const;
        ApplicationContainer InstallTorus3D(NodeContainer nodes, uint32_t x, uint32_t y, uint32_t z) const;
        ApplicationContainer InstallDragonfly(NodeContainer nodes, uint32_t groups, uint32_t routers) const;

        // Simulation control
        void StartCollectiveOperations(ApplicationContainer apps, Time startTime = Seconds(1.0)) const;
        void ScheduleBroadcast(ApplicationContainer apps, uint32_t rootRank, uint32_t dataSize, Time time) const;
        void ScheduleAllreduce(ApplicationContainer apps, uint32_t dataSize, Time time) const;
        void ScheduleBarrier(ApplicationContainer apps, Time time) const;

        // Analysis and reporting
        void CollectPerformanceMetrics(ApplicationContainer apps) const;
        void GeneratePerformanceReport(ApplicationContainer apps, std::string filename) const;
        void CompareTopologyPerformance(NodeContainer fatTreeNodes, NodeContainer torusNodes,
            NodeContainer dragonflyNodes) const;

    private:
        void ConfigureNode(Ptr<MpiResearchApplication> app, uint32_t rank,
            const NetworkNodeInfo& nodeInfo) const;
        NetworkNodeInfo CreateNodeInfo(Ptr<Node> node, uint32_t rank) const;
        void SetupNodeConnections(ApplicationContainer apps) const;

        ObjectFactory m_factory;
        NetworkTopology m_topology;
        uint32_t m_worldSize;
        Time m_computationDelay;
        Time m_communicationDelay;
        bool m_detailedLogging;
    };

} // namespace ns3

#endif /* MPI_RESEARCH_HELPER_H */