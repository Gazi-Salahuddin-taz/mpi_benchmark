#ifndef MPI_RESEARCH_SCENARIO_H
#define MPI_RESEARCH_SCENARIO_H

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/csma-module.h"
#include "ns3/bridge-helper.h"
#include "mpi-research-application.h"
#include <vector>
#include <map>
#include <fstream>

namespace ns3 {

/**
 * Base class for MPI research scenarios
 */
class MpiResearchScenario : public Object {
public:
    enum TopologyType {
        FAT_TREE,
        TORUS_2D,
        TORUS_3D,
        DRAGONFLY,
        STAR,
        LINEAR,
        RING
    };

    MpiResearchScenario();
    virtual ~MpiResearchScenario();

    static TypeId GetTypeId(void);

    void RunSimulation(Time duration = Seconds(30.0));
    virtual void ConfigureTopology() = 0;
    virtual void InstallApplications();
    virtual void ScheduleMpiOperations();
    virtual void CollectResults();

    // Configuration methods
    void SetTopologyType(TopologyType topology);
    void SetWorldSize(uint32_t size);
    void EnableMpiLogging(bool enable);
    void SetComputationDelay(Time delay);
    void SetCommunicationDelay(Time delay);
    void SetDataSizeDistribution(uint32_t minSize, uint32_t maxSize);

protected:
    virtual void CreateTopology() = 0;
    virtual void SetupIpAddressing() = 0;
    virtual void SetupRouting() = 0;

    // Common utilities
    Ptr<Node> GetNodeSafe(NodeContainer& container, uint32_t index, const std::string& context);
    void BuildNeighborMappings();
    void WriteCommonResults();

    // Topology parameters
    TopologyType m_topologyType;
    uint32_t m_worldSize;
    NodeContainer m_computeNodes;
    NodeContainer m_allNodes;
    NetDeviceContainer m_allDevices;
    std::vector<NetDeviceContainer> m_links;

    // MPI applications
    std::vector<Ptr<MpiResearchApplication>> m_mpiApps;
    ApplicationContainer m_applications;

    // Configuration
    bool m_enableMpiLogging;
    Time m_computationDelay;
    Time m_communicationDelay;
    uint32_t m_minDataSize;
    uint32_t m_maxDataSize;
    std::string m_resultsDir;

    // Statistics
    Time m_simulationStart;
    Time m_simulationEnd;
};

} // namespace ns3

#endif // MPI_RESEARCH_SCENARIO_H