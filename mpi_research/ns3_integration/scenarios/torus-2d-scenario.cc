#include "torus-2d-scenario.h"
#include <cmath>
#include <sstream>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("Torus2DScenario");
NS_OBJECT_ENSURE_REGISTERED(Torus2DScenario);

TypeId Torus2DScenario::GetTypeId(void) {
    static TypeId tid = TypeId("ns3::Torus2DScenario")
        .SetParent<MpiResearchScenario>()
        .AddConstructor<Torus2DScenario>()
        .AddAttribute("GridWidth", "Width of the torus grid",
                     UintegerValue(4),
                     MakeUintegerAccessor(&Torus2DScenario::m_gridWidth),
                     MakeUintegerChecker<uint32_t>())
        .AddAttribute("GridHeight", "Height of the torus grid", 
                     UintegerValue(4),
                     MakeUintegerAccessor(&Torus2DScenario::m_gridHeight),
                     MakeUintegerChecker<uint32_t>());
    return tid;
}

Torus2DScenario::Torus2DScenario(uint32_t width, uint32_t height)
    : m_gridWidth(width), m_gridHeight(height) {
    
    NS_LOG_FUNCTION(this << width << height);
    
    m_totalComputeNodes = width * height;
    SetWorldSize(m_totalComputeNodes);
    SetTopologyType(TORUS_2D);
}

void Torus2DScenario::CreateTopology() {
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Creating 2D Torus topology " << m_gridWidth << "x" << m_gridHeight);

    // Create compute nodes
    m_computeNodes.Create(m_totalComputeNodes);
    m_allNodes.Add(m_computeNodes);

    // Configure torus links
    PointToPointHelper p2pTorus;
    p2pTorus.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gbps")));
    p2pTorus.SetChannelAttribute("Delay", TimeValue(MicroSeconds(2)));
    p2pTorus.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("1000p"));

    // Connect nodes in torus pattern
    for (uint32_t y = 0; y < m_gridHeight; ++y) {
        for (uint32_t x = 0; x < m_gridWidth; ++x) {
            uint32_t currentIndex = y * m_gridWidth + x;
            
            // Connect to right neighbor (with wrap-around)
            uint32_t rightX = (x + 1) % m_gridWidth;
            uint32_t rightIndex = y * m_gridWidth + rightX;
            
            if (currentIndex < rightIndex) { // Avoid duplicate links
                NetDeviceContainer link = p2pTorus.Install(
                    m_computeNodes.Get(currentIndex), 
                    m_computeNodes.Get(rightIndex)
                );
                m_links.push_back(link);
            }

            // Connect to bottom neighbor (with wrap-around)
            uint32_t bottomY = (y + 1) % m_gridHeight;
            uint32_t bottomIndex = bottomY * m_gridWidth + x;
            
            if (currentIndex < bottomIndex) { // Avoid duplicate links
                NetDeviceContainer link = p2pTorus.Install(
                    m_computeNodes.Get(currentIndex), 
                    m_computeNodes.Get(bottomIndex)
                );
                m_links.push_back(link);
            }
        }
    }

    NS_LOG_INFO("2D Torus created with " << m_links.size() << " links");
}

void Torus2DScenario::SetupIpAddressing() {
    NS_LOG_FUNCTION(this);

    Ipv4AddressHelper address;
    address.SetBase("10.1.0.0", "255.255.0.0");

    for (uint32_t i = 0; i < m_links.size(); ++i) {
        Ipv4InterfaceContainer interfaces = address.Assign(m_links[i]);
        address.NewNetwork();
    }
}

void Torus2DScenario::SetupRouting() {
    NS_LOG_FUNCTION(this);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
}

void Torus2DScenario::CollectResults() {
    NS_LOG_FUNCTION(this);

    system(("mkdir -p " + m_resultsDir).c_str());

    std::ofstream topoFile(m_resultsDir + "/torus_2d_topology.txt");
    topoFile << "2D Torus Topology Analysis\n";
    topoFile << "==========================\n";
    topoFile << "Grid dimensions: " << m_gridWidth << " x " << m_gridHeight << "\n";
    topoFile << "Total nodes: " << m_totalComputeNodes << "\n";
    topoFile << "Total links: " << m_links.size() << "\n";
    topoFile << "Diameter: " << (m_gridWidth/2 + m_gridHeight/2) << "\n";
    topoFile.close();

    WriteCommonResults();
}

} // namespace ns3