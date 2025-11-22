#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "fat-tree-scenario.h"
#include "torus-2d-scenario.h"
#include "dragonfly-scenario.h"
#include "star-scenario.h"
#include <iostream>
#include <memory>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("MpiResearchRunner");

/**
 * Main runner for MPI research scenarios
 */
int main(int argc, char* argv[]) {
    // Default configuration
    std::string topology = "fat-tree";
    uint32_t size = 16;
    double duration = 30.0;
    bool enableLogging = true;
    std::string resultsDir = "results";
    
    // Fat tree specific
    uint32_t k = 4;
    
    // Torus specific
    uint32_t width = 4;
    uint32_t height = 4;
    
    // Dragonfly specific
    uint32_t groups = 4;
    uint32_t routersPerGroup = 4;
    uint32_t nodesPerRouter = 4;

    CommandLine cmd;
    cmd.AddValue("topology", "Network topology (fat-tree, torus-2d, dragonfly, star)", topology);
    cmd.AddValue("size", "Number of MPI processes", size);
    cmd.AddValue("duration", "Simulation duration in seconds", duration);
    cmd.AddValue("logging", "Enable MPI detailed logging", enableLogging);
    cmd.AddValue("results", "Results directory", resultsDir);
    
    // Fat tree parameters
    cmd.AddValue("k", "Fat tree k parameter", k);
    
    // Torus parameters
    cmd.AddValue("width", "Torus grid width", width);
    cmd.AddValue("height", "Torus grid height", height);
    
    // Dragonfly parameters
    cmd.AddValue("groups", "Dragonfly groups", groups);
    cmd.AddValue("routers", "Routers per group", routersPerGroup);
    cmd.AddValue("nodes", "Nodes per router", nodesPerRouter);
    
    cmd.Parse(argc, argv);

    // Configure logging
    LogComponentEnable("MpiResearchRunner", LOG_LEVEL_INFO);
    LogComponentEnable("MpiResearchApplication", enableLogging ? LOG_LEVEL_INFO : LOG_LEVEL_WARN);
    
    if (enableLogging) {
        LogComponentEnable("FatTreeScenario", LOG_LEVEL_INFO);
        LogComponentEnable("Torus2DScenario", LOG_LEVEL_INFO);
        LogComponentEnable("DragonflyScenario", LOG_LEVEL_INFO);
    }

    std::unique_ptr<MpiResearchScenario> scenario;

    try {
        NS_LOG_INFO("Creating " << topology << " scenario with " << size << " processes");
        
        if (topology == "fat-tree") {
            // Calculate k from size or use provided k
            if (size > 0) {
                // Find smallest k that can accommodate the size
                uint32_t calculated_k = 2;
                while (calculated_k * calculated_k * calculated_k / 4 < size && calculated_k <= 16) {
                    calculated_k += 2;
                }
                if (calculated_k <= 16) {
                    k = calculated_k;
                }
            }
            scenario = std::make_unique<FatTreeScenario>(k);
            
        } else if (topology == "torus-2d") {
            // Calculate grid dimensions from size
            if (size > 0) {
                width = static_cast<uint32_t>(std::sqrt(size));
                height = (size + width - 1) / width;
            }
            scenario = std::make_unique<Torus2DScenario>(width, height);
            
        } else if (topology == "dragonfly") {
            // Calculate dragonfly parameters from size
            if (size > 0) {
                groups = static_cast<uint32_t>(std::cbrt(size));
                routersPerGroup = groups;
                nodesPerRouter = (size + groups * routersPerGroup - 1) / (groups * routersPerGroup);
            }
            scenario = std::make_unique<DragonflyScenario>(groups, routersPerGroup, nodesPerRouter);
            
        } else if (topology == "star") {
            scenario = std::make_unique<StarScenario>(size);
            
        } else {
            NS_FATAL_ERROR("Unknown topology: " << topology);
        }

        // Configure scenario
        scenario->EnableMpiLogging(enableLogging);
        scenario->SetComputationDelay(MilliSeconds(2));
        scenario->SetCommunicationDelay(MicroSeconds(50));
        scenario->SetDataSizeDistribution(1024, 1048576); // 1KB to 1MB

        // Run simulation
        NS_LOG_INFO("Starting simulation with duration " << duration << " seconds");
        scenario->RunSimulation(Seconds(duration));
        
        NS_LOG_INFO("Simulation completed successfully");
        std::cout << "MPI Research Simulation Completed!" << std::endl;
        std::cout << "Topology: " << topology << std::endl;
        std::cout << "World Size: " << size << std::endl;
        std::cout << "Results saved to: " << resultsDir << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Simulation failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}