#ifndef MPI_RESEARCH_APPLICATION_H
#define MPI_RESEARCH_APPLICATION_H

#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/ipv4-address.h"
#include "ns3/socket.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/nstime.h"
#include "ns3/random-variable-stream.h"
#include "ns3/vector.h"
#include "ns3/node-container.h"
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <chrono>

namespace ns3 {

    // MPI Collective Operation Types
    enum MpiCollectiveType {
        MPI_BROADCAST,
        MPI_ALLREDUCE,
        MPI_REDUCE,
        MPI_ALLGATHER,
        MPI_BARRIER,
        MPI_SCATTER,
        MPI_GATHER
    };

    // MPI Operation Status
    enum MpiOperationStatus {
        MPI_OP_PENDING,
        MPI_OP_COMPLETED,
        MPI_OP_FAILED
    };

    // Network Topology Types
    enum NetworkTopology {
        FAT_TREE,
        TORUS_2D,
        TORUS_3D,
        DRAGONFLY,
        CLOS,
        UNKNOWN
    };

    // Performance Metrics for NS-3 Simulation
    struct MpiPerformanceMetrics {
        Time executionTime;
        Time communicationTime;
        Time computationTime;
        double bandwidthUtilization;
        double networkEfficiency;
        uint64_t messagesSent;
        uint64_t dataVolume;
        double loadImbalance;
        std::vector<Time> messageLatencies;
        std::vector<double> linkUtilizations;

        MpiPerformanceMetrics()
            : executionTime(0), communicationTime(0), computationTime(0),
            bandwidthUtilization(0.0), networkEfficiency(0.0),
            messagesSent(0), dataVolume(0), loadImbalance(0.0) {
        }
    };

    // MPI Collective Operation Request
    struct MpiCollectiveRequest {
        uint32_t operationId;
        MpiCollectiveType operationType;
        uint32_t rootRank;
        uint32_t dataSize;
        Time startTime;
        Time completionTime;
        MpiOperationStatus status;
        std::vector<uint32_t> participants;
        MpiPerformanceMetrics metrics;

        MpiCollectiveRequest()
            : operationId(0), operationType(MPI_BROADCAST), rootRank(0),
            dataSize(0), status(MPI_OP_PENDING) {
        }
    };

    // Network Node Information
    struct NetworkNodeInfo {
        uint32_t nodeId;
        uint32_t rank;
        Ipv4Address address;
        Vector position;
        std::string hostname;
        std::vector<uint32_t> neighbors;
        double processingPower; // MIPS
        double memoryCapacity;  // MB

        NetworkNodeInfo()
            : nodeId(0), rank(0), processingPower(1000.0), memoryCapacity(8192.0) {
        }
    };

    // MPI Research Application for NS-3
    class MpiResearchApplication : public Application {
    public:
        static TypeId GetTypeId(void);
        MpiResearchApplication();
        virtual ~MpiResearchApplication();

        // Application interface
        void StartApplication(void) override;
        void StopApplication(void) override;

        // MPI Collective Operations Simulation
        void SimulateBroadcast(uint32_t rootRank, uint32_t dataSize);
        void SimulateAllreduce(uint32_t dataSize);
        void SimulateReduce(uint32_t rootRank, uint32_t dataSize);
        void SimulateAllgather(uint32_t dataSize);
        void SimulateBarrier();

        // Advanced collective operations
        void SimulateTopologyAwareBroadcast(uint32_t rootRank, uint32_t dataSize);
        void SimulateHierarchicalAllreduce(uint32_t dataSize);
        void SimulatePipelineBroadcast(uint32_t rootRank, uint32_t dataSize);

        // Performance analysis
        MpiPerformanceMetrics GetLastOperationMetrics() const;
        std::vector<MpiPerformanceMetrics> GetOperationHistory() const;
        void ResetPerformanceMetrics();

        // Configuration
        void SetRank(uint32_t rank);
        void SetWorldSize(uint32_t size);
        void SetNetworkTopology(NetworkTopology topology);
        void SetComputationDelay(Time delay);
        void EnableDetailedLogging(bool enable);

        // Network configuration
        void SetNodeInformation(const NetworkNodeInfo& info);
        void AddNeighbor(uint32_t neighborRank, Ipv4Address address);

    private:
        void DoInitialize(void) override;

        // Socket handling
        void CreateSocket();
        void SendPacket(Ptr<Packet> packet, Ipv4Address destination, uint32_t protocolId);
        void ReceivePacket(Ptr<Socket> socket);

        // MPI operation implementations
        void ExecuteBinomialTreeBroadcast(uint32_t rootRank, uint32_t dataSize);
        void ExecutePipelineRingBroadcast(uint32_t rootRank, uint32_t dataSize);
        void ExecuteTopologyAwareBroadcast(uint32_t rootRank, uint32_t dataSize);
        void ExecuteRingAllreduce(uint32_t dataSize);
        void ExecuteTreeAllreduce(uint32_t dataSize);

        // Communication primitives
        void SendDataToRank(uint32_t destRank, uint32_t dataSize, uint32_t operationId);
        void ReceiveDataFromRank(uint32_t srcRank, uint32_t operationId);

        // Performance measurement
        void StartOperationTimer(uint32_t operationId);
        void StopOperationTimer(uint32_t operationId);
        void RecordMessageLatency(Time latency);
        void CalculateBandwidthUtilization();

        // Utility methods
        uint32_t GetRankFromAddress(Ipv4Address address) const;
        Ipv4Address GetAddressFromRank(uint32_t rank) const;
        bool IsLocalNode(uint32_t rank) const;
        double CalculateCommunicationCost(uint32_t srcRank, uint32_t destRank) const;

        // Event scheduling
        void ScheduleComputation(uint32_t operationId, Time duration);
        void ScheduleCommunication(uint32_t operationId, uint32_t destRank, Time duration);

        // Protocol handlers
        void HandleBroadcastProtocol(Ptr<Packet> packet, Ipv4Address source);
        void HandleAllreduceProtocol(Ptr<Packet> packet, Ipv4Address source);
        void HandleControlProtocol(Ptr<Packet> packet, Ipv4Address source);

        // Member variables
        uint32_t m_rank;
        uint32_t m_worldSize;
        NetworkTopology m_topology;
        NetworkNodeInfo m_nodeInfo;

        Ptr<Socket> m_socket;
        uint16_t m_port;

        Time m_computationDelay;
        Time m_communicationDelay;
        bool m_detailedLogging;

        std::map<uint32_t, Ipv4Address> m_rankToAddress;
        std::map<Ipv4Address, uint32_t> m_addressToRank;

        std::map<uint32_t, MpiCollectiveRequest> m_activeOperations;
        std::vector<MpiPerformanceMetrics> m_operationHistory;

        Ptr<ExponentialRandomVariable> m_delayVariable;
        Ptr<UniformRandomVariable> m_dataSizeVariable;

        uint32_t m_nextOperationId;
        Time m_simulationStartTime;

        // Statistics
        uint64_t m_totalMessagesSent;
        uint64_t m_totalDataSent;
        double m_totalCommunicationTime;
        std::vector<Time> m_allMessageLatencies;
    };

} // namespace ns3

#endif /* MPI_RESEARCH_APPLICATION_H */