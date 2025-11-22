#include "mpi-research-application.h"
#include "ns3/log.h"
#include "ns3/ipv4-address.h"
#include "ns3/nstime.h"
#include "ns3/inet-socket-address.h"
#include "ns3/socket-factory.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/pointer.h"
#include "ns3/simulator.h"
#include "ns3/random-variable-stream.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace ns3 {

    NS_LOG_COMPONENT_DEFINE("MpiResearchApplication");
    NS_OBJECT_ENSURE_REGISTERED(MpiResearchApplication);

    TypeId MpiResearchApplication::GetTypeId(void) {
        static TypeId tid = TypeId("MpiResearchApplication")
            .SetParent<Application>()
            .SetGroupName("Applications")
            .AddConstructor<MpiResearchApplication>()
            .AddAttribute("Port", "Port number for MPI communication",
                UintegerValue(12345),
                MakeUintegerAccessor(&MpiResearchApplication::m_port),
                MakeUintegerChecker<uint16_t>())
            .AddAttribute("Rank", "MPI rank of this node",
                UintegerValue(0),
                MakeUintegerAccessor(&MpiResearchApplication::SetRank),
                MakeUintegerChecker<uint32_t>())
            .AddAttribute("WorldSize", "Total number of MPI processes",
                UintegerValue(1),
                MakeUintegerAccessor(&MpiResearchApplication::SetWorldSize),
                MakeUintegerChecker<uint32_t>())
            .AddAttribute("ComputationDelay", "Base computation delay per operation",
                TimeValue(MilliSeconds(1)),
                MakeTimeAccessor(&MpiResearchApplication::m_computationDelay),
                MakeTimeChecker())
            .AddAttribute("CommunicationDelay", "Base communication delay per message",
                TimeValue(MicroSeconds(100)),
                MakeTimeAccessor(&MpiResearchApplication::m_communicationDelay),
                MakeTimeChecker())
            .AddAttribute("EnableDetailedLogging", "Enable detailed operation logging",
                BooleanValue(false),
                MakeBooleanAccessor(&MpiResearchApplication::EnableDetailedLogging),
                MakeBooleanChecker())
            .AddAttribute("DelayVariable", "Random variable for delay modeling",
                PointerValue(),
                MakePointerAccessor(&MpiResearchApplication::m_delayVariable),
                MakePointerChecker<RandomVariableStream>())
            .AddAttribute("DataSizeVariable", "Random variable for data size modeling",
                PointerValue(),
                MakePointerAccessor(&MpiResearchApplication::m_dataSizeVariable),
                MakePointerChecker<RandomVariableStream>());
        return tid;
    }

    MpiResearchApplication::MpiResearchApplication()
        : m_rank(0),
        m_worldSize(1),
        m_topology(UNKNOWN),
        m_port(12345),
        m_computationDelay(MilliSeconds(1)),
        m_communicationDelay(MicroSeconds(100)),
        m_detailedLogging(false),
        m_nextOperationId(1),
        m_totalMessagesSent(0),
        m_totalDataSent(0),
        m_totalCommunicationTime(0) {

        NS_LOG_FUNCTION(this);

        // Initialize random variables
        m_delayVariable = CreateObject<ExponentialRandomVariable>();
        m_delayVariable->SetAttribute("Mean", DoubleValue(0.001)); // 1ms mean

        m_dataSizeVariable = CreateObject<UniformRandomVariable>();
        m_dataSizeVariable->SetAttribute("Min", DoubleValue(1024.0));  // 1KB
        m_dataSizeVariable->SetAttribute("Max", DoubleValue(1048576.0)); // 1MB
    }

    MpiResearchApplication::~MpiResearchApplication() {
        NS_LOG_FUNCTION(this);
    }

    void MpiResearchApplication::DoInitialize(void) {
        NS_LOG_FUNCTION(this);
        Application::DoInitialize();

        m_simulationStartTime = Simulator::Now();

        if (m_detailedLogging) {
            NS_LOG_INFO("MpiResearchApplication initialized - Rank: " << m_rank
                << ", World Size: " << m_worldSize);
        }
    }

    void MpiResearchApplication::StartApplication(void) {
        NS_LOG_FUNCTION(this);

        CreateSocket();

        if (m_detailedLogging) {
            NS_LOG_INFO("MpiResearchApplication started on rank " << m_rank);
        }
    }

    void MpiResearchApplication::StopApplication(void) {
        NS_LOG_FUNCTION(this);

        if (m_socket) {
            m_socket->Close();
            m_socket->SetRecvCallback(MakeNullCallback<void, Ptr<Socket>>());
        }

        // Log final statistics
        if (m_detailedLogging) {
            NS_LOG_INFO("MpiResearchApplication stopped - Rank: " << m_rank);
            NS_LOG_INFO("Total messages sent: " << m_totalMessagesSent);
            NS_LOG_INFO("Total data sent: " << m_totalDataSent << " bytes");
            NS_LOG_INFO("Total communication time: " << m_totalCommunicationTime.GetSeconds() << "s");
        }
    }

    void MpiResearchApplication::CreateSocket() {
        NS_LOG_FUNCTION(this);

        TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
        m_socket = Socket::CreateSocket(GetNode(), tid);

        InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_port);
        m_socket->Bind(local);
        m_socket->SetRecvCallback(MakeCallback(&MpiResearchApplication::ReceivePacket, this));

        if (m_detailedLogging) {
            NS_LOG_INFO("Socket created for MPI communication on port " << m_port);
        }
    }

    void MpiResearchApplication::ReceivePacket(Ptr<Socket> socket) {
        NS_LOG_FUNCTION(this << socket);

        Ptr<Packet> packet;
        Address from;

        while ((packet = socket->RecvFrom(from))) {
            InetSocketAddress address = InetSocketAddress::ConvertFrom(from);
            Ipv4Address sourceAddress = address.GetIpv4();
            uint16_t sourcePort = address.GetPort();

            if (m_detailedLogging) {
                NS_LOG_INFO("Received packet from " << sourceAddress << ":" << sourcePort
                    << " size: " << packet->GetSize() << " bytes");
            }

            // Record message latency (simplified)
            Time now = Simulator::Now();
            RecordMessageLatency(now - m_simulationStartTime);

            // Handle different protocol types based on packet content
            HandleBroadcastProtocol(packet, sourceAddress);
        }
    }

    void MpiResearchApplication::SendPacket(Ptr<Packet> packet, Ipv4Address destination, uint32_t protocolId) {
        NS_LOG_FUNCTION(this << destination << protocolId);

        if (!m_socket) {
            NS_LOG_WARN("Socket not initialized, cannot send packet");
            return;
        }

        InetSocketAddress remote = InetSocketAddress(destination, m_port);
        m_socket->Connect(remote);

        int bytesSent = m_socket->Send(packet);
        if (bytesSent > 0) {
            m_totalMessagesSent++;
            m_totalDataSent += bytesSent;

            if (m_detailedLogging) {
                NS_LOG_INFO("Sent " << bytesSent << " bytes to " << destination);
            }
        }
        else {
            NS_LOG_WARN("Failed to send packet to " << destination);
        }
    }

    void MpiResearchApplication::SimulateBroadcast(uint32_t rootRank, uint32_t dataSize) {
        NS_LOG_FUNCTION(this << rootRank << dataSize);

        uint32_t operationId = m_nextOperationId++;
        MpiCollectiveRequest request;
        request.operationId = operationId;
        request.operationType = MPI_BROADCAST;
        request.rootRank = rootRank;
        request.dataSize = dataSize;
        request.startTime = Simulator::Now();
        request.status = MPI_OP_PENDING;

        m_activeOperations[operationId] = request;

        if (m_detailedLogging) {
            NS_LOG_INFO("Starting broadcast operation " << operationId
                << " with root " << rootRank << ", data size: " << dataSize);
        }

        // Choose broadcast algorithm based on topology and data size
        if (m_topology == FAT_TREE && dataSize > 4096) {
            ExecuteTopologyAwareBroadcast(rootRank, dataSize);
        }
        else if (dataSize < 1024) {
            ExecuteBinomialTreeBroadcast(rootRank, dataSize);
        }
        else {
            ExecutePipelineRingBroadcast(rootRank, dataSize);
        }

        StartOperationTimer(operationId);
    }

    void MpiResearchApplication::ExecuteBinomialTreeBroadcast(uint32_t rootRank, uint32_t dataSize) {
        NS_LOG_FUNCTION(this << rootRank << dataSize);

        if (m_rank == rootRank) {
            // Root process: send to other processes in binomial tree pattern
            uint32_t mask = 1;
            while (mask < m_worldSize) {
                uint32_t destRank = (m_rank + mask) % m_worldSize;
                if (destRank < m_worldSize) {
                    ScheduleCommunication(m_nextOperationId, destRank,
                        m_communicationDelay * (dataSize / 1024));

                    if (m_detailedLogging) {
                        NS_LOG_INFO("Root " << m_rank << " sending to " << destRank
                            << " in binomial tree (mask: " << mask << ")");
                    }
                }
                mask <<= 1;
            }
        }
        else {
            // Non-root processes: wait to receive data
            uint32_t relativeRank = (m_rank - rootRank + m_worldSize) % m_worldSize;
            uint32_t mask = 1;

            while (mask < m_worldSize) {
                if (relativeRank & mask) {
                    uint32_t sourceRank = (m_rank - mask + m_worldSize) % m_worldSize;

                    if (m_detailedLogging) {
                        NS_LOG_INFO("Process " << m_rank << " waiting from " << sourceRank);
                    }
                    break;
                }
                mask <<= 1;
            }
        }
    }

    void MpiResearchApplication::ExecutePipelineRingBroadcast(uint32_t rootRank, uint32_t dataSize) {
        NS_LOG_FUNCTION(this << rootRank << dataSize);

        // Simplified pipeline ring implementation
        uint32_t nextRank = (m_rank + 1) % m_worldSize;
        uint32_t prevRank = (m_rank - 1 + m_worldSize) % m_worldSize;

        if (m_rank == rootRank) {
            // Start the pipeline
            ScheduleCommunication(m_nextOperationId, nextRank,
                m_communicationDelay * (dataSize / 1024));

            if (m_detailedLogging) {
                NS_LOG_INFO("Pipeline root " << m_rank << " starting broadcast to " << nextRank);
            }
        }
        else {
            // Receive from previous and send to next (if not last)
            if (m_rank != (rootRank - 1 + m_worldSize) % m_worldSize) {
                ScheduleCommunication(m_nextOperationId, nextRank,
                    m_communicationDelay * (dataSize / 1024));

                if (m_detailedLogging) {
                    NS_LOG_INFO("Process " << m_rank << " forwarding in pipeline to " << nextRank);
                }
            }
        }
    }

    void MpiResearchApplication::ExecuteTopologyAwareBroadcast(uint32_t rootRank, uint32_t dataSize) {
        NS_LOG_FUNCTION(this << rootRank << dataSize);

        // Topology-aware broadcast based on detected network topology
        switch (m_topology) {
        case FAT_TREE:
            // Fat-tree specific broadcast logic
            if (m_detailedLogging) {
                NS_LOG_INFO("Executing Fat-Tree optimized broadcast");
            }
            // Implementation would consider pod and core switches
            break;

        case TORUS_2D:
        case TORUS_3D:
            // Torus specific broadcast logic
            if (m_detailedLogging) {
                NS_LOG_INFO("Executing Torus optimized broadcast");
            }
            // Implementation would use dimension-ordered routing
            break;

        case DRAGONFLY:
            // Dragonfly specific broadcast logic
            if (m_detailedLogging) {
                NS_LOG_INFO("Executing Dragonfly optimized broadcast");
            }
            break;

        default:
            // Fallback to binomial tree
            ExecuteBinomialTreeBroadcast(rootRank, dataSize);
            break;
        }
    }

    void MpiResearchApplication::SimulateAllreduce(uint32_t dataSize) {
        NS_LOG_FUNCTION(this << dataSize);

        uint32_t operationId = m_nextOperationId++;
        MpiCollectiveRequest request;
        request.operationId = operationId;
        request.operationType = MPI_ALLREDUCE;
        request.dataSize = dataSize;
        request.startTime = Simulator::Now();
        request.status = MPI_OP_PENDING;

        m_activeOperations[operationId] = request;

        if (m_detailedLogging) {
            NS_LOG_INFO("Starting allreduce operation " << operationId
                << ", data size: " << dataSize);
        }

        // Choose allreduce algorithm based on data size
        if (dataSize < 8192) {
            ExecuteRingAllreduce(dataSize);
        }
        else {
            ExecuteTreeAllreduce(dataSize);
        }

        StartOperationTimer(operationId);
    }

    void MpiResearchApplication::ExecuteRingAllreduce(uint32_t dataSize) {
        NS_LOG_FUNCTION(this << dataSize);

        // Simplified ring allreduce implementation
        uint32_t nextRank = (m_rank + 1) % m_worldSize;
        uint32_t prevRank = (m_rank - 1 + m_worldSize) % m_worldSize;

        // Phase 1: Reduce-scatter
        for (uint32_t step = 0; step < m_worldSize - 1; ++step) {
            ScheduleCommunication(m_nextOperationId, nextRank,
                m_communicationDelay * (dataSize / m_worldSize / 1024));

            ScheduleComputation(m_nextOperationId, m_computationDelay * (dataSize / m_worldSize));
        }

        // Phase 2: Allgather
        for (uint32_t step = 0; step < m_worldSize - 1; ++step) {
            ScheduleCommunication(m_nextOperationId, nextRank,
                m_communicationDelay * (dataSize / m_worldSize / 1024));
        }

        if (m_detailedLogging) {
            NS_LOG_INFO("Process " << m_rank << " participating in ring allreduce");
        }
    }

    void MpiResearchApplication::ExecuteTreeAllreduce(uint32_t dataSize) {
        NS_LOG_FUNCTION(this << dataSize);

        // Tree-based allreduce implementation
        // This would use a binomial or k-ary tree for reduction

        if (m_detailedLogging) {
            NS_LOG_INFO("Process " << m_rank << " participating in tree allreduce");
        }

        // Implementation would involve:
        // 1. Local reduction within subtrees
        // 2. Global reduction along tree
        // 3. Result broadcast back to all processes
    }

    void MpiResearchApplication::SimulateReduce(uint32_t rootRank, uint32_t dataSize) {
        NS_LOG_FUNCTION(this << rootRank << dataSize);

        uint32_t operationId = m_nextOperationId++;
        MpiCollectiveRequest request;
        request.operationId = operationId;
        request.operationType = MPI_REDUCE;
        request.rootRank = rootRank;
        request.dataSize = dataSize;
        request.startTime = Simulator::Now();
        request.status = MPI_OP_PENDING;

        m_activeOperations[operationId] = request;

        if (m_detailedLogging) {
            NS_LOG_INFO("Starting reduce operation " << operationId
                << " with root " << rootRank << ", data size: " << dataSize);
        }

        StartOperationTimer(operationId);
    }

    void MpiResearchApplication::SimulateAllgather(uint32_t dataSize) {
        NS_LOG_FUNCTION(this << dataSize);

        uint32_t operationId = m_nextOperationId++;
        MpiCollectiveRequest request;
        request.operationId = operationId;
        request.operationType = MPI_ALLGATHER;
        request.dataSize = dataSize;
        request.startTime = Simulator::Now();
        request.status = MPI_OP_PENDING;

        m_activeOperations[operationId] = request;

        if (m_detailedLogging) {
            NS_LOG_INFO("Starting allgather operation " << operationId
                << ", data size: " << dataSize);
        }

        StartOperationTimer(operationId);
    }

    void MpiResearchApplication::SimulateBarrier() {
        NS_LOG_FUNCTION(this);

        uint32_t operationId = m_nextOperationId++;
        MpiCollectiveRequest request;
        request.operationId = operationId;
        request.operationType = MPI_BARRIER;
        request.startTime = Simulator::Now();
        request.status = MPI_OP_PENDING;

        m_activeOperations[operationId] = request;

        if (m_detailedLogging) {
            NS_LOG_INFO("Starting barrier operation " << operationId);
        }

        // Barrier implementation using tree or butterfly
        StartOperationTimer(operationId);
    }

    void MpiResearchApplication::SimulateTopologyAwareBroadcast(uint32_t rootRank, uint32_t dataSize) {
        NS_LOG_FUNCTION(this << rootRank << dataSize);
        ExecuteTopologyAwareBroadcast(rootRank, dataSize);
    }

    void MpiResearchApplication::SimulateHierarchicalAllreduce(uint32_t dataSize) {
        NS_LOG_FUNCTION(this << dataSize);

        // Hierarchical allreduce for multi-level networks
        if (m_detailedLogging) {
            NS_LOG_INFO("Starting hierarchical allreduce with data size: " << dataSize);
        }
    }

    void MpiResearchApplication::SimulatePipelineBroadcast(uint32_t rootRank, uint32_t dataSize) {
        NS_LOG_FUNCTION(this << rootRank << dataSize);
        ExecutePipelineRingBroadcast(rootRank, dataSize);
    }

    void MpiResearchApplication::ScheduleComputation(uint32_t operationId, Time duration) {
        NS_LOG_FUNCTION(this << operationId << duration);

        Simulator::Schedule(duration, &MpiResearchApplication::StopOperationTimer, this, operationId);

        if (m_detailedLogging) {
            NS_LOG_INFO("Scheduled computation for operation " << operationId
                << ", duration: " << duration.GetSeconds() << "s");
        }
    }

    void MpiResearchApplication::ScheduleCommunication(uint32_t operationId, uint32_t destRank, Time duration) {
        NS_LOG_FUNCTION(this << operationId << destRank << duration);

        // Simulate communication by scheduling a timer
        Simulator::Schedule(duration, &MpiResearchApplication::RecordCommunicationComplete,
            this, operationId, destRank);

        m_totalCommunicationTime += duration;

        if (m_detailedLogging) {
            NS_LOG_INFO("Scheduled communication for operation " << operationId
                << " to rank " << destRank << ", duration: " << duration.GetSeconds() << "s");
        }
    }

    void MpiResearchApplication::RecordCommunicationComplete(uint32_t operationId, uint32_t destRank) {
        NS_LOG_FUNCTION(this << operationId << destRank);

        if (m_detailedLogging) {
            NS_LOG_INFO("Communication completed for operation " << operationId
                << " to rank " << destRank);
        }
    }

    void MpiResearchApplication::StartOperationTimer(uint32_t operationId) {
        NS_LOG_FUNCTION(this << operationId);

        auto it = m_activeOperations.find(operationId);
        if (it != m_activeOperations.end()) {
            it->second.startTime = Simulator::Now();
        }
    }

    void MpiResearchApplication::StopOperationTimer(uint32_t operationId) {
        NS_LOG_FUNCTION(this << operationId);

        auto it = m_activeOperations.find(operationId);
        if (it != m_activeOperations.end()) {
            it->second.completionTime = Simulator::Now();
            it->second.status = MPI_OP_COMPLETED;

            // Calculate metrics
            it->second.metrics.executionTime = it->second.completionTime - it->second.startTime;
            it->second.metrics.dataVolume = it->second.dataSize;

            // Store in history
            m_operationHistory.push_back(it->second.metrics);

            if (m_detailedLogging) {
                NS_LOG_INFO("Operation " << operationId << " completed in "
                    << it->second.metrics.executionTime.GetSeconds() << "s");
            }

            // Remove from active operations
            m_activeOperations.erase(it);
        }
    }

    void MpiResearchApplication::RecordMessageLatency(Time latency) {
        m_allMessageLatencies.push_back(latency);
    }

    void MpiResearchApplication::HandleBroadcastProtocol(Ptr<Packet> packet, Ipv4Address source) {
        NS_LOG_FUNCTION(this << packet << source);

        // Process broadcast protocol messages
        // This would parse the packet and handle different message types
    }

    void MpiResearchApplication::HandleAllreduceProtocol(Ptr<Packet> packet, Ipv4Address source) {
        NS_LOG_FUNCTION(this << packet << source);

        // Process allreduce protocol messages
    }

    void MpiResearchApplication::HandleControlProtocol(Ptr<Packet> packet, Ipv4Address source) {
        NS_LOG_FUNCTION(this << packet << source);

        // Process control protocol messages (barrier, synchronization, etc.)
    }

    // Configuration methods
    void MpiResearchApplication::SetRank(uint32_t rank) {
        m_rank = rank;
    }

    void MpiResearchApplication::SetWorldSize(uint32_t size) {
        m_worldSize = size;
    }

    void MpiResearchApplication::SetNetworkTopology(NetworkTopology topology) {
        m_topology = topology;
    }

    void MpiResearchApplication::SetComputationDelay(Time delay) {
        m_computationDelay = delay;
    }

    void MpiResearchApplication::EnableDetailedLogging(bool enable) {
        m_detailedLogging = enable;
    }

    void MpiResearchApplication::SetNodeInformation(const NetworkNodeInfo& info) {
        m_nodeInfo = info;
    }

    void MpiResearchApplication::AddNeighbor(uint32_t neighborRank, Ipv4Address address) {
        m_rankToAddress[neighborRank] = address;
        m_addressToRank[address] = neighborRank;
        m_nodeInfo.neighbors.push_back(neighborRank);
    }

    // Performance analysis methods
    MpiPerformanceMetrics MpiResearchApplication::GetLastOperationMetrics() const {
        if (m_operationHistory.empty()) {
            return MpiPerformanceMetrics();
        }
        return m_operationHistory.back();
    }

    std::vector<MpiPerformanceMetrics> MpiResearchApplication::GetOperationHistory() const {
        return m_operationHistory;
    }

    void MpiResearchApplication::ResetPerformanceMetrics() {
        m_operationHistory.clear();
        m_totalMessagesSent = 0;
        m_totalDataSent = 0;
        m_totalCommunicationTime = 0;
        m_allMessageLatencies.clear();
    }

    // Utility methods
    uint32_t MpiResearchApplication::GetRankFromAddress(Ipv4Address address) const {
        auto it = m_addressToRank.find(address);
        if (it != m_addressToRank.end()) {
            return it->second;
        }
        return m_worldSize; // Invalid rank
    }

    Ipv4Address MpiResearchApplication::GetAddressFromRank(uint32_t rank) const {
        auto it = m_rankToAddress.find(rank);
        if (it != m_rankToAddress.end()) {
            return it->second;
        }
        return Ipv4Address::GetZero();
    }

    bool MpiResearchApplication::IsLocalNode(uint32_t rank) const {
        return (rank == m_rank);
    }

    double MpiResearchApplication::CalculateCommunicationCost(uint32_t srcRank, uint32_t destRank) const {
        // Simplified communication cost model
        // In practice, this would consider network topology, link bandwidth, etc.
        if (srcRank == destRank) return 0.0;

        // Base cost + distance factor
        double baseCost = 1.0;
        uint32_t distance = std::abs(static_cast<int>(srcRank - destRank));

        return baseCost + (distance * 0.1);
    }

} // namespace ns3