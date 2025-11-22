#ifndef PERFORMANCE_MEASUREMENT_H
#define PERFORMANCE_MEASUREMENT_H

#include <mpi.h>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <functional>
#include "../core/collective_optimizer.h"

namespace TopologyAwareResearch {

    class PerformanceTimer {
    private:
        std::chrono::high_resolution_clock::time_point start_time_;
        std::map<std::string, double> timings_;
        std::string current_section_;

    public:
        PerformanceTimer();
        ~PerformanceTimer();

        void start(const std::string& section_name);
        void stop();
        void reset();

        double get_elapsed_time(const std::string& section_name) const;
        std::map<std::string, double> get_all_timings() const;
        void print_timings() const;

        // Advanced timing utilities
        static double get_current_time();
        static void busy_wait_us(int microseconds);
    };

    class BandwidthMeasurer {
    private:
        MPI_Comm comm_;
        int iterations_;
        int warmup_iterations_;

    public:
        BandwidthMeasurer(MPI_Comm comm);
        ~BandwidthMeasurer();

        double measure_point_to_point_bandwidth(int src_rank, int dst_rank,
            int message_size, int iterations = 10);
        std::vector<std::vector<double>> measure_all_to_all_bandwidth(int message_size);
        double measure_collective_bandwidth(int collective_type, int message_size,
            int root = 0, int iterations = 10);

        void set_iterations(int iterations) { iterations_ = iterations; }
        void set_warmup_iterations(int warmup) { warmup_iterations_ = warmup; }

    private:
        void warmup(int src_rank, int dst_rank, int message_size);
    };

    class LatencyMeasurer {
    private:
        MPI_Comm comm_;
        int iterations_;
        int warmup_iterations_;

    public:
        LatencyMeasurer(MPI_Comm comm);
        ~LatencyMeasurer();

        double measure_point_to_point_latency(int src_rank, int dst_rank,
            int iterations = 1000);
        std::vector<std::vector<double>> measure_all_to_all_latency();
        double measure_collective_latency(int collective_type, int message_size = 0,
            int root = 0, int iterations = 100);

        void set_iterations(int iterations) { iterations_ = iterations; }

    private:
        double ping_pong_latency(int rank1, int rank2, int iterations);
    };

    class EnergyMonitor {
    private:
        bool energy_available_;
        double power_consumption_;
        std::chrono::high_resolution_clock::time_point start_time_;

    public:
        EnergyMonitor();
        ~EnergyMonitor();

        void start_measurement();
        double stop_measurement();
        double get_current_power() const;
        double estimate_energy_consumption(double execution_time, int processes) const;

        // Advanced energy estimation models
        double estimate_communication_energy(double communication_time, int data_volume) const;
        double estimate_computation_energy(double computation_time) const;
        double estimate_idle_energy(double idle_time) const;

    private:
        bool initialize_energy_measurement();
        double read_power_consumption();
    };

    class StatisticalAnalyzer {
    private:
        std::vector<double> samples_;
        double confidence_level_;

    public:
        StatisticalAnalyzer();
        ~StatisticalAnalyzer();

        void add_sample(double value);
        void clear_samples();

        // Statistical calculations
        double calculate_mean() const;
        double calculate_median() const;
        double calculate_standard_deviation() const;
        double calculate_variance() const;
        double calculate_confidence_interval() const;
        double calculate_min() const;
        double calculate_max() const;

        // Distribution analysis
        bool is_normal_distribution(double significance = 0.05) const;
        std::vector<double> get_quartiles() const;
        std::vector<std::pair<double, int>> get_histogram(int bins = 10) const;

        // Outlier detection
        std::vector<double> detect_outliers(double threshold = 1.5) const;
        bool remove_outliers(double threshold = 1.5);

        void set_confidence_level(double level) { confidence_level_ = level; }

    private:
        double calculate_z_score(double value) const;
        double calculate_interquartile_range() const;
    };

    class PerformanceProfiler {
    private:
        MPI_Comm comm_;
        int world_rank_;
        int world_size_;
        std::map<std::string, double> start_times_;
        std::map<std::string, PerformanceMetrics> profile_data_;
        bool detailed_profiling_;

    public:
        PerformanceProfiler(MPI_Comm comm);
        ~PerformanceProfiler();

        void start_profiling(const std::string& operation_name);
        void stop_profiling(const std::string& operation_name);
        void record_metrics(const std::string& operation_name, const PerformanceMetrics& metrics);

        // Analysis and reporting
        void generate_profile_report(const std::string& filename) const;
        void compare_operations() const;
        PerformanceMetrics get_operation_metrics(const std::string& operation_name) const;

        // Advanced profiling
        void profile_collective_operation(const std::string& operation_name,
            std::function<void()> operation,
            int iterations = 10);
        void profile_point_to_point(int src, int dst, int message_size, int iterations = 10);

        void set_detailed_profiling(bool detailed) { detailed_profiling_ = detailed; }

    private:
        void gather_profile_data();
        PerformanceMetrics calculate_statistics(const std::vector<PerformanceMetrics>& samples) const;
    };

} // namespace TopologyAwareResearch

#endif