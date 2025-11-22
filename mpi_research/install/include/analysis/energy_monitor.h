#ifndef ENERGY_MONITOR_H
#define ENERGY_MONITOR_H

#include <mpi.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

namespace TopologyAwareResearch {

class EnergyMonitor {
public:
    EnergyMonitor();
    ~EnergyMonitor();

    // Energy measurement methods
    void start_measurement();
    double stop_measurement();

    // Power monitoring
    double get_current_power() const;
    double get_total_energy() const;

    // System information
    std::string get_energy_source() const;
    bool is_energy_measurement_available() const;

    // Configuration
    void set_measurement_interval(double interval);
    void enable_power_capping(bool enable);

private:
    double measurement_start_time_;
    double total_energy_consumed_;
    double measurement_interval_;
    bool measurement_available_;

    // Platform-specific energy measurement
    double read_energy_consumption();
    bool initialize_energy_measurement();

    // File-based energy reading (for systems with energy sensors)
    std::string energy_file_path_;
    double read_energy_from_file(const std::string& path);

    // Fallback: estimate energy based on CPU time and model
    double estimate_energy_consumption(double execution_time);
};

} // namespace TopologyAwareResearch

#endif