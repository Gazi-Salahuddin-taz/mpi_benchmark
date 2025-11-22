#ifndef STATISTICAL_ANALYZER_H
#define STATISTICAL_ANALYZER_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace TopologyAwareResearch {

class StatisticalAnalyzer {
public:
    StatisticalAnalyzer();
    ~StatisticalAnalyzer();

    // Data management
    void add_sample(double value);
    void clear_samples();
    size_t get_sample_count() const;

    // Descriptive statistics
    double calculate_mean() const;
    double calculate_median() const;
    double calculate_standard_deviation() const;
    double calculate_variance() const;
    double calculate_min() const;
    double calculate_max() const;
    double calculate_range() const;

    // Confidence intervals
    double calculate_confidence_interval() const;
    //double calculate_confidence_interval(double confidence_level = 0.95) const;

    // Statistical tests
    bool is_outlier(double value, double threshold = 2.0) const;
    double calculate_skewness() const;
    double calculate_kurtosis() const;
    std::vector<double> detect_outliers(double threshold = 2.0) const;

    // Data access
    const std::vector<double>& get_samples() const;
    std::vector<double> get_sorted_samples() const;

private:
    std::vector<double> samples_;

    double calculate_t_value(double confidence_level, int degrees_freedom) const;
};

} // namespace TopologyAwareResearch

#endif