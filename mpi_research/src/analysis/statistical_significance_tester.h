#ifndef STATISTICAL_SIGNIFICANCE_TESTER_H
#define STATISTICAL_SIGNIFICANCE_TESTER_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace TopologyAwareResearch {

class StatisticalSignificanceTester {
public:
    StatisticalSignificanceTester();
    ~StatisticalSignificanceTester();

    // Statistical tests
    double calculate_t_test(const std::vector<double>& sample1, const std::vector<double>& sample2);
    double calculate_mann_whitney_u(const std::vector<double>& sample1, const std::vector<double>& sample2);
    double calculate_wilcoxon_signed_rank(const std::vector<double>& sample1, const std::vector<double>& sample2);
    double calculate_anova(const std::vector<std::vector<double>>& samples);

    // Effect size calculations
    double calculate_cohens_d(const std::vector<double>& sample1, const std::vector<double>& sample2);
    double calculate_hedges_g(const std::vector<double>& sample1, const std::vector<double>& sample2);

    // Confidence intervals
    std::pair<double, double> calculate_mean_confidence_interval(const std::vector<double>& sample, double confidence_level = 0.95);

    // Statistical power
    double calculate_statistical_power(const std::vector<double>& sample1, const std::vector<double>& sample2, double alpha = 0.05);

    // Significance testing
    bool is_statistically_significant(const std::vector<double>& sample1, const std::vector<double>& sample2, double alpha = 0.05);

private:
    double calculate_mean(const std::vector<double>& sample) const;
    double calculate_variance(const std::vector<double>& sample, double mean) const;
    double calculate_standard_deviation(double variance) const;
    double calculate_standard_error(double std_dev, size_t sample_size) const;
};

} // namespace TopologyAwareResearch

#endif