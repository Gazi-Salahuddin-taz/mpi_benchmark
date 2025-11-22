#include "statistical_analyzer.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace TopologyAwareResearch {

StatisticalAnalyzer::StatisticalAnalyzer() {}

StatisticalAnalyzer::~StatisticalAnalyzer() {}

void StatisticalAnalyzer::add_sample(double value) {
    samples_.push_back(value);
}

void StatisticalAnalyzer::clear_samples() {
    samples_.clear();
}

size_t StatisticalAnalyzer::get_sample_count() const {
    return samples_.size();
}

double StatisticalAnalyzer::calculate_mean() const {
    if (samples_.empty()) return 0.0;

    double sum = std::accumulate(samples_.begin(), samples_.end(), 0.0);
    return sum / static_cast<double>(samples_.size());
}

double StatisticalAnalyzer::calculate_median() const {
    if (samples_.empty()) return 0.0;

    std::vector<double> sorted_samples = samples_;
    std::sort(sorted_samples.begin(), sorted_samples.end());

    size_t n = sorted_samples.size();
    if (n % 2 == 0) {
        return (sorted_samples[n/2 - 1] + sorted_samples[n/2]) / 2.0;
    } else {
        return sorted_samples[n/2];
    }
}

double StatisticalAnalyzer::calculate_variance() const {
    if (samples_.size() <= 1) return 0.0;

    double mean = calculate_mean();
    double sum_sq = 0.0;

    for (double sample : samples_) {
        double diff = sample - mean;
        sum_sq += diff * diff;
    }

    return sum_sq / (samples_.size() - 1); // Sample variance
}

double StatisticalAnalyzer::calculate_standard_deviation() const {
    return std::sqrt(calculate_variance());
}

double StatisticalAnalyzer::calculate_min() const {
    if (samples_.empty()) return 0.0;
    return *std::min_element(samples_.begin(), samples_.end());
}

double StatisticalAnalyzer::calculate_max() const {
    if (samples_.empty()) return 0.0;
    return *std::max_element(samples_.begin(), samples_.end());
}

double StatisticalAnalyzer::calculate_range() const {
    return calculate_max() - calculate_min();
}

double StatisticalAnalyzer::calculate_confidence_interval() const {
    if (samples_.size() <= 1) return 0.0;

    double standard_error = calculate_standard_deviation() / std::sqrt(static_cast<double>(samples_.size()));
    double t_value = calculate_t_value(0.95, samples_.size() - 1);

    return t_value * standard_error;
}

bool StatisticalAnalyzer::is_outlier(double value, double threshold) const {
    if (samples_.size() < 2) return false;

    double mean = calculate_mean();
    double std_dev = calculate_standard_deviation();

    if (std_dev == 0.0) return false;

    double z_score = std::abs((value - mean) / std_dev);
    return z_score > threshold;
}

double StatisticalAnalyzer::calculate_skewness() const {
    if (samples_.size() < 3) return 0.0;

    double mean = calculate_mean();
    double std_dev = calculate_standard_deviation();
    if (std_dev == 0.0) return 0.0;

    double sum_cubed_deviations = 0.0;
    for (double sample : samples_) {
        double deviation = sample - mean;
        sum_cubed_deviations += deviation * deviation * deviation;
    }

    double n = static_cast<double>(samples_.size());
    return (sum_cubed_deviations / n) / std::pow(std_dev, 3);
}

double StatisticalAnalyzer::calculate_kurtosis() const {
    if (samples_.size() < 4) return 0.0;

    double mean = calculate_mean();
    double std_dev = calculate_standard_deviation();
    if (std_dev == 0.0) return 0.0;

    double sum_fourth_deviations = 0.0;
    for (double sample : samples_) {
        double deviation = sample - mean;
        sum_fourth_deviations += deviation * deviation * deviation * deviation;
    }

    double n = static_cast<double>(samples_.size());
    return (sum_fourth_deviations / n) / std::pow(std_dev, 4) - 3.0; // Excess kurtosis
}

const std::vector<double>& StatisticalAnalyzer::get_samples() const {
    return samples_;
}

std::vector<double> StatisticalAnalyzer::get_sorted_samples() const {
    std::vector<double> sorted = samples_;
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}

std::vector<double> StatisticalAnalyzer::detect_outliers(double threshold) const {
    std::vector<double> outliers;

    if (samples_.size() < 4) return outliers;

    std::vector<double> sorted_samples = samples_;
    std::sort(sorted_samples.begin(), sorted_samples.end());

    size_t n = sorted_samples.size();

    // Calculate Q1 (25th percentile) and Q3 (75th percentile)
    double q1, q3;

    if (n % 2 == 0) {
        // Even number of elements
        q1 = sorted_samples[n/4];
        q3 = sorted_samples[3*n/4];
    } else {
        // Odd number of elements
        q1 = sorted_samples[n/4];
        q3 = sorted_samples[3*n/4];
    }

    double iqr = q3 - q1;

    double lower_bound = q1 - threshold * iqr;
    double upper_bound = q3 + threshold * iqr;

    for (double sample : samples_) {
        if (sample < lower_bound || sample > upper_bound) {
            outliers.push_back(sample);
        }
    }

    return outliers;
}

double StatisticalAnalyzer::calculate_t_value(double confidence_level, int degrees_freedom) const {
    // Simplified t-distribution values for common confidence levels
    if (degrees_freedom >= 30) {
        // For large samples, approximate with normal distribution
        if (confidence_level == 0.90) return 1.645;
        if (confidence_level == 0.95) return 1.960;
        if (confidence_level == 0.99) return 2.576;
    } else {
        // For small samples, use conservative t-values
        if (confidence_level == 0.90) return 1.697;
        if (confidence_level == 0.95) return 2.042;
        if (confidence_level == 0.99) return 2.750;
    }

    // Default to 95% confidence
    return 2.0;
}

} // namespace TopologyAwareResearch