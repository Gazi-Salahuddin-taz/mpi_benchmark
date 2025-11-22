#include "statistical_significance_tester.h"
#include <cmath>
#include <algorithm>

namespace TopologyAwareResearch {

StatisticalSignificanceTester::StatisticalSignificanceTester() {}

StatisticalSignificanceTester::~StatisticalSignificanceTester() {}

double StatisticalSignificanceTester::calculate_mean(const std::vector<double>& sample) const {
    if (sample.empty()) return 0.0;
    double sum = std::accumulate(sample.begin(), sample.end(), 0.0);
    return sum / sample.size();
}

double StatisticalSignificanceTester::calculate_variance(const std::vector<double>& sample, double mean) const {
    if (sample.size() <= 1) return 0.0;
    double sum_sq = 0.0;
    for (double value : sample) {
        double diff = value - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / (sample.size() - 1);
}

double StatisticalSignificanceTester::calculate_standard_deviation(double variance) const {
    return std::sqrt(variance);
}

double StatisticalSignificanceTester::calculate_standard_error(double std_dev, size_t sample_size) const {
    if (sample_size == 0) return 0.0;
    return std_dev / std::sqrt(sample_size);
}

double StatisticalSignificanceTester::calculate_t_test(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    double mean1 = calculate_mean(sample1);
    double mean2 = calculate_mean(sample2);

    double var1 = calculate_variance(sample1, mean1);
    double var2 = calculate_variance(sample2, mean2);

    int n1 = sample1.size();
    int n2 = sample2.size();

    // Pooled standard deviation
    double pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
    double standard_error = std::sqrt(pooled_variance * (1.0 / n1 + 1.0 / n2));

    if (standard_error == 0.0) return 0.0;

    return (mean1 - mean2) / standard_error;
}

double StatisticalSignificanceTester::calculate_mann_whitney_u(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    // Combine samples
    std::vector<double> combined = sample1;
    combined.insert(combined.end(), sample2.begin(), sample2.end());

    // Assign ranks
    std::vector<size_t> indices(combined.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        return combined[i] < combined[j];
    });

    // Calculate rank sums
    double rank_sum1 = 0.0;
    for (size_t i = 0; i < sample1.size(); ++i) {
        // Find rank of sample1[i] in combined sorted array
        auto it = std::find(indices.begin(), indices.end(), i);
        if (it != indices.end()) {
            rank_sum1 += std::distance(indices.begin(), it) + 1;
        }
    }

    double u1 = rank_sum1 - (sample1.size() * (sample1.size() + 1)) / 2.0;
    double u2 = sample1.size() * sample2.size() - u1;

    return std::min(u1, u2);
}

double StatisticalSignificanceTester::calculate_wilcoxon_signed_rank(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    if (sample1.size() != sample2.size()) return 0.0;

    std::vector<double> differences;
    for (size_t i = 0; i < sample1.size(); ++i) {
        differences.push_back(sample1[i] - sample2[i]);
    }

    // Remove zero differences
    differences.erase(std::remove_if(differences.begin(), differences.end(),
        [](double d) { return d == 0.0; }), differences.end());

    // Rank absolute differences
    std::vector<size_t> indices(differences.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        return std::abs(differences[i]) < std::abs(differences[j]);
    });

    double w_plus = 0.0, w_minus = 0.0;
    for (size_t i = 0; i < indices.size(); ++i) {
        double diff = differences[indices[i]];
        double rank = i + 1;

        if (diff > 0) {
            w_plus += rank;
        } else {
            w_minus += rank;
        }
    }

    return std::min(w_plus, w_minus);
}

double StatisticalSignificanceTester::calculate_anova(const std::vector<std::vector<double>>& samples) {
    if (samples.size() < 2) return 0.0;

    // Calculate overall mean
    std::vector<double> all_data;
    for (const auto& sample : samples) {
        all_data.insert(all_data.end(), sample.begin(), sample.end());
    }
    double overall_mean = calculate_mean(all_data);

    // Between-group sum of squares
    double ss_between = 0.0;
    for (const auto& sample : samples) {
        double group_mean = calculate_mean(sample);
        ss_between += sample.size() * (group_mean - overall_mean) * (group_mean - overall_mean);
    }

    // Within-group sum of squares
    double ss_within = 0.0;
    for (const auto& sample : samples) {
        double group_mean = calculate_mean(sample);
        for (double value : sample) {
            ss_within += (value - group_mean) * (value - group_mean);
        }
    }

    // F-statistic
    double df_between = samples.size() - 1;
    double df_within = all_data.size() - samples.size();

    if (df_within == 0.0 || ss_within == 0.0) return 0.0;

    double ms_between = ss_between / df_between;
    double ms_within = ss_within / df_within;

    return ms_between / ms_within;
}

double StatisticalSignificanceTester::calculate_cohens_d(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    double mean1 = calculate_mean(sample1);
    double mean2 = calculate_mean(sample2);

    double var1 = calculate_variance(sample1, mean1);
    double var2 = calculate_variance(sample2, mean2);

    int n1 = sample1.size();
    int n2 = sample2.size();

    double pooled_stddev = std::sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2));

    if (pooled_stddev == 0.0) return 0.0;

    return (mean1 - mean2) / pooled_stddev;
}

double StatisticalSignificanceTester::calculate_hedges_g(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    double cohens_d = calculate_cohens_d(sample1, sample2);

    int n1 = sample1.size();
    int n2 = sample2.size();

    // Hedges' g correction for small sample sizes
    double correction = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0));

    return cohens_d * correction;
}

std::pair<double, double> StatisticalSignificanceTester::calculate_mean_confidence_interval(const std::vector<double>& sample, double confidence_level) {
    if (sample.empty()) return {0.0, 0.0};

    double mean = calculate_mean(sample);
    double std_dev = calculate_standard_deviation(calculate_variance(sample, mean));
    double standard_error = calculate_standard_error(std_dev, sample.size());

    // Z-value for confidence level (simplified - for large samples)
    double z_value = 1.96; // 95% confidence

    if (confidence_level == 0.90) {
        z_value = 1.645;
    } else if (confidence_level == 0.99) {
        z_value = 2.576;
    }

    double margin_of_error = z_value * standard_error;

    return {mean - margin_of_error, mean + margin_of_error};
}

double StatisticalSignificanceTester::calculate_statistical_power(const std::vector<double>& sample1, const std::vector<double>& sample2, double alpha) {
    double cohens_d = calculate_cohens_d(sample1, sample2);
    int n1 = sample1.size();
    int n2 = sample2.size();

    // Simplified power calculation
    double effect_size = std::abs(cohens_d);
    double total_n = n1 + n2;

    // Approximate power using effect size and sample size
    double power = 1.0 - std::exp(-effect_size * std::sqrt(total_n) / 2.0);

    return std::min(power, 1.0);
}

bool StatisticalSignificanceTester::is_statistically_significant(const std::vector<double>& sample1, const std::vector<double>& sample2, double alpha) {
    if (sample1.size() < 2 || sample2.size() < 2) return false;

    // Use t-test for significance
    double t_statistic = calculate_t_test(sample1, sample2);
    int degrees_freedom = sample1.size() + sample2.size() - 2;

    // Simplified p-value calculation (for two-tailed test)
    double critical_value = 2.0; // Approximate for alpha=0.05

    if (alpha == 0.01) {
        critical_value = 2.576;
    } else if (alpha == 0.10) {
        critical_value = 1.645;
    }

    return std::abs(t_statistic) > critical_value;
}

} // namespace TopologyAwareResearch