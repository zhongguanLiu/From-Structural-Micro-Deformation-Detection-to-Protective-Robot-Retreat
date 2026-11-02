#ifndef DEFORM_MONITOR_V3_RISK_MAP_RISK_MATH_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_RISK_MATH_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace deform_monitor_v3 {
namespace risk_map {

inline double ClampProbability(double p, double floor) {
  const double f = std::max(1.0e-15, std::min(0.1, floor));
  return std::max(f, std::min(1.0 - f, p));
}

inline double Sigmoid(double x) {
  if (x >= 0.0) {
    return 1.0 / (1.0 + std::exp(-x));
  }
  const double e = std::exp(x);
  return e / (1.0 + e);
}

inline double Logit(double p, double floor = 1.0e-9) {
  const double q = ClampProbability(p, floor);
  return std::log(q) - std::log1p(-q);
}

inline double LogSumExp(const std::vector<double>& terms) {
  if (terms.empty()) {
    return -std::numeric_limits<double>::infinity();
  }
  const double maximum = *std::max_element(terms.begin(), terms.end());
  if (!std::isfinite(maximum)) {
    return maximum;
  }
  double scaled_sum = 0.0;
  for (const double term : terms) {
    scaled_sum += std::exp(term - maximum);
  }
  return maximum + std::log(scaled_sum);
}

inline double LogPoissonPmf(int count, double rate) {
  if (count < 0 || !(rate > 0.0) || !std::isfinite(rate)) {
    return -std::numeric_limits<double>::infinity();
  }
  return count * std::log(rate) - rate - std::lgamma(count + 1.0);
}

inline double LogMixtureObservedCount(int count, double rate,
                                      double visibility_probability,
                                      double probability_floor) {
  const double pi = ClampProbability(visibility_probability, probability_floor);
  if (count > 0) {
    return std::log(pi) + LogPoissonPmf(count, rate);
  }
  return LogSumExp({std::log(pi) - rate, std::log1p(-pi)});
}

}  // namespace risk_map
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_RISK_MAP_RISK_MATH_HPP
