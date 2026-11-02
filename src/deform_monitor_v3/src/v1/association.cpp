#include "deform_monitor_v3/v1/association.hpp"

#include "deform_monitor_v3/v1/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace deform_monitor_v3 {
namespace v1 {

AssociationResult AssociationModel::Evaluate(
    std::vector<AssociationFeature> candidates,
    bool expected_visible) const {
  AssociationResult result;
  // Association is a probability-bearing public boundary.  Non-finite
  // features, empty identities, or duplicate identities must fail closed;
  // allowing any of them into sorting/softmax would make the selected tail
  // platform-dependent or count one physical alternative more than once.
  for (const AssociationFeature& candidate : candidates) {
    if (candidate.candidate_id.empty() ||
        !std::isfinite(candidate.distance_lower_bound) ||
        !std::isfinite(candidate.continuous_projection) ||
        !std::isfinite(candidate.projector_chordal_distance) ||
        !std::isfinite(candidate.log_support_count_ratio) ||
        !std::isfinite(candidate.overlap) ||
        !std::isfinite(candidate.independent_visibility) ||
        !std::isfinite(candidate.prediction_continuity)) {
      return result;
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const AssociationFeature& lhs,
                                                     const AssociationFeature& rhs) {
    return lhs.candidate_id < rhs.candidate_id;
  });
  if (std::adjacent_find(
          candidates.begin(), candidates.end(),
          [](const AssociationFeature& lhs, const AssociationFeature& rhs) {
            return lhs.candidate_id == rhs.candidate_id;
          }) != candidates.end()) {
    return result;
  }
  double tail_log_score = -std::numeric_limits<double>::infinity();
  if (candidates.size() > static_cast<size_t>(config_.resource.association_max_candidates)) {
    std::stable_sort(candidates.begin(), candidates.end(), [&](const AssociationFeature& lhs,
                                                               const AssociationFeature& rhs) {
      Eigen::Matrix<double, 7, 1> lf, rf;
      lf << lhs.distance_lower_bound, lhs.continuous_projection,
          lhs.projector_chordal_distance, lhs.log_support_count_ratio,
          lhs.overlap, lhs.independent_visibility, lhs.prediction_continuity;
      rf << rhs.distance_lower_bound, rhs.continuous_projection,
          rhs.projector_chordal_distance, rhs.log_support_count_ratio,
          rhs.overlap, rhs.independent_visibility, rhs.prediction_continuity;
      const double ls = config_.calibration.association_candidate_bias +
                        config_.calibration.association_weights.dot(lf);
      const double rs = config_.calibration.association_candidate_bias +
                        config_.calibration.association_weights.dot(rf);
      if (std::abs(ls - rs) > config_.numerical.tie_tolerance) return ls > rs;
      return lhs.candidate_id < rhs.candidate_id;
    });
    std::vector<double> tail_scores;
    for (size_t i = static_cast<size_t>(config_.resource.association_max_candidates);
         i < candidates.size(); ++i) {
      Eigen::Matrix<double, 7, 1> feature;
      feature << candidates[i].distance_lower_bound,
          candidates[i].continuous_projection,
          candidates[i].projector_chordal_distance,
          candidates[i].log_support_count_ratio,
          candidates[i].overlap, candidates[i].independent_visibility,
          candidates[i].prediction_continuity;
      tail_scores.push_back(config_.calibration.association_candidate_bias +
                            config_.calibration.association_weights.dot(feature));
    }
    tail_log_score = LogSumExp(tail_scores);
    candidates.resize(static_cast<size_t>(config_.resource.association_max_candidates));
    std::sort(candidates.begin(), candidates.end(), [](const AssociationFeature& lhs,
                                                       const AssociationFeature& rhs) {
      return lhs.candidate_id < rhs.candidate_id;
    });
  }

  std::vector<double> scores;
  for (const AssociationFeature& candidate : candidates) {
    Eigen::Matrix<double, 7, 1> feature;
    feature << candidate.distance_lower_bound, candidate.continuous_projection,
        candidate.projector_chordal_distance, candidate.log_support_count_ratio,
        candidate.overlap, candidate.independent_visibility,
        candidate.prediction_continuity;
    AssociationCategory category;
    category.candidate_id = candidate.candidate_id;
    category.geometric_candidate = true;
    category.semantics = AssociationSemantics::OTHER;
    category.log_score = config_.calibration.association_candidate_bias +
                         config_.calibration.association_weights.dot(feature);
    result.categories.push_back(category);
    scores.push_back(category.log_score);
  }
  AssociationCategory missing;
  missing.semantics = AssociationSemantics::SOURCE_MISSING;
  missing.log_score = expected_visible
                          ? config_.calibration.association_missing_bias
                          : -std::numeric_limits<double>::infinity();
  result.categories.push_back(missing);
  scores.push_back(missing.log_score);
  AssociationCategory other;
  other.semantics = AssociationSemantics::OTHER;
  // Candidates beyond the deterministic cap are unresolved alternatives.
  // Their full softmax numerator is transferred to OTHER rather than being
  // silently dropped and renormalized into retained candidates.
  other.log_score = LogSumExp(
      {config_.calibration.association_other_bias, tail_log_score});
  result.categories.push_back(other);
  scores.push_back(other.log_score);

  const std::vector<double> probabilities = Softmax(scores);
  if (probabilities.size() != result.categories.size()) return result;
  result.entropy = 0.0;
  for (size_t i = 0; i < probabilities.size(); ++i) {
    result.categories[i].probability = probabilities[i];
    if (probabilities[i] > 0.0) result.entropy -= probabilities[i] * std::log(probabilities[i]);
    if (!result.categories[i].geometric_candidate &&
        result.categories[i].semantics == AssociationSemantics::OTHER)
      result.other_mass = probabilities[i];
  }
  result.valid = !probabilities.empty() && std::abs(
      std::accumulate(probabilities.begin(), probabilities.end(), 0.0) - 1.0) <=
      config_.numerical.normalization_tolerance;
  return result;
}

}  // namespace v1
}  // namespace deform_monitor_v3
