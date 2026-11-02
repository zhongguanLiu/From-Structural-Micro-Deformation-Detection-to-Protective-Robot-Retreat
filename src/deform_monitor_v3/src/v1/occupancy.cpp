#include "deform_monitor_v3/v1/occupancy.hpp"

#include "deform_monitor_v3/v1/deterministic.hpp"
#include "deform_monitor_v3/v1/math.hpp"

#include <boost/math/distributions/beta.hpp>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <set>

namespace deform_monitor_v3 {
namespace v1 {

std::vector<VoxelKey> OccupancyEventGraph::TraceRay(
    const Eigen::Vector3d& origin_R, const Eigen::Vector3d& endpoint_R,
    bool include_endpoint, bool* resource_exceeded) const {
  std::vector<VoxelKey> cells;
  if (resource_exceeded) *resource_exceeded = false;
  const double voxel = config_.resource.occupancy_voxel_m;
  if (!(voxel > 0.0) || !origin_R.allFinite() || !endpoint_R.allFinite()) return cells;
  VoxelKey current;
  VoxelKey target;
  if (!TryGridKey(origin_R, config_.resource.grid_origin_m, voxel, &current) ||
      !TryGridKey(endpoint_R, config_.resource.grid_origin_m, voxel, &target)) {
    if (resource_exceeded) *resource_exceeded = true;
    return cells;
  }
  cells.push_back(current);
  if (current == target) {
    if (!include_endpoint) cells.clear();
    return cells;
  }
  const Eigen::Vector3d direction = endpoint_R - origin_R;
  std::array<int, 3> step{{0, 0, 0}};
  Eigen::Vector3d t_max = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d t_delta = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  const int32_t current_values[3] = {current.x, current.y, current.z};
  for (int axis = 0; axis < 3; ++axis) {
    if (direction[axis] > 0.0) step[axis] = 1;
    else if (direction[axis] < 0.0) step[axis] = -1;
    if (step[axis] != 0) {
      const double boundary_index = static_cast<double>(current_values[axis] +
          (step[axis] > 0 ? 1 : 0));
      const double boundary = config_.resource.grid_origin_m[axis] + voxel * boundary_index;
      t_max[axis] = (boundary - origin_R[axis]) / direction[axis];
      t_delta[axis] = voxel / std::abs(direction[axis]);
    }
  }
  const uint64_t distance_x = static_cast<uint64_t>(std::llabs(
      static_cast<int64_t>(target.x) - static_cast<int64_t>(current.x)));
  const uint64_t distance_y = static_cast<uint64_t>(std::llabs(
      static_cast<int64_t>(target.y) - static_cast<int64_t>(current.y)));
  const uint64_t distance_z = static_cast<uint64_t>(std::llabs(
      static_cast<int64_t>(target.z) - static_cast<int64_t>(current.z)));
  const uint64_t required_upper = distance_x + distance_y + distance_z + 1U;
  if (required_upper > static_cast<uint64_t>(config_.resource.max_tiles)) {
    if (resource_exceeded) *resource_exceeded = true;
    cells.clear();
    return cells;
  }
  const size_t hard_limit = static_cast<size_t>(required_upper + 3U);
  for (size_t iteration = 0; iteration < hard_limit && !(current == target); ++iteration) {
    // A floating-point endpoint can lie numerically just below an exact voxel
    // boundary.  Once an axis has reached the canonical target cell it must
    // not cross that boundary again while another tied axis catches up.
    // Otherwise the DDA overshoots the target and falsely reports a resource
    // failure for an ordinary finite ray.
    if (current.x == target.x)
      t_max.x() = std::numeric_limits<double>::infinity();
    if (current.y == target.y)
      t_max.y() = std::numeric_limits<double>::infinity();
    if (current.z == target.z)
      t_max.z() = std::numeric_limits<double>::infinity();
    // At an edge/corner crossing all tied axes advance in the same DDA event.
    // Advancing them one at a time would fabricate intermediate cells touched
    // only at a zero-measure boundary and then use those cells as FREE
    // evidence.  The comparison is relative to the ray parameter so the
    // decision is deterministic across metric scales.
    const double next = t_max.minCoeff();
    const double tie = config_.numerical.tie_tolerance *
                       std::max(1.0, std::abs(next));
    if (t_max.x() <= next + tie) {
      current.x += step[0];
      t_max.x() += t_delta.x();
    }
    if (t_max.y() <= next + tie) {
      current.y += step[1];
      t_max.y() += t_delta.y();
    }
    if (t_max.z() <= next + tie) {
      current.z += step[2];
      t_max.z() += t_delta.z();
    }
    cells.push_back(current);
  }
  if (!(current == target)) {
    if (resource_exceeded) *resource_exceeded = true;
    cells.clear();
    return cells;
  }
  if (!include_endpoint && !cells.empty() && cells.back() == target) cells.pop_back();
  return cells;
}

bool OccupancyEventGraph::ReferenceLineOfSightVisible(
    const Eigen::Vector3d& origin_R,
    const Eigen::Vector3d& endpoint_R) const {
  if (reference_.empty()) return false;
  bool resource_exceeded = false;
  const std::vector<VoxelKey> ray = TraceRay(
      origin_R, endpoint_R, false, &resource_exceeded);
  if (resource_exceeded) return false;
  for (const VoxelKey& key : ray) {
    const auto blocker = reference_.find(key);
    if (blocker != reference_.end() &&
        blocker->second.knowledge == KnowledgeState::OCCUPIED)
      return false;
  }
  return true;
}

ProbabilityInterval OccupancyEventGraph::BetaPosterior(double successes,
                                                       double failures) const {
  const double temper = config_.calibration.reference_beta_tempering;
  const double a = 1.0 + temper * successes;
  const double b = 1.0 + temper * failures;
  ProbabilityInterval result;
  result.mean = a / (a + b);
  try {
    const boost::math::beta_distribution<double> distribution(a, b);
    result.lower = boost::math::quantile(distribution,
        config_.calibration.occupancy_interval_alpha / 2.0);
    result.upper = boost::math::quantile(distribution,
        1.0 - config_.calibration.occupancy_interval_alpha / 2.0);
  } catch (...) {
    result.lower = 0.0; result.upper = 1.0;
  }
  result.calibrated = config_.calibration.validated;
  result.mean_identified = true;
  return result;
}

std::pair<double, double> OccupancyEventGraph::EventRates(
    HazardSemantics semantics) const {
  if (semantics == HazardSemantics::NOVEL_OCCUPANCY)
    return {config_.calibration.novel_occupancy_birth_rate_hz,
            config_.calibration.novel_occupancy_death_rate_hz};
  if (semantics == HazardSemantics::REFERENCE_PERSISTENCE)
    return {config_.calibration.reference_persistence_birth_rate_hz,
            config_.calibration.reference_persistence_death_rate_hz};
  return {config_.calibration.unknown_occupancy_birth_rate_hz,
          config_.calibration.unknown_occupancy_death_rate_hz};
}

ProbabilityInterval OccupancyEventGraph::PredictEvent(
    const EventState& event, const ros::Time& stamp) const {
  ProbabilityInterval result = event.posterior;
  if (event.last_stamp.isZero()) return result;
  const double delta_t = (stamp - event.last_stamp).toSec();
  if (!(delta_t >= 0.0) || !std::isfinite(delta_t)) {
    result.lower = 0.0;
    result.upper = 1.0;
    result.calibrated = false;
    result.mean_identified = false;
    return result;
  }
  const auto rates = EventRates(event.semantics);
  const double total_rate = rates.first + rates.second;
  const double stationary = rates.first / total_rate;
  const double retention = std::exp(-total_rate * delta_t);
  const auto propagate = [&](double probability) {
    return stationary + (probability - stationary) * retention;
  };
  result.lower = propagate(event.posterior.lower);
  result.mean = propagate(event.posterior.mean);
  result.upper = propagate(event.posterior.upper);
  result.calibrated = event.posterior.calibrated &&
                      config_.calibration.validated;
  result.mean_identified = event.posterior.mean_identified;
  return result;
}

ProbabilityInterval OccupancyEventGraph::PredictCauseProbability(
    const VoxelKey& key, const StableId& cause_id,
    const ros::Time& arrival_stamp) const {
  const auto voxel_events = event_states_.find(key);
  if (voxel_events != event_states_.end()) {
    const auto event = voxel_events->second.find(cause_id);
    if (event != voxel_events->second.end())
      return PredictEvent(event->second, arrival_stamp);
  }
  const auto voxel_causes = total_causes_.find(key);
  if (voxel_causes != total_causes_.end()) {
    const auto cause = voxel_causes->second.find(cause_id);
    if (cause != voxel_causes->second.end()) {
      // Structural projections are de-duplicated before the occupancy risk
      // path. A cause without an EventState is therefore an explicit
      // protective/diagnostic carrier; retaining its published interval is
      // the only justified fail-safe propagation.
      return cause->second;
    }
  }
  ProbabilityInterval unavailable;
  unavailable.lower = 0.0;
  unavailable.mean = 0.0;
  unavailable.upper = 1.0;
  unavailable.calibrated = false;
  return unavailable;
}

ProbabilityInterval OccupancyEventGraph::UpdateEventPosterior(
    const ProbabilityInterval& prior, const CurrentCount& current) const {
  // E is the number of occupancy-role ray exposures in this non-overlapping
  // frame increment. Visibility pi comes from the disjoint association-role
  // coverage and therefore does not reuse the endpoint count k.
  const double exposure = static_cast<double>(current.traversed) +
                          static_cast<double>(current.endpoints);
  if (!(exposure > 0.0)) return prior;
  if (!(current.visibility > 0.0)) {
    ProbabilityInterval unresolved = prior;
    if (current.endpoints > 0U) {
      unresolved.lower = 0.0;
      unresolved.upper = 1.0;
      unresolved.calibrated = false;
      unresolved.mean_identified = false;
    }
    return unresolved;
  }
  const int hits = static_cast<int>(current.endpoints);
  const double log_h1 = LogVisibilityMixedNegativeBinomial(
      hits, exposure * config_.calibration.nb_occupied_mean_scale,
      config_.calibration.nb_dispersion, current.visibility,
      config_.numerical.probability_floor);
  const double log_h0 = LogVisibilityMixedNegativeBinomial(
      hits, exposure * config_.calibration.nb_clutter_mean,
      config_.calibration.nb_dispersion, current.visibility,
      config_.numerical.probability_floor);
  if ((!std::isfinite(log_h1) && !std::isfinite(log_h0)) ||
      std::isnan(log_h1 - log_h0)) {
    ProbabilityInterval unresolved = prior;
    unresolved.lower = 0.0;
    unresolved.upper = 1.0;
    unresolved.calibrated = false;
    unresolved.mean_identified = false;
    return unresolved;
  }
  const double log_bayes_factor = log_h1 - log_h0;
  const auto update_prior = [&](double prior_value) {
    const double prior = ClampProbability(
        prior_value, config_.numerical.probability_floor);
    const double log_odds = std::log(prior) - std::log1p(-prior) +
                            log_bayes_factor;
    return 1.0 / (1.0 + std::exp(
        -std::max(-700.0, std::min(700.0, log_odds))));
  };
  ProbabilityInterval result;
  result.mean = update_prior(prior.mean);
  result.lower = update_prior(prior.lower);
  result.upper = update_prior(prior.upper);
  result.calibrated = config_.calibration.validated &&
                      prior.calibrated;
  result.mean_identified = prior.mean_identified;
  if (!result.calibrated) {
    result.lower = 0.0;
    result.upper = 1.0;
  }
  return result;
}

ProbabilityInterval OccupancyEventGraph::AggregateCauses(
    const std::map<StableId, ProbabilityInterval>& causes) const {
  ProbabilityInterval result;
  result.upper = 0.0;
  result.calibrated = true;
  result.mean_identified = true;
  if (causes.empty()) return result;
  if (causes.size() == 1U) return causes.begin()->second;
  for (const auto& item : causes) {
    const ProbabilityInterval& probability = item.second;
    // Dependence between different causes is not invented. The maximum lower
    // endpoint and Boole upper endpoint are valid without independence.
    result.lower = std::max(result.lower, probability.lower);
    result.mean += probability.mean;
    result.upper += probability.upper;
    result.calibrated = result.calibrated && probability.calibrated;
  }
  result.upper = std::min(1.0, result.upper);
  // Distinct causes have no frozen copula.  Their union mean is not the sum
  // of marginal means; expose only the Frechet/Boole interval and place its
  // conservative endpoint in the legacy mean slot.
  result.mean = result.upper;
  result.mean_identified = false;
  return result;
}

void OccupancyEventGraph::RefreshOutputProbabilities() {
  for (OccupancyCell& cell : output_) {
    const auto total = total_causes_.find(cell.key);
    cell.total_occupancy = total == total_causes_.end()
        ? ProbabilityInterval{0.0, 0.0, 0.0,
                              config_.calibration.validated, true}
        : AggregateCauses(total->second);
    if (cell.new_occupancy_applicable) {
      const auto novel = new_causes_.find(cell.key);
      cell.new_occupancy = novel == new_causes_.end()
          ? ProbabilityInterval{0.0, 0.0, 0.0,
                                config_.calibration.validated, true}
          : AggregateCauses(novel->second);
    } else {
      // Explicit unavailability. The wire contract carries the applicability
      // bit, so [0,0,1] cannot be mistaken for a usable estimate.
      cell.new_occupancy = ProbabilityInterval();
      cell.new_occupancy.mean = 0.0;
      cell.new_occupancy.calibrated = false;
    }
    cell.cause_ids.clear();
    cell.cause_probabilities.clear();
    cell.cause_semantics.clear();
    cell.cause_source_ids.clear();
    if (total != total_causes_.end())
      for (const auto& cause : total->second)
        if (cause.second.upper > 0.0) {
          cell.cause_ids.push_back(cause.first);
          cell.cause_probabilities.push_back(cause.second);
          const auto voxel_semantics = cause_semantics_.find(cell.key);
          const auto semantics = voxel_semantics == cause_semantics_.end()
              ? std::map<StableId, HazardSemantics>::const_iterator()
              : voxel_semantics->second.find(cause.first);
          cell.cause_semantics.push_back(
              voxel_semantics != cause_semantics_.end() &&
                      semantics != voxel_semantics->second.end()
                  ? semantics->second
                  : HazardSemantics::OTHER_ENVELOPE);
          const auto voxel_events = event_states_.find(cell.key);
          const auto event = voxel_events == event_states_.end()
              ? std::map<StableId, EventState>::const_iterator()
              : voxel_events->second.find(cause.first);
          if (voxel_events != event_states_.end() &&
              event != voxel_events->second.end())
            cell.cause_source_ids.emplace_back(event->second.source_ids.begin(),
                                               event->second.source_ids.end());
          else
            cell.cause_source_ids.emplace_back();
        }
  }
}

StableId OccupancyEventGraph::CauseId(const VoxelKey& key) const {
  CanonicalBytes bytes;
  bytes.AddString(config_.resource.session_id); bytes.AddU32(reference_epoch_);
  bytes.AddI64(key.x); bytes.AddI64(key.y); bytes.AddI64(key.z);
  return MakeStableId("occupancy-event-v1", bytes);
}

StableId OccupancyEventGraph::GduCauseId(const StableId& gdu_id) const {
  CanonicalBytes fields;
  fields.AddString(config_.resource.session_id);
  fields.AddId(gdu_id);
  return MakeStableId("hazard-cause-v1", fields);
}

StableId OccupancyEventGraph::ReferenceCauseId(const VoxelKey& key) const {
  CanonicalBytes fields;
  fields.AddString(config_.resource.session_id);
  fields.AddU32(reference_epoch_);
  fields.AddI64(key.x);
  fields.AddI64(key.y);
  fields.AddI64(key.z);
  return MakeStableId("reference-occupancy-cause-v1", fields);
}

bool OccupancyEventGraph::BuildReference(const std::deque<FrameBundle>& frames,
                                         const ReferenceSnapshot& structural_reference,
                                         uint32_t reference_epoch,
                                         std::vector<ReasonCode>* reasons) {
  reference_.clear(); total_causes_.clear(); new_causes_.clear();
  cause_semantics_.clear(); event_states_.clear();
  output_.clear(); reference_epoch_ = reference_epoch;
  if (frames.size() < static_cast<size_t>(config_.resource.reference_min_frames)) {
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_INSUFFICIENT_EXPOSURE);
    return false;
  }
  const auto reference_resource_failure =
      [&](const char* stage, size_t value, size_t limit) {
        std::cerr << "[deform_v1] occupancy reference resource failure stage="
                  << stage << " value=" << value << " limit=" << limit
                  << std::endl;
        if (reasons)
          reasons->push_back(ReasonCode::REFERENCE_RESOURCE_LIMIT);
      };
  for (const FrameBundle& frame : frames) {
    std::map<VoxelKey, std::pair<uint32_t, uint32_t>> per_frame;
    for (const CanonicalPoint& point : frame.points) {
      // Reference occupancy must use the same disjoint OCCUPANCY support as
      // online occupancy.  Reusing pose/association/measurement points here
      // would violate the frozen evidence partition and inflate ray work by
      // roughly ten times in the configured 10% occupancy split.
      if (point.role != PointRole::OCCUPANCY) continue;
      bool resource_exceeded = false;
      const std::vector<VoxelKey> ray = TraceRay(
          frame.pose.origin_L_R, point.position_R, true,
          &resource_exceeded);
      if (resource_exceeded) {
        reference_resource_failure("ray_trace", 1U, 0U);
        reference_.clear();
        return false;
      }
      if (ray.empty()) continue;
      for (size_t i = 0; i + 1U < ray.size(); ++i) ++per_frame[ray[i]].first;
      ++per_frame[ray.back()].second;
      if (per_frame.size() > config_.resource.max_tiles) {
        reference_resource_failure("per_frame_tiles", per_frame.size(),
                                   config_.resource.max_tiles);
        reference_.clear();
        return false;
      }
    }
    for (const auto& count : per_frame) {
      ReferenceCount& reference = reference_[count.first];
      const double total = static_cast<double>(count.second.first) +
                           static_cast<double>(count.second.second);
      if (!(total > 0.0)) continue;
      // One frame contributes one tempered exposure group.  The within-frame
      // occupied/free ray ratio is a group statistic, not a collection of
      // independent Bernoulli trials.
      reference.occupied_groups +=
          static_cast<double>(count.second.second) / total;
      reference.free_groups +=
          static_cast<double>(count.second.first) / total;
      ++reference.exposures;
    }
    if (reference_.size() > config_.resource.max_tiles) {
      reference_resource_failure("reference_tiles", reference_.size(),
                                 config_.resource.max_tiles);
      reference_.clear();
      return false;
    }
  }
  for (auto& item : reference_) {
    ReferenceCount& reference = item.second;
    reference.occupancy = BetaPosterior(reference.occupied_groups,
                                        reference.free_groups);
    if (reference.exposures < static_cast<uint32_t>(config_.resource.reference_min_frames))
      reference.knowledge = KnowledgeState::UNKNOWN;
    else if (reference.occupancy.upper <= config_.calibration.reference_free_threshold)
      reference.knowledge = KnowledgeState::FREE;
    else if (reference.occupancy.lower >= config_.calibration.reference_occupied_threshold)
      reference.knowledge = KnowledgeState::OCCUPIED;
    else reference.knowledge = KnowledgeState::UNKNOWN;
  }
  // Determine structural ownership from the complete frozen surfel carrier,
  // not only from its centroid voxel.  A cell touched by two GDUs cannot be
  // assigned to either one, but it is still structurally represented: making
  // it an independent REFERENCE_PERSISTENCE event would count the same frozen
  // surface once as occupancy and again through both structural projections.
  std::map<VoxelKey, std::set<StableId>> structural_owners;
  for (const Surfel& surfel : structural_reference.surfels) {
    if (surfel.gdu_index >= structural_reference.gdus.size() ||
        !surfel.center_R.allFinite() ||
        !std::isfinite(surfel.support_radius_m) ||
        surfel.support_radius_m < 0.0) {
      reference_resource_failure("invalid_structural_surfel",
                                 surfel.gdu_index,
                                 structural_reference.gdus.size());
      reference_.clear();
      return false;
    }
    VoxelKey minimum;
    VoxelKey maximum;
    const Eigen::Vector3d extent = Eigen::Vector3d::Constant(
        surfel.support_radius_m);
    if (!TryGridKey(surfel.center_R - extent,
                    config_.resource.grid_origin_m,
                    config_.resource.occupancy_voxel_m, &minimum) ||
        !TryGridKey(surfel.center_R + extent,
                    config_.resource.grid_origin_m,
                    config_.resource.occupancy_voxel_m, &maximum)) {
      reference_resource_failure("structural_grid_domain", 1U, 0U);
      reference_.clear();
      return false;
    }
    const int64_t span_x = static_cast<int64_t>(maximum.x) - minimum.x + 1;
    const int64_t span_y = static_cast<int64_t>(maximum.y) - minimum.y + 1;
    const int64_t span_z = static_cast<int64_t>(maximum.z) - minimum.z + 1;
    const long double candidate_cells = static_cast<long double>(span_x) *
        static_cast<long double>(span_y) * static_cast<long double>(span_z);
    if (span_x <= 0 || span_y <= 0 || span_z <= 0 ||
        candidate_cells >
            static_cast<long double>(config_.resource.max_tiles)) {
      reference_resource_failure(
          "structural_surfel_span",
          candidate_cells > static_cast<long double>(
                                std::numeric_limits<size_t>::max())
              ? std::numeric_limits<size_t>::max()
              : static_cast<size_t>(candidate_cells),
          config_.resource.max_tiles);
      reference_.clear();
      return false;
    }
    const StableId cause = GduCauseId(
        structural_reference.gdus[surfel.gdu_index].id);
    const double half_extent = 0.5 * config_.resource.occupancy_voxel_m;
    for (int64_t x = minimum.x; x <= static_cast<int64_t>(maximum.x); ++x)
      for (int64_t y = minimum.y; y <= static_cast<int64_t>(maximum.y); ++y)
        for (int64_t z = minimum.z; z <= static_cast<int64_t>(maximum.z); ++z) {
          const VoxelKey key{static_cast<int32_t>(x), static_cast<int32_t>(y),
                             static_cast<int32_t>(z)};
          if (reference_.count(key) == 0U) continue;
          const Eigen::Vector3d center = config_.resource.grid_origin_m +
              config_.resource.occupancy_voxel_m *
                  (Eigen::Vector3d(static_cast<double>(x),
                                   static_cast<double>(y),
                                   static_cast<double>(z)) +
                   Eigen::Vector3d::Constant(0.5));
          const Eigen::Vector3d closest = surfel.center_R.cwiseMax(
              center - Eigen::Vector3d::Constant(half_extent)).cwiseMin(
              center + Eigen::Vector3d::Constant(half_extent));
          if ((closest - surfel.center_R).squaredNorm() <=
              surfel.support_radius_m * surfel.support_radius_m +
                  config_.numerical.tie_tolerance)
            structural_owners[key].insert(cause);
        }
  }
  for (auto& item : reference_) {
    const auto owner = structural_owners.find(item.first);
    if (owner == structural_owners.end()) continue;
    item.second.structural_geometry_represented = true;
    if (owner->second.size() == 1U)
      item.second.owner_cause_id = *owner->second.begin();
  }
  return !reference_.empty();
}

bool OccupancyEventGraph::Update(
    const FrameBundle& frame,
    const std::vector<HazardGeometry,
        Eigen::aligned_allocator<HazardGeometry>>& structural_claims,
    std::vector<ReasonCode>* reasons) {
  if (reference_.empty() || frame.validity == Validity::INVALID) {
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_NOT_FROZEN);
    return false;
  }
  std::map<VoxelKey, CurrentCount> current;
  size_t current_evidence_links = 0U;
  const auto append_evidence = [&](const VoxelKey& key,
                                   const StableId& evidence_id) {
    const auto existing_cell = current.find(key);
    if (existing_cell != current.end() &&
        existing_cell->second.evidence_ids.count(evidence_id) > 0U)
      return true;
    if (current_evidence_links >= config_.resource.max_evidence_links)
      return false;
    current[key].evidence_ids.insert(evidence_id);
    ++current_evidence_links;
    return true;
  };
  const bool free_space_evidence_valid =
      frame.validity == Validity::VALID;
  const auto endpoint_claimed = [&](const Eigen::Vector3d& endpoint_R) {
    for (const HazardGeometry& claim : structural_claims) {
      if (std::abs(claim.horizon_s) > config_.numerical.tie_tolerance)
        continue;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(
          0.5 * (claim.covariance_R + claim.covariance_R.transpose()),
          Eigen::EigenvaluesOnly);
      if (eigen.info() != Eigen::Success) continue;
      const double sigma = std::sqrt(std::max(
          0.0, eigen.eigenvalues().maxCoeff()));
      const double extent = claim.radius_m + claim.model_envelope_m +
                            3.0 * sigma;
      if ((endpoint_R - claim.center_R).squaredNorm() <= extent * extent)
        return true;
    }
    return false;
  };
  // Visibility comes from a disjoint support role, not from the endpoint
  // count k consumed by the negative-binomial likelihood.
  std::map<VoxelKey, uint32_t> visibility_coverage;
  for (const CanonicalPoint& point : frame.points) {
    if (!free_space_evidence_valid) break;
    if (point.role != PointRole::ASSOCIATION) continue;
    bool resource_exceeded = false;
    const std::vector<VoxelKey> ray = TraceRay(
        frame.pose.origin_L_R, point.position_R, true,
        &resource_exceeded);
    if (resource_exceeded) {
      if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
      return false;
    }
    for (const VoxelKey& key : ray) ++visibility_coverage[key];
    if (visibility_coverage.size() > config_.resource.max_tiles) {
      if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
      return false;
    }
  }
  for (const CanonicalPoint& point : frame.points) {
    if (point.role != PointRole::OCCUPANCY) continue;
    if (!free_space_evidence_valid) {
      // A geometrically unresolved scan still protects its measured endpoint,
      // but never fabricates traversed FREE cells from an untrusted ray.
      VoxelKey endpoint;
      if (!TryGridKey(point.position_R, config_.resource.grid_origin_m,
                      config_.resource.occupancy_voxel_m, &endpoint)) {
        if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
        return false;
      }
      ++current[endpoint].endpoints;
      if (!append_evidence(endpoint, point.evidence_id)) {
        if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
        return false;
      }
      continue;
    }
    bool resource_exceeded = false;
    const std::vector<VoxelKey> ray = TraceRay(
        frame.pose.origin_L_R, point.position_R, true,
        &resource_exceeded);
    if (resource_exceeded) {
      if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
      return false;
    }
    if (ray.empty()) continue;
    for (size_t i = 0; i + 1U < ray.size(); ++i) {
      ++current[ray[i]].traversed;
      if (!append_evidence(ray[i], point.evidence_id)) {
        if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
        return false;
      }
    }
    // A structurally explained return is already carried by the GDU/OTHER
    // HazardCause. Keep its traversed free-space evidence, but do not create a
    // second independent PRESENT event at the endpoint.
    if (!endpoint_claimed(point.position_R)) {
      ++current[ray.back()].endpoints;
      if (!append_evidence(ray.back(), point.evidence_id)) {
        if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
        return false;
      }
    }
    if (current.size() > config_.resource.max_tiles) {
      if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
      return false;
    }
  }
  std::set<VoxelKey> keys;
  for (const auto& item : reference_) keys.insert(item.first);
  for (const auto& item : current) keys.insert(item.first);
  for (const auto& item : event_states_) keys.insert(item.first);
  if (keys.size() > config_.resource.max_tiles) {
    if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
    return false;
  }
  total_causes_.clear();
  new_causes_.clear();
  cause_semantics_.clear();
  output_.clear(); output_.reserve(keys.size());
  for (const VoxelKey& key : keys) {
    const auto reference_found = reference_.find(key);
    ReferenceCount reference;
    if (reference_found != reference_.end()) {
      reference = reference_found->second;
    } else {
      reference.occupancy.lower = 0.0;
      reference.occupancy.mean = 0.5;
      reference.occupancy.upper = 1.0;
      reference.occupancy.calibrated = false;
      reference.occupancy.mean_identified = false;
      reference.knowledge = KnowledgeState::UNKNOWN;
    }
    CurrentCount observed;
    const auto current_found = current.find(key);
    if (current_found != current.end()) observed = current_found->second;
    const Eigen::Vector3d cell_center = config_.resource.grid_origin_m +
        config_.resource.occupancy_voxel_m *
        (Eigen::Vector3d(static_cast<double>(key.x), static_cast<double>(key.y),
                         static_cast<double>(key.z)) +
         Eigen::Vector3d::Constant(0.5));
    const double range_m = (cell_center - frame.pose.origin_L_R).norm();
    const bool within_calibrated_range =
        range_m >= config_.calibration.visibility_min_range_m &&
        range_m <= config_.calibration.visibility_max_range_m;
    const auto coverage = visibility_coverage.find(key);
    const uint32_t coverage_count = coverage == visibility_coverage.end()
        ? 0U : coverage->second;
    // visibility_coverage was built once from the disjoint ASSOCIATION rays.
    // A traversed cell is empirically visible in this frame; re-tracing a ray
    // through every reference cell would be O(N_cells * ray_length), and a
    // frozen occupied cell on that observed ray is change evidence rather
    // than a reason to discard visibility.
    observed.visibility = within_calibrated_range
        ? 1.0 - std::exp(-static_cast<double>(coverage_count) /
                         config_.calibration.visibility_exposure_scale)
        : 0.0;
    if (free_space_evidence_valid && observed.visibility == 0.0 &&
        observed.endpoints > 0U && reasons)
      reasons->push_back(ReasonCode::OUTSIDE_CALIBRATED_RANGE);

    OccupancyCell cell;
    cell.key = key;
    cell.reference_occupancy = reference.occupancy;
    cell.reference_cause_id = reference.owner_cause_id;
    cell.knowledge = reference.knowledge;
    cell.new_occupancy_applicable =
        reference.knowledge == KnowledgeState::FREE;
    cell.exposure_count = coverage_count;
    cell.visibility = observed.visibility;
    if (!free_space_evidence_valid && observed.endpoints > 0U) {
      // This is an ephemeral, fail-safe carrier for a real endpoint whose ray
      // and structural interpretation are outside the admitted input profile.
      // It is deliberately distinct from the persistent NOVEL event so an
      // untrusted frame cannot update or clear that calibrated chain.
      CanonicalBytes protective_fields;
      protective_fields.AddString(config_.resource.session_id);
      protective_fields.AddU32(reference_epoch_);
      protective_fields.AddI64(key.x);
      protective_fields.AddI64(key.y);
      protective_fields.AddI64(key.z);
      protective_fields.AddString("input-quality-endpoint-envelope-v1");
      const StableId protective_cause =
          MakeStableId("occupancy-protective-cause-v1", protective_fields);
      ProbabilityInterval protective;
      protective.lower = 0.0;
      protective.mean = 1.0;
      protective.upper = 1.0;
      protective.calibrated = false;
      protective.mean_identified = true;
      total_causes_[key][protective_cause] = protective;
      if (reference.knowledge == KnowledgeState::FREE)
        new_causes_[key][protective_cause] = protective;
      cause_semantics_[key][protective_cause] =
          HazardSemantics::OTHER_ENVELOPE;
      cell.last_observed = frame.stamp;
      // Do not let this endpoint enter the normal event likelihood below.
      observed = CurrentCount();
    }
    output_.push_back(cell);

    if (reference.structural_geometry_represented) {
      // The GDU U/A/C mixture is the sole PRESENT carrier; its probability is
      // inserted below by ProjectStructuralHazards. This remains true for an
      // overlap without a unique owner: every involved GDU is projected and
      // no extra reference-persistence event is born.
      continue;
    }
    StableId cause;
    HazardSemantics semantics = HazardSemantics::OTHER_ENVELOPE;
    ProbabilityInterval initial;
    if (reference.knowledge == KnowledgeState::FREE) {
      cause = CauseId(key);
      semantics = HazardSemantics::NOVEL_OCCUPANCY;
      initial.mean = config_.calibration.new_occupancy_prior;
      initial.lower = config_.calibration.validated ? initial.mean : 0.0;
      initial.upper = config_.calibration.validated ? initial.mean : 1.0;
      initial.calibrated = config_.calibration.validated;
      initial.mean_identified = true;
    } else if (reference.knowledge == KnowledgeState::OCCUPIED &&
               reference_found != reference_.end()) {
      cause = ReferenceCauseId(key);
      semantics = HazardSemantics::REFERENCE_PERSISTENCE;
      initial = reference.occupancy;
    } else {
      cause = CauseId(key);
      semantics = HazardSemantics::OTHER_ENVELOPE;
      initial = reference.occupancy;
    }
    const auto existing_voxel = event_states_.find(key);
    const bool event_exists = existing_voxel != event_states_.end() &&
        existing_voxel->second.count(cause) > 0U;
    // UNKNOWN is absence of sufficient free/occupied evidence, not evidence
    // that a persistent obstacle exists.  Only a frozen OCCUPIED cell births
    // a reference-persistence event without a current endpoint.  UNKNOWN
    // remains protective in the occupancy knowledge layer and can birth an
    // event later when an actual endpoint is observed.
    const bool frozen_presence_carrier =
        reference_found != reference_.end() &&
        reference.knowledge == KnowledgeState::OCCUPIED;
    // A KNOWN_FREE singleton event is born only from an unexplained endpoint;
    // instantiating the calibrated birth prior in every free voxel would turn
    // map resolution into thousands of fictitious independent hazards.
    if (!event_exists && !frozen_presence_carrier &&
        observed.endpoints == 0U) {
      if (observed.visibility > 0.0 &&
          observed.traversed + observed.endpoints > 0U)
        output_.back().last_observed = frame.stamp;
      continue;
    }
    EventState& event = event_states_[key][cause];
    if (event.last_stamp.isZero()) {
      event.posterior = initial;
      event.semantics = semantics;
    }
    if (event.semantics != semantics) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
    const ProbabilityInterval prior = PredictEvent(event, frame.stamp);
    event.posterior = UpdateEventPosterior(prior, observed);
    event.last_stamp = frame.stamp;
    if (observed.visibility > 0.0 &&
        observed.traversed + observed.endpoints > 0U) {
      event.last_evidence_stamp = frame.stamp;
      event.source_ids = observed.evidence_ids;
    }
    output_.back().last_observed = event.last_evidence_stamp;
    total_causes_[key][cause] = event.posterior;
    if (reference.knowledge == KnowledgeState::FREE)
      new_causes_[key][cause] = event.posterior;
    cause_semantics_[key][cause] = semantics;
  }
  // max_events bounds logical HazardCauseId entries, not their spatial
  // projections.  One structural GDU cause may legitimately occupy thousands
  // of voxels; counting every (voxel,cause) projection as a new event both
  // violates cause deduplication and makes map resolution consume the causal
  // ledger.
  std::set<StableId> event_causes;
  size_t retained_evidence_links = 0U;
  for (const auto& voxel : event_states_) {
    for (const auto& event : voxel.second) {
      event_causes.insert(event.first);
      if (event.second.source_ids.size() >
          config_.resource.max_evidence_links -
              std::min(config_.resource.max_evidence_links,
                       retained_evidence_links)) {
        std::cerr << "[deform_v1] resource stage=retained_evidence_links value="
                  << retained_evidence_links << " limit="
                  << config_.resource.max_evidence_links << std::endl;
        if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
        return false;
      }
      retained_evidence_links += event.second.source_ids.size();
    }
  }
  if (event_causes.size() > config_.resource.max_events) {
    std::cerr << "[deform_v1] resource stage=occupancy_causes value="
              << event_causes.size() << " limit="
              << config_.resource.max_events << std::endl;
    if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
    return false;
  }
  RefreshOutputProbabilities();
  return true;
}

bool OccupancyEventGraph::ProjectStructuralHazards(
    const std::vector<HazardGeometry, Eigen::aligned_allocator<HazardGeometry>>& hazards,
    std::vector<ReasonCode>* reasons) {
  // A component may contain many surfel primitives in the same voxel.  Those
  // primitives are one mutually exclusive component event, not independent
  // evidence.  Collapse primitive multiplicity before projecting probability.
  struct ComponentProjection {
    ProbabilityInterval probability{0.0, 0.0, 0.0, true, true};
    StructuralState state = StructuralState::U;
    HazardSemantics semantics = HazardSemantics::OTHER_ENVELOPE;
  };
  using ComponentProbability = std::map<StableId, ComponentProjection>;
  std::map<VoxelKey, std::map<StableId, ComponentProbability>> projection;
  std::map<std::pair<VoxelKey, StableId>, double> other_mass;
  size_t projection_entries = 0U;
  for (const HazardGeometry& hazard : hazards) {
    if (std::abs(hazard.horizon_s) > config_.numerical.tie_tolerance) continue;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(
        0.5 * (hazard.covariance_R + hazard.covariance_R.transpose()),
        Eigen::EigenvaluesOnly);
    const double sigma = eigen.info() == Eigen::Success
        ? std::sqrt(std::max(0.0, eigen.eigenvalues().maxCoeff())) : 0.0;
    const double extent = hazard.radius_m + hazard.model_envelope_m + 3.0 * sigma;
    VoxelKey minimum;
    VoxelKey maximum;
    if (!std::isfinite(extent) || extent < 0.0 ||
        !TryGridKey(hazard.center_R - Eigen::Vector3d::Constant(extent),
                    config_.resource.grid_origin_m,
                    config_.resource.occupancy_voxel_m, &minimum) ||
        !TryGridKey(hazard.center_R + Eigen::Vector3d::Constant(extent),
                    config_.resource.grid_origin_m,
                    config_.resource.occupancy_voxel_m, &maximum)) {
      if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
      return false;
    }
    const int64_t span_x = static_cast<int64_t>(maximum.x) - minimum.x + 1;
    const int64_t span_y = static_cast<int64_t>(maximum.y) - minimum.y + 1;
    const int64_t span_z = static_cast<int64_t>(maximum.z) - minimum.z + 1;
    const long double candidate_cells = static_cast<long double>(span_x) *
        static_cast<long double>(span_y) * static_cast<long double>(span_z);
    if (span_x <= 0 || span_y <= 0 || span_z <= 0 ||
        candidate_cells > static_cast<long double>(config_.resource.max_tiles)) {
      std::cerr << "[deform_v1] projection resource stage=single_hazard_span"
                << " span=" << span_x << "x" << span_y << "x" << span_z
                << " extent=" << extent << " limit="
                << config_.resource.max_tiles << std::endl;
      if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
      return false;
    }
    for (int64_t x = minimum.x; x <= static_cast<int64_t>(maximum.x); ++x)
      for (int64_t y = minimum.y; y <= static_cast<int64_t>(maximum.y); ++y)
        for (int64_t z = minimum.z; z <= static_cast<int64_t>(maximum.z); ++z) {
          const VoxelKey key{static_cast<int32_t>(x), static_cast<int32_t>(y),
                             static_cast<int32_t>(z)};
          // This layer projects inferred causes only into cells whose frozen
          // reference already has an occupancy interpretation.  A geometric
          // envelope reaching an unobserved cell does not turn UNKNOWN into a
          // newly observed voxel; navigation already treats UNKNOWN
          // protectively.  Frozen FREE cells receive change occupancy, while
          // structurally owned carrier cells receive their canonical GDU
          // cause so baseline geometry is not represented by a second event.
          const auto reference_cell = reference_.find(key);
          const bool known_free = reference_cell != reference_.end() &&
              reference_cell->second.knowledge == KnowledgeState::FREE;
          const bool owned_structural_carrier =
              hazard.semantics == HazardSemantics::STRUCTURAL_DEFORMATION &&
              reference_cell != reference_.end() &&
              reference_cell->second.structural_geometry_represented;
          if (!known_free && !owned_structural_carrier) continue;
          const Eigen::Vector3d center = config_.resource.grid_origin_m +
              config_.resource.occupancy_voxel_m *
                  (Eigen::Vector3d(static_cast<double>(x),
                                   static_cast<double>(y),
                                   static_cast<double>(z)) +
                   Eigen::Vector3d::Constant(0.5));
          const Eigen::Vector3d closest = hazard.center_R.cwiseMax(
              center - Eigen::Vector3d::Constant(
                           0.5 * config_.resource.occupancy_voxel_m)).cwiseMin(
              center + Eigen::Vector3d::Constant(
                           0.5 * config_.resource.occupancy_voxel_m));
          if ((closest - hazard.center_R).squaredNorm() > extent * extent) continue;
          auto inserted = projection[key][hazard.cause_id].emplace(
              hazard.component_id, ComponentProjection());
          if (inserted.second &&
              ++projection_entries >
                  config_.resource.max_hazard_projection_entries) {
            std::cerr << "[deform_v1] projection resource stage=component_voxels"
                      << " value=" << projection_entries << " limit="
                      << config_.resource.max_hazard_projection_entries
                      << " voxels=" << projection.size() << std::endl;
            if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
            return false;
          }
          ComponentProjection& projected = inserted.first->second;
          projected.state = hazard.structural_state;
          projected.semantics = hazard.semantics;
          ProbabilityInterval& component = projected.probability;
          component.lower = std::max(component.lower, hazard.existence.lower);
          component.mean = std::max(component.mean, hazard.existence.mean);
          component.upper = std::max(component.upper, hazard.existence.upper);
          component.calibrated = component.calibrated &&
                                 hazard.existence.calibrated;
          component.mean_identified = component.mean_identified &&
                                      hazard.existence.mean_identified;
          const auto key_cause = std::make_pair(key, hazard.cause_id);
          other_mass[key_cause] = std::max(other_mass[key_cause],
                                           hazard.other_mass);
        }
    if (projection.size() > config_.resource.max_tiles) {
      std::cerr << "[deform_v1] projection resource stage=projected_voxels"
                << " value=" << projection.size() << " limit="
                << config_.resource.max_tiles << std::endl;
      if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
      return false;
    }
  }
  std::set<VoxelKey> projected_output_keys;
  for (const OccupancyCell& cell : output_)
    projected_output_keys.insert(cell.key);
  for (const auto& voxel : projection)
    projected_output_keys.insert(voxel.first);
  if (projected_output_keys.size() > config_.resource.max_tiles) {
    std::cerr << "[deform_v1] projection resource stage=union_voxels"
              << " value=" << projected_output_keys.size() << " limit="
              << config_.resource.max_tiles << std::endl;
    if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
    return false;
  }
  for (const auto& voxel : projection) {
    const VoxelKey& key = voxel.first;
    auto cell = std::lower_bound(output_.begin(), output_.end(), key,
        [](const OccupancyCell& lhs, const VoxelKey& rhs) { return lhs.key < rhs; });
    if (cell == output_.end() || !(cell->key == key)) {
      OccupancyCell unknown;
      unknown.key = key;
      unknown.reference_occupancy = ProbabilityInterval();
      unknown.total_occupancy = ProbabilityInterval();
      unknown.new_occupancy = ProbabilityInterval();
      unknown.knowledge = KnowledgeState::UNKNOWN;
      unknown.new_occupancy_applicable = false;
      cell = output_.insert(cell, unknown);
    }
    for (const auto& cause : voxel.second) {
      ProbabilityInterval total_projection;
      total_projection.upper = 0.0;
      total_projection.calibrated = true;
      total_projection.mean_identified = true;
      ProbabilityInterval new_projection;
      new_projection.upper = 0.0;
      new_projection.calibrated = true;
      new_projection.mean_identified = true;
      for (const auto& component : cause.second) {
        const ProbabilityInterval& probability = component.second.probability;
        total_projection.lower += probability.lower;
        total_projection.mean += probability.mean;
        total_projection.upper += probability.upper;
        total_projection.calibrated = total_projection.calibrated &&
                                      probability.calibrated;
        total_projection.mean_identified =
            total_projection.mean_identified &&
            probability.mean_identified;
        const bool eligible_new =
            component.second.semantics !=
                HazardSemantics::STRUCTURAL_DEFORMATION ||
            component.second.state != StructuralState::U;
        if (eligible_new) {
          new_projection.lower += probability.lower;
          new_projection.mean += probability.mean;
          new_projection.upper += probability.upper;
          new_projection.calibrated = new_projection.calibrated &&
                                      probability.calibrated;
          new_projection.mean_identified =
              new_projection.mean_identified &&
              probability.mean_identified;
        }
      }
      total_projection.lower = std::min(1.0, total_projection.lower);
      total_projection.mean = std::min(1.0, total_projection.mean);
      total_projection.upper = std::min(1.0, total_projection.upper +
          other_mass[std::make_pair(key, cause.first)]);
      new_projection.lower = std::min(1.0, new_projection.lower);
      new_projection.mean = std::min(1.0, new_projection.mean);
      new_projection.upper = std::min(1.0, new_projection.upper +
          other_mass[std::make_pair(key, cause.first)]);
      if (other_mass[std::make_pair(key, cause.first)] >
          config_.numerical.probability_floor) {
        total_projection.mean_identified = false;
        new_projection.mean_identified = false;
      }
      if (!total_projection.mean_identified)
        total_projection.mean = total_projection.upper;
      if (!new_projection.mean_identified)
        new_projection.mean = new_projection.upper;
      total_causes_[key][cause.first] = total_projection;
      HazardSemantics projected_semantics = HazardSemantics::OTHER_ENVELOPE;
      for (const auto& component : cause.second) {
        if (component.second.semantics == HazardSemantics::STRUCTURAL_DEFORMATION) {
          projected_semantics = HazardSemantics::STRUCTURAL_DEFORMATION;
          break;
        }
        projected_semantics = component.second.semantics;
      }
      cause_semantics_[key][cause.first] = projected_semantics;
      if (cell->new_occupancy_applicable && new_projection.upper > 0.0)
        new_causes_[key][cause.first] = new_projection;
    }
  }
  if (output_.size() > config_.resource.max_tiles) {
    std::cerr << "[deform_v1] projection resource stage=output_voxels"
              << " value=" << output_.size() << " limit="
              << config_.resource.max_tiles << std::endl;
    if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
    return false;
  }
  RefreshOutputProbabilities();
  return true;
}

const OccupancyCell* OccupancyEventGraph::Find(const VoxelKey& key) const {
  const auto found = std::lower_bound(output_.begin(), output_.end(), key,
      [](const OccupancyCell& cell, const VoxelKey& value) { return cell.key < value; });
  return found != output_.end() && found->key == key ? &*found : nullptr;
}

}  // namespace v1
}  // namespace deform_monitor_v3
