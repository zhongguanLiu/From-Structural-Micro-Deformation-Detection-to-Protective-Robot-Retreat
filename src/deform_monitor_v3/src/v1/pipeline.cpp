#include "deform_monitor_v3/v1/pipeline.hpp"

#include "deform_monitor_v3/v1/config.hpp"
#include "deform_monitor_v3/v1/deterministic.hpp"
#include "deform_monitor_v3/v1/math.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>

namespace deform_monitor_v3 {
namespace v1 {

std::map<StableId, StableId> MatchDisjointPatchPairs(
    DisjointPatchDescriptors association,
    DisjointPatchDescriptors measurement,
    double maximum_distance_m,
    double tie_tolerance,
    DisjointPatchMatchingStats* stats) {
  std::map<StableId, StableId> output;
  if (stats) stats->distance_evaluations = 0U;
  if (!std::isfinite(maximum_distance_m) || maximum_distance_m < 0.0 ||
      !std::isfinite(tie_tolerance) || tie_tolerance < 0.0)
    return output;
  const auto valid = [](const DisjointPatchDescriptor& patch) {
    return !patch.id.empty() && patch.center_R.allFinite() &&
        patch.gdu_index != std::numeric_limits<uint32_t>::max();
  };
  if (!std::all_of(association.begin(), association.end(), valid) ||
      !std::all_of(measurement.begin(), measurement.end(), valid))
    return output;
  const auto canonical = [](const DisjointPatchDescriptor& lhs,
                            const DisjointPatchDescriptor& rhs) {
    return lhs.id < rhs.id;
  };
  std::sort(association.begin(), association.end(), canonical);
  std::sort(measurement.begin(), measurement.end(), canonical);
  const auto duplicate_id = [](const DisjointPatchDescriptor& lhs,
                               const DisjointPatchDescriptor& rhs) {
    return lhs.id == rhs.id;
  };
  if (std::adjacent_find(association.begin(), association.end(), duplicate_id) !=
          association.end() ||
      std::adjacent_find(measurement.begin(), measurement.end(), duplicate_id) !=
          measurement.end())
    return output;

  struct Candidate {
    size_t measurement = 0U;
    double distance_m = 0.0;
  };
  const double matching_cell_size_m = maximum_distance_m + tie_tolerance;
  std::map<uint32_t, std::map<VoxelKey, std::vector<size_t>>>
      measurement_cells_by_gdu;
  if (!(matching_cell_size_m > 0.0) ||
      !std::isfinite(matching_cell_size_m))
    return output;
  for (size_t m = 0U; m < measurement.size(); ++m) {
    VoxelKey key;
    if (!TryGridKey(measurement[m].center_R, Eigen::Vector3d::Zero(),
                    matching_cell_size_m, &key))
      return {};
    measurement_cells_by_gdu[measurement[m].gdu_index][key].push_back(m);
  }
  std::vector<std::vector<Candidate>> candidates(association.size());
  for (size_t a = 0U; a < association.size(); ++a) {
    const auto same_gdu =
        measurement_cells_by_gdu.find(association[a].gdu_index);
    if (same_gdu == measurement_cells_by_gdu.end()) continue;
    VoxelKey center;
    if (!TryGridKey(association[a].center_R, Eigen::Vector3d::Zero(),
                    matching_cell_size_m, &center))
      return {};
    for (int64_t dx = -1; dx <= 1; ++dx)
      for (int64_t dy = -1; dy <= 1; ++dy)
        for (int64_t dz = -1; dz <= 1; ++dz) {
          const int64_t x = static_cast<int64_t>(center.x) + dx;
          const int64_t y = static_cast<int64_t>(center.y) + dy;
          const int64_t z = static_cast<int64_t>(center.z) + dz;
          if (x < std::numeric_limits<int32_t>::min() ||
              x > std::numeric_limits<int32_t>::max() ||
              y < std::numeric_limits<int32_t>::min() ||
              y > std::numeric_limits<int32_t>::max() ||
              z < std::numeric_limits<int32_t>::min() ||
              z > std::numeric_limits<int32_t>::max())
            continue;
          const auto cell = same_gdu->second.find(VoxelKey{
              static_cast<int32_t>(x), static_cast<int32_t>(y),
              static_cast<int32_t>(z)});
          if (cell == same_gdu->second.end()) continue;
          for (size_t m : cell->second) {
            if (stats) ++stats->distance_evaluations;
            const double distance =
                (association[a].center_R - measurement[m].center_R).norm();
            if (distance <= maximum_distance_m + tie_tolerance)
              candidates[a].push_back(Candidate{m, distance});
          }
        }
    std::sort(candidates[a].begin(), candidates[a].end(),
              [&](const Candidate& lhs, const Candidate& rhs) {
                if (std::abs(lhs.distance_m - rhs.distance_m) > tie_tolerance)
                  return lhs.distance_m < rhs.distance_m;
                return measurement[lhs.measurement].id <
                       measurement[rhs.measurement].id;
              });
  }

  // Scarce association patches are augmented first.  The recursive
  // reassignment is the standard bipartite augmenting-path construction, so
  // the final number of pairs is maximum rather than merely greedy.
  std::vector<size_t> order(association.size());
  std::iota(order.begin(), order.end(), 0U);
  std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
    if (candidates[lhs].size() != candidates[rhs].size())
      return candidates[lhs].size() < candidates[rhs].size();
    return association[lhs].id < association[rhs].id;
  });
  const size_t unmatched = std::numeric_limits<size_t>::max();
  std::vector<size_t> measurement_owner(measurement.size(), unmatched);
  std::function<bool(size_t, std::vector<bool>*)> augment =
      [&](size_t association_index, std::vector<bool>* visited) {
        for (const Candidate& candidate : candidates[association_index]) {
          if ((*visited)[candidate.measurement]) continue;
          (*visited)[candidate.measurement] = true;
          const size_t previous = measurement_owner[candidate.measurement];
          if (previous == unmatched || augment(previous, visited)) {
            measurement_owner[candidate.measurement] = association_index;
            return true;
          }
        }
        return false;
      };
  for (size_t association_index : order) {
    std::vector<bool> visited(measurement.size(), false);
    static_cast<void>(augment(association_index, &visited));
  }
  for (size_t measurement_index = 0U;
       measurement_index < measurement_owner.size(); ++measurement_index) {
    const size_t association_index = measurement_owner[measurement_index];
    if (association_index == unmatched) continue;
    output.emplace(association[association_index].id,
                   measurement[measurement_index].id);
  }
  return output;
}

bool ReferenceSurfelSpatialIndex::Build(
    const ReferenceSnapshot& reference,
    const Eigen::Vector3d& grid_origin_R,
    double cell_size_m) {
  valid_ = false;
  surfels_by_cell_.clear();
  owner_candidates_by_patch_cell_.clear();
  owner_patch_cell_size_m_ = 0.0;
  owner_search_radius_m_ = 0.0;
  owner_candidate_cache_valid_ = false;
  grid_origin_R_.setZero();
  cell_size_m_ = 0.0;
  if (!grid_origin_R.allFinite() || !(cell_size_m > 0.0) ||
      !std::isfinite(cell_size_m)) {
    return false;
  }
  for (uint32_t index = 0U; index < reference.surfels.size(); ++index) {
    VoxelKey key;
    if (!TryGridKey(reference.surfels[index].center_R, grid_origin_R,
                    cell_size_m, &key)) {
      surfels_by_cell_.clear();
      return false;
    }
    surfels_by_cell_[key].push_back(index);
  }
  for (auto& cell : surfels_by_cell_) {
    std::sort(cell.second.begin(), cell.second.end(),
              [&](uint32_t lhs, uint32_t rhs) {
                return reference.surfels[lhs].id <
                       reference.surfels[rhs].id;
              });
  }
  grid_origin_R_ = grid_origin_R;
  cell_size_m_ = cell_size_m;
  valid_ = true;
  return true;
}

bool ReferenceSurfelSpatialIndex::BuildOwnerCandidateCache(
    const ReferenceSnapshot& reference,
    const Eigen::Vector3d& grid_origin_R,
    double patch_cell_size_m,
    double owner_radius_m,
    double tie_tolerance) {
  valid_ = false;
  surfels_by_cell_.clear();
  owner_candidates_by_patch_cell_.clear();
  owner_patch_cell_size_m_ = 0.0;
  owner_search_radius_m_ = 0.0;
  owner_candidate_cache_valid_ = false;
  if (!grid_origin_R.allFinite() || !(patch_cell_size_m > 0.0) ||
      !std::isfinite(patch_cell_size_m) || !(owner_radius_m >= 0.0) ||
      !std::isfinite(owner_radius_m) || !(tie_tolerance >= 0.0) ||
      !std::isfinite(tie_tolerance) || reference.gdus.size() > 64U) {
    return false;
  }
  const double search_radius = owner_radius_m + tie_tolerance;
  if (!std::isfinite(search_radius)) return false;
  owner_candidates_by_patch_cell_.reserve(reference.surfels.size() * 8U);
  const Eigen::Vector3d extent = Eigen::Vector3d::Constant(search_radius);
  for (uint32_t index = 0U; index < reference.surfels.size(); ++index) {
    const Eigen::Vector3d& center_R = reference.surfels[index].center_R;
    VoxelKey minimum;
    VoxelKey maximum;
    if (!center_R.allFinite() ||
        !TryGridKey(center_R - extent, grid_origin_R, patch_cell_size_m,
                    &minimum) ||
        !TryGridKey(center_R + extent, grid_origin_R, patch_cell_size_m,
                    &maximum)) {
      owner_candidates_by_patch_cell_.clear();
      return false;
    }
    for (int64_t x = minimum.x; x <= static_cast<int64_t>(maximum.x); ++x)
      for (int64_t y = minimum.y; y <= static_cast<int64_t>(maximum.y); ++y)
        for (int64_t z = minimum.z; z <= static_cast<int64_t>(maximum.z); ++z) {
          const VoxelKey key{static_cast<int32_t>(x),
                             static_cast<int32_t>(y),
                             static_cast<int32_t>(z)};
          const Eigen::Vector3d lower = grid_origin_R + patch_cell_size_m *
              Eigen::Vector3d(static_cast<double>(key.x),
                              static_cast<double>(key.y),
                              static_cast<double>(key.z));
          const Eigen::Vector3d upper =
              lower + Eigen::Vector3d::Constant(patch_cell_size_m);
          Eigen::Vector3d delta = Eigen::Vector3d::Zero();
          for (int axis = 0; axis < 3; ++axis) {
            if (center_R[axis] < lower[axis])
              delta[axis] = lower[axis] - center_R[axis];
            else if (center_R[axis] > upper[axis])
              delta[axis] = center_R[axis] - upper[axis];
          }
          if (delta.squaredNorm() <= search_radius * search_radius)
            owner_candidates_by_patch_cell_[key].push_back(index);
        }
  }
  grid_origin_R_ = grid_origin_R;
  owner_patch_cell_size_m_ = patch_cell_size_m;
  owner_search_radius_m_ = search_radius;
  owner_candidate_cache_valid_ = true;
  valid_ = true;
  return true;
}

bool ReferenceSurfelSpatialIndex::NearbySurfels(
    const Eigen::Vector3d& position_R,
    double radius_m,
    std::vector<uint32_t>* surfel_indices) const {
  if (!surfel_indices) return false;
  surfel_indices->clear();
  if (!valid_ || !position_R.allFinite() || !std::isfinite(radius_m) ||
      radius_m < 0.0) {
    return false;
  }
  VoxelKey center;
  if (!TryGridKey(position_R, grid_origin_R_, cell_size_m_, &center))
    return false;
  const int64_t radius_cells = static_cast<int64_t>(
      std::ceil(radius_m / cell_size_m_));
  for (int64_t dx = -radius_cells; dx <= radius_cells; ++dx)
    for (int64_t dy = -radius_cells; dy <= radius_cells; ++dy)
      for (int64_t dz = -radius_cells; dz <= radius_cells; ++dz) {
        const int64_t x = static_cast<int64_t>(center.x) + dx;
        const int64_t y = static_cast<int64_t>(center.y) + dy;
        const int64_t z = static_cast<int64_t>(center.z) + dz;
        if (x < std::numeric_limits<int32_t>::min() ||
            x > std::numeric_limits<int32_t>::max() ||
            y < std::numeric_limits<int32_t>::min() ||
            y > std::numeric_limits<int32_t>::max() ||
            z < std::numeric_limits<int32_t>::min() ||
            z > std::numeric_limits<int32_t>::max()) {
          continue;
        }
        const auto cell = surfels_by_cell_.find(VoxelKey{
            static_cast<int32_t>(x), static_cast<int32_t>(y),
            static_cast<int32_t>(z)});
        if (cell == surfels_by_cell_.end()) continue;
        surfel_indices->insert(surfel_indices->end(), cell->second.begin(),
                               cell->second.end());
      }
  return true;
}

PatchOwnership ReferenceSurfelSpatialIndex::QueryOwner(
    const ReferenceSnapshot& reference,
    const Eigen::Vector3d& position_R,
    double neighborhood_radius_m,
    double tie_tolerance,
    double ownership_resolution_m) const {
  PatchOwnership result;
  if (!std::isfinite(tie_tolerance) || tie_tolerance < 0.0 ||
      !std::isfinite(ownership_resolution_m) ||
      ownership_resolution_m < 0.0) {
    return result;
  }
  std::vector<uint32_t> nearby_fallback;
  const std::vector<uint32_t>* nearby = nullptr;
  const double owner_search_radius_m =
      neighborhood_radius_m + tie_tolerance;
  if (!std::isfinite(owner_search_radius_m))
    return result;
  if (owner_candidate_cache_valid_ &&
      owner_search_radius_m <= owner_search_radius_m_) {
    VoxelKey key;
    if (!TryGridKey(position_R, grid_origin_R_, owner_patch_cell_size_m_, &key))
      return result;
    const auto found = owner_candidates_by_patch_cell_.find(key);
    if (found == owner_candidates_by_patch_cell_.end()) return result;
    nearby = &found->second;
  } else {
    if (!NearbySurfels(position_R, owner_search_radius_m, &nearby_fallback))
      return result;
    // The exhaustive implementation visits the frozen reference in storage
    // order. Preserve that order exactly. Cached lists already have this order
    // because BuildOwnerCandidateCache appends increasing surfel indices.
    std::sort(nearby_fallback.begin(), nearby_fallback.end());
    nearby = &nearby_fallback;
  }
  double nearest = std::numeric_limits<double>::infinity();
  std::array<double, 64> nearest_by_gdu;
  nearest_by_gdu.fill(std::numeric_limits<double>::infinity());
  for (uint32_t index : *nearby) {
    if (index >= reference.surfels.size()) continue;
    const double distance =
        (position_R - reference.surfels[index].center_R).norm();
    if (distance <= neighborhood_radius_m) {
      const uint32_t gdu = reference.surfels[index].gdu_index;
      if (gdu < nearest_by_gdu.size() && distance < nearest_by_gdu[gdu])
        nearest_by_gdu[gdu] = distance;
    }
    if (distance < nearest - tie_tolerance ||
        (std::abs(distance - nearest) <= tie_tolerance &&
         (result.owner_surfel == std::numeric_limits<uint32_t>::max() ||
          reference.surfels[index].id <
              reference.surfels[result.owner_surfel].id))) {
      nearest = distance;
      result.owner_surfel = index;
    }
  }
  if (nearest > neighborhood_radius_m)
    result.owner_surfel = std::numeric_limits<uint32_t>::max();
  std::pair<double, uint32_t> first{
      std::numeric_limits<double>::infinity(), 0U};
  std::pair<double, uint32_t> second{
      std::numeric_limits<double>::infinity(), 0U};
  for (uint32_t gdu = 0U;
       gdu < reference.gdus.size() && gdu < nearest_by_gdu.size(); ++gdu) {
    if (!std::isfinite(nearest_by_gdu[gdu])) continue;
    const std::pair<double, uint32_t> candidate{nearest_by_gdu[gdu], gdu};
    if (candidate < first) {
      second = first;
      first = candidate;
    } else if (candidate < second) {
      second = candidate;
    }
  }
  if (std::isfinite(second.first) &&
      second.first - first.first <=
          ownership_resolution_m) {
    result.owner_surfel = std::numeric_limits<uint32_t>::max();
    result.cross_gdu_ambiguous = true;
  }
  return result;
}

std::vector<bool> ReferenceSurfelSpatialIndex::PoseControlExclusionMask(
    const ReferenceSnapshot& reference,
    uint32_t target_gdu_index,
    double radius_m) const {
  std::vector<bool> excluded(reference.surfels.size(), false);
  if (target_gdu_index >= reference.gdus.size() ||
      !std::isfinite(radius_m) || radius_m < 0.0) {
    return excluded;
  }
  const double radius_squared = radius_m * radius_m;
  std::vector<uint32_t> nearby;
  for (uint32_t target_surfel :
       reference.gdus[target_gdu_index].surfel_indices) {
    if (target_surfel >= reference.surfels.size()) continue;
    if (!NearbySurfels(reference.surfels[target_surfel].center_R, radius_m,
                       &nearby)) {
      return std::vector<bool>(reference.surfels.size(), false);
    }
    for (uint32_t owner : nearby) {
      if (owner >= reference.surfels.size()) continue;
      if ((reference.surfels[owner].center_R -
           reference.surfels[target_surfel].center_R).squaredNorm() <=
          radius_squared) {
        excluded[owner] = true;
      }
    }
  }
  return excluded;
}

namespace {

struct PatchAccumulator {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points;
  std::vector<StableId> evidence_ids;
};

Eigen::Vector4d PatchGeometryProbability(const Eigen::Vector3d& eigenvalues,
                                         const CalibrationConfig& calibration) {
  const double scale = std::max(1.0e-15, eigenvalues[2]);
  Eigen::Vector4d feature;
  feature << (eigenvalues[1] - eigenvalues[0]) / scale,
             (eigenvalues[2] - eigenvalues[1]) / scale,
             eigenvalues[0] / scale, 1.0;
  const Eigen::Vector4d logits = calibration.geometry_softmax_weights * feature +
                                 calibration.geometry_softmax_bias;
  const std::vector<double> probabilities = Softmax(
      {logits[0], logits[1], logits[2], logits[3]});
  Eigen::Vector4d output;
  for (int i = 0; i < 4; ++i) output[i] = probabilities[static_cast<size_t>(i)];
  return output;
}

Eigen::MatrixXd ObservableAxes(const Eigen::Matrix3d& projector,
                               double tolerance) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(projector);
  if (eigen.info() != Eigen::Success) return Eigen::MatrixXd(3, 0);
  std::vector<int> selected;
  for (int i = 0; i < 3; ++i) if (eigen.eigenvalues()[i] > 0.5) selected.push_back(i);
  Eigen::MatrixXd axes(3, static_cast<int>(selected.size()));
  for (size_t i = 0; i < selected.size(); ++i) {
    Eigen::Vector3d axis = eigen.eigenvectors().col(selected[i]);
    int sign_axis = 0;
    if (std::abs(axis.y()) > std::abs(axis[sign_axis]) + tolerance) sign_axis = 1;
    if (std::abs(axis.z()) > std::abs(axis[sign_axis]) + tolerance) sign_axis = 2;
    if (axis[sign_axis] < 0.0) axis = -axis;
    axes.col(static_cast<int>(i)) = axis;
  }
  return axes;
}

bool ValidProbability(double value, double tolerance) {
  return std::isfinite(value) && value >= -tolerance &&
      value <= 1.0 + tolerance;
}

bool ValidInterval(const ProbabilityInterval& interval, double tolerance) {
  return ValidProbability(interval.lower, tolerance) &&
      ValidProbability(interval.mean, tolerance) &&
      ValidProbability(interval.upper, tolerance) &&
      interval.lower <= interval.mean + tolerance &&
      interval.mean <= interval.upper + tolerance;
}

bool ValidateFrozenReferenceNumerical(const ReferenceSnapshot& reference,
                                      const V1Config& config) {
  if (!reference.frozen || reference.id.empty() ||
      reference.world_frame != config.resource.world_frame)
    return false;
  for (const Surfel& surfel : reference.surfels) {
    if (surfel.id.empty() || !surfel.center_R.allFinite() ||
        !std::isfinite(surfel.support_radius_m) ||
        surfel.support_radius_m < 0.0 ||
        !ValidateSymmetricPsd(surfel.covariance_R,
                              config.numerical.psd_eigenvalue_tolerance))
      return false;
  }
  return true;
}

bool ValidateSnapshotNumerical(const V1Snapshot& snapshot,
                               const V1Config& config,
                               bool validate_frozen_reference) {
  const double tolerance = config.numerical.normalization_tolerance;
  if (snapshot.common.snapshot_id.empty() ||
      snapshot.common.session_id != config.resource.session_id ||
      snapshot.common.world_frame != config.resource.world_frame ||
      snapshot.common.measurement_stamp.isZero() ||
      snapshot.common.publish_stamp.isZero() ||
      snapshot.common.valid_from != snapshot.common.measurement_stamp ||
      snapshot.common.valid_until < snapshot.common.valid_from ||
      snapshot.common.prediction_valid_until < snapshot.common.valid_from ||
      !snapshot.reference.frozen || snapshot.reference.id.empty() ||
      snapshot.reference.epoch != snapshot.common.reference_epoch ||
      snapshot.reference.world_frame != snapshot.common.world_frame ||
      snapshot.deformation.size() != snapshot.reference.gdus.size() ||
      snapshot.interfaces.size() != snapshot.reference.interfaces.size()) {
    return false;
  }
  if (validate_frozen_reference &&
      !ValidateFrozenReferenceNumerical(snapshot.reference, config))
    return false;
  for (size_t gdu_index = 0U; gdu_index < snapshot.deformation.size();
       ++gdu_index) {
    const GduPosterior& posterior = snapshot.deformation[gdu_index];
    const GduDefinition& definition = snapshot.reference.gdus[gdu_index];
    const int coefficient_count = definition.coordinate_rank * static_cast<int>(
        definition.control_surfel_indices.size() +
        definition.interface_indices.size());
    double state_sum = posterior.other_mass;
    if (posterior.gdu_id != definition.id ||
        (definition.monitorable && coefficient_count <= 0) ||
        (!definition.monitorable && coefficient_count != 0) ||
        posterior.observable_projector.rows() != coefficient_count ||
        posterior.observable_projector.cols() != coefficient_count ||
        posterior.nullspace.rows() != coefficient_count ||
        !ValidProbability(posterior.other_mass, tolerance) ||
        !ValidInterval(ProbabilityInterval{
            posterior.task_scale_deformation_lower,
            posterior.task_scale_deformation_probability,
            posterior.task_scale_deformation_upper, false}, tolerance) ||
        !ValidProbability(posterior.task_displacement_probability, tolerance) ||
        !ValidProbability(posterior.current_velocity_exceedance_probability,
                          tolerance) ||
        !std::isfinite(posterior.continuous_valid_observation_duration_s) ||
        posterior.continuous_valid_observation_duration_s < 0.0 ||
        !ValidProbability(posterior.task_interface_jump_probability, tolerance) ||
        !posterior.observable_projector.allFinite() ||
        !posterior.nullspace.allFinite() ||
        !std::isfinite(posterior.reference_information_ratio) ||
        !std::isfinite(posterior.effective_spatial_support) ||
        !std::isfinite(posterior.effective_temporal_support) ||
        (posterior.minimum_detectable_effect_defined &&
         !std::isfinite(posterior.minimum_detectable_effect_m)) ||
        (posterior.velocity_minimum_detectable_effect_defined &&
         !std::isfinite(posterior.velocity_minimum_detectable_effect_mps)) ||
        (posterior.interface_jump_minimum_detectable_effect_defined &&
         !std::isfinite(posterior.interface_jump_minimum_detectable_effect_m)) ||
        (posterior.task_power_complete &&
         !std::isfinite(posterior.task_power_worst_normalized_mde))) {
      return false;
    }
    for (double probability : posterior.state_probability) {
      if (!ValidProbability(probability, tolerance)) return false;
      state_sum += probability;
    }
    if (std::abs(state_sum - 1.0) > tolerance) return false;
    double component_sum = posterior.other_mass;
    std::array<double, 3> component_state_sum{{0.0, 0.0, 0.0}};
    for (const GaussianComponent& component : posterior.components) {
      const int physical_dimension = component.state == StructuralState::U
          ? 0
          : (component.state == StructuralState::A
                 ? 2 * coefficient_count : coefficient_count);
      const int expected_state_dimension = physical_dimension + 15;
      if (component.mean.size() != expected_state_dimension ||
          component.interface_modes.size() !=
              definition.interface_indices.size() ||
          component.observable_projector.rows() != coefficient_count ||
          component.observable_projector.cols() != coefficient_count ||
          component.observable_nullspace.rows() != coefficient_count ||
          !ValidProbability(component.weight, tolerance) ||
          !component.mean.allFinite() || !component.covariance.allFinite() ||
          component.covariance.rows() != component.mean.size() ||
          component.covariance.cols() != component.mean.size() ||
          !ValidateSymmetricPsd(component.covariance,
                                config.numerical.psd_eigenvalue_tolerance) ||
          component.sqrt_information.cols() != component.mean.size() ||
          !component.sqrt_information.allFinite() ||
          !component.observable_projector.allFinite() ||
          !component.observable_nullspace.allFinite() ||
          (component.minimum_detectable_effect_defined &&
           !std::isfinite(component.minimum_detectable_effect_m))) {
        return false;
      }
      const Eigen::MatrixXd precision =
          component.sqrt_information.transpose() *
          component.sqrt_information;
      const double covariance_scale =
          std::max(1.0, component.covariance.norm());
      if ((component.covariance * precision * component.covariance -
           component.covariance).norm() / covariance_scale >
          config.numerical.solve_residual_tolerance) {
        return false;
      }
      component_sum += component.weight;
      component_state_sum[static_cast<size_t>(component.state)] +=
          component.weight;
    }
    if (std::abs(component_sum - 1.0) > tolerance) return false;
    for (size_t state = 0U; state < component_state_sum.size(); ++state)
      if (std::abs(component_state_sum[state] -
                   posterior.state_probability[state]) > tolerance)
        return false;
  }
  for (size_t interface_index = 0U;
       interface_index < snapshot.interfaces.size(); ++interface_index) {
    const InterfacePosterior& interface = snapshot.interfaces[interface_index];
    double probability_sum = interface.other_mass;
    if (interface.interface_id !=
            snapshot.reference.interfaces[interface_index].id ||
        interface.validity == Validity::INVALID ||
        !ValidProbability(interface.other_mass, tolerance) ||
        !interface.jump_mean_R.allFinite() ||
        !ValidateSymmetricPsd(interface.jump_covariance_R,
                              config.numerical.psd_eigenvalue_tolerance) ||
        !interface.left_query_position_R.allFinite() ||
        !interface.right_query_position_R.allFinite()) {
      return false;
    }
    for (double probability : interface.joint_probability) {
      if (!ValidProbability(probability, tolerance)) return false;
      probability_sum += probability;
    }
    if (std::abs(probability_sum - 1.0) > tolerance) return false;
  }
  for (const HazardGeometry& hazard : snapshot.hazards) {
    if (hazard.cause_id.empty() || hazard.component_id.empty() ||
        !std::isfinite(hazard.horizon_s) || !(hazard.horizon_s >= 0.0) ||
        hazard.horizon_s > config.resource.prediction_horizons_s.back() +
                               config.numerical.tie_tolerance ||
        (hazard.semantics == HazardSemantics::STRUCTURAL_DEFORMATION &&
         hazard.gdu_index >= snapshot.reference.gdus.size()) ||
        !hazard.center_R.allFinite() ||
        !ValidateSymmetricPsd(hazard.covariance_R,
                              config.numerical.psd_eigenvalue_tolerance) ||
        !std::isfinite(hazard.radius_m) || !(hazard.radius_m >= 0.0) ||
        !ValidInterval(hazard.existence, tolerance) ||
        !std::isfinite(hazard.model_envelope_m) ||
        !(hazard.model_envelope_m >= 0.0) ||
        !ValidProbability(hazard.other_mass, tolerance)) {
      return false;
    }
  }
  for (const OccupancyCell& cell : snapshot.occupancy) {
    if (!ValidInterval(cell.reference_occupancy, tolerance) ||
        !ValidInterval(cell.total_occupancy, tolerance) ||
        !ValidInterval(cell.new_occupancy, tolerance) ||
        !std::isfinite(cell.visibility) || cell.visibility < -tolerance ||
        cell.visibility > 1.0 + tolerance ||
        cell.cause_ids.size() != cell.cause_probabilities.size() ||
        cell.cause_ids.size() != cell.cause_semantics.size() ||
        cell.cause_ids.size() != cell.cause_source_ids.size()) {
      return false;
    }
    std::set<StableId> unique_causes;
    for (size_t cause_index = 0U; cause_index < cell.cause_ids.size();
         ++cause_index) {
      if (cell.cause_ids[cause_index].empty() ||
          !unique_causes.insert(cell.cause_ids[cause_index]).second ||
          !ValidInterval(cell.cause_probabilities[cause_index], tolerance))
        return false;
      for (const StableId& source : cell.cause_source_ids[cause_index])
        if (source.empty()) return false;
    }
  }
  for (const ObservationExplanation& explanation : snapshot.explanations) {
    if (explanation.cause_id.empty() ||
        !ValidInterval(explanation.probability, tolerance) ||
        !std::isfinite(explanation.protective_radius_m) ||
        explanation.protective_radius_m < 0.0) {
      return false;
    }
    if (explanation.protective_geometry_available) {
      for (double value : explanation.protective_center_R)
        if (!std::isfinite(value)) return false;
      for (double value : explanation.protective_covariance_R)
        if (!std::isfinite(value)) return false;
    }
  }
  return true;
}

bool HasReason(const std::vector<ReasonCode>& reasons, ReasonCode reason) {
  return std::find(reasons.begin(), reasons.end(), reason) != reasons.end();
}

}  // namespace

V1Pipeline::V1Pipeline(const V1Config& config)
    : config_(config), filter_(config), reference_visibility_(config),
      association_(config) {
  if (config_.config_digest.empty()) config_.config_digest = CanonicalConfigDigest(config_);
}

std::deque<FrameBundle> V1Pipeline::CropFrames(
    const std::deque<FrameBundle>& frames) const {
  std::deque<FrameBundle> cropped;
  for (const FrameBundle& source : frames) {
    FrameBundle frame = source;
    frame.points.clear();
    for (const CanonicalPoint& point : source.points)
      if (roi_.Contains(point.position_R)) frame.points.push_back(point);
    if (!frame.points.empty()) cropped.push_back(std::move(frame));
  }
  return cropped;
}

bool V1Pipeline::ValidateReferenceSelfConsistency(
    const std::deque<FrameBundle>& frames, const ReferenceSnapshot& reference,
    std::vector<ReasonCode>* reasons) const {
  if (!reference.frozen || reference.surfels.empty()) {
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_SELF_CONSISTENCY_FAILED);
    return false;
  }
  using FrameCenterMap = std::map<StableId, std::pair<Eigen::Vector3d, size_t>>;
  std::map<VoxelKey, FrameCenterMap> centers_by_cell;
  for (const FrameBundle& frame : frames) {
    for (const CanonicalPoint& point : frame.points) {
      VoxelKey key;
      if (!TryGridKey(point.position_R, config_.resource.grid_origin_m,
                      config_.resource.surfel_seed_spacing_m, &key)) {
        if (reasons) reasons->push_back(ReasonCode::REFERENCE_RESOURCE_LIMIT);
        return false;
      }
      FrameCenterMap& by_frame = centers_by_cell[key];
      auto inserted = by_frame.emplace(
          frame.frame_id,
          std::make_pair(Eigen::Vector3d::Zero(), size_t{0U}));
      inserted.first->second.first += point.position_R;
      ++inserted.first->second.second;
    }
  }
  double normalized_squared_sum = 0.0;
  size_t held_out_tests = 0U;
  std::map<StableId, size_t> tests_by_frame;
  std::map<VoxelKey, std::vector<const Surfel*>> reference_surfels_by_cell;
  for (const Surfel& surfel : reference.surfels) {
    VoxelKey key;
    if (!TryGridKey(surfel.center_R, config_.resource.grid_origin_m,
                    config_.resource.surfel_seed_spacing_m, &key)) {
      if (reasons) reasons->push_back(ReasonCode::REFERENCE_RESOURCE_LIMIT);
      return false;
    }
    reference_surfels_by_cell[key].push_back(&surfel);
  }
  const int64_t nearest_radius_cells = 2;
  for (auto& cell : centers_by_cell) {
    if (cell.second.size() < 3U) continue;
    Eigen::Vector3d all_center = Eigen::Vector3d::Zero();
    for (auto& frame_center : cell.second) {
      frame_center.second.first /=
          static_cast<double>(frame_center.second.second);
      all_center += frame_center.second.first;
    }
    all_center /= static_cast<double>(cell.second.size());
    const Surfel* nearest_surfel = nullptr;
    double nearest_distance = std::numeric_limits<double>::infinity();
    for (int64_t dx = -nearest_radius_cells; dx <= nearest_radius_cells; ++dx)
      for (int64_t dy = -nearest_radius_cells; dy <= nearest_radius_cells; ++dy)
        for (int64_t dz = -nearest_radius_cells; dz <= nearest_radius_cells; ++dz) {
          const int64_t x = static_cast<int64_t>(cell.first.x) + dx;
          const int64_t y = static_cast<int64_t>(cell.first.y) + dy;
          const int64_t z = static_cast<int64_t>(cell.first.z) + dz;
          if (x < std::numeric_limits<int32_t>::min() ||
              x > std::numeric_limits<int32_t>::max() ||
              y < std::numeric_limits<int32_t>::min() ||
              y > std::numeric_limits<int32_t>::max() ||
              z < std::numeric_limits<int32_t>::min() ||
              z > std::numeric_limits<int32_t>::max())
            continue;
          const auto nearby = reference_surfels_by_cell.find(
              VoxelKey{static_cast<int32_t>(x), static_cast<int32_t>(y),
                       static_cast<int32_t>(z)});
          if (nearby == reference_surfels_by_cell.end()) continue;
          for (const Surfel* surfel : nearby->second) {
            const double distance = (all_center - surfel->center_R).norm();
            if (distance < nearest_distance - config_.numerical.tie_tolerance ||
                (std::abs(distance - nearest_distance) <=
                     config_.numerical.tie_tolerance &&
                 (!nearest_surfel || surfel->id < nearest_surfel->id))) {
              nearest_distance = distance;
              nearest_surfel = surfel;
            }
          }
        }
    if (!nearest_surfel ||
        nearest_distance > std::sqrt(3.0) *
                               config_.resource.surfel_seed_spacing_m)
      continue;
    const Eigen::Matrix3d projector = nearest_surfel->observable_projector_R;
    const double exposure_count = static_cast<double>(
        nearest_surfel->exposure_count);
    if (!(exposure_count > 1.0)) continue;
    // surfel.covariance_R stores covariance of the multi-exposure reference
    // mean, S/n.  A held-out exposure minus the mean of the remaining n-1
    // exposures has covariance S*(1+1/(n-1)) =
    // covariance(mean)*n^2/(n-1).  Using S/n directly overstates a static
    // inconsistency by approximately n^2/(n-1).
    const Eigen::Matrix3d leave_one_covariance =
        nearest_surfel->covariance_R * exposure_count * exposure_count /
        (exposure_count - 1.0);
    const double reference_variance = std::max(
        config_.calibration.independent_point_sigma_m *
            config_.calibration.independent_point_sigma_m,
        (projector * leave_one_covariance * projector.transpose()).trace());
    for (const auto& held_out : cell.second) {
      Eigen::Vector3d other_center = Eigen::Vector3d::Zero();
      for (const auto& other : cell.second)
        if (other.first != held_out.first) other_center += other.second.first;
      other_center /= static_cast<double>(cell.second.size() - 1U);
      const Eigen::Vector3d residual = projector *
          (held_out.second.first - other_center);
      normalized_squared_sum += residual.squaredNorm() / reference_variance;
      ++held_out_tests;
      ++tests_by_frame[held_out.first];
    }
  }
  bool every_frame_tested = tests_by_frame.size() == frames.size();
  for (const FrameBundle& frame : frames)
    every_frame_tested = every_frame_tested && tests_by_frame[frame.frame_id] > 0U;
  const double normalized_rms = held_out_tests > 0U
      ? std::sqrt(normalized_squared_sum / static_cast<double>(held_out_tests))
      : std::numeric_limits<double>::infinity();
  if (!every_frame_tested ||
      normalized_rms > config_.calibration.reference_stability_sigma) {
    if (reasons)
      reasons->push_back(ReasonCode::REFERENCE_SELF_CONSISTENCY_FAILED);
    return false;
  }
  for (const Surfel& surfel : reference.surfels) {
    if (!ValidateSymmetricPsd(surfel.covariance_R,
                              config_.numerical.psd_eigenvalue_tolerance) ||
        !std::isfinite(surfel.support_radius_m) ||
        surfel.support_radius_m < 0.0 ||
        !surfel.geometry_probability.allFinite() ||
        (surfel.geometry_probability.array() < 0.0).any() ||
        std::abs(surfel.geometry_probability.sum() - 1.0) >
            config_.numerical.normalization_tolerance) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
  }
  for (const GduDefinition& gdu : reference.gdus) {
    if (!gdu.reference_bias_covariance_R.allFinite() ||
        !ValidateSymmetricPsd(gdu.reference_bias_covariance_R,
                              config_.numerical.psd_eigenvalue_tolerance)) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
  }
  return true;
}

bool V1Pipeline::TryBoundedReference(bool historical, std::vector<ReasonCode>* reasons) {
  const auto begin = std::chrono::steady_clock::now();
  const size_t minimum = static_cast<size_t>(config_.resource.reference_min_frames);
  const size_t maximum = 5U * minimum;
  // Select by time/scan count only, before observing geometry or residuals.
  // Crop only this bounded window, not the whole historical point buffer.
  std::deque<FrameBundle> selected;
  size_t new_frames = 0U;
  for (auto frame = raw_buffer_.rbegin(); frame != raw_buffer_.rend(); ++frame) {
    if (historical ? frame->stamp >= reference_cutoff_ : frame->stamp < reference_cutoff_)
      continue;
    if (!historical && frame->stamp > bounded_last_attempt_stamp_) ++new_frames;
    selected.push_front(*frame);
    // At high input rates 5N scans may span less than the minimum duration.
    // Extend only far enough to meet that duration, still within raw_buffer_'s
    // hard point/time budgets, rather than waiting forever at a fixed count.
    if (selected.size() >= maximum &&
        (selected.back().stamp - selected.front().stamp).toSec() >=
            config_.resource.reference_min_duration_s) break;
  }
  const double duration = selected.size() > 1U
      ? (selected.back().stamp - selected.front().stamp).toSec() : 0.0;
  std::ostringstream detail;
  detail << "source=" << (historical ? "recent_history" : "post_request_baseline")
         << " frames=" << selected.size() << "/min=" << minimum
         << " span_s=" << duration << "/min=" << config_.resource.reference_min_duration_s
         << " builds=" << reference_build_attempts_;
  if (selected.size() < minimum || duration < config_.resource.reference_min_duration_s ||
      (historical && (reference_cutoff_ - selected.back().stamp).toSec() >
                         config_.resource.max_update_gap_s)) {
    reference_diagnostics_ = "WAITING_DATA " + detail.str();
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_INSUFFICIENT_EXPOSURE);
    return false;
  }
  if (!historical && !bounded_last_attempt_stamp_.isZero() && new_frames < minimum) {
    // Preserve the last failure/coverage report during the retry cooldown.
    return false;
  }
  if (!historical) bounded_last_attempt_stamp_ = selected.back().stamp;
  selected = CropFrames(selected);
  detail << " roi_frames=" << selected.size();
  for (size_t i = 1U; i < selected.size(); ++i) {
    if ((selected[i].stamp - selected[i - 1U].stamp).toSec() > config_.resource.max_update_gap_s) {
      reference_diagnostics_ = "INPUT_GAP " + detail.str();
      if (reasons) reasons->push_back(ReasonCode::INPUT_TIME_INVALID);
      return false;
    }
  }
  StableIdRegistryTransaction ids;
  if (!ids.active()) {
    if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
    return false;
  }
  ++reference_build_attempts_;
  std::cerr << "[deform_v1] reference BUILDING " << detail.str()
            << " (one candidate; keep baseline static)" << std::endl;
  ReferenceBuilder builder(config_);
  ReferenceSnapshot candidate = builder.Build(selected, reference_epoch_);
  std::vector<ReasonCode> local_reasons = candidate.reasons;
  const bool consistent = candidate.frozen &&
      ValidateReferenceSelfConsistency(selected, candidate, &local_reasons);
  size_t monitorable_gdus = 0U, monitorable_surfels = 0U;
  for (const auto& gdu : candidate.gdus) {
    if (gdu.monitorable) {
      ++monitorable_gdus;
      monitorable_surfels += gdu.surfel_indices.size();
    }
  }
  const bool healthy = StableIdRegistryHealthy();
  const bool success = consistent && healthy && ids.Commit();
  std::ostringstream report;
  report << (success ? "FROZEN " : "REJECTED ") << detail.str()
         << " completed_builds=" << reference_build_attempts_
         << " surfels=" << candidate.surfels.size()
         << " monitorable_surfels=" << monitorable_surfels
         << " monitorable_gdus=" << monitorable_gdus << "/" << candidate.gdus.size()
         << " build_wall_s=" << std::chrono::duration<double>(
                std::chrono::steady_clock::now() - begin).count() << " reasons=";
  for (ReasonCode reason : local_reasons) report << static_cast<int>(reason) << ",";
  reference_diagnostics_ = report.str();
  std::cerr << "[deform_v1] reference " << reference_diagnostics_ << std::endl;
  if (!success) {
    if (reasons) {
      for (ReasonCode reason : local_reasons)
        if (!HasReason(*reasons, reason)) reasons->push_back(reason);
      if (!healthy || consistent)
        reasons->push_back(StableIdRegistryResourceExhausted()
            ? ReasonCode::RESOURCE_LIMIT : ReasonCode::INVALID_ID_COLLISION);
    }
    return false;
  }
  reference_ = std::move(candidate);
  reference_frames_ = std::move(selected);
  reference_prepared_ = true;
  reference_from_fallback_ = !historical;
  if (!historical) {
    reference_.reasons.push_back(ReasonCode::PRE_REFERENCE_CHANGE_UNOBSERVABLE);
    if (reasons) reasons->push_back(ReasonCode::PRE_REFERENCE_CHANGE_UNOBSERVABLE);
  }
  return true;
}

bool V1Pipeline::SelectHistoricalReference(std::vector<ReasonCode>* reasons) {
  if (config_.resource.bounded_reference_search &&
      trigger_source_ == TriggerSource::HUMAN_OR_EXTERNAL_TASK)
    return TryBoundedReference(true, reasons);
  StableIdRegistryTransaction search_ids;
  if (!search_ids.active()) {
    if (reasons) reasons->push_back(
        StableIdRegistryResourceExhausted()
            ? ReasonCode::RESOURCE_LIMIT
            : ReasonCode::INVALID_ID_COLLISION);
    return false;
  }
  const std::deque<FrameBundle> cropped = CropFrames(raw_buffer_);
  std::deque<FrameBundle> best_frames;
  bool found = false;
  bool registry_failed = false;
  bool registry_resource_failure = false;
  std::vector<ReasonCode> rejected_reasons;
  ReferenceBuilder builder(config_);
  for (size_t end_reverse = 0; end_reverse < cropped.size(); ++end_reverse) {
    const size_t end = cropped.size() - 1U - end_reverse;
    const ros::Time historical_cutoff =
        trigger_source_ == TriggerSource::ALGORITHM_RESIDUAL
            ? selection_support_cutoff_
            : reference_cutoff_;
    if (!(cropped[end].stamp < historical_cutoff)) continue;
    // A historical segment is evidence-eligible only if the unobserved gap to
    // the monitoring trigger can still be propagated by the registered online
    // model.  Selecting an older global-map segment would prepare a reference
    // that MONITOR_UPDATE must subsequently reject forever, while also
    // suppressing the current-dwell fallback.  Such data remains useful as
    // mapping context, but it is not a continuous before/after baseline.
    if ((reference_cutoff_ - cropped[end].stamp).toSec() >
        config_.resource.max_update_gap_s +
            config_.numerical.tie_tolerance) {
      continue;
    }
    bool gap_valid = true;
    std::vector<std::pair<ReferenceSnapshot, std::deque<FrameBundle>>> at_same_end;
    for (size_t start_reverse = 0; start_reverse <= end; ++start_reverse) {
      const size_t start = end - start_reverse;
      if (start < end && (cropped[start + 1U].stamp - cropped[start].stamp).toSec() >
                         config_.resource.max_update_gap_s) gap_valid = false;
      if (!gap_valid) break;
      const size_t count = end - start + 1U;
      const double duration = (cropped[end].stamp - cropped[start].stamp).toSec();
      if (count < static_cast<size_t>(config_.resource.reference_min_frames) ||
          duration < config_.resource.reference_min_duration_s) continue;
      std::deque<FrameBundle> candidate(cropped.begin() + static_cast<long>(start),
                                        cropped.begin() + static_cast<long>(end + 1U));
      ReferenceSnapshot snapshot = builder.Build(candidate, reference_epoch_);
      if (!StableIdRegistryHealthy()) {
        registry_resource_failure = StableIdRegistryResourceExhausted();
        registry_failed = true;
        break;
      }
      std::vector<ReasonCode> local_reasons = snapshot.reasons;
      if (snapshot.frozen && ValidateReferenceSelfConsistency(candidate, snapshot,
                                                              &local_reasons)) {
        at_same_end.push_back(std::make_pair(snapshot, candidate));
        // Residual-triggered selection must keep the first compact historical
        // window so the same residual cannot choose a more favourable
        // baseline.  An external, pre-registered task is different: its
        // stable dwell occurs before motion and may need a longer window for
        // sparse distant surfaces.  Compare only a bounded 5N family using
        // reference geometry; current deformation residuals never enter this
        // selection.
        if (trigger_source_ == TriggerSource::ALGORITHM_RESIDUAL)
          break;
      } else
        rejected_reasons.insert(rejected_reasons.end(), local_reasons.begin(),
                                local_reasons.end());
      if (trigger_source_ == TriggerSource::HUMAN_OR_EXTERNAL_TASK &&
          count >= 5U * static_cast<size_t>(
                            config_.resource.reference_min_frames))
        break;
    }
    if (registry_failed) break;
    if (!at_same_end.empty()) {
      std::sort(at_same_end.begin(), at_same_end.end(), [&](const auto& lhs, const auto& rhs) {
        if (trigger_source_ == TriggerSource::HUMAN_OR_EXTERNAL_TASK) {
          const auto monitorable_count = [](const ReferenceSnapshot& value) {
            return static_cast<size_t>(std::count_if(
                value.gdus.begin(), value.gdus.end(),
                [](const GduDefinition& gdu) { return gdu.monitorable; }));
          };
          const size_t lhs_monitorable = monitorable_count(lhs.first);
          const size_t rhs_monitorable = monitorable_count(rhs.first);
          if (lhs_monitorable != rhs_monitorable)
            return lhs_monitorable > rhs_monitorable;
          if (lhs.first.surfels.size() != rhs.first.surfels.size())
            return lhs.first.surfels.size() > rhs.first.surfels.size();
          if (lhs.second.size() != rhs.second.size())
            return lhs.second.size() < rhs.second.size();
        }
        return lhs.first.id < rhs.first.id;
      });
      best_frames = at_same_end.front().second;
      found = true;
      break;  // end stamps are visited newest first.
    }
  }
  search_ids.Rollback();
  if (registry_failed) {
    if (reasons) reasons->push_back(
        registry_resource_failure ? ReasonCode::RESOURCE_LIMIT
                                  : ReasonCode::INVALID_ID_COLLISION);
    return false;
  }
  if (!found) {
    if (rejected_reasons.empty())
      rejected_reasons.push_back(ReasonCode::REFERENCE_INSUFFICIENT_EXPOSURE);
    if (reasons) {
      for (ReasonCode reason : rejected_reasons)
        if (!HasReason(*reasons, reason)) reasons->push_back(reason);
    }
    return false;
  }
  // Candidate-window search may create many identities that are never part of
  // the frozen reference.  Rebuild only the selected window in a fresh atomic
  // collision-audit transaction so search artifacts cannot consume its bound.
  StableIdRegistryTransaction selected_ids;
  if (!selected_ids.active()) {
    if (reasons) reasons->push_back(
        StableIdRegistryResourceExhausted()
            ? ReasonCode::RESOURCE_LIMIT
            : ReasonCode::INVALID_ID_COLLISION);
    return false;
  }
  ReferenceSnapshot selected = builder.Build(best_frames, reference_epoch_);
  std::vector<ReasonCode> selected_reasons = selected.reasons;
  if (!selected.frozen ||
      !ValidateReferenceSelfConsistency(best_frames, selected,
                                        &selected_reasons) ||
      !StableIdRegistryHealthy() || !selected_ids.Commit()) {
    if (reasons) {
      if (!StableIdRegistryHealthy())
        reasons->push_back(StableIdRegistryResourceExhausted()
                               ? ReasonCode::RESOURCE_LIMIT
                               : ReasonCode::INVALID_ID_COLLISION);
      for (ReasonCode reason : selected_reasons)
        if (!HasReason(*reasons, reason)) reasons->push_back(reason);
    }
    return false;
  }
  reference_ = selected; reference_frames_ = best_frames;
  reference_prepared_ = true; reference_from_fallback_ = false;
  return true;
}

bool V1Pipeline::TryFallbackReference(std::vector<ReasonCode>* reasons) {
  if (config_.resource.bounded_reference_search &&
      trigger_source_ == TriggerSource::HUMAN_OR_EXTERNAL_TASK)
    return TryBoundedReference(false, reasons);
  StableIdRegistryTransaction search_ids;
  if (!search_ids.active()) {
    if (reasons) reasons->push_back(
        StableIdRegistryResourceExhausted()
            ? ReasonCode::RESOURCE_LIMIT
            : ReasonCode::INVALID_ID_COLLISION);
    return false;
  }
  const std::deque<FrameBundle> cropped = CropFrames(raw_buffer_);
  const bool current_dwell =
      trigger_source_ == TriggerSource::EXTERNAL_TASK_CURRENT_DWELL;
  const size_t minimum_frames = static_cast<size_t>(
      config_.resource.reference_min_frames);
  ReferenceBuilder builder(config_);
  // A non-repetitive LiDAR may not revisit every 10 cm cell in only N scans.
  // The explicit stationary-dwell contract therefore freezes one fixed 5N
  // window.  Building all nested N..5N candidates was both a hidden
  // reference-selection rule and quadratic work; on a no-prior full sensor
  // field it rebuilt essentially the same 15k-point scene 25 times.  The
  // newest complete 5N window is deterministic at each attempt and depends
  // only on pre-motion reference evidence, never on a monitoring residual.
  const size_t current_dwell_frames = 5U * minimum_frames;
  if (current_dwell) {
    std::deque<FrameBundle> eligible;
    for (const FrameBundle& frame : cropped)
      if (frame.stamp >= reference_cutoff_) eligible.push_back(frame);
    if (eligible.size() < current_dwell_frames) return false;
    std::deque<FrameBundle> selected_frames(
        eligible.end() - static_cast<long>(current_dwell_frames),
        eligible.end());
    for (size_t index = 1U; index < selected_frames.size(); ++index) {
      if ((selected_frames[index].stamp -
           selected_frames[index - 1U].stamp).toSec() >
          config_.resource.max_update_gap_s) {
        return false;
      }
    }
    if ((selected_frames.back().stamp -
         selected_frames.front().stamp).toSec() <
        config_.resource.reference_min_duration_s) {
      return false;
    }
    ReferenceSnapshot selected =
        builder.Build(selected_frames, reference_epoch_);
    std::vector<ReasonCode> selected_reasons = selected.reasons;
    const bool consistent =
        selected.frozen &&
        ValidateReferenceSelfConsistency(selected_frames, selected,
                                         &selected_reasons);
    const bool registry_healthy = StableIdRegistryHealthy();
    if (!consistent || !registry_healthy || !search_ids.Commit()) {
      if (reasons) {
        if (!registry_healthy)
          reasons->push_back(StableIdRegistryResourceExhausted()
                                 ? ReasonCode::RESOURCE_LIMIT
                                 : ReasonCode::INVALID_ID_COLLISION);
        for (ReasonCode reason : selected_reasons)
          if (!HasReason(*reasons, reason)) reasons->push_back(reason);
      }
      return false;
    }
    reference_ = selected;
    reference_frames_ = selected_frames;
    reference_prepared_ = true;
    reference_from_fallback_ = true;
    return true;
  }
  std::deque<FrameBundle> selected_frames;
  bool found = false;
  bool registry_failed = false;
  bool registry_resource_failure = false;
  for (size_t end = 0; end < cropped.size(); ++end) {
    if (cropped[end].stamp < reference_cutoff_) continue;
    std::vector<std::pair<ReferenceSnapshot, std::deque<FrameBundle>>> at_same_end;
    bool gap_valid = true;
    for (size_t start_reverse = 0; start_reverse <= end; ++start_reverse) {
      const size_t start = end - start_reverse;
      if (cropped[start].stamp < reference_cutoff_) break;
      if (start < end && (cropped[start + 1U].stamp - cropped[start].stamp).toSec() >
                         config_.resource.max_update_gap_s) gap_valid = false;
      if (!gap_valid) break;
      const size_t candidate_count = end - start + 1U;
      if (candidate_count < minimum_frames ||
          (cropped[end].stamp - cropped[start].stamp).toSec() <
              config_.resource.reference_min_duration_s) continue;
      std::deque<FrameBundle> candidate(cropped.begin() + static_cast<long>(start),
                                        cropped.begin() + static_cast<long>(end + 1U));
      ReferenceSnapshot snapshot = builder.Build(candidate, reference_epoch_);
      if (!StableIdRegistryHealthy()) {
        registry_resource_failure = StableIdRegistryResourceExhausted();
        registry_failed = true;
        break;
      }
      std::vector<ReasonCode> local_reasons;
      if (snapshot.frozen && ValidateReferenceSelfConsistency(candidate, snapshot,
                                                              &local_reasons)) {
        at_same_end.push_back(std::make_pair(snapshot, candidate));
        break;
      }
    }
    if (registry_failed) break;
    if (!at_same_end.empty()) {
      selected_frames = at_same_end.front().second;
      found = true;
      break;  // end stamps are visited oldest first: first dwell window.
    }
  }
  search_ids.Rollback();
  if (registry_failed) {
    if (reasons) reasons->push_back(
        registry_resource_failure ? ReasonCode::RESOURCE_LIMIT
                                  : ReasonCode::INVALID_ID_COLLISION);
    return false;
  }
  if (!found) return false;

  StableIdRegistryTransaction selected_ids;
  if (!selected_ids.active()) {
    if (reasons) reasons->push_back(
        StableIdRegistryResourceExhausted()
            ? ReasonCode::RESOURCE_LIMIT
            : ReasonCode::INVALID_ID_COLLISION);
    return false;
  }
  ReferenceSnapshot selected = builder.Build(selected_frames, reference_epoch_);
  std::vector<ReasonCode> selected_reasons;
  if (!selected.frozen ||
      !ValidateReferenceSelfConsistency(selected_frames, selected,
                                        &selected_reasons) ||
      !StableIdRegistryHealthy() || !selected_ids.Commit()) {
    if (reasons) {
      if (!StableIdRegistryHealthy())
        reasons->push_back(StableIdRegistryResourceExhausted()
                               ? ReasonCode::RESOURCE_LIMIT
                               : ReasonCode::INVALID_ID_COLLISION);
      for (ReasonCode reason : selected_reasons)
        if (!HasReason(*reasons, reason)) reasons->push_back(reason);
    }
    return false;
  }
  reference_ = selected;
  reference_frames_ = selected_frames;
  reference_prepared_ = true;
  reference_from_fallback_ = true;
  if (trigger_source_ != TriggerSource::EXTERNAL_TASK_CURRENT_DWELL) {
    reference_.reasons.push_back(
        ReasonCode::PRE_REFERENCE_CHANGE_UNOBSERVABLE);
    if (reasons)
      reasons->push_back(ReasonCode::PRE_REFERENCE_CHANGE_UNOBSERVABLE);
  }
  return true;
}

PhaseResponse V1Pipeline::SetPhase(const PhaseRequest& request) {
  PhaseResponse response; response.phase = phase_; response.reference_epoch = reference_epoch_;
  if (request.session_id != config_.resource.session_id || request.effective_stamp.isZero() ||
      request.selection_support_cutoff.isZero() ||
      request.selection_support_cutoff > request.effective_stamp ||
      !request.roi.minimum_R.allFinite() || !request.roi.maximum_R.allFinite() ||
      (request.roi.minimum_R.array() >= request.roi.maximum_R.array()).any()) {
    response.reasons.push_back(ReasonCode::INPUT_SCHEMA_INVALID); return response;
  }
  if (!request.expected_previous_snapshot.empty() &&
      request.expected_previous_snapshot != last_snapshot_.common.snapshot_id) {
    response.reasons.push_back(ReasonCode::QUERY_CONTRACT_INVALID); return response;
  }
  if (request.target == MonitoringPhase::MONITOR_REFERENCE &&
      phase_ == MonitoringPhase::MAPPING_CONTEXT) {
    ++reference_epoch_;
    roi_ = request.roi; reference_cutoff_ = request.effective_stamp;
    selection_support_cutoff_ = request.selection_support_cutoff;
    trigger_source_ = request.trigger_source; reference_prepared_ = false;
    bounded_last_attempt_stamp_ = ros::Time();
    reference_build_attempts_ = 0U;
    reference_diagnostics_.clear();
    first_monitor_update_pending_ = false;
    reference_from_fallback_ = false; reference_ = ReferenceSnapshot();
    ClearMeasurementReferenceCache();
    reference_frames_.clear(); phase_ = MonitoringPhase::MONITOR_REFERENCE;
    // A current-dwell task deliberately establishes its baseline after this
    // request.  Historical context may have a different crop or robot-motion
    // regime and is therefore not exchangeable reference evidence.
    if (trigger_source_ != TriggerSource::EXTERNAL_TASK_CURRENT_DWELL)
      SelectHistoricalReference(&response.reasons);
    response.accepted = true; response.phase = phase_;
    response.reference_epoch = reference_epoch_;
    if (reference_prepared_) response.reference_snapshot_id = reference_.id;
    return response;
  }
  if (request.target == MonitoringPhase::MONITOR_UPDATE &&
      phase_ == MonitoringPhase::MONITOR_REFERENCE) {
    if (request.selection_support_cutoff != selection_support_cutoff_) {
      response.reasons.push_back(ReasonCode::QUERY_CONTRACT_INVALID);
      return response;
    }
    if (!reference_prepared_) {
      response.reasons.push_back(ReasonCode::REFERENCE_NOT_FROZEN); return response;
    }
    if ((request.effective_stamp - reference_.end_stamp).toSec() < 0.0 ||
        (has_accepted_frame_ &&
         (request.effective_stamp - last_accepted_stamp_).toSec() >
             config_.resource.max_update_gap_s)) {
      response.reasons.push_back(ReasonCode::INPUT_TIME_INVALID); return response;
    }
    // The frozen-reference status and this service request travel through the
    // same single-threaded ROS callback queue as FAST-LIO frames.  Therefore
    // additional phase-1 frames may have been accepted before the request is
    // serviced even when the client calls immediately.  Initialize the
    // dynamic clock at the newest accepted phase-1 stamp.  Geometry remains
    // anchored to the frozen reference, so the first phase-2 measurement can
    // still detect any displacement accrued in this unobserved hand-off; only
    // velocity over the hand-off interval is intentionally not inferred.
    const ros::Time initialization_stamp =
        has_accepted_frame_ && last_accepted_stamp_ > reference_.end_stamp
            ? last_accepted_stamp_ : reference_.end_stamp;
    // All entries depend only on the now-frozen reference.  Prepare them
    // before the first monitoring transaction so no current residual can
    // influence the index, exclusion mask, or deterministic surfel order.
    if (!PrepareMeasurementReferenceCache(&response.reasons))
      return response;
    UacGaussianSumFilter candidate_filter(config_);
    OccupancyEventGraph candidate_occupancy(config_);
    StableIdRegistryTransaction initialization_ids;
    if (!initialization_ids.active()) {
      response.reasons.push_back(StableIdRegistryResourceExhausted()
                                     ? ReasonCode::RESOURCE_LIMIT
                                     : ReasonCode::INVALID_ID_COLLISION);
      return response;
    }
    if (!candidate_filter.Initialize(reference_, initialization_stamp, &response.reasons) ||
        !candidate_occupancy.BuildReference(reference_frames_, reference_,
                                            reference_epoch_,
                                            &response.reasons) ||
        !StableIdRegistryHealthy() || !initialization_ids.Commit()) {
      if (!StableIdRegistryHealthy())
        response.reasons.push_back(StableIdRegistryResourceExhausted()
                                       ? ReasonCode::RESOURCE_LIMIT
                                       : ReasonCode::INVALID_ID_COLLISION);
      return response;
    }
    // Initialization candidates are no longer needed after the complete
    // reference/filter transaction has committed.  Transfer their ownership
    // instead of deep-copying the frozen caches a second time.
    filter_ = std::move(candidate_filter);
    reference_visibility_ = std::move(candidate_occupancy);
    phase_ = MonitoringPhase::MONITOR_UPDATE;
    first_monitor_update_pending_ = true;
    response.accepted = true; response.phase = phase_;
    response.reference_epoch = reference_epoch_;
    response.reference_snapshot_id = reference_.id;
    return response;
  }
  if (request.target == MonitoringPhase::CLOSED && phase_ != MonitoringPhase::CLOSED) {
    phase_ = MonitoringPhase::CLOSED; response.accepted = true; response.phase = phase_;
    return response;
  }
  response.reasons.push_back(ReasonCode::INPUT_SCHEMA_INVALID);
  return response;
}

std::vector<V1Pipeline::CurrentPatch, Eigen::aligned_allocator<V1Pipeline::CurrentPatch>>
V1Pipeline::BuildPatches(const FrameBundle& frame, PointRole role) const {
  std::map<VoxelKey, PatchAccumulator> accumulators;
  for (const CanonicalPoint& point : frame.points) {
    if (point.role != role || !roi_.Contains(point.position_R)) continue;
    VoxelKey key;
    if (!TryGridKey(point.position_R, config_.resource.grid_origin_m,
                    config_.resource.surfel_seed_spacing_m, &key))
      return {};
    accumulators[key].points.push_back(point.position_R);
    accumulators[key].evidence_ids.push_back(point.evidence_id);
  }
  std::vector<CurrentPatch, Eigen::aligned_allocator<CurrentPatch>> patches;
  for (const auto& item : accumulators) {
    if (item.second.points.empty()) continue;
    CurrentPatch patch; patch.key = item.first; patch.role = role;
    for (const Eigen::Vector3d& point : item.second.points) patch.center_R += point;
    patch.center_R /= static_cast<double>(item.second.points.size());
    Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d& point : item.second.points) {
      const Eigen::Vector3d delta = point - patch.center_R;
      scatter += delta * delta.transpose();
    }
    if (item.second.points.size() > 1U)
      scatter /= static_cast<double>(item.second.points.size() - 1U);
    patch.centroid_covariance_R = scatter /
        static_cast<double>(std::max<size_t>(1U, item.second.points.size()));
    patch.centroid_covariance_R.diagonal().array() +=
        config_.calibration.independent_point_sigma_m *
        config_.calibration.independent_point_sigma_m /
        static_cast<double>(item.second.points.size());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(scatter);
    if (eigen.info() == Eigen::Success) {
      const Eigen::Vector4d probability = PatchGeometryProbability(
          eigen.eigenvalues(), config_.calibration);
      const Eigen::Vector3d normal = eigen.eigenvectors().col(0);
      const Eigen::Vector3d tangent = eigen.eigenvectors().col(2);
      patch.projector_R = probability[0] * normal * normal.transpose() +
          probability[1] * (Eigen::Matrix3d::Identity() - tangent * tangent.transpose()) +
          probability[2] * Eigen::Matrix3d::Identity();
    }
    patch.count = static_cast<uint32_t>(item.second.points.size());
    CanonicalBytes fields;
    fields.AddId(frame.frame_id); fields.AddU8(static_cast<uint8_t>(role));
    fields.AddI64(item.first.x); fields.AddI64(item.first.y); fields.AddI64(item.first.z);
    std::vector<StableId> ids = item.second.evidence_ids; std::sort(ids.begin(), ids.end());
    for (const StableId& id : ids) fields.AddId(id);
    patch.id = MakeStableId("current-patch-v1", fields);
    patches.push_back(patch);
  }
  std::sort(patches.begin(), patches.end(), [](const CurrentPatch& lhs,
                                               const CurrentPatch& rhs) {
    return lhs.id < rhs.id;
  });
  return patches;
}

void V1Pipeline::AssignPatchOwners(
    std::vector<CurrentPatch, Eigen::aligned_allocator<CurrentPatch>>* patches) const {
  if (!patches) return;
  for (CurrentPatch& patch : *patches) {
    const double ownership_resolution_m =
        config_.calibration.reference_stability_sigma *
        config_.calibration.independent_point_sigma_m;
    const PatchOwnership ownership = reference_owner_index_.QueryOwner(
        reference_, patch.center_R,
        config_.resource.graph_neighbor_radius_m,
        config_.numerical.tie_tolerance, ownership_resolution_m);
    patch.owner_surfel = ownership.owner_surfel;
    // Only a distance tie within the calibrated spatial resolution is
    // unresolved.  Merely having another local GDU inside the association
    // radius is expected after spatial tiling and must not discard all
    // control/association/measurement evidence.
    patch.cross_gdu_ambiguous = ownership.cross_gdu_ambiguous;
  }
}

void V1Pipeline::ClearMeasurementReferenceCache() {
  reference_owner_index_ = ReferenceSurfelSpatialIndex();
  pose_control_index_ = ReferenceSurfelSpatialIndex();
  pose_control_excluded_cache_.clear();
  gdu_surfels_by_id_cache_.clear();
  measurement_reference_cache_valid_ = false;
  frozen_reference_numerically_validated_ = false;
}

bool V1Pipeline::PrepareMeasurementReferenceCache(
    std::vector<ReasonCode>* reasons) {
  ClearMeasurementReferenceCache();
  if (!reference_.frozen || reference_.surfels.empty()) {
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_NOT_FROZEN);
    return false;
  }
  // This O(N-surfel) audit belongs to the freeze boundary. reference_ cannot
  // change during MONITOR_UPDATE, so repeating eigenvalue/finite checks in
  // every snapshot transaction adds work without adding a failure mode.
  if (!ValidateFrozenReferenceNumerical(reference_, config_)) {
    if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    return false;
  }
  if (!reference_owner_index_.BuildOwnerCandidateCache(
          reference_, config_.resource.grid_origin_m,
          config_.resource.surfel_seed_spacing_m,
          config_.resource.graph_neighbor_radius_m,
          config_.numerical.tie_tolerance)) {
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_RESOURCE_LIMIT);
    return false;
  }
  // The exclusion radius is deliberately much wider than the association
  // radius.  A second exact index with one exclusion-radius cell avoids
  // visiting millions of 10 cm cells while retaining the final Euclidean
  // acceptance test.
  const double exclusion_cell_size_m = std::max(
      config_.resource.pose_control_exclusion_radius_m,
      config_.resource.surfel_seed_spacing_m);
  if (!pose_control_index_.Build(reference_, config_.resource.grid_origin_m,
                                 exclusion_cell_size_m)) {
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_RESOURCE_LIMIT);
    ClearMeasurementReferenceCache();
    return false;
  }
  pose_control_excluded_cache_.resize(reference_.gdus.size());
  gdu_surfels_by_id_cache_.resize(reference_.gdus.size());
  for (uint32_t target = 0U; target < reference_.gdus.size(); ++target) {
    pose_control_excluded_cache_[target] =
        pose_control_index_.PoseControlExclusionMask(
            reference_, target,
            config_.resource.pose_control_exclusion_radius_m);
    gdu_surfels_by_id_cache_[target] =
        reference_.gdus[target].surfel_indices;
    std::sort(gdu_surfels_by_id_cache_[target].begin(),
              gdu_surfels_by_id_cache_[target].end(),
              [&](uint32_t lhs, uint32_t rhs) {
                if (lhs >= reference_.surfels.size() ||
                    rhs >= reference_.surfels.size()) {
                  return lhs < rhs;
                }
                return reference_.surfels[lhs].id <
                       reference_.surfels[rhs].id;
              });
  }
  measurement_reference_cache_valid_ = true;
  frozen_reference_numerically_validated_ = true;
  return true;
}

bool V1Pipeline::BuildMeasurements(
    const FrameBundle& frame, const UacGaussianSumFilter& predicted_filter,
    std::vector<std::vector<AssociationMeasurementSet>>* measurements_by_gdu,
    std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>* nuisance_by_gdu,
    std::vector<ObservationExplanation>* explanations,
    std::vector<ReasonCode>* reasons,
    bool include_unmonitorable_matching_for_test,
    MeasurementBuildStats* build_stats) const {
  if (!measurements_by_gdu || !nuisance_by_gdu || !explanations) return false;
  if (build_stats) *build_stats = MeasurementBuildStats();
  if (!measurement_reference_cache_valid_ ||
      pose_control_excluded_cache_.size() != reference_.gdus.size() ||
      gdu_surfels_by_id_cache_.size() != reference_.gdus.size()) {
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_NOT_FROZEN);
    return false;
  }
  explanations->clear();
  auto pose_control_patches = BuildPatches(frame, PointRole::POSE_CONTROL);
  auto association_patches = BuildPatches(frame, PointRole::ASSOCIATION);
  auto measurement_patches = BuildPatches(frame, PointRole::MEASUREMENT);
  if (pose_control_patches.size() > config_.resource.max_tiles ||
      association_patches.size() > config_.resource.max_tiles ||
      measurement_patches.size() > config_.resource.max_tiles) {
    if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
    return false;
  }
  const bool association_support_available = !association_patches.empty();
  const bool measurement_support_available = !measurement_patches.empty();
  if ((!association_support_available || !measurement_support_available) && reasons)
    reasons->push_back(ReasonCode::INSUFFICIENT_SUPPORT);
  AssignPatchOwners(&pose_control_patches);
  AssignPatchOwners(&association_patches);
  AssignPatchOwners(&measurement_patches);
  DisjointPatchDescriptors association_descriptors;
  DisjointPatchDescriptors measurement_descriptors;
  std::map<StableId, const CurrentPatch*> association_patch_by_id;
  std::map<StableId, const CurrentPatch*> measurement_patch_by_id;
  std::vector<std::vector<const CurrentPatch*>> association_patches_by_surfel(
      reference_.surfels.size());
  for (const CurrentPatch& patch : association_patches) {
    if (patch.owner_surfel >= reference_.surfels.size() ||
        patch.cross_gdu_ambiguous)
      continue;
    const uint32_t owner_gdu =
        reference_.surfels[patch.owner_surfel].gdu_index;
    if (owner_gdu >= reference_.gdus.size() ||
        (!include_unmonitorable_matching_for_test &&
         !reference_.gdus[owner_gdu].monitorable)) {
      continue;
    }
    DisjointPatchDescriptor descriptor;
    descriptor.id = patch.id;
    descriptor.gdu_index = owner_gdu;
    descriptor.center_R = patch.center_R;
    association_descriptors.push_back(descriptor);
    association_patch_by_id.emplace(patch.id, &patch);
    association_patches_by_surfel[patch.owner_surfel].push_back(&patch);
  }
  for (const CurrentPatch& patch : measurement_patches) {
    if (patch.owner_surfel >= reference_.surfels.size() ||
        patch.cross_gdu_ambiguous)
      continue;
    const uint32_t owner_gdu =
        reference_.surfels[patch.owner_surfel].gdu_index;
    if (owner_gdu >= reference_.gdus.size() ||
        (!include_unmonitorable_matching_for_test &&
         !reference_.gdus[owner_gdu].monitorable)) {
      continue;
    }
    DisjointPatchDescriptor descriptor;
    descriptor.id = patch.id;
    descriptor.gdu_index = owner_gdu;
    descriptor.center_R = patch.center_R;
    measurement_descriptors.push_back(descriptor);
    measurement_patch_by_id.emplace(patch.id, &patch);
  }
  DisjointPatchMatchingStats matching_stats;
  if (build_stats) {
    build_stats->association_descriptor_count = association_descriptors.size();
    build_stats->measurement_descriptor_count = measurement_descriptors.size();
  }
  const std::map<StableId, StableId> paired_measurement_id =
      MatchDisjointPatchPairs(
          association_descriptors, measurement_descriptors,
          2.0 * config_.resource.surfel_seed_spacing_m,
          config_.numerical.tie_tolerance,
          build_stats ? &matching_stats : nullptr);
  if (build_stats)
    build_stats->matching_distance_evaluations =
        matching_stats.distance_evaluations;
  measurements_by_gdu->assign(reference_.gdus.size(), {});
  nuisance_by_gdu->assign(reference_.gdus.size(), FrameNuisance());
  std::vector<std::set<StableId>> measurement_patch_ids_by_gdu(
      reference_.gdus.size());
  // Association and control eligibility read only previous-state masses and
  // current mixture field moments.  They must not recompute the full
  // fixed-lag/MDE/QMC posterior that is required only for the published
  // detection product after the measurement update.
  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>>
      predicted_posterior;
  if (!predicted_filter.PredictiveMixture(
          0.0, &predicted_posterior, reasons) ||
      predicted_posterior.size() != reference_.gdus.size()) {
    if (reasons && predicted_posterior.size() != reference_.gdus.size())
      reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    return false;
  }
  const auto correction_rotation = [](const Eigen::Matrix<double, 6, 1>& xi)
      -> Eigen::Matrix3d {
    const Eigen::Vector3d omega = xi.head<3>();
    const double angle = omega.norm();
    if (!(angle > 0.0)) return Eigen::Matrix3d::Identity().eval();
    return Eigen::AngleAxisd(angle, omega / angle).toRotationMatrix();
  };
  const auto corrected_point = [&](const Eigen::Vector3d& point,
                                   const FrameNuisance& nuisance) {
    return frame.pose.origin_L_R +
        correction_rotation(nuisance.pose_correction) *
            (point - frame.pose.origin_L_R) +
        nuisance.pose_correction.tail<3>();
  };

  // Per-GDU control-only SE(3) correction. FAST-LIO supplies the initial
  // registered pose and correspondences only; the GDU's frozen spatial
  // neighbourhood and all ASSOCIATION/MEASUREMENT points are excluded from
  // this solve.
  for (uint32_t target = 0; target < reference_.gdus.size(); ++target) {
    FrameNuisance& nuisance = (*nuisance_by_gdu)[target];
    CanonicalBytes nuisance_fields;
    nuisance_fields.AddId(frame.frame_id);
    // Every leave-one-GDU estimate is a view of the same physical pose
    // error of this LiDAR frame.  The mean/covariance remains target-specific
    // because its control set excludes that target, but the correlation id
    // must be shared across GDUs so no consumer treats them as independent.
    nuisance_fields.AddString("control-only-se3-shared-frame-correlation-v2");
    nuisance.nuisance_correlation_id =
        MakeStableId("nuisance-correlation-v1", nuisance_fields);
    if (!reference_.gdus[target].monitorable) continue;
    if (config_.resource.upstream_stationary_pose_freeze_verified) {
      // The upstream stationary adapter applies one frozen body-to-world
      // transform to every reference and monitoring scan.  That common rigid
      // transform cancels from relative deformation.  Estimating another
      // GDU-local SE(3) correction here would use the same geometry twice.
      nuisance.pose_covariance.setZero();
      nuisance.pose_correction.setZero();
      nuisance.control_patch_count = 0U;
      nuisance.control_registration_valid = true;
      nuisance.deformation_measurement_valid = true;
      nuisance.pose_temporal_bound_applied = false;
      continue;
    }
    std::vector<const CurrentPatch*> controls;
    for (const CurrentPatch& patch : pose_control_patches) {
      if (patch.owner_surfel >= reference_.surfels.size() ||
          patch.cross_gdu_ambiguous) continue;
      if (pose_control_excluded_cache_[target][patch.owner_surfel]) continue;
      const uint32_t owner_gdu =
          reference_.surfels[patch.owner_surfel].gdu_index;
      if (owner_gdu >= predicted_posterior.size()) continue;
      if (reference_.surfels[patch.owner_surfel]
              .observable_projector_R.trace() <= 0.5)
        continue;
      // Stability is read from the predicted (previous-data) posterior, never
      // from the current residual used by this registration.  Requiring the
      // clearing threshold here creates a cold-start deadlock because the
      // initialized U probability is intentionally below 0.99.  A control
      // carrier is eligible when U is the unique dominant previous state;
      // DETECTED/CLEARED admission remains governed by the stricter task-power
      // rules and is not weakened by this registration choice.
      const std::array<double, 3>& state_probability =
          predicted_posterior[owner_gdu].state_probability;
      if (!(state_probability[0] >
            std::max(state_probability[1], state_probability[2]) +
                config_.numerical.tie_tolerance))
        continue;
      controls.push_back(&patch);
    }
    std::sort(controls.begin(), controls.end(),
              [](const CurrentPatch* lhs, const CurrentPatch* rhs) {
                return lhs->id < rhs->id;
              });
    if (controls.size() > static_cast<size_t>(
            config_.resource.max_patches_per_gdu))
      controls.resize(static_cast<size_t>(
          config_.resource.max_patches_per_gdu));
    nuisance.control_patch_count = static_cast<uint32_t>(controls.size());
    if (controls.size() < static_cast<size_t>(
            config_.resource.pose_control_min_patches)) {
      if (reasons) reasons->push_back(ReasonCode::POSE_CONTROL_INSUFFICIENT);
      continue;
    }
    Eigen::Matrix3d correction_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d correction_t = Eigen::Vector3d::Zero();
    Eigen::MatrixXd final_design;
    Eigen::VectorXd final_residual;
    bool solve_valid = true;
    for (int iteration = 0; iteration < 3; ++iteration) {
      final_design = Eigen::MatrixXd::Zero(
          3 * static_cast<int>(controls.size()), 6);
      final_residual = Eigen::VectorXd::Zero(
          3 * static_cast<int>(controls.size()));
      const Eigen::Matrix3d rotation = correction_R;
      for (size_t i = 0; i < controls.size(); ++i) {
        const CurrentPatch& patch = *controls[i];
        const Surfel& surfel = reference_.surfels[patch.owner_surfel];
        const Eigen::Vector3d rotated =
            rotation * (patch.center_R - frame.pose.origin_L_R);
        const Eigen::Vector3d position = frame.pose.origin_L_R + rotated +
            correction_t;
        const Eigen::Vector3d residual = position - surfel.center_R;
        const double norm = residual.norm();
        const double huber = norm <=
            config_.calibration.pose_control_huber_scale_m ? 1.0 :
            config_.calibration.pose_control_huber_scale_m /
                std::max(norm, config_.numerical.probability_floor);
        const double sigma = std::sqrt(std::max(
            config_.numerical.cholesky_jitter_initial,
            (patch.centroid_covariance_R + surfel.covariance_R).trace() /
                3.0));
        const double whitening = std::sqrt(huber) / sigma;
        final_design.block<3, 3>(3 * static_cast<int>(i), 0) =
            -whitening * Skew(rotated);
        final_design.block<3, 3>(3 * static_cast<int>(i), 3) =
            whitening * Eigen::Matrix3d::Identity();
        final_residual.segment<3>(3 * static_cast<int>(i)) =
            whitening * residual;
      }
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(
          final_design, Eigen::ComputeThinU | Eigen::ComputeThinV);
      if (svd.info() != Eigen::Success || svd.rank() != 6 ||
          !(svd.singularValues()[5] > 0.0) ||
          svd.singularValues()[0] / svd.singularValues()[5] >
              config_.calibration.pose_control_max_condition_number) {
        solve_valid = false;
        break;
      }
      const Eigen::Matrix<double, 6, 1> increment =
          svd.solve(-final_residual);
      if (!increment.allFinite()) {
        solve_valid = false;
        break;
      }
      const Eigen::Vector3d incremental_rotation = increment.head<3>();
      const double incremental_angle = incremental_rotation.norm();
      if (incremental_angle > 0.0) {
        correction_R = Eigen::AngleAxisd(
            incremental_angle,
            incremental_rotation / incremental_angle).toRotationMatrix() *
            correction_R;
      }
      correction_t += increment.tail<3>();
      if (increment.norm() <= config_.numerical.solve_residual_tolerance)
        break;
    }
    if (!solve_valid) {
      if (reasons) reasons->push_back(ReasonCode::POSE_CONTROL_INSUFFICIENT);
      continue;
    }
    // Re-linearize once at the accepted correction.  The last iteration's
    // design/residual were evaluated before its increment was applied; using
    // them for covariance would violate the documented "final residual"
    // covariance and make the result depend on whether convergence happened
    // on iteration two or three.
    final_design = Eigen::MatrixXd::Zero(
        3 * static_cast<int>(controls.size()), 6);
    final_residual = Eigen::VectorXd::Zero(
        3 * static_cast<int>(controls.size()));
    const Eigen::Matrix3d final_rotation = correction_R;
    for (size_t i = 0; i < controls.size(); ++i) {
      const CurrentPatch& patch = *controls[i];
      const Surfel& surfel = reference_.surfels[patch.owner_surfel];
      const Eigen::Vector3d rotated =
          final_rotation * (patch.center_R - frame.pose.origin_L_R);
      const Eigen::Vector3d position = frame.pose.origin_L_R + rotated +
          correction_t;
      const Eigen::Vector3d residual = position - surfel.center_R;
      const double residual_norm = residual.norm();
      const double huber = residual_norm <=
          config_.calibration.pose_control_huber_scale_m ? 1.0 :
          config_.calibration.pose_control_huber_scale_m /
              std::max(residual_norm, config_.numerical.probability_floor);
      const double sigma = std::sqrt(std::max(
          config_.numerical.cholesky_jitter_initial,
          (patch.centroid_covariance_R + surfel.covariance_R).trace() /
              3.0));
      const double whitening = std::sqrt(huber) / sigma;
      final_design.block<3, 3>(3 * static_cast<int>(i), 0) =
          -whitening * Skew(rotated);
      final_design.block<3, 3>(3 * static_cast<int>(i), 3) =
          whitening * Eigen::Matrix3d::Identity();
      final_residual.segment<3>(3 * static_cast<int>(i)) =
          whitening * residual;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> covariance_svd(
        final_design, Eigen::ComputeThinV);
    if (covariance_svd.info() != Eigen::Success ||
        covariance_svd.rank() != 6 ||
        !(covariance_svd.singularValues()[5] > 0.0) ||
        covariance_svd.singularValues()[0] /
                covariance_svd.singularValues()[5] >
            config_.calibration.pose_control_max_condition_number) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
    const double residual_variance = std::max(
        config_.numerical.cholesky_jitter_initial,
        final_residual.squaredNorm() /
            static_cast<double>(std::max<int>(1, final_design.rows() - 6)));
    Eigen::Matrix<double, 6, 6> diagonal =
        Eigen::Matrix<double, 6, 6>::Zero();
    for (int mode = 0; mode < 6; ++mode)
      diagonal(mode, mode) = residual_variance /
          std::max(config_.numerical.cholesky_jitter_initial,
                   covariance_svd.singularValues()[mode] *
                       covariance_svd.singularValues()[mode]);
    const Eigen::AngleAxisd final_angle_axis(correction_R);
    nuisance.pose_correction.head<3>() = final_angle_axis.angle() *
        final_angle_axis.axis();
    nuisance.pose_correction.tail<3>() = correction_t;
    nuisance.pose_covariance = covariance_svd.matrixV() * diagonal *
        covariance_svd.matrixV().transpose();
    nuisance.control_registration_valid = true;
    nuisance.deformation_measurement_valid = true;
    nuisance.pose_temporal_bound_applied = false;
  }

  auto append_protective_explanation = [&](const CurrentPatch& patch,
                                           const std::string& provenance) {
    ObservationExplanation explanation;
    explanation.source_evidence_id = patch.id;
    explanation.semantics = AssociationSemantics::OTHER;
    explanation.probability.lower = config_.calibration.validated ? 1.0 : 0.0;
    explanation.probability.mean = 1.0;
    explanation.probability.upper = 1.0;
    explanation.probability.calibrated = config_.calibration.validated;
    explanation.probability.mean_identified = true;
    explanation.committed_to_posterior = false;
    explanation.protective_geometry_available = true;
    explanation.cross_gdu_assignment_envelope =
        provenance == "cross-gdu-assignment-envelope-v1";
    for (int axis = 0; axis < 3; ++axis)
      explanation.protective_center_R[static_cast<size_t>(axis)] =
          patch.center_R[axis];
    for (int row = 0; row < 3; ++row)
      for (int column = 0; column < 3; ++column)
        explanation.protective_covariance_R[
            static_cast<size_t>(3 * row + column)] =
            patch.centroid_covariance_R(row, column);
    explanation.protective_radius_m =
        config_.resource.graph_neighbor_radius_m;
    VoxelKey explanation_key;
    const bool key_valid = TryGridKey(
        patch.center_R, config_.resource.grid_origin_m,
        config_.resource.surfel_seed_spacing_m, &explanation_key);
    CanonicalBytes fields;
    if (key_valid) {
      // Diagnostic identity only: this groups unassigned patches without
      // creating an occupancy event or a navigation hazard inside the
      // deformation detector.
      fields.AddString(config_.resource.session_id);
      fields.AddU32(reference_epoch_);
      fields.AddI64(explanation_key.x);
      fields.AddI64(explanation_key.y);
      fields.AddI64(explanation_key.z);
      fields.AddString(provenance);
      explanation.cause_id =
          MakeStableId("detection-explanation-v1", fields);
    } else {
      fields.AddId(patch.id);
      fields.AddString(provenance);
      explanation.cause_id =
          MakeStableId("detection-explanation-v1", fields);
    }
    explanations->push_back(explanation);
  };
  if (frame.validity != Validity::VALID) {
    // An unresolved deskew/input-health frame is not a calibrated structural
    // observation.  It may still contain real occupied endpoints, so preserve
    // the current patches as conservative OTHER geometry, but do not form a
    // residual, SOURCE_MISSING factor, background estimate, or positive
    // effective support from it.  OccupancyEventGraph applies the matching
    // endpoint-only/no-free-space rule to the disjoint occupancy role.
    for (const CurrentPatch& patch : association_patches)
      append_protective_explanation(patch, "input-quality-envelope-v1");
    for (const CurrentPatch& patch : measurement_patches)
      append_protective_explanation(patch, "input-quality-envelope-v1");
    for (uint32_t g = 0; g < reference_.gdus.size(); ++g) {
      FrameNuisance& nuisance = (*nuisance_by_gdu)[g];
      nuisance.pose_covariance = frame.pose.covariance;
      nuisance.pose_covariance.bottomRightCorner<3, 3>().diagonal().array() +=
          config_.calibration.pose_cross_covariance_bound_m2;
      CanonicalBytes nuisance_fields;
      nuisance_fields.AddId(frame.frame_id);
      nuisance_fields.AddString("input-quality-unavailable-v1");
      nuisance.nuisance_correlation_id =
          MakeStableId("nuisance-correlation-v1", nuisance_fields);
      nuisance.pose_temporal_bound_applied = true;
    }
    return true;
  }
  for (const CurrentPatch& patch : association_patches) {
    if (!patch.cross_gdu_ambiguous) continue;
    append_protective_explanation(
        patch, "cross-gdu-assignment-envelope-v1");
    if (reasons) reasons->push_back(ReasonCode::ASSOCIATION_AMBIGUOUS);
  }
  for (const CurrentPatch& patch : measurement_patches) {
    if (!patch.cross_gdu_ambiguous) continue;
    append_protective_explanation(
        patch, "cross-gdu-assignment-envelope-v1");
    if (reasons) reasons->push_back(ReasonCode::ASSOCIATION_AMBIGUOUS);
  }
  for (const CurrentPatch& patch : association_patches)
    if (patch.owner_surfel == std::numeric_limits<uint32_t>::max() &&
        !patch.cross_gdu_ambiguous)
      append_protective_explanation(
          patch, "outside-reference-envelope-v1");
  for (const CurrentPatch& patch : measurement_patches)
    if (patch.owner_surfel == std::numeric_limits<uint32_t>::max() &&
        !patch.cross_gdu_ambiguous)
      append_protective_explanation(
          patch, "outside-reference-envelope-v1");

  if (!association_support_available || !measurement_support_available) {
    // This frame contains no valid split-support likelihood.  Advance the
    // explicitly correlated pose nuisance, but submit no SOURCE_MISSING,
    // geometric, or OTHER factor: absence of a role sample is not a physical
    // observation and must not erase the resolved posterior into OTHER.
    for (uint32_t g = 0; g < reference_.gdus.size(); ++g) {
      FrameNuisance& nuisance = (*nuisance_by_gdu)[g];
      nuisance.pose_covariance = frame.pose.covariance;
      nuisance.pose_covariance.bottomRightCorner<3, 3>().diagonal().array() +=
          config_.calibration.pose_cross_covariance_bound_m2;
      CanonicalBytes nuisance_fields;
      nuisance_fields.AddId(frame.frame_id);
      nuisance_fields.AddString("split-support-unavailable-v1");
      nuisance.nuisance_correlation_id =
          MakeStableId("nuisance-correlation-v1", nuisance_fields);
      nuisance.pose_temporal_bound_applied = true;
    }
    return true;
  }

  // The online contract is 3..max_patches_per_gdu aggregated spatial
  // sources, not every frozen surfel in a large connected wall/floor GDU.
  // Prefer surfels that actually own a disjoint association or measurement
  // patch in this frame, then fill deterministically by frozen SurfelId so a
  // fully missing source can still be tested.  This bounded selection is made
  // before any likelihood or explanation is constructed; truncating factors
  // afterwards would change normalization and reuse a partially evaluated
  // observation.
  std::vector<bool> has_current_support(reference_.surfels.size(), false);
  for (const CurrentPatch& patch : association_patches)
    if (patch.owner_surfel < has_current_support.size())
      has_current_support[patch.owner_surfel] = true;
  for (const CurrentPatch& patch : measurement_patches)
    if (patch.owner_surfel < has_current_support.size())
      has_current_support[patch.owner_surfel] = true;
  std::vector<bool> selected_surfel(reference_.surfels.size(), false);
  for (uint32_t gdu_index = 0U; gdu_index < reference_.gdus.size();
       ++gdu_index) {
    const GduDefinition& gdu = reference_.gdus[gdu_index];
    if (!gdu.monitorable) continue;
    size_t selected_count = 0U;
    const size_t selection_limit = static_cast<size_t>(
        config_.resource.max_patches_per_gdu);
    // This is the same total order as sorting by
    // (has_current_support descending, frozen SurfelId ascending), but the
    // invariant SurfelId order was frozen once with the reference.
    for (int support_pass = 1; support_pass >= 0 &&
         selected_count < selection_limit; --support_pass) {
      for (uint32_t surfel_index : gdu_surfels_by_id_cache_[gdu_index]) {
        if (surfel_index >= selected_surfel.size()) continue;
        const bool supported = has_current_support[surfel_index];
        if (supported != (support_pass == 1)) continue;
        selected_surfel[surfel_index] = true;
        ++selected_count;
        if (selected_count >= selection_limit) break;
      }
    }
  }

  for (uint32_t s = 0; s < reference_.surfels.size(); ++s) {
    if (!selected_surfel[s]) continue;
    const Surfel& surfel = reference_.surfels[s];
    const uint32_t gdu = surfel.gdu_index;
    if (gdu >= nuisance_by_gdu->size() ||
        !reference_.gdus[gdu].monitorable ||
        !(*nuisance_by_gdu)[gdu].control_registration_valid)
      continue;
    const FrameNuisance& control_nuisance = (*nuisance_by_gdu)[gdu];
    const Eigen::Vector3d corrected_origin = frame.pose.origin_L_R +
        control_nuisance.pose_correction.tail<3>();
    const double range = (surfel.center_R - corrected_origin).norm();
    // SOURCE_MISSING is evidence only if the disjoint ASSOCIATION support was
    // actually sampled.  A sparse 15/30/45/10 split with zero association patches
    // is missing *data*, not evidence that a visible reference source vanished.
    const bool expected_visible = association_support_available &&
        range >= config_.calibration.visibility_min_range_m &&
        range <= config_.calibration.visibility_max_range_m &&
        reference_visibility_.ReferenceLineOfSightVisible(
            corrected_origin, surfel.center_R);
    std::vector<AssociationFeature> features;
    // This position is exactly a frozen reference surfel. Reuse the basis
    // cached by filter initialization and evaluate its predictive field once
    // per source, rather than rebuilding the metric-graph POU once per
    // association candidate.
    FieldQueryResult prediction;
    if (!association_patches_by_surfel[s].empty())
      prediction = predicted_filter.QueryReferenceSurfelWithPosterior(
          gdu, s, predicted_posterior[gdu]);
    for (const CurrentPatch* patch_pointer :
         association_patches_by_surfel[s]) {
      const CurrentPatch& patch = *patch_pointer;
      const Eigen::Vector3d association_center =
          corrected_point(patch.center_R, control_nuisance);
      const Eigen::Vector3d delta = association_center - surfel.center_R;
      const double distance = delta.norm();
      const Eigen::Matrix3d tangent_projector =
          Eigen::Matrix3d::Identity() - surfel.observable_projector_R;
      AssociationFeature feature;
      feature.candidate_id = patch.id;
      feature.distance_lower_bound = distance / config_.resource.graph_neighbor_radius_m;
      feature.continuous_projection = std::exp(-
          (tangent_projector * delta).norm() /
          config_.resource.surfel_seed_spacing_m);
      feature.projector_chordal_distance =
          (patch.projector_R - surfel.observable_projector_R).norm();
      feature.log_support_count_ratio = std::log(
          (static_cast<double>(patch.count) + 1.0) /
          (static_cast<double>(surfel.point_count) + 1.0));
      feature.overlap = std::exp(-distance / config_.resource.surfel_seed_spacing_m);
      feature.independent_visibility = expected_visible ? 1.0 : 0.0;
      feature.prediction_continuity = std::exp(-
          (delta - prediction.mean_displacement_R).norm() /
          config_.resource.surfel_seed_spacing_m);
      features.push_back(feature);
    }
    AssociationMeasurementSet set;
    CanonicalBytes source_fields; source_fields.AddId(frame.frame_id); source_fields.AddId(surfel.id);
    set.source_evidence_id = MakeStableId("selection-source-v1", source_fields);
    // SOURCE_MISSING remains a mutually exclusive alternative even when
    // clutter-like candidates exist.  Removing it in non-empty candidate sets
    // would condition on the selected support and bias deformation evidence.
    set.selection = association_.Evaluate(features, expected_visible);
    if (!set.selection.valid) {
      // A malformed multinomial is not equivalent to an unobserved source.
      // Skipping it would delete probability mass and still let the frame
      // update the remaining GDUs. Abort the candidate transaction instead.
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
    if (set.selection.entropy > config_.calibration.association_entropy_max && reasons)
      reasons->push_back(ReasonCode::ASSOCIATION_AMBIGUOUS);
    Validity basis_validity = Validity::UNKNOWN;
    std::vector<ReasonCode> basis_reasons;
    Eigen::MatrixXd vector_basis;
    const bool basis_available = predicted_filter.ReferenceSurfelBasis(
        gdu, s, &vector_basis, &basis_validity, &basis_reasons);
    std::array<int, 4> hypothesis_observable_rank{{0, 0, 0, 0}};
    if (basis_available && basis_validity == Validity::VALID &&
        vector_basis.size() != 0) {
      for (int h = 0; h < 4; ++h)
        hypothesis_observable_rank[static_cast<size_t>(h)] =
            ObservableAxes(
                surfel.hypothesis_projector_R[static_cast<size_t>(h)],
                config_.numerical.tie_tolerance).cols();
    }
    for (const AssociationCategory& category : set.selection.categories) {
      if (category.geometric_candidate) continue;
      ObservationExplanation explanation;
      explanation.source_evidence_id = set.source_evidence_id;
      explanation.semantics = category.semantics;
      explanation.probability.mean = category.probability;
      explanation.probability.lower = config_.calibration.validated
          ? category.probability : 0.0;
      explanation.probability.upper = config_.calibration.validated
          ? category.probability : 1.0;
      explanation.probability.calibrated = config_.calibration.validated;
      explanation.probability.mean_identified = true;
      explanation.committed_to_posterior = true;
      CanonicalBytes explanation_fields;
      explanation_fields.AddId(set.source_evidence_id);
      explanation_fields.AddU8(static_cast<uint8_t>(category.semantics));
      explanation.cause_id = MakeStableId("hazard-cause-v1", explanation_fields);
      explanations->push_back(explanation);
    }

    for (const AssociationCategory& category : set.selection.categories) {
      if (!category.geometric_candidate) continue;
      const auto association_patch =
          association_patch_by_id.find(category.candidate_id);
      if (association_patch == association_patch_by_id.end()) continue;
      const auto paired_id = paired_measurement_id.find(category.candidate_id);
      if (!measurement_support_available ||
          paired_id == paired_measurement_id.end())
        continue;
      const auto paired_patch = measurement_patch_by_id.find(paired_id->second);
      if (paired_patch == measurement_patch_by_id.end())
        continue;
      const CurrentPatch* measurement_patch = paired_patch->second;
      measurement_patch_ids_by_gdu[gdu].insert(measurement_patch->id);
      FieldMeasurement measurement;
      CanonicalBytes evidence_fields; evidence_fields.AddId(set.source_evidence_id);
      evidence_fields.AddId(category.candidate_id); evidence_fields.AddId(measurement_patch->id);
      measurement.evidence_id = MakeStableId("evidence-v1", evidence_fields);
      measurement.candidate_id = category.candidate_id; measurement.gdu_index = gdu;
      measurement.position_R = corrected_point(
          measurement_patch->center_R, control_nuisance);
      measurement.raw_residual_R = measurement.position_R - surfel.center_R;
      measurement.association_probability = category.probability;
      if (!basis_available ||
          basis_validity != Validity::VALID || vector_basis.size() == 0) {
        continue;
      }
      Eigen::Matrix<double, 3, 6> point_pose_jacobian;
      point_pose_jacobian.leftCols<3>() =
          -Skew(measurement.position_R - corrected_origin);
      point_pose_jacobian.rightCols<3>().setIdentity();
      const Eigen::Matrix3d corrected_R_RL =
          correction_rotation(control_nuisance.pose_correction) *
          frame.pose.R_RL;
      const Eigen::Matrix3d R_RI = corrected_R_RL *
                                   config_.resource.extrinsic.R_IL.transpose();
      const Eigen::Vector3d point_from_lidar_I = R_RI.transpose() *
          (measurement.position_R - corrected_origin);
      Eigen::Matrix<double, 3, 6> point_extrinsic_jacobian;
      // Frozen extrinsic perturbation convention is
      // [IMU-frame left rotation of R_IL; IMU-frame additive t_IL].
      point_extrinsic_jacobian.leftCols<3>() =
          -R_RI * Skew(point_from_lidar_I);
      point_extrinsic_jacobian.rightCols<3>() = R_RI;
      for (int h = 0; h < 4; ++h) {
        const Eigen::Matrix3d projector =
            surfel.hypothesis_projector_R[static_cast<size_t>(h)];
        HypothesisMeasurement& model = measurement.hypothesis[static_cast<size_t>(h)];
        model.observable_rank =
            hypothesis_observable_rank[static_cast<size_t>(h)];
        // All geometry hypotheses are densities on the same 3-D raw residual
        // measure.  Observable coordinates receive the deformation and
        // shared-nuisance designs; the orthogonal complement is integrated
        // with a calibrated broad sampling density.  Projecting to ranks
        // 1/2/3 and comparing those bare densities would mix incompatible
        // physical units and bias the geometry posterior.
        model.residual_m = measurement.raw_residual_R;
        model.design = projector * vector_basis;
        model.reference_bias_design = projector;
        model.extrinsic_design = projector * point_extrinsic_jacobian;
        model.pose_design = projector * point_pose_jacobian;
        const Eigen::Matrix3d complement = Eigen::Matrix3d::Identity() - projector;
        // The frozen surfel mean uncertainty is an independent reference
        // term.  The epoch-common reference bias is carried separately in the
        // 3-D persistent nuisance state; omitting the former or adding the
        // latter here would respectively undercount or double-count reference
        // error.
        model.independent_covariance_m2 = projector *
            (measurement_patch->centroid_covariance_R +
             surfel.covariance_R) * projector.transpose() +
            config_.calibration.unobservable_residual_sigma_m *
                config_.calibration.unobservable_residual_sigma_m *
                complement * complement.transpose();
        model.independent_covariance_m2 = 0.5 *
            (model.independent_covariance_m2 +
             model.independent_covariance_m2.transpose());
      }
      set.candidates.push_back(measurement);
    }
    if (!set.candidates.empty()) {
      (*measurements_by_gdu)[gdu].push_back(set);
    } else {
      // The role hash deliberately samples only part of the endpoints for
      // ASSOCIATION. Therefore an empty feature set is absence of association
      // evidence, not evidence that the frozen source disappeared. Likewise,
      // a geometric association without a disjoint MEASUREMENT partner is
      // incomplete split-support data. Keep explanations diagnostic-only and
      // submit no SOURCE_MISSING/OTHER/formal factor.
      for (ObservationExplanation& explanation : *explanations) {
        if (explanation.source_evidence_id == set.source_evidence_id)
          explanation.committed_to_posterior = false;
      }
    }
  }

  for (uint32_t g = 0; g < reference_.gdus.size(); ++g) {
    const size_t patch_count = measurement_patch_ids_by_gdu[g].size();
    if (patch_count > static_cast<size_t>(config_.resource.max_patches_per_gdu)) {
      if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
      return false;
    }
    if ((*nuisance_by_gdu)[g].control_registration_valid &&
        patch_count < static_cast<size_t>(config_.resource.min_patches_per_gdu)) {
      (*nuisance_by_gdu)[g].deformation_measurement_valid = false;
      if (reasons) reasons->push_back(ReasonCode::INSUFFICIENT_SUPPORT);
    }
  }
  return true;
}

CommonHeader V1Pipeline::BuildCommonHeader(
    const FrameBundle& frame, uint32_t approximation_flags,
    const std::vector<ReasonCode>& reasons) const {
  CommonHeader common;
  common.session_id = config_.resource.session_id;
  common.reference_epoch = reference_epoch_;
  common.accepted_frame_index = frame.accepted_frame_index;
  common.measurement_stamp = frame.stamp; common.publish_stamp = ros::Time::now();
  common.world_frame = config_.resource.world_frame;
  common.software_revision = config_.software_revision;
  common.config_digest = config_.config_digest;
  common.calibration_version = config_.calibration.version;
  common.robot_geometry_revision.clear();
  common.input_capability_profile = 0U;  // FRAME_XYZI_REGISTERED
  common.sensor_scope = "LIDAR_ONLY_NO_SEMANTICS_NO_OBJECT_ID";
  common.hazard_coverage_mask = 0U;
  common.uncovered_hazards.clear();
  common.probabilities_calibrated = config_.calibration.validated &&
      frame.validity == Validity::VALID &&
      reference_.validity == Validity::VALID;
  common.validity = CombineValidity(
      config_.calibration.validated ? Validity::VALID
                                    : Validity::INCONCLUSIVE,
      CombineValidity(frame.validity, reference_.validity));
  common.reasons = reasons;
  common.reasons.insert(common.reasons.end(), reference_.reasons.begin(),
                        reference_.reasons.end());
  if (!config_.calibration.validated && !HasReason(common.reasons, ReasonCode::CALIBRATION_UNSET))
    common.reasons.push_back(ReasonCode::CALIBRATION_UNSET);
  if (HasReason(common.reasons, ReasonCode::INVALID_NUMERICAL))
    common.validity = CombineValidity(common.validity, Validity::INVALID);
  else if (HasReason(common.reasons,
                     ReasonCode::OUTSIDE_CALIBRATED_RANGE))
    common.validity = CombineValidity(common.validity,
                                      Validity::INCONCLUSIVE);
  else if (HasReason(common.reasons, ReasonCode::ASSOCIATION_AMBIGUOUS) ||
           HasReason(common.reasons, ReasonCode::INSUFFICIENT_SUPPORT) ||
           HasReason(common.reasons, ReasonCode::COMMON_MODE_AMBIGUOUS) ||
           HasReason(common.reasons,
                     ReasonCode::DEFORMATION_REFERENCE_AMBIGUOUS))
    common.validity = CombineValidity(common.validity,
                                      Validity::INCONCLUSIVE);
  if ((approximation_flags &
       APPROX_EXTRINSIC_CROSS_GDU_COVARIANCE_OMITTED) != 0U) {
    common.validity = CombineValidity(common.validity,
                                      Validity::INCONCLUSIVE);
    if (!HasReason(common.reasons,
                   ReasonCode::INCONCLUSIVE_EXTRINSIC_COVARIANCE))
      common.reasons.push_back(
          ReasonCode::INCONCLUSIVE_EXTRINSIC_COVARIANCE);
  }
  common.valid_from = frame.stamp;
  common.valid_until = frame.stamp + ros::Duration(config_.resource.map_ttl_s);
  common.prediction_valid_until = frame.stamp + ros::Duration(
      config_.resource.prediction_horizons_s.back());
  common.dropped_input_count = dropped_input_count_;
  common.approximation_flags = approximation_flags;
  CanonicalBytes fields; fields.AddString(common.session_id); fields.AddId(reference_.id);
  fields.AddId(frame.frame_id); fields.AddString(config_.config_digest);
  common.snapshot_id = MakeStableId("snapshot-v1", fields);
  std::sort(common.reasons.begin(), common.reasons.end(), [](ReasonCode lhs, ReasonCode rhs) {
    return static_cast<uint16_t>(lhs) < static_cast<uint16_t>(rhs);
  });
  common.reasons.erase(std::unique(common.reasons.begin(), common.reasons.end()),
                       common.reasons.end());
  return common;
}

void V1Pipeline::TrimRawBuffer(std::vector<ReasonCode>* reasons) {
  while (!raw_buffer_.empty() &&
         (raw_buffer_.back().stamp - raw_buffer_.front().stamp).toSec() >
             config_.resource.raw_frame_buffer_s)
    raw_buffer_.pop_front();
  size_t buffered_points = 0U;
  for (const FrameBundle& frame : raw_buffer_) {
    if (buffered_points > config_.resource.max_raw_buffer_points -
                              std::min(config_.resource.max_raw_buffer_points,
                                       frame.points.size())) {
      buffered_points = config_.resource.max_raw_buffer_points + 1U;
      break;
    }
    buffered_points += frame.points.size();
  }
  bool evicted_for_capacity = false;
  while (!raw_buffer_.empty() &&
         buffered_points > config_.resource.max_raw_buffer_points) {
    buffered_points -= raw_buffer_.front().points.size();
    raw_buffer_.pop_front();
    evicted_for_capacity = true;
  }
  if (evicted_for_capacity) {
    ++resource_failure_count_;
    if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
  }
}

bool V1Pipeline::AcceptFrame(const FrameBundle& frame,
                             V1Snapshot* published_snapshot,
                             std::vector<ReasonCode>* reasons) {
  if (phase_ == MonitoringPhase::CLOSED || frame.session_id != config_.resource.session_id ||
      frame.world_frame != config_.resource.world_frame || frame.validity == Validity::INVALID ||
      frame.stamp.isZero()) {
    if (reasons) reasons->push_back(ReasonCode::INPUT_SCHEMA_INVALID);
    ++dropped_input_count_; return false;
  }
  if (frame.points.size() > config_.resource.max_points_per_frame) {
    if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
    ++resource_failure_count_;
    ++dropped_input_count_;
    return false;
  }
  VoxelKey checked_key;
  if (!TryGridKey(frame.pose.origin_L_R, config_.resource.grid_origin_m,
                  config_.resource.surfel_seed_spacing_m, &checked_key)) {
    if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
    ++resource_failure_count_;
    ++dropped_input_count_;
    return false;
  }
  for (const CanonicalPoint& point : frame.points) {
    if (!TryGridKey(point.position_R, config_.resource.grid_origin_m,
                    config_.resource.surfel_seed_spacing_m, &checked_key)) {
      if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
      ++resource_failure_count_;
      ++dropped_input_count_;
      return false;
    }
  }
  // Selection-support frames are not accepted monitoring observations.  This
  // guard must precede the monotone-index commit; otherwise a rejected frame
  // consumes the index while the node correctly reuses it, permanently
  // deadlocking every later input on the ordering check.
  if (phase_ == MonitoringPhase::MONITOR_UPDATE &&
      trigger_source_ == TriggerSource::ALGORITHM_RESIDUAL &&
      frame.stamp <= selection_support_cutoff_) {
    if (reasons)
      reasons->push_back(ReasonCode::INCONCLUSIVE_SELECTION_TRIGGERED);
    ++dropped_input_count_;
    return false;
  }
  if (accepted_frame_ids_.count(frame.frame_id) > 0U) {
    if (reasons) reasons->push_back(ReasonCode::INPUT_DUPLICATE);
    ++dropped_input_count_; return false;
  }
  if (has_accepted_frame_ && (frame.stamp <= last_accepted_stamp_ ||
      frame.accepted_frame_index <= last_accepted_frame_index_)) {
    if (reasons) reasons->push_back(ReasonCode::INPUT_TIME_INVALID);
    ++dropped_input_count_; return false;
  }
  has_accepted_frame_ = true; last_accepted_stamp_ = frame.stamp;
  last_accepted_frame_index_ = frame.accepted_frame_index;
  accepted_frame_ids_.insert(frame.frame_id);
  accepted_frame_id_order_.push_back(frame.frame_id);
  // Strictly increasing stamp/index already reject any older retransmission.
  // Retain only recent identities so a long monitoring session cannot grow an
  // unbounded duplicate-classification set.
  const size_t recent_identity_limit = std::max<size_t>(
      2U, static_cast<size_t>(config_.resource.reference_min_frames) +
              static_cast<size_t>(config_.resource.fixed_lag_frames) + 1U);
  while (accepted_frame_id_order_.size() > recent_identity_limit) {
    accepted_frame_ids_.erase(accepted_frame_id_order_.front());
    accepted_frame_id_order_.pop_front();
  }
  if (phase_ == MonitoringPhase::MAPPING_CONTEXT ||
      phase_ == MonitoringPhase::MONITOR_REFERENCE) {
    raw_buffer_.push_back(frame); TrimRawBuffer(reasons);
    if (phase_ == MonitoringPhase::MONITOR_REFERENCE && !reference_prepared_)
      TryFallbackReference(reasons);
    return true;
  }

  StableIdRegistryTransaction candidate_ids;
  if (!candidate_ids.active()) {
    const bool resource_exhausted = StableIdRegistryResourceExhausted();
    if (reasons) reasons->push_back(resource_exhausted
                                        ? ReasonCode::RESOURCE_LIMIT
                                        : ReasonCode::INVALID_ID_COLLISION);
    if (resource_exhausted) ++resource_failure_count_;
    ++dropped_input_count_;
    return true;
  }
  const auto begin = std::chrono::steady_clock::now();
  UacGaussianSumFilter candidate_filter = filter_;
  const auto after_filter_copy = std::chrono::steady_clock::now();
  std::vector<ReasonCode> local_reasons = frame.reasons;
  // The first observation after a synchronous reference-to-update hand-off
  // may arrive after phase-1 sensor callbacks already advanced the input
  // stream.  SetPhase has already built and validated the complete initial U
  // posterior.  Rebuilding all GDU mixtures here is mathematically redundant
  // and, for a no-prior full scene, exceeded the transaction deadline.  Move
  // only the still-unobserved filter clock, then perform the normal absolute
  // geometry measurement against the unchanged frozen reference.  No
  // velocity is fabricated over the hand-off interval.  Every later frame
  // uses the ordinary bounded CTMC prediction.
  const bool temporal_ready = first_monitor_update_pending_
      ? candidate_filter.ReanchorFirstObservation(frame.stamp, &local_reasons)
      : candidate_filter.Predict(frame.stamp, &local_reasons);
  const auto after_prediction = std::chrono::steady_clock::now();
  if (!temporal_ready) {
    if (reasons) reasons->insert(reasons->end(), local_reasons.begin(), local_reasons.end());
    ++dropped_input_count_; return true;
  }
  std::vector<std::vector<AssociationMeasurementSet>> measurements;
  std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>> nuisance;
  std::vector<ObservationExplanation> explanations;
  const bool measurements_built = BuildMeasurements(
      frame, candidate_filter, &measurements, &nuisance,
      &explanations, &local_reasons);
  const auto after_measurements = std::chrono::steady_clock::now();
  if (!measurements_built)
    std::cerr << "[deform_v1] resource stage=build_measurements" << std::endl;
  const bool posterior_updated = measurements_built &&
      candidate_filter.Update(measurements, nuisance, frame.stamp,
                              &local_reasons);
  const auto after_update = std::chrono::steady_clock::now();
  if (measurements_built && !posterior_updated)
    std::cerr << "[deform_v1] resource stage=posterior_update" << std::endl;
  if (!measurements_built || !posterior_updated) {
    if (reasons) reasons->insert(reasons->end(), local_reasons.begin(), local_reasons.end());
    ++dropped_input_count_; return true;
  }
  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>>
      candidate_deformation = candidate_filter.Posterior();
  V1Snapshot snapshot;
  uint32_t approximation = candidate_filter.approximation_flags() |
                           APPROX_POSE_CROSS_COVARIANCE_BOUND;
  if (std::any_of(explanations.begin(), explanations.end(),
                  [](const ObservationExplanation& explanation) {
                    return explanation.cross_gdu_assignment_envelope;
                  }))
    approximation |= APPROX_CROSS_GDU_ASSOCIATION_TO_ENVELOPE;
  if (config_.resource.extrinsic.covariance.norm() >
      config_.numerical.psd_eigenvalue_tolerance)
    approximation |= APPROX_EXTRINSIC_CROSS_GDU_COVARIANCE_OMITTED;
  snapshot.reference = reference_;
  snapshot.deformation = std::move(candidate_deformation);
  snapshot.interfaces = candidate_filter.InterfacePosteriors(
      snapshot.deformation);
  if (trigger_source_ == TriggerSource::ALGORITHM_RESIDUAL) {
    for (GduPosterior& posterior : snapshot.deformation)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_SELECTION_TRIGGERED;
  }
  if (HasReason(local_reasons, ReasonCode::ASSOCIATION_AMBIGUOUS))
    for (GduPosterior& posterior : snapshot.deformation)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_ASSOCIATION;
  if (HasReason(local_reasons, ReasonCode::INSUFFICIENT_SUPPORT))
    for (GduPosterior& posterior : snapshot.deformation)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_LOW_POWER;
  if (HasReason(local_reasons, ReasonCode::COMMON_MODE_AMBIGUOUS))
    for (GduPosterior& posterior : snapshot.deformation)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_SLAM;
  if (frame.validity != Validity::VALID)
    for (GduPosterior& posterior : snapshot.deformation)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_SLAM;
  if ((approximation & APPROX_EXTRINSIC_CROSS_GDU_COVARIANCE_OMITTED) != 0U)
    for (GduPosterior& posterior : snapshot.deformation)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_SLAM;
  if (reference_.validity != Validity::VALID)
    for (GduPosterior& posterior : snapshot.deformation)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_REFERENCE;
  std::sort(explanations.begin(), explanations.end(),
      [](const ObservationExplanation& lhs, const ObservationExplanation& rhs) {
        if (lhs.cause_id != rhs.cause_id) return lhs.cause_id < rhs.cause_id;
        return lhs.source_evidence_id < rhs.source_evidence_id;
      });
  if (explanations.size() > config_.resource.max_events) {
    std::cerr << "[deform_v1] resource stage=committed_explanations value="
              << explanations.size() << " limit="
              << config_.resource.max_events << std::endl;
    local_reasons.push_back(ReasonCode::RESOURCE_LIMIT);
    if (reasons)
      reasons->insert(reasons->end(), local_reasons.begin(),
                      local_reasons.end());
    ++resource_failure_count_;
    ++dropped_input_count_;
    return true;
  }
  snapshot.explanations = explanations;
  // Freeze the public validity header only after every reason-producing stage
  // in the transaction has succeeded.  This prevents the layers and the
  // common header from describing different causal states.
  snapshot.common = BuildCommonHeader(frame, approximation, local_reasons);
  if (trigger_source_ == TriggerSource::ALGORITHM_RESIDUAL) {
    snapshot.common.validity = CombineValidity(
        snapshot.common.validity, Validity::INCONCLUSIVE);
    if (!HasReason(snapshot.common.reasons,
                   ReasonCode::INCONCLUSIVE_SELECTION_TRIGGERED))
      snapshot.common.reasons.push_back(
          ReasonCode::INCONCLUSIVE_SELECTION_TRIGGERED);
    std::sort(snapshot.common.reasons.begin(), snapshot.common.reasons.end(),
              [](ReasonCode lhs, ReasonCode rhs) {
                return static_cast<uint16_t>(lhs) <
                       static_cast<uint16_t>(rhs);
              });
    snapshot.common.reasons.erase(
        std::unique(snapshot.common.reasons.begin(),
                    snapshot.common.reasons.end()),
        snapshot.common.reasons.end());
  }
  if (!frozen_reference_numerically_validated_ ||
      snapshot.reference.id != reference_.id ||
      snapshot.reference.epoch != reference_.epoch ||
      snapshot.reference.surfels.size() != reference_.surfels.size() ||
      snapshot.reference.gdus.size() != reference_.gdus.size() ||
      snapshot.reference.interfaces.size() != reference_.interfaces.size() ||
      !ValidateSnapshotNumerical(snapshot, config_, false)) {
    local_reasons.push_back(ReasonCode::INVALID_NUMERICAL);
    if (reasons)
      reasons->insert(reasons->end(), local_reasons.begin(),
                      local_reasons.end());
    ++dropped_input_count_;
    return true;
  }
  if (!StableIdRegistryHealthy()) {
    std::cerr << "[deform_v1] resource stage=stable_id_registry" << std::endl;
    local_reasons.push_back(StableIdRegistryResourceExhausted()
                                ? ReasonCode::RESOURCE_LIMIT
                                : ReasonCode::INVALID_ID_COLLISION);
    if (reasons)
      reasons->insert(reasons->end(), local_reasons.begin(), local_reasons.end());
    if (StableIdRegistryResourceExhausted()) ++resource_failure_count_;
    ++dropped_input_count_;
    return true;
  }
  const auto after_products = std::chrono::steady_clock::now();
  const auto end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
  // One second is already slower than the formal detector input cadence and
  // therefore deserves a research-runtime breakdown even though the much
  // wider deadline below is only the fail-closed resource bound.  Keeping
  // the warning boundary independent of that bound lets headless runs expose
  // which exact mathematical stage, rather than RViz, is limiting cadence.
  if (elapsed_ms > 1000.0 && elapsed_ms <= config_.resource.deadline_ms) {
    std::cerr << "[deform_v1] slow transaction elapsed_ms=" << elapsed_ms
              << " stages_ms={filter_copy:"
              << std::chrono::duration<double, std::milli>(
                     after_filter_copy - begin).count()
              << ",predict:"
              << std::chrono::duration<double, std::milli>(
                     after_prediction - after_filter_copy).count()
              << ",measurements:"
              << std::chrono::duration<double, std::milli>(
                     after_measurements - after_prediction).count()
              << ",update:"
              << std::chrono::duration<double, std::milli>(
                     after_update - after_measurements).count()
              << ",products:"
              << std::chrono::duration<double, std::milli>(
                     after_products - after_update).count()
              << "}" << std::endl;
  }
  if (elapsed_ms > config_.resource.deadline_ms) {
    std::cerr << "[deform_v1] deadline elapsed_ms=" << elapsed_ms
              << " limit_ms=" << config_.resource.deadline_ms
              << " stages_ms={filter_copy:"
              << std::chrono::duration<double, std::milli>(
                     after_filter_copy - begin).count()
              << ",predict:"
              << std::chrono::duration<double, std::milli>(
                     after_prediction - after_filter_copy).count()
              << ",measurements:"
              << std::chrono::duration<double, std::milli>(
                     after_measurements - after_prediction).count()
              << ",update:"
              << std::chrono::duration<double, std::milli>(
                     after_update - after_measurements).count()
              << ",products:"
              << std::chrono::duration<double, std::milli>(
                     after_products - after_update).count()
              << "}" << std::endl;
    local_reasons.push_back(ReasonCode::DEADLINE_EXCEEDED);
    if (reasons) reasons->insert(reasons->end(), local_reasons.begin(), local_reasons.end());
    ++deadline_failure_count_;
    ++dropped_input_count_;
    return true;
  }
  if (!candidate_ids.Commit()) {
    local_reasons.push_back(StableIdRegistryResourceExhausted()
                                ? ReasonCode::RESOURCE_LIMIT
                                : ReasonCode::INVALID_ID_COLLISION);
    if (reasons)
      reasons->insert(reasons->end(), local_reasons.begin(),
                      local_reasons.end());
    if (StableIdRegistryResourceExhausted()) ++resource_failure_count_;
    ++dropped_input_count_;
    return true;
  }
  // The candidate was the only mutable object used by this transaction.  It
  // is now committed and has no remaining readers, so moving is exactly
  // equivalent to the former value copy while avoiding a second copy of the
  // Gaussian/history state and its immutable reference caches.
  filter_ = std::move(candidate_filter);
  first_monitor_update_pending_ = false;
  last_snapshot_ = std::move(snapshot);
  if (published_snapshot) *published_snapshot = last_snapshot_;
  return true;
}

}  // namespace v1
}  // namespace deform_monitor_v3
