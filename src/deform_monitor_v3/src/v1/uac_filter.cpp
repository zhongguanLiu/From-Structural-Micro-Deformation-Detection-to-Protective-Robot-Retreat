#include "deform_monitor_v3/v1/uac_filter.hpp"

#include "deform_monitor_v3/v1/deterministic.hpp"
#include "deform_monitor_v3/v1/math.hpp"

#include <Eigen/Eigenvalues>

#include <boost/random/sobol.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <system_error>
#include <thread>

namespace deform_monitor_v3 {
namespace v1 {
namespace {

double ClampUnitProbability(double value) {
  if (!std::isfinite(value)) return value;
  return std::max(0.0, std::min(1.0, value));
}

int StateDimension(StructuralState state, int coefficient_count) {
  const int physical = state == StructuralState::U ? 0
      : (state == StructuralState::A ? 2 * coefficient_count : coefficient_count);
  // Persistent ordering is [physical, epoch-static reference bias(3),
  // epoch-static fixed-extrinsic nuisance(6), temporally correlated LiDAR pose
  // nuisance(6)].  The pose block remains in
  // the state across the fixed lag instead of being re-created independently
  // at every frame (DEC-027/DEC-028).
  return physical + 15;
}

int PhysicalDimension(StructuralState state, int coefficient_count) {
  if (state == StructuralState::U) return 0;
  return state == StructuralState::A ? 2 * coefficient_count : coefficient_count;
}

int LinkModeIndex(InterfaceJointState mode) {
  if (mode == InterfaceJointState::LINK_COHERENT) return 0;
  if (mode == InterfaceJointState::LINK_RELATIVE_MOTION) return 1;
  if (mode == InterfaceJointState::LINK_UNRESOLVED) return 2;
  return -1;
}

bool SameModeVector(const std::vector<InterfaceJointState>& lhs,
                    const std::vector<InterfaceJointState>& rhs) {
  return lhs == rhs;
}

void RefreshLineageDigest(GaussianComponent* component) {
  CanonicalBytes bytes;
  for (const StableId& id : component->active_fixed_lag_lineage) bytes.AddId(id);
  component->association_lineage = component->active_fixed_lag_lineage.empty()
      ? StableId() : MakeStableId("association-lineage-v1", bytes);
}

void AppendCurrentFrameLineage(const StableId& evidence_id,
                               const StableId& category_id,
                               uint8_t semantics,
                               GaussianComponent* component) {
  if (!component || component->active_fixed_lag_lineage.empty()) return;
  CanonicalBytes bytes;
  bytes.AddId(component->active_fixed_lag_lineage.back());
  bytes.AddId(evidence_id);
  bytes.AddId(category_id);
  bytes.AddU8(semantics);
  component->active_fixed_lag_lineage.back() =
      MakeStableId("association-frame-branch-v1", bytes);
  RefreshLineageDigest(component);
}

struct PredictRawBranch {
  StructuralState destination = StructuralState::U;
  std::vector<InterfaceJointState> modes;
  double probability = 0.0;
};

struct PredictTransitionPlan {
  std::vector<PredictRawBranch> branches;
  double pruned_probability = 0.0;
  double legal_total = 0.0;
};

struct PredictPlanKey {
  StructuralState source = StructuralState::U;
  std::vector<InterfaceJointState> modes;

  bool operator<(const PredictPlanKey& other) const {
    if (source != other.source) return source < other.source;
    return modes < other.modes;
  }
};

struct PredictKernelKey {
  StructuralState source = StructuralState::U;
  StructuralState destination = StructuralState::U;
  std::vector<InterfaceJointState> modes;

  bool operator<(const PredictKernelKey& other) const {
    if (source != other.source) return source < other.source;
    if (destination != other.destination)
      return destination < other.destination;
    return modes < other.modes;
  }
};

}  // namespace

bool UacGaussianSumFilter::Initialize(const ReferenceSnapshot& reference,
                                      const ros::Time& stamp,
                                      std::vector<ReasonCode>* reasons) {
  initialized_ = false;
  first_observation_reanchor_available_ = false;
  approximation_flags_ = APPROX_QMC_INTERVAL |
      APPROX_PATH_CAUSE_COMMON_RANK;
  states_.clear();
  shared_interfaces_.clear();
  reference_ = reference;
  if (!reference.frozen || reference.gdus.empty()) {
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_NOT_FROZEN);
    return false;
  }
  if (reference.gdus.size() > config_.resource.max_gdu ||
      reference.interfaces.size() > config_.resource.max_interfaces) {
    std::cerr << "[deform_v1] filter reference resource failure gdus="
              << reference.gdus.size() << "/" << config_.resource.max_gdu
              << " interfaces=" << reference.interfaces.size() << "/"
              << config_.resource.max_interfaces << std::endl;
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_RESOURCE_LIMIT);
    return false;
  }
  std::vector<uint32_t> interface_owner_count(reference.interfaces.size(), 0U);
  for (uint32_t g = 0U; g < reference.gdus.size(); ++g) {
    const GduDefinition& definition = reference.gdus[g];
    const size_t controls = definition.control_surfel_indices.size();
    const bool coordinate_contract = definition.monitorable
        ? (definition.coordinate_rank >= 1U &&
           definition.coordinate_rank <= 3U &&
           controls >= config_.resource.min_controls_per_gdu &&
           controls <= config_.resource.max_controls_per_gdu &&
           definition.coordinate_axes_R.allFinite() &&
           (definition.coordinate_axes_R.leftCols(definition.coordinate_rank)
                    .transpose() *
                definition.coordinate_axes_R.leftCols(
                    definition.coordinate_rank) -
            Eigen::MatrixXd::Identity(definition.coordinate_rank,
                                      definition.coordinate_rank)).norm() <=
               10.0 * config_.numerical.psd_eigenvalue_tolerance)
        : (definition.coordinate_rank == 0U && controls == 0U &&
           definition.interface_indices.empty());
    if (!coordinate_contract ||
        definition.interface_indices.size() >
            config_.resource.max_interfaces_per_gdu) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
    std::set<uint32_t> unique_surfels;
    for (uint32_t surfel : definition.surfel_indices) {
      if (surfel >= reference.surfels.size() ||
          reference.surfels[surfel].gdu_index != g ||
          !unique_surfels.insert(surfel).second) {
        if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
        return false;
      }
    }
    std::set<uint32_t> unique_controls;
    for (uint32_t surfel : definition.control_surfel_indices) {
      if (unique_surfels.count(surfel) == 0U ||
          !unique_controls.insert(surfel).second) {
        if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
        return false;
      }
    }
    std::set<uint32_t> unique_interfaces;
    for (uint32_t interface_index : definition.interface_indices) {
      if (interface_index >= reference.interfaces.size() ||
          !unique_interfaces.insert(interface_index).second) {
        if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
        return false;
      }
      const InterfaceDefinition& interface =
          reference.interfaces[interface_index];
      if (interface.cross_gdu || interface.left_gdu != g ||
          interface.right_gdu != g) {
        if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
        return false;
      }
      ++interface_owner_count[interface_index];
    }
  }
  if (std::any_of(interface_owner_count.begin(), interface_owner_count.end(),
                  [](uint32_t count) { return count != 1U; })) {
    if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    return false;
  }
  PouField field(&reference_, &config_);
  states_.resize(reference_.gdus.size());
  for (uint32_t g = 0; g < reference_.gdus.size(); ++g) {
    GduState& state = states_[g];
    state.coefficient_count = field.CoefficientCount(g);
    const int n = state.coefficient_count;
    const GduDefinition& gdu = reference_.gdus[g];
    auto frozen_cache = std::make_shared<GduState::FrozenReferenceCache>();
    frozen_cache->task_bases.reserve(gdu.surfel_indices.size());
    frozen_cache->task_metric = Eigen::MatrixXd::Zero(n, n);
    frozen_cache->surfel_bases.reserve(gdu.surfel_indices.size());
    for (uint32_t surfel_index : gdu.surfel_indices) {
      Validity validity = Validity::UNKNOWN;
      std::vector<ReasonCode> basis_reasons;
      double attachment_distance_m =
          std::numeric_limits<double>::infinity();
      const Eigen::MatrixXd basis = field.VectorBasis(
          g, reference_.surfels[surfel_index].center_R,
          &validity, &basis_reasons, &attachment_distance_m);
      GduState::FrozenReferenceSurfelBasis cached_basis;
      cached_basis.basis = basis;
      cached_basis.validity = validity;
      cached_basis.reasons = basis_reasons;
      cached_basis.attachment_distance_m = attachment_distance_m;
      const size_t cached_offset = frozen_cache->surfel_bases.size();
      if (!frozen_cache->surfel_basis_offsets.emplace(
              surfel_index, cached_offset).second) {
        if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
        return false;
      }
      frozen_cache->surfel_bases.push_back(std::move(cached_basis));
      // Preserve the original Posterior() admission contract exactly.  Its
      // task metric accepted every VALID n-column basis; the task-event QMC
      // additionally required a finite physical 3-by-n evaluation.
      if (validity != Validity::VALID || basis.cols() != n) continue;
      Eigen::MatrixXd smooth_basis = basis;
      const int interface_first_column =
          gdu.coordinate_rank * static_cast<int>(
              gdu.control_surfel_indices.size());
      if (interface_first_column < n)
        smooth_basis.rightCols(n - interface_first_column).setZero();
      frozen_cache->task_metric +=
          smooth_basis.transpose() * smooth_basis;
      ++frozen_cache->task_quadrature_count;
      if (basis.rows() == 3 && basis.allFinite())
        frozen_cache->task_bases.push_back(std::move(smooth_basis));
    }
    if (frozen_cache->task_quadrature_count > 0U)
      frozen_cache->task_metric /=
          static_cast<double>(frozen_cache->task_quadrature_count);
    state.frozen_reference_cache = std::move(frozen_cache);
    state.reference_bias_prior_covariance =
        reference_.gdus[g].reference_bias_covariance_R;
    if (!ValidateSymmetricPsd(state.reference_bias_prior_covariance,
                              config_.numerical.psd_eigenvalue_tolerance)) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
    GaussianComponent component;
    component.state = StructuralState::U;
    component.geometry = reference_.gdus[g].coordinate_geometry;
    component.weight = 1.0;
    component.mean = Eigen::VectorXd::Zero(15);
    component.covariance = Eigen::MatrixXd::Zero(15, 15);
    component.covariance.topLeftCorner<3, 3>() =
        state.reference_bias_prior_covariance;
    component.covariance.block<6, 6>(3, 3) =
        config_.resource.extrinsic.covariance;
    component.interface_modes.assign(
        reference_.gdus[g].interface_indices.size(),
        InterfaceJointState::LINK_COHERENT);
    for (size_t local = 0; local < component.interface_modes.size(); ++local) {
      const uint32_t interface_index =
          reference_.gdus[g].interface_indices[local];
      if (interface_index >= reference_.interfaces.size() ||
          !reference_.interfaces[interface_index].identifiable)
        component.interface_modes[local] =
            InterfaceJointState::LINK_UNRESOLVED;
    }
    state.components.push_back(component);
    // Static connectivity is initialized once per reference-only competition
    // group.  A group carries one finite coherence prior: its registered
    // scales/locations are mutually exclusive alternatives, not independent
    // opportunities.  U forbids LINK_RELATIVE_MOTION, so the initial joint
    // prior is conditioned on the legal U subset.
    for (const CoherenceCompetitionGroup& group :
         CoherenceCompetitionGroups(g)) {
      bool identifiable_group = true;
      for (size_t local : group.local_interface_indices) {
        const uint32_t index = reference_.gdus[g].interface_indices[local];
        identifiable_group = identifiable_group &&
            index < reference_.interfaces.size() &&
            reference_.interfaces[index].identifiable;
      }
      if (!identifiable_group) continue;
      std::vector<GaussianComponent, Eigen::aligned_allocator<GaussianComponent>> expanded;
      const double cut = 1.0 - group.p_link;
      const double coherent = group.p_link *
          config_.calibration.interface_initial_link_probability[0];
      const double unresolved = group.p_link *
          config_.calibration.interface_initial_link_probability[2];
      const double legal_sum = cut + coherent + unresolved;
      if (!(legal_sum > 0.0) ||
          group.local_interface_indices.empty() ||
          group.local_interface_indices.size() !=
              group.normalized_weights.size()) {
        if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
        return false;
      }
      for (const GaussianComponent& parent : state.components) {
        if (cut > 0.0) {
          GaussianComponent child = parent;
          for (size_t local : group.local_interface_indices)
            child.interface_modes[local] =
                InterfaceJointState::CUT_NOT_APPLICABLE;
          child.weight *= cut / legal_sum;
          expanded.push_back(child);
        }
        if (coherent > 0.0) {
          GaussianComponent child = parent;
          for (size_t local : group.local_interface_indices)
            child.interface_modes[local] = InterfaceJointState::LINK_COHERENT;
          child.weight *= coherent / legal_sum;
          expanded.push_back(child);
        }
        if (unresolved > 0.0) {
          for (size_t candidate = 0;
               candidate < group.local_interface_indices.size(); ++candidate) {
            if (!(group.normalized_weights[candidate] > 0.0)) continue;
            GaussianComponent child = parent;
            for (size_t local : group.local_interface_indices)
              child.interface_modes[local] = InterfaceJointState::LINK_COHERENT;
            child.interface_modes[group.local_interface_indices[candidate]] =
                InterfaceJointState::LINK_UNRESOLVED;
            child.weight *= unresolved / legal_sum *
                group.normalized_weights[candidate];
            expanded.push_back(child);
          }
        }
      }
      state.components.swap(expanded);
      MergePruneNormalize(&state);
    }
    for (GaussianComponent& component : state.components) {
      component.component_id = ComponentId(component);
    }
    if (!RefreshDerived(&state, reasons)) return false;
  }
  last_stamp_ = stamp;
  shared_interfaces_.resize(reference_.interfaces.size());
  if (!SynchronizeSharedInterfaces(reasons)) return false;
  initialized_ = true;
  first_observation_reanchor_available_ = true;
  return true;
}

bool UacGaussianSumFilter::ReanchorFirstObservation(
    const ros::Time& stamp, std::vector<ReasonCode>* reasons) {
  if (!initialized_ || !first_observation_reanchor_available_ ||
      stamp.isZero() || stamp < last_stamp_) {
    if (reasons)
      reasons->push_back(!initialized_ ? ReasonCode::REFERENCE_NOT_FROZEN
                                      : ReasonCode::INPUT_TIME_INVALID);
    return false;
  }
  for (const GduState& state : states_) {
    const bool has_history = state.current_measurement_valid ||
        state.has_pose_prior || !state.last_pose_stamp.isZero() ||
        !state.last_valid_velocity_stamp.isZero() ||
        state.continuous_valid_velocity_duration_s != 0.0 ||
        std::any_of(state.information_weight_history.begin(),
                    state.information_weight_history.end(),
                    [](const auto& history) { return !history.empty(); }) ||
        std::any_of(state.observable_frame_history.begin(),
                    state.observable_frame_history.end(),
                    [](const auto& history) { return !history.empty(); });
    if (has_history) {
      if (reasons) reasons->push_back(ReasonCode::INPUT_TIME_INVALID);
      return false;
    }
  }
  last_stamp_ = stamp;
  first_observation_reanchor_available_ = false;
  return true;
}

std::vector<UacGaussianSumFilter::CoherenceCompetitionGroup>
UacGaussianSumFilter::CoherenceCompetitionGroups(uint32_t gdu_index) const {
  std::vector<CoherenceCompetitionGroup> output;
  if (gdu_index >= reference_.gdus.size()) return output;
  const GduDefinition& gdu = reference_.gdus[gdu_index];
  std::map<StableId, size_t> group_location;
  for (size_t local = 0; local < gdu.interface_indices.size(); ++local) {
    const uint32_t interface_index = gdu.interface_indices[local];
    if (interface_index >= reference_.interfaces.size()) return {};
    const InterfaceDefinition& interface = reference_.interfaces[interface_index];
    const StableId group_id = interface.competition_group_id.empty()
        ? interface.id : interface.competition_group_id;
    auto inserted = group_location.emplace(group_id, output.size());
    if (inserted.second) {
      CoherenceCompetitionGroup group;
      group.id = group_id;
      group.p_link = interface.p_link;
      output.push_back(group);
    }
    CoherenceCompetitionGroup& group = output[inserted.first->second];
    group.p_link = std::min(group.p_link, interface.p_link);
    group.local_interface_indices.push_back(local);
    group.normalized_weights.push_back(interface.competition_weight);
  }
  for (CoherenceCompetitionGroup& group : output) {
    double total = 0.0;
    for (double weight : group.normalized_weights)
      if (std::isfinite(weight) && weight > 0.0) total += weight;
    if (!(total > 0.0)) return {};
    for (double& weight : group.normalized_weights) weight /= total;
  }
  return output;
}

UacGaussianSumFilter::LinearKernel UacGaussianSumFilter::BuildKernel(
    StructuralState source, StructuralState destination,
    const std::vector<InterfaceJointState>& destination_modes,
    uint32_t gdu_index, double delta_t) const {
  const int n = states_[gdu_index].coefficient_count;
  const int source_physical_dim = PhysicalDimension(source, n);
  const int destination_physical_dim = PhysicalDimension(destination, n);
  const int source_dim = StateDimension(source, n);
  const int destination_dim = StateDimension(destination, n);
  LinearKernel kernel;
  kernel.F = Eigen::MatrixXd::Zero(destination_dim, source_dim);
  kernel.Q = Eigen::MatrixXd::Zero(destination_dim, destination_dim);
  kernel.F.block(destination_physical_dim, source_physical_dim, 15, 15).setIdentity();
  if (destination == StructuralState::U) return kernel;
  if (source == StructuralState::U && destination == StructuralState::A) {
    kernel.Q.topLeftCorner(n, n).diagonal().array() =
        config_.calibration.active_birth_displacement_variance;
    kernel.Q.block(n, n, n, n).diagonal().array() =
        config_.calibration.active_birth_velocity_variance;
  } else if (source == StructuralState::U && destination == StructuralState::C) {
    kernel.Q.topLeftCorner(n, n).diagonal().array() =
        config_.calibration.changed_birth_displacement_variance;
  } else if (source == StructuralState::A && destination == StructuralState::A) {
    kernel.F.topLeftCorner(n, n).setIdentity();
    kernel.F.block(0, n, n, n) = delta_t * Eigen::MatrixXd::Identity(n, n);
    kernel.F.block(n, n, n, n).setIdentity();
    const double q = config_.calibration.active_acceleration_psd;
    kernel.Q.topLeftCorner(n, n).diagonal().array() = q * delta_t * delta_t * delta_t / 3.0;
    kernel.Q.block(0, n, n, n).diagonal().array() = q * delta_t * delta_t / 2.0;
    kernel.Q.block(n, 0, n, n).diagonal().array() = q * delta_t * delta_t / 2.0;
    kernel.Q.block(n, n, n, n).diagonal().array() = q * delta_t;
  } else if (source == StructuralState::A && destination == StructuralState::C) {
    kernel.F.leftCols(n).setIdentity();
    kernel.F.block(0, n, n, n) = delta_t * Eigen::MatrixXd::Identity(n, n);
    kernel.Q.topLeftCorner(n, n).diagonal().array() = config_.calibration.active_acceleration_psd *
                                  delta_t * delta_t * delta_t / 3.0 +
                                  config_.calibration.changed_random_walk_psd * delta_t;
  } else if (source == StructuralState::C && destination == StructuralState::C) {
    kernel.F.setIdentity();
    kernel.Q.topLeftCorner(n, n).diagonal().array() =
        config_.calibration.changed_random_walk_psd * delta_t;
  } else if (source == StructuralState::C && destination == StructuralState::A) {
    kernel.F.topRows(n).setIdentity();
    kernel.Q.topLeftCorner(n, n).diagonal().array() =
        config_.calibration.changed_random_walk_psd * delta_t;
    kernel.Q.block(n, n, n, n).diagonal().array() =
        config_.calibration.active_birth_velocity_variance;
  }

  if (destination_physical_dim == 0 || kernel.Q.rows() == 0)
    return kernel;
  // Apply the normalized interface innovation product in the frozen
  
  // by field evaluation; coherent/relative LINK modes constrain the same
  // preallocated jump coefficient without changing state dimension.
  // UNRESOLVED is not a weak pseudo-measurement: it broadens the jump process
  // into a protective kernel.  For the singular U->A birth, the jump-position
  // slab is injected once here while the velocity birth remains the base
  // prior.  The coherent U->A branch already has an exact zero jump position.
  const GduDefinition& gdu = reference_.gdus[gdu_index];
  const int coordinate_rank = gdu.coordinate_rank;
  for (size_t i = 0; i < destination_modes.size(); ++i) {
    const InterfaceJointState mode = destination_modes[i];
    if (mode != InterfaceJointState::LINK_UNRESOLVED &&
        !(source == StructuralState::U &&
          destination == StructuralState::A &&
          mode == InterfaceJointState::LINK_RELATIVE_MOTION))
      continue;
    const int jump_column = coordinate_rank * static_cast<int>(
        gdu.control_surfel_indices.size() + i);
    if (coordinate_rank <= 0 || jump_column + coordinate_rank > n) continue;
    const double variance =
        mode == InterfaceJointState::LINK_RELATIVE_MOTION
            ? config_.calibration.interface_relative_process_variance
            : config_.calibration.interface_unresolved_process_variance;
    kernel.Q.block(jump_column, jump_column, coordinate_rank,
                   coordinate_rank).diagonal().array() += variance;
  }
  if (source == StructuralState::U && destination == StructuralState::A) {
    for (size_t i = 0; i < destination_modes.size(); ++i) {
      const int jump_column = coordinate_rank * static_cast<int>(
          gdu.control_surfel_indices.size() + i);
      if (coordinate_rank <= 0 || jump_column + coordinate_rank > n) continue;
      double variance = 0.0;
      if (destination_modes[i] == InterfaceJointState::LINK_RELATIVE_MOTION)
        variance = config_.calibration.interface_relative_process_variance;
      else if (destination_modes[i] == InterfaceJointState::LINK_UNRESOLVED)
        variance = config_.calibration.interface_unresolved_process_variance;
      kernel.Q.block(jump_column, jump_column, coordinate_rank,
                     coordinate_rank).setZero();
      kernel.Q.block(jump_column, jump_column, coordinate_rank,
                     coordinate_rank).diagonal().array() = variance;
    }
    return kernel;
  }
  int constraint_count = 0;
  for (InterfaceJointState mode : destination_modes)
    if (mode == InterfaceJointState::LINK_COHERENT ||
        mode == InterfaceJointState::LINK_RELATIVE_MOTION)
      constraint_count += coordinate_rank;
  if (constraint_count == 0) return kernel;
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(constraint_count, destination_physical_dim);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(constraint_count, source_physical_dim);
  Eigen::MatrixXd Q_gamma = Eigen::MatrixXd::Zero(constraint_count, constraint_count);
  int row = 0;
  for (size_t i = 0; i < destination_modes.size(); ++i) {
    const InterfaceJointState mode = destination_modes[i];
    if (mode != InterfaceJointState::LINK_COHERENT &&
        mode != InterfaceJointState::LINK_RELATIVE_MOTION)
      continue;
    const int jump_column = coordinate_rank * static_cast<int>(
        gdu.control_surfel_indices.size() + i);
    L.block(row, jump_column, coordinate_rank, coordinate_rank).setIdentity();
    double rho = 0.0;
    double variance = config_.calibration.interface_coherent_process_variance;
    if (mode == InterfaceJointState::LINK_COHERENT) {
      rho = config_.calibration.interface_coherent_rho;
      const uint32_t interface_index = gdu.interface_indices[i];
      const double allocation = interface_index < reference_.interfaces.size()
          ? reference_.interfaces[interface_index].competition_weight : 1.0;
      // All non-selected candidates remain coherent.  Allocate the coherent
      // innovation variance with the same finite group measure so their
      // summed field uncertainty cannot grow merely because the library was
      // refined.
      variance = config_.calibration.interface_coherent_process_variance *
          std::max(config_.numerical.probability_floor, allocation);
    } else if (mode == InterfaceJointState::LINK_RELATIVE_MOTION) {
      rho = config_.calibration.interface_relative_rho;
      variance = config_.calibration.interface_relative_process_variance;
    }
    if (source != StructuralState::U)
      R.block(row, jump_column, coordinate_rank, coordinate_rank) =
          rho * Eigen::MatrixXd::Identity(coordinate_rank, coordinate_rank);
    Q_gamma.block(row, row, coordinate_rank, coordinate_rank) =
        variance * Eigen::MatrixXd::Identity(coordinate_rank, coordinate_rank);
    row += coordinate_rank;
  }
  const Eigen::MatrixXd F0 = kernel.F.topLeftCorner(
      destination_physical_dim, source_physical_dim);
  const Eigen::MatrixXd Q0 = 0.5 * (
      kernel.Q.topLeftCorner(destination_physical_dim,
                             destination_physical_dim) +
      kernel.Q.topLeftCorner(destination_physical_dim,
                             destination_physical_dim).transpose());
  const Eigen::MatrixXd innovation =
      0.5 * (L * Q0 * L.transpose() + Q_gamma +
             (L * Q0 * L.transpose() + Q_gamma).transpose());
  Eigen::MatrixXd innovation_lower;
  if (!StableCholesky(innovation, config_.numerical,
                      &innovation_lower)) {
    kernel.valid = false;
    return kernel;
  }
  // K = Q0 L' S^{-1}, formed by a solve rather than any explicit inverse.
  const Eigen::MatrixXd forward =
      innovation_lower.triangularView<Eigen::Lower>().solve(L * Q0);
  const Eigen::MatrixXd gain =
      innovation_lower.transpose().triangularView<Eigen::Upper>()
          .solve(forward).transpose();
  const Eigen::MatrixXd F_effective = F0 + gain * (R - L * F0);
  const Eigen::MatrixXd Q_effective = Q0 - gain * L * Q0;
  kernel.F.topLeftCorner(destination_physical_dim, source_physical_dim) = F_effective;
  kernel.Q.topLeftCorner(destination_physical_dim, destination_physical_dim) =
      0.5 * (Q_effective + Q_effective.transpose());
  return kernel;
}

bool UacGaussianSumFilter::AdvancePoseNuisance(
    GduState* state,
    const Eigen::Matrix<double, 6, 6>& current_pose_covariance,
    const ros::Time& stamp, std::vector<ReasonCode>* reasons) const {
  if (!state || !ValidateSymmetricPsd(current_pose_covariance,
                                      config_.numerical.psd_eigenvalue_tolerance)) {
    if (reasons) reasons->push_back(ReasonCode::INPUT_POSE_COVARIANCE_INVALID);
    return false;
  }
  Eigen::MatrixXd current_lower_dynamic;
  if (!StableCholesky(current_pose_covariance, config_.numerical,
                      &current_lower_dynamic) ||
      current_lower_dynamic.rows() != 6) {
    if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    return false;
  }
  const Eigen::Matrix<double, 6, 6> current_lower = current_lower_dynamic;
  Eigen::Matrix<double, 6, 6> transition =
      Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> process = current_pose_covariance;
  if (state->has_pose_prior) {
    const double delta_t = (stamp - state->last_pose_stamp).toSec();
    if (!(delta_t > 0.0)) {
      if (reasons) reasons->push_back(ReasonCode::INPUT_TIME_INVALID);
      return false;
    }
    Eigen::MatrixXd previous_lower_dynamic;
    if (!StableCholesky(state->last_pose_prior_covariance, config_.numerical,
                        &previous_lower_dynamic) ||
        previous_lower_dynamic.rows() != 6) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
    const Eigen::Matrix<double, 6, 6> previous_lower = previous_lower_dynamic;
    const double rho = std::exp(-delta_t / config_.calibration.pose_ou_tau_s);
    // K_{k,k-1}=L_k rho L_{k-1}^T.  Therefore the conditional transition is
    // K_{k,k-1} Sigma_{k-1}^{-1}=rho L_k L_{k-1}^{-1}.
    transition = rho * current_lower *
        previous_lower.triangularView<Eigen::Lower>().solve(
            Eigen::Matrix<double, 6, 6>::Identity());
    process = (1.0 - rho * rho) * current_pose_covariance;
  }
  for (GaussianComponent& component : state->components) {
    const int dimension = static_cast<int>(component.mean.size());
    if (dimension < 15 || component.covariance.rows() != dimension) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
    const int pose_offset = dimension - 6;
    Eigen::MatrixXd mapping = Eigen::MatrixXd::Identity(dimension, dimension);
    mapping.block<6, 6>(pose_offset, pose_offset) = transition;
    component.mean = mapping * component.mean;
    component.covariance = mapping * component.covariance * mapping.transpose();
    component.covariance.block<6, 6>(pose_offset, pose_offset) += process;
    component.covariance = 0.5 *
        (component.covariance + component.covariance.transpose());
  }
  state->last_pose_prior_covariance = current_pose_covariance;
  state->last_pose_stamp = stamp;
  state->has_pose_prior = true;
  return true;
}

bool UacGaussianSumFilter::BuildCurrentObservableInformation(
    GduState* state, const std::vector<AssociationMeasurementSet>& measurements,
    const FrameNuisance& nuisance, const ros::Time& stamp,
    std::vector<ReasonCode>* reasons) const {
  if (!state) return false;
  const int n = state->coefficient_count;
  for (Eigen::MatrixXd& information : state->current_observable_information)
    information = Eigen::MatrixXd::Zero(n, n);
  state->reference_information_ratio = {{1.0, 1.0, 1.0, 1.0}};
  state->reference_confounding = {{false, false, false, false}};
  state->effective_spatial_support = {{0.0, 0.0, 0.0, 0.0}};
  state->effective_temporal_support = {{0.0, 0.0, 0.0, 0.0}};
  const auto record_effective_support =
      [this, state, &stamp](int geometry_index,
                            const std::vector<double>& source_weights) {
        double total_weight = 0.0;
        double squared_weight_sum = 0.0;
        for (double weight : source_weights) {
          if (!(weight > 0.0) || !std::isfinite(weight)) continue;
          total_weight += weight;
          squared_weight_sum += weight * weight;
        }
        if (squared_weight_sum > 0.0) {
          state->effective_spatial_support[static_cast<size_t>(geometry_index)] =
              total_weight * total_weight / squared_weight_sum;
        }

        auto& history = state->information_weight_history[
            static_cast<size_t>(geometry_index)];
        // Replacing a repeated timestamp makes the statistic idempotent if an
        // upstream caller retries the same transaction before publication.
        if (!history.empty() && history.back().first == stamp)
          history.back().second = total_weight;
        else
          history.emplace_back(stamp, total_weight);
        while (history.size() >
               static_cast<size_t>(config_.resource.fixed_lag_frames))
          history.pop_front();

        double temporal_weight = 0.0;
        for (const auto& sample : history)
          if (sample.second > 0.0 && std::isfinite(sample.second))
            temporal_weight += sample.second;
        if (!(temporal_weight > 0.0)) return;
        double correlated_square_sum = 0.0;
        for (const auto& lhs : history) {
          if (!(lhs.second > 0.0) || !std::isfinite(lhs.second)) continue;
          for (const auto& rhs : history) {
            if (!(rhs.second > 0.0) || !std::isfinite(rhs.second)) continue;
            const double separation_s =
                std::abs((lhs.first - rhs.first).toSec());
            const double correlation = std::exp(
                -separation_s / config_.calibration.pose_ou_tau_s);
            correlated_square_sum += lhs.second * rhs.second * correlation;
          }
        }
        if (correlated_square_sum > 0.0 &&
            std::isfinite(correlated_square_sum)) {
          state->effective_temporal_support[
              static_cast<size_t>(geometry_index)] =
              temporal_weight * temporal_weight / correlated_square_sum;
        }
      };
  const auto record_observable_block =
      [this, state](int geometry_index,
                    const ObservableFrameBlock& block) {
        auto& history = state->observable_frame_history[
            static_cast<size_t>(geometry_index)];
        if (!history.empty() && history.back().stamp == block.stamp)
          history.back() = block;
        else
          history.push_back(block);
        while (history.size() >
               static_cast<size_t>(config_.resource.fixed_lag_frames))
          history.pop_front();
      };
  for (int geometry_index = 0; geometry_index < 4; ++geometry_index) {
    int rows = 0;
    for (const AssociationMeasurementSet& set : measurements) {
      std::map<StableId, double> category_probability;
      for (const AssociationCategory& category : set.selection.categories)
        if (category.geometric_candidate)
          category_probability[category.candidate_id] = category.probability;
      for (const FieldMeasurement& measurement : set.candidates) {
        const auto probability = category_probability.find(measurement.candidate_id);
        const HypothesisMeasurement& model =
            measurement.hypothesis[static_cast<size_t>(geometry_index)];
        if (probability != category_probability.end() && probability->second > 0.0 &&
            model.residual_m.size() > 0) rows += model.residual_m.size();
      }
    }
    if (rows == 0) {
      record_effective_support(geometry_index, {});
      ObservableFrameBlock empty_block;
      empty_block.stamp = stamp;
      empty_block.design = Eigen::MatrixXd::Zero(0, n);
      record_observable_block(geometry_index, empty_block);
      continue;
    }
    Eigen::MatrixXd design = Eigen::MatrixXd::Zero(rows, n);
    Eigen::MatrixXd nuisance_design = Eigen::MatrixXd::Zero(rows, 15);
    Eigen::MatrixXd independent = Eigen::MatrixXd::Zero(rows, rows);
    std::vector<std::pair<int, int>> source_row_blocks;
    int row = 0;
    for (const AssociationMeasurementSet& set : measurements) {
      const int source_first_row = row;
      std::map<StableId, double> category_probability;
      for (const AssociationCategory& category : set.selection.categories)
        if (category.geometric_candidate)
          category_probability[category.candidate_id] = category.probability;
      for (const FieldMeasurement& measurement : set.candidates) {
        const auto probability = category_probability.find(measurement.candidate_id);
        const HypothesisMeasurement& model =
            measurement.hypothesis[static_cast<size_t>(geometry_index)];
        if (probability == category_probability.end() || !(probability->second > 0.0) ||
            model.residual_m.size() <= 0 || model.design.cols() != n) continue;
        const int dimension = model.residual_m.size();
        const double root_probability = std::sqrt(probability->second);
        design.block(row, 0, dimension, n) = root_probability * model.design;
        nuisance_design.block(row, 0, dimension, 3) =
            root_probability * model.reference_bias_design;
        nuisance_design.block(row, 3, dimension, 6) =
            root_probability * model.extrinsic_design;
        nuisance_design.block(row, 9, dimension, 6) =
            root_probability * model.pose_design;
        independent.block(row, row, dimension, dimension) =
            model.independent_covariance_m2;
        row += dimension;
      }
      if (row > source_first_row)
        source_row_blocks.emplace_back(source_first_row, row - source_first_row);
    }
    if (row != rows) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
    Eigen::MatrixXd nuisance_covariance = Eigen::MatrixXd::Zero(15, 15);
    nuisance_covariance.topLeftCorner<3, 3>() =
        state->reference_bias_prior_covariance;
    nuisance_covariance.block<6, 6>(3, 3) = config_.resource.extrinsic.covariance;
    nuisance_covariance.block<6, 6>(9, 9) = nuisance.pose_covariance;
    ObservableFrameBlock observable_block;
    observable_block.stamp = stamp;
    observable_block.design = design;
    observable_block.independent_covariance = independent;
    observable_block.reference_bias_design = nuisance_design.leftCols(3);
    observable_block.extrinsic_design = nuisance_design.block(0, 3, rows, 6);
    observable_block.pose_design = nuisance_design.block(0, 9, rows, 6);
    observable_block.pose_covariance = nuisance.pose_covariance;
    record_observable_block(geometry_index, observable_block);
    Eigen::MatrixXd marginal = independent +
        nuisance_design * nuisance_covariance * nuisance_design.transpose();
    Eigen::MatrixXd lower;
    if (!StableCholesky(marginal, config_.numerical, &lower)) {
      record_effective_support(geometry_index, {});
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
    const Eigen::MatrixXd whitened =
        lower.triangularView<Eigen::Lower>().solve(design);
    std::vector<double> source_weights;
    source_weights.reserve(source_row_blocks.size());
    for (const auto& block : source_row_blocks)
      source_weights.push_back(
          whitened.block(block.first, 0, block.second, n).squaredNorm());
    record_effective_support(geometry_index, source_weights);
    const Eigen::MatrixXd information = whitened.transpose() * whitened;
    state->current_observable_information[static_cast<size_t>(geometry_index)] =
        information;

    // Reference/deformation confounding is measured against the information
    // that the same rows would provide if the epoch-static reference bias
    // were known.  On the positive range of I_known, the generalized matrix
    // I_known^{-1/2} I_marginal I_known^{-1/2} has dimensionless eigenvalues
    // in [0,1]; its smallest eigenvalue is the retained-information ratio.
    Eigen::MatrixXd covariance_without_reference = nuisance_covariance;
    covariance_without_reference.topLeftCorner<3, 3>().setZero();
    const Eigen::MatrixXd marginal_without_reference = independent +
        nuisance_design * covariance_without_reference *
            nuisance_design.transpose();
    Eigen::MatrixXd known_lower;
    if (!StableCholesky(marginal_without_reference, config_.numerical,
                        &known_lower)) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
    const Eigen::MatrixXd known_whitened =
        known_lower.triangularView<Eigen::Lower>().solve(design);
    const Eigen::MatrixXd known_information =
        known_whitened.transpose() * known_whitened;
    double ratio = 0.0;
    if (!MinimumRetainedInformationRatio(
            known_information, information,
            config_.numerical.psd_eigenvalue_tolerance, &ratio)) continue;
    state->reference_information_ratio[static_cast<size_t>(geometry_index)] =
        ratio;
    state->reference_confounding[static_cast<size_t>(geometry_index)] =
        ratio + config_.numerical.tie_tolerance <
        config_.calibration.reference_confounding_min_information_ratio;
    if (geometry_index < 3 &&
        state->reference_confounding[static_cast<size_t>(geometry_index)] &&
        reasons)
      reasons->push_back(ReasonCode::DEFORMATION_REFERENCE_AMBIGUOUS);
  }
  return true;
}

UacGaussianSumFilter::TaskEventProbability
UacGaussianSumFilter::ComponentTaskScaleProbability(
    uint32_t gdu_index, const GaussianComponent& component) const {
  TaskEventProbability output;
  if (component.state == StructuralState::U || gdu_index >= states_.size())
    return output;
  const int n = states_[gdu_index].coefficient_count;
  const int physical_dimension = PhysicalDimension(component.state, n);
  if (n <= 0 || physical_dimension <= 0 ||
      component.mean.size() < physical_dimension ||
      component.covariance.rows() < physical_dimension)
    return output;
  const GduDefinition& gdu = reference_.gdus[gdu_index];
  if (gdu.surfel_indices.empty()) return output;
  // The smooth physical bases are frozen-reference invariants populated by
  // Initialize().  Component-specific mean/covariance and deterministic QMC
  // sampling remain below and are intentionally not cached.
  if (!states_[gdu_index].frozen_reference_cache) return output;
  const std::vector<Eigen::MatrixXd>& bases =
      states_[gdu_index].frozen_reference_cache->task_bases;
  if (bases.empty()) return output;
  const Eigen::MatrixXd coefficient_covariance =
      0.5 * (component.covariance.topLeftCorner(physical_dimension,
                                                physical_dimension) +
             component.covariance.topLeftCorner(physical_dimension,
                                                 physical_dimension).transpose());
  Eigen::MatrixXd square_root;
  if (!DeterministicPsdSquareRoot(coefficient_covariance,
                                  config_.numerical, &square_root)) {
    output.displacement = 1.0;
    output.current_velocity = component.state == StructuralState::A ? 1.0 : 0.0;
    output.interface_jump = gdu.interface_indices.empty() ? 0.0 : 1.0;
    output.event_union_without_velocity = 1.0;
    output.event_union = 1.0;
    return output;
  }
  boost::random::sobol engine(static_cast<unsigned int>(physical_dimension));
  CanonicalBytes shift_fields;
  shift_fields.AddId(component.component_id);
  shift_fields.AddId(reference_.gdus[gdu_index].id);
  shift_fields.AddString("task-scale-sobol-v1");
  std::vector<uint64_t> digital_shifts(
      static_cast<size_t>(physical_dimension), 0U);
  for (int dimension = 0; dimension < physical_dimension; ++dimension) {
    CanonicalBytes dimension_fields = shift_fields;
    dimension_fields.AddI64(dimension);
    const std::array<uint8_t, 32> digest = Sha256(dimension_fields.bytes());
    uint64_t word = 0U;
    for (size_t byte = 0; byte < 8U; ++byte)
      word = (word << 8U) | digest[byte];
    digital_shifts[static_cast<size_t>(dimension)] =
        word & static_cast<uint64_t>(engine.max());
  }
  const auto field_rms = [&bases](const Eigen::VectorXd& coefficient) {
    double squared_norm_sum = 0.0;
    for (const Eigen::MatrixXd& basis : bases)
      squared_norm_sum += (basis * coefficient).squaredNorm();
    return std::sqrt(squared_norm_sum /
                     static_cast<double>(bases.size()));
  };
  double displacement_count = 0.0;
  double velocity_count = 0.0;
  double interface_count = 0.0;
  double union_without_velocity_count = 0.0;
  double union_count = 0.0;
  for (int sample = 0; sample < config_.resource.prediction_samples; ++sample) {
    Eigen::VectorXd standard_normal(physical_dimension);
    for (int dimension = 0; dimension < physical_dimension; ++dimension) {
      const uint64_t raw = static_cast<uint64_t>(engine()) ^
          digital_shifts[static_cast<size_t>(dimension)];
      const long double denominator = static_cast<long double>(engine.max()) + 1.0L;
      const double uniform = static_cast<double>(
          (static_cast<long double>(raw) + 0.5L) / denominator);
      standard_normal[dimension] = InverseNormalCdf(
          std::max(1.0e-15, std::min(1.0 - 1.0e-15, uniform)));
    }
    const Eigen::VectorXd physical_sample =
        component.mean.head(physical_dimension) + square_root * standard_normal;
    const Eigen::VectorXd displacement = physical_sample.head(n);
    const bool displacement_event =
        field_rms(displacement) >=
        config_.calibration.task_displacement_scale_m;
    bool interface_event = false;
    for (size_t interface = 0; interface < component.interface_modes.size();
         ++interface) {
      if (component.interface_modes[interface] ==
          InterfaceJointState::CUT_NOT_APPLICABLE) continue;
      const int coordinate_rank = gdu.coordinate_rank;
      const int offset = coordinate_rank * static_cast<int>(
          gdu.control_surfel_indices.size() + interface);
      if (coordinate_rank <= 0 || offset + coordinate_rank > n) continue;
      const Eigen::Vector3d jump = gdu.coordinate_axes_R.leftCols(coordinate_rank) *
          displacement.segment(offset, coordinate_rank);
      const double jump_norm_squared =
          jump.dot(config_.calibration.task_interface_metric * jump);
      if (jump_norm_squared >=
          config_.calibration.task_interface_jump_scale_m *
          config_.calibration.task_interface_jump_scale_m) {
        interface_event = true;
        break;
      }
    }
    bool current_velocity_event = false;
    if (component.state == StructuralState::A) {
      const Eigen::VectorXd velocity = physical_sample.segment(n, n);
      current_velocity_event = field_rms(velocity) >=
          config_.calibration.task_velocity_scale_mps;
    }
    displacement_count += displacement_event ? 1.0 : 0.0;
    interface_count += interface_event ? 1.0 : 0.0;
    velocity_count += current_velocity_event ? 1.0 : 0.0;
    union_without_velocity_count +=
        (displacement_event || interface_event) ? 1.0 : 0.0;
    union_count += (displacement_event || interface_event ||
                    current_velocity_event) ? 1.0 : 0.0;
  }
  const double denominator =
      static_cast<double>(config_.resource.prediction_samples);
  output.displacement = displacement_count / denominator;
  output.current_velocity = velocity_count / denominator;
  output.interface_jump = interface_count / denominator;
  output.event_union_without_velocity =
      union_without_velocity_count / denominator;
  output.event_union = union_count / denominator;
  return output;
}

bool UacGaussianSumFilter::SynchronizeSharedInterfaces(
    std::vector<ReasonCode>* reasons) {
  if (shared_interfaces_.size() != reference_.interfaces.size())
    shared_interfaces_.resize(reference_.interfaces.size());
  struct SideMoments {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::array<double, 4> mass{{0.0, 0.0, 0.0, 0.0}};
    std::array<Eigen::Vector3d, 4> mean{{
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}};
    std::array<Eigen::Matrix3d, 4> covariance{{
        Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(),
        Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()}};
    double other = 0.0;
  };
  for (uint32_t interface_index = 0;
       interface_index < reference_.interfaces.size(); ++interface_index) {
    const InterfaceDefinition& definition = reference_.interfaces[interface_index];
    if (!definition.cross_gdu) {
      const uint32_t gdu_index = definition.left_gdu;
      if (gdu_index >= reference_.gdus.size() ||
          definition.right_gdu != gdu_index) {
        if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
        return false;
      }
      const GduDefinition& gdu = reference_.gdus[gdu_index];
      const auto location = std::find(gdu.interface_indices.begin(),
                                      gdu.interface_indices.end(), interface_index);
      if (location == gdu.interface_indices.end()) {
        if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
        return false;
      }
      const size_t local = static_cast<size_t>(
          location - gdu.interface_indices.begin());
      const int coordinate_rank = gdu.coordinate_rank;
      const int coefficient = coordinate_rank * static_cast<int>(
          gdu.control_surfel_indices.size() + local);
      if (coordinate_rank <= 0 || coefficient + coordinate_rank >
          states_[gdu_index].coefficient_count) {
        if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
        return false;
      }
      const Eigen::MatrixXd axes =
          gdu.coordinate_axes_R.leftCols(coordinate_rank);
      SideMoments local_moments;
      local_moments.other = states_[gdu_index].other_mass;
      for (const GaussianComponent& component : states_[gdu_index].components) {
        if (local >= component.interface_modes.size()) return false;
        const size_t mode = static_cast<size_t>(component.interface_modes[local]);
        local_moments.mass[mode] += component.weight;
        if (component.state != StructuralState::U && mode != 0U)
          local_moments.mean[mode] += component.weight *
              axes * component.mean.segment(coefficient, coordinate_rank);
      }
      for (size_t mode = 0; mode < 4U; ++mode)
        if (local_moments.mass[mode] > 0.0)
          local_moments.mean[mode] /= local_moments.mass[mode];
      for (const GaussianComponent& component : states_[gdu_index].components) {
        const size_t mode = static_cast<size_t>(component.interface_modes[local]);
        Eigen::Vector3d value = Eigen::Vector3d::Zero();
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        if (component.state != StructuralState::U && mode != 0U) {
          value = axes * component.mean.segment(
              coefficient, coordinate_rank);
          covariance = axes * component.covariance.block(
              coefficient, coefficient, coordinate_rank, coordinate_rank) *
              axes.transpose();
        }
        const Eigen::Vector3d delta = value - local_moments.mean[mode];
        local_moments.covariance[mode] += component.weight *
            (covariance + delta * delta.transpose());
      }
      SharedInterfaceState shared;
      shared.other_mass = local_moments.other;
      for (size_t mode = 0; mode < 4U; ++mode) {
        shared.probability[mode] = local_moments.mass[mode];
        shared.mean[mode] = local_moments.mean[mode];
        if (local_moments.mass[mode] > 0.0)
          shared.covariance[mode] =
              local_moments.covariance[mode] / local_moments.mass[mode];
      }
      shared_interfaces_[interface_index] = shared;
      continue;
    }
    // Initialize() rejects cross-GDU jump coordinates for the frozen V1.1
    // model.  Keep this guard so a corrupted snapshot cannot silently enter
    // an unregistered synchronization approximation.
    if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    return false;
  }
  return true;
}

std::vector<std::pair<std::vector<InterfaceJointState>, double>>
UacGaussianSumFilter::CoherenceBranches(const GaussianComponent& parent,
                                        StructuralState destination,
                                        uint32_t gdu_index,
                                        double* pruned_probability) const {
  const std::vector<CoherenceCompetitionGroup> groups =
      CoherenceCompetitionGroups(gdu_index);
  return CoherenceBranchesFromGroups(parent, destination, gdu_index, groups,
                                     pruned_probability);
}

std::vector<std::pair<std::vector<InterfaceJointState>, double>>
UacGaussianSumFilter::CoherenceBranchesFromGroups(
    const GaussianComponent& parent, StructuralState destination,
    uint32_t gdu_index,
    const std::vector<CoherenceCompetitionGroup>& groups,
    double* pruned_probability) const {
  if (pruned_probability) *pruned_probability = 0.0;
  if (parent.interface_modes.size() !=
          reference_.gdus[gdu_index].interface_indices.size() ||
      (!parent.interface_modes.empty() && groups.empty())) {
    if (pruned_probability) *pruned_probability = 1.0;
    return {};
  }
  std::vector<std::pair<std::vector<InterfaceJointState>, double>> branches(1);
  branches[0].first = parent.interface_modes;
  branches[0].second = 1.0;
  const size_t intermediate_cap = static_cast<size_t>(
      config_.resource.mixture_max_components * 4);
  for (const CoherenceCompetitionGroup& group : groups) {
    std::vector<std::pair<std::vector<InterfaceJointState>, double>> expanded;
    bool all_cut = true;
    bool any_cut = false;
    size_t active_local = std::numeric_limits<size_t>::max();
    InterfaceJointState active_mode = InterfaceJointState::LINK_COHERENT;
    for (size_t local : group.local_interface_indices) {
      const InterfaceJointState mode = parent.interface_modes[local];
      any_cut = any_cut || mode == InterfaceJointState::CUT_NOT_APPLICABLE;
      all_cut = all_cut && mode == InterfaceJointState::CUT_NOT_APPLICABLE;
      if (mode == InterfaceJointState::LINK_RELATIVE_MOTION ||
          mode == InterfaceJointState::LINK_UNRESOLVED) {
        if (active_local != std::numeric_limits<size_t>::max()) {
          if (pruned_probability)
            for (const auto& branch : branches)
              *pruned_probability += branch.second;
          return {};
        }
        active_local = local;
        active_mode = mode;
      }
    }
    if (any_cut != all_cut) {
      if (pruned_probability)
        for (const auto& branch : branches)
          *pruned_probability += branch.second;
      return {};
    }
    if (all_cut) {
      expanded = branches;
    } else {
      const int source_mode = active_local == std::numeric_limits<size_t>::max()
          ? 0 : LinkModeIndex(active_mode);
      double legal_row_sum = 0.0;
      for (int target_mode = 0; target_mode < 3; ++target_mode) {
        if (destination == StructuralState::U && target_mode == 1) continue;
        legal_row_sum += config_.calibration.interface_coherence_transition(
            source_mode, target_mode);
      }
      if (!(legal_row_sum > 0.0)) {
        if (pruned_probability)
          for (const auto& branch : branches)
            *pruned_probability += branch.second;
        return {};
      }
      for (const auto& branch : branches) {
        for (int target_mode = 0; target_mode < 3; ++target_mode) {
          if (destination == StructuralState::U && target_mode == 1) continue;
          const double transition_probability =
              config_.calibration.interface_coherence_transition(
                  source_mode, target_mode) / legal_row_sum;
          if (!(transition_probability > 0.0)) continue;
          if (target_mode == 0) {
            auto child = branch;
            for (size_t local : group.local_interface_indices)
              child.first[local] = InterfaceJointState::LINK_COHERENT;
            child.second *= transition_probability;
            expanded.push_back(child);
            continue;
          }
          const InterfaceJointState target = target_mode == 1
              ? InterfaceJointState::LINK_RELATIVE_MOTION
              : InterfaceJointState::LINK_UNRESOLVED;
          if (active_local != std::numeric_limits<size_t>::max()) {
            auto child = branch;
            for (size_t local : group.local_interface_indices)
              child.first[local] = InterfaceJointState::LINK_COHERENT;
            child.first[active_local] = target;
            child.second *= transition_probability;
            expanded.push_back(child);
          } else {
            for (size_t candidate = 0;
                 candidate < group.local_interface_indices.size(); ++candidate) {
              const double allocation = group.normalized_weights[candidate];
              if (!(allocation > 0.0)) continue;
              auto child = branch;
              for (size_t local : group.local_interface_indices)
                child.first[local] = InterfaceJointState::LINK_COHERENT;
              child.first[group.local_interface_indices[candidate]] = target;
              child.second *= transition_probability * allocation;
              expanded.push_back(child);
            }
          }
        }
      }
    }
    std::sort(expanded.begin(), expanded.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.second != rhs.second) return lhs.second > rhs.second;
      return lhs.first < rhs.first;
    });
    if (expanded.size() > intermediate_cap) {
      if (pruned_probability)
        for (size_t i = intermediate_cap; i < expanded.size(); ++i)
          *pruned_probability += expanded[i].second;
      expanded.resize(intermediate_cap);
    }
    branches.swap(expanded);
  }
  return branches;
}

GaussianComponent UacGaussianSumFilter::ApplyKernel(
    const GaussianComponent& parent, StructuralState destination,
    const std::vector<InterfaceJointState>& modes,
    const LinearKernel& kernel, bool assign_transient_component_id,
    bool* valid) const {
  if (valid) *valid = kernel.valid;
  GaussianComponent child;
  if (assign_transient_component_id) {
    child = parent;
  } else {
    // Prediction only carries persistent probabilistic state.  The parent's
    // information/observability matrices are derived products: mean and
    // covariance are replaced below, and RefreshDerived reconstructs the
    // final information factor after all substeps.  Copying those dense
    // matrices into every raw branch was therefore exact but redundant.
    child.state = parent.state;
    child.geometry = parent.geometry;
    child.interface_modes = parent.interface_modes;
    child.association_lineage = parent.association_lineage;
    child.active_fixed_lag_lineage = parent.active_fixed_lag_lineage;
    child.contamination_scale = parent.contamination_scale;
    child.parent_id = parent.parent_id;
    child.weight = parent.weight;
  }
  if (!kernel.valid) return child;
  child.parent_id = parent.component_id;
  child.state = destination;
  child.interface_modes = modes;
  child.contamination_scale = 0;
  child.mean = kernel.F * parent.mean;
  child.covariance = kernel.F * parent.covariance * kernel.F.transpose() + kernel.Q;
  child.covariance = 0.5 * (child.covariance + child.covariance.transpose());
  // MergePruneNormalize neither reads component_id nor includes it in the
  // discrete merge key.  It deterministically recomputes the retained ID
  // after lineage/moment merging.  The uncached reference path keeps the old
  // transient assignment so tests can prove that eliding it is field-exact.
  if (assign_transient_component_id)
    child.component_id = ComponentId(child);
  return child;
}

bool UacGaussianSumFilter::Predict(const ros::Time& stamp,
                                   std::vector<ReasonCode>* reasons) {
  return PredictInterval(stamp, true, reasons);
}

bool UacGaussianSumFilter::PredictInterval(
    const ros::Time& stamp, bool enforce_update_gap,
    std::vector<ReasonCode>* reasons) {
  return PredictIntervalImpl(stamp, enforce_update_gap, true, true, reasons);
}

bool UacGaussianSumFilter::PredictIntervalImpl(
    const ros::Time& stamp, bool enforce_update_gap,
    bool use_exact_interval_caches, bool parallelize_independent_gdus,
    std::vector<ReasonCode>* reasons) {
  if (!initialized_) {
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_NOT_FROZEN);
    return false;
  }
  const double interval = (stamp - last_stamp_).toSec();
  const double maximum_interval = enforce_update_gap
      ? config_.resource.max_update_gap_s
      : config_.resource.prediction_horizons_s.back();
  if (!(interval > 0.0) || interval > maximum_interval) {
    if (reasons) reasons->push_back(interval <= 0.0 ? ReasonCode::INPUT_TIME_INVALID
                                                   : ReasonCode::UNKNOWN_PREDICTION);
    return false;
  }
  const int substeps = std::max(1, static_cast<int>(std::ceil(
      interval / config_.resource.ctmc_max_substep_s)));
  const double delta = interval / static_cast<double>(substeps);
  const Eigen::Matrix3d transition = CtmcTransition(
      UacGenerator(config_.calibration.ctmc_rates), delta);
  if (!transition.allFinite()) {
    if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    return false;
  }
  approximation_flags_ |= APPROX_CTMC_ENDPOINT_RESET;

  // During one prediction interval, delta, transition, the frozen reference,
  // and the calibration are immutable.  Consequently (i) coherence branches
  // depend only on [GDU, source mode vector, destination state], and (ii) a
  // linear kernel depends only on [GDU, source state, destination state,
  // destination mode vector].  Cache those exact values across the CTMC
  // substeps; no posterior mean/covariance enters either construction.
  std::vector<std::vector<CoherenceCompetitionGroup>> competition_groups;
  std::vector<std::map<PredictPlanKey, PredictTransitionPlan>> plan_caches;
  std::vector<std::map<PredictKernelKey, LinearKernel>> kernel_caches;
  if (use_exact_interval_caches) {
    competition_groups.resize(states_.size());
    plan_caches.resize(states_.size());
    kernel_caches.resize(states_.size());
    for (uint32_t g = 0U; g < states_.size(); ++g)
      competition_groups[g] = CoherenceCompetitionGroups(g);
  }

  const auto advance_one_substep = [&](uint32_t g,
                                       uint32_t* deferred_flags) -> bool {
    GduState& state = states_[g];
    std::vector<GaussianComponent, Eigen::aligned_allocator<GaussianComponent>> next;
    double next_other = state.other_mass;
    for (const GaussianComponent& parent : state.components) {
      PredictTransitionPlan uncached_plan;
      const PredictTransitionPlan* plan = &uncached_plan;
      const auto build_plan = [&](PredictTransitionPlan* output) {
        for (int destination_index = 0; destination_index < 3;
             ++destination_index) {
          const StructuralState destination =
              static_cast<StructuralState>(destination_index);
          const double structural_probability = transition(
              static_cast<int>(parent.state), destination_index);
          if (!(structural_probability > 0.0)) continue;
          double coherence_pruned = 0.0;
          const auto coherence = use_exact_interval_caches
              ? CoherenceBranchesFromGroups(
                    parent, destination, g, competition_groups[g],
                    &coherence_pruned)
              : CoherenceBranches(parent, destination, g,
                                  &coherence_pruned);
          output->pruned_probability +=
              structural_probability * coherence_pruned;
          for (const auto& branch : coherence)
            output->branches.push_back(PredictRawBranch{
                destination, branch.first,
                structural_probability * branch.second});
        }
        output->legal_total = output->pruned_probability;
        for (const PredictRawBranch& branch : output->branches)
          output->legal_total += branch.probability;
      };
      if (use_exact_interval_caches) {
        const PredictPlanKey key{parent.state, parent.interface_modes};
        auto& cache = plan_caches[g];
        auto found = cache.find(key);
        if (found == cache.end()) {
          PredictTransitionPlan value;
          build_plan(&value);
          found = cache.emplace(key, std::move(value)).first;
        }
        plan = &found->second;
      } else {
        build_plan(&uncached_plan);
      }

      if (!(plan->legal_total > 0.0)) {
        next_other += parent.weight;
        continue;
      }
      next_other += parent.weight * plan->pruned_probability /
                    plan->legal_total;
      for (const PredictRawBranch& branch : plan->branches) {
        LinearKernel uncached_kernel;
        const LinearKernel* kernel = &uncached_kernel;
        if (use_exact_interval_caches) {
          const PredictKernelKey key{parent.state, branch.destination,
                                     branch.modes};
          auto& cache = kernel_caches[g];
          auto found = cache.find(key);
          if (found == cache.end()) {
            LinearKernel value = BuildKernel(
                parent.state, branch.destination, branch.modes, g, delta);
            found = cache.emplace(key, std::move(value)).first;
          }
          kernel = &found->second;
        } else {
          uncached_kernel = BuildKernel(
              parent.state, branch.destination, branch.modes, g, delta);
        }
        bool kernel_valid = false;
        GaussianComponent child = ApplyKernel(
            parent, branch.destination, branch.modes, *kernel,
            !use_exact_interval_caches, &kernel_valid);
        if (!kernel_valid) return false;
        child.weight = parent.weight * branch.probability /
                       plan->legal_total;
        if (parent.state == StructuralState::C &&
            branch.destination == StructuralState::U) {
          *deferred_flags |= APPROX_CTMC_COLLAPSED_MULTIJUMP;
        }
        next.push_back(std::move(child));
      }
    }
    state.components.swap(next);
    state.other_mass = next_other;
    MergePruneNormalize(&state, deferred_flags);
    return true;
  };

  struct GduAdvanceResult {
    bool valid = true;
    uint32_t approximation_flags = APPROX_NONE;
  };
  std::vector<GduAdvanceResult> advance_results(states_.size());
  if (!parallelize_independent_gdus || states_.size() < 2U) {
    // Retained serial order is the exact reference path used by the
    // field-by-field regression below.
    for (int step = 0; step < substeps; ++step) {
      for (uint32_t g = 0U; g < states_.size(); ++g) {
        if (!advance_one_substep(g,
                &advance_results[g].approximation_flags)) {
          advance_results[g].valid = false;
          break;
        }
      }
      if (std::any_of(advance_results.begin(), advance_results.end(),
                      [](const GduAdvanceResult& result) {
                        return !result.valid;
                      }))
        break;
    }
  } else {
    // A GDU has no transition, moment or cache dependency on any other GDU;
    // shared-interface products are explicitly delayed until after this
    // block (cross-GDU interfaces are rejected by Initialize).  Each worker
    // therefore owns one states_[g] and its same-index caches for all
    // substeps.  Results remain in canonical GDU-index order.
    std::atomic<size_t> next_gdu(0U);
    const size_t worker_count = std::min<size_t>(
        states_.size(), 8U);
    const auto worker = [&]() {
      while (true) {
        const size_t index = next_gdu.fetch_add(1U);
        if (index >= states_.size()) return;
        GduAdvanceResult& result = advance_results[index];
        for (int step = 0; step < substeps; ++step) {
          if (!advance_one_substep(
                  static_cast<uint32_t>(index),
                  &result.approximation_flags)) {
            result.valid = false;
            break;
          }
        }
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(worker_count > 0U ? worker_count - 1U : 0U);
    for (size_t worker_index = 1U; worker_index < worker_count;
         ++worker_index) {
      try {
        workers.emplace_back(worker);
      } catch (const std::system_error&) {
        // Already-created workers plus the calling thread still consume the
        // same atomic work queue, so thread-resource pressure changes only
        // latency, never the posterior or failure semantics.
        break;
      }
    }
    worker();
    for (std::thread& thread : workers) thread.join();
  }
  for (const GduAdvanceResult& result : advance_results) {
    approximation_flags_ |= result.approximation_flags;
    if (!result.valid) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
  }
  // sqrt_information and shared-interface marginals are derived output
  // products; neither participates in the next CTMC kernel, branch weight or
  // moment merge.  Re-factorizing every component and re-aggregating every
  // interface at every 0.05 s substep is therefore exactly redundant.  Do it
  // once after the final substep, preserving the same final posterior while
  // avoiding O(substeps) repeated matrix decompositions.
  for (GduState& state : states_)
    if (!RefreshDerived(&state, reasons)) return false;
  if (!SynchronizeSharedInterfaces(reasons)) return false;
  last_stamp_ = stamp;
  first_observation_reanchor_available_ = false;
  return true;
}

std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>>
UacGaussianSumFilter::MixtureOnlyPosterior() const {
  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>> output;
  output.reserve(states_.size());
  for (size_t gdu_index = 0; gdu_index < states_.size(); ++gdu_index) {
    GduPosterior posterior;
    posterior.gdu_id = reference_.gdus[gdu_index].id;
    posterior.state_probability = {{0.0, 0.0, 0.0}};
    posterior.other_mass = states_[gdu_index].other_mass;
    posterior.nuisance_correlation_id =
        states_[gdu_index].nuisance_correlation_id;
    posterior.components = states_[gdu_index].components;
    for (const GaussianComponent& component : posterior.components)
      posterior.state_probability[static_cast<size_t>(component.state)] +=
          component.weight;
    output.push_back(std::move(posterior));
  }
  return output;
}

bool UacGaussianSumFilter::PredictiveMixture(
    double horizon_s,
    std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>>* mixture,
    std::vector<ReasonCode>* reasons) const {
  if (!mixture) {
    if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    return false;
  }
  mixture->clear();
  if (!initialized_) {
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_NOT_FROZEN);
    return false;
  }
  if (!std::isfinite(horizon_s) || horizon_s < 0.0 ||
      horizon_s > config_.resource.prediction_horizons_s.back() +
                      config_.numerical.tie_tolerance) {
    if (reasons) reasons->push_back(ReasonCode::UNKNOWN_PREDICTION);
    return false;
  }
  if (horizon_s <= config_.numerical.tie_tolerance) {
    *mixture = MixtureOnlyPosterior();
    return true;
  }
  UacGaussianSumFilter predicted(*this);
  if (!predicted.PredictInterval(
          last_stamp_ + ros::Duration(horizon_s), false, reasons)) {
    return false;
  }
  *mixture = predicted.MixtureOnlyPosterior();
  return true;
}

std::vector<GaussianComponent, Eigen::aligned_allocator<GaussianComponent>>
UacGaussianSumFilter::RobustMeasurementUpdate(
    const GaussianComponent& component, const FieldMeasurement& measurement,
    double category_probability, double* log_normalizer,
    bool* numerical_failure) const {
  std::vector<GaussianComponent, Eigen::aligned_allocator<GaussianComponent>> output;
  if (numerical_failure) *numerical_failure = false;
  if (!(category_probability > 0.0)) return output;
  const HypothesisMeasurement& model =
      measurement.hypothesis[static_cast<size_t>(component.geometry)];
  const int density_dimension = model.residual_m.size();
  if (density_dimension <= 0 ||
      model.independent_covariance_m2.rows() != density_dimension ||
      model.independent_covariance_m2.cols() != density_dimension ||
      model.design.rows() != density_dimension) {
    if (numerical_failure) *numerical_failure = true;
    return output;
  }
  Eigen::MatrixXd H;
  if (component.state == StructuralState::U)
    H = Eigen::MatrixXd::Zero(density_dimension, 15);
  else if (component.state == StructuralState::A) {
    H = Eigen::MatrixXd::Zero(density_dimension,
                              2 * model.design.cols() + 15);
    H.leftCols(model.design.cols()) = model.design;
  } else {
    H = Eigen::MatrixXd::Zero(density_dimension,
                              model.design.cols() + 15);
    H.leftCols(model.design.cols()) = model.design;
  }
  if (model.reference_bias_design.rows() != density_dimension ||
      model.reference_bias_design.cols() != 3) {
    if (numerical_failure) *numerical_failure = true;
    return output;
  }
  if (model.extrinsic_design.rows() != density_dimension ||
      model.extrinsic_design.cols() != 6 ||
      model.pose_design.rows() != density_dimension || model.pose_design.cols() != 6) {
    if (numerical_failure) *numerical_failure = true;
    return output;
  }
  H.block(0, H.cols() - 15, density_dimension, 3) = model.reference_bias_design;
  H.block(0, H.cols() - 12, density_dimension, 6) = model.extrinsic_design;
  H.rightCols(6) = model.pose_design;
  if (H.cols() != component.mean.size()) {
    if (numerical_failure) *numerical_failure = true;
    return output;
  }
  const Eigen::VectorXd innovation = model.residual_m - H * component.mean;
  const Eigen::MatrixXd latent_covariance = H * component.covariance * H.transpose();
  const auto scales = GaussianStudentTScaleMixture(
      config_.calibration.contamination_probability,
      config_.calibration.student_t_dof, 8);
  std::vector<double> log_weights;
  for (const RobustScaleNode& scale : scales) {
    if (!(scale.prior_weight > 0.0) || !(scale.precision > 0.0)) continue;
    const double scale2 = scale.scale_index == 0 ? 1.0
        : config_.calibration.student_t_scale * config_.calibration.student_t_scale;
    const Eigen::MatrixXd measurement_covariance =
        model.independent_covariance_m2 * scale2 / scale.precision;
    const Eigen::MatrixXd innovation_covariance = latent_covariance + measurement_covariance;
    Eigen::MatrixXd lower;
    if (!StableCholesky(innovation_covariance, config_.numerical, &lower)) {
      if (numerical_failure) *numerical_failure = true;
      return {};
    }
    const Eigen::VectorXd whitened = lower.triangularView<Eigen::Lower>().solve(innovation);
    const double log_determinant = 2.0 * lower.diagonal().array().log().sum();
    const double log_likelihood = -0.5 *
        (static_cast<double>(innovation.size()) * std::log(2.0 * M_PI) +
         log_determinant + whitened.squaredNorm());
    // Every association alternative is represented as a dimensionless Bayes
    // factor on the same 3-D residual measure. OTHER is the frozen broad
    // reference density (factor 1); comparing a raw m^-3 Gaussian density to
    // a dimensionless missing probability would otherwise make posterior
    // mass depend on the chosen length unit.
    const double reference_variance =
        config_.calibration.unobservable_residual_sigma_m *
        config_.calibration.unobservable_residual_sigma_m;
    const double reference_log_density = -0.5 *
        (static_cast<double>(innovation.size()) *
             std::log(2.0 * M_PI * reference_variance) +
         model.residual_m.squaredNorm() / reference_variance);
    const double log_bayes_factor = log_likelihood -
                                    reference_log_density;
    GaussianComponent child = component;
    child.contamination_scale = scale.scale_index;
    AppendCurrentFrameLineage(measurement.evidence_id,
                              measurement.candidate_id,
                              static_cast<uint8_t>(AssociationSemantics::OTHER) + 1U,
                              &child);
    if (H.cols() > 0 && std::isfinite(log_likelihood)) {
      const Eigen::MatrixXd covariance_design_transpose =
          component.covariance * H.transpose();
      const Eigen::MatrixXd solved_lower =
          lower.triangularView<Eigen::Lower>().solve(
              covariance_design_transpose.transpose());
      const Eigen::MatrixXd gain =
          lower.transpose().triangularView<Eigen::Upper>().solve(
              solved_lower).transpose();
      child.mean = component.mean + gain * innovation;
      const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(H.cols(), H.cols());
      const Eigen::MatrixXd update = identity - gain * H;
      child.covariance = update * component.covariance * update.transpose() +
                         gain * measurement_covariance * gain.transpose();
      child.covariance = 0.5 * (child.covariance + child.covariance.transpose());
    }
    child.component_id = ComponentId(child);
    const double log_weight = std::log(component.weight) + std::log(category_probability) +
                              std::log(scale.prior_weight) +
                              log_bayes_factor;
    log_weights.push_back(log_weight);
    child.weight = log_weight;
    output.push_back(child);
  }
  if (log_normalizer) *log_normalizer = LogSumExp(log_weights);
  return output;
}

bool UacGaussianSumFilter::Update(
    const std::vector<std::vector<AssociationMeasurementSet>>& measurements_by_gdu,
    const std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>& nuisance_by_gdu,
    const ros::Time& stamp, std::vector<ReasonCode>* reasons) {
  return UpdateImpl(measurements_by_gdu, nuisance_by_gdu, stamp, true,
                    reasons);
}

bool UacGaussianSumFilter::UpdateImpl(
    const std::vector<std::vector<AssociationMeasurementSet>>& measurements_by_gdu,
    const std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>& nuisance_by_gdu,
    const ros::Time& stamp, bool parallelize_independent_gdus,
    std::vector<ReasonCode>* reasons) {
  if (!initialized_ || measurements_by_gdu.size() != states_.size() ||
      nuisance_by_gdu.size() != states_.size() || stamp != last_stamp_) {
    if (reasons) reasons->push_back(ReasonCode::INPUT_SCHEMA_INVALID);
    return false;
  }

  struct GduUpdateResult {
    std::unique_ptr<GduState> state;
    std::vector<ReasonCode> reasons;
    uint32_t approximation_flags = APPROX_NONE;
    bool valid = true;
  };
  std::vector<GduUpdateResult> results(states_.size());
  const auto copy_update_state = [](const GduState& source,
                                    GduState* destination) {
    // The POU/task/reference-basis members are immutable epoch caches and can
    // be very large.  Update never reads them through its private state copy
    // (task QMC reads the canonical states_[g] cache), so copying them once
    // per frame would waste both memory bandwidth and latency.  Copy exactly
    // the mutable update/history posterior plus the two small frozen scalars
    // required by the update algebra.
    destination->coefficient_count = source.coefficient_count;
    destination->reference_bias_prior_covariance =
        source.reference_bias_prior_covariance;
    destination->last_pose_prior_covariance =
        source.last_pose_prior_covariance;
    destination->last_pose_stamp = source.last_pose_stamp;
    destination->has_pose_prior = source.has_pose_prior;
    destination->nuisance_correlation_id = source.nuisance_correlation_id;
    destination->current_observable_information =
        source.current_observable_information;
    destination->reference_information_ratio =
        source.reference_information_ratio;
    destination->reference_confounding = source.reference_confounding;
    destination->effective_spatial_support = source.effective_spatial_support;
    destination->effective_temporal_support =
        source.effective_temporal_support;
    destination->information_weight_history =
        source.information_weight_history;
    destination->observable_frame_history = source.observable_frame_history;
    destination->other_mass = source.other_mass;
    destination->consecutive_source_missing =
        source.consecutive_source_missing;
    destination->current_measurement_valid = source.current_measurement_valid;
    destination->continuous_valid_velocity_duration_s =
        source.continuous_valid_velocity_duration_s;
    destination->last_valid_velocity_stamp =
        source.last_valid_velocity_stamp;
    destination->components = source.components;
  };
  const auto commit_update_state = [](GduState* source,
                                      GduState* destination) {
    destination->last_pose_prior_covariance =
        std::move(source->last_pose_prior_covariance);
    destination->last_pose_stamp = source->last_pose_stamp;
    destination->has_pose_prior = source->has_pose_prior;
    destination->nuisance_correlation_id = source->nuisance_correlation_id;
    destination->current_observable_information =
        std::move(source->current_observable_information);
    destination->reference_information_ratio =
        source->reference_information_ratio;
    destination->reference_confounding = source->reference_confounding;
    destination->effective_spatial_support = source->effective_spatial_support;
    destination->effective_temporal_support =
        source->effective_temporal_support;
    destination->information_weight_history =
        std::move(source->information_weight_history);
    destination->observable_frame_history =
        std::move(source->observable_frame_history);
    destination->other_mass = source->other_mass;
    destination->consecutive_source_missing =
        source->consecutive_source_missing;
    destination->current_measurement_valid = source->current_measurement_valid;
    destination->continuous_valid_velocity_duration_s =
        source->continuous_valid_velocity_duration_s;
    destination->last_valid_velocity_stamp =
        source->last_valid_velocity_stamp;
    destination->components = std::move(source->components);
  };
  const auto update_one_gdu = [&](uint32_t g) {
    GduUpdateResult& result = results[g];
    // Work on a private copy and commit only after every GDU has succeeded.
    // Besides making a failed update atomic, this prevents a fast worker for
    // a later GDU from becoming externally observable when an earlier GDU
    // fails.  GDU-local copies also rule out false sharing through Eigen's
    // dynamic storage while all immutable config/reference caches remain
    // shared read-only.
    result.state.reset(new GduState());
    copy_update_state(states_[g], result.state.get());
    GduState& state = *result.state;
    std::vector<ReasonCode>* local_reasons = &result.reasons;
    double missing_mass = 0.0;
    double geometric_mass = 0.0;
    for (const AssociationMeasurementSet& set : measurements_by_gdu[g]) {
      for (const AssociationCategory& category : set.selection.categories) {
        if (category.semantics == AssociationSemantics::SOURCE_MISSING)
          missing_mass += category.probability;
        else if (category.geometric_candidate)
          geometric_mass += category.probability;
      }
    }
    const bool source_missing_frame = !measurements_by_gdu[g].empty() &&
        missing_mass > geometric_mass + config_.numerical.tie_tolerance;
    if (source_missing_frame)
      ++state.consecutive_source_missing;
    else
      state.consecutive_source_missing = 0;
    state.current_measurement_valid =
        nuisance_by_gdu[g].control_registration_valid &&
        nuisance_by_gdu[g].deformation_measurement_valid &&
        !source_missing_frame;
    if (!state.current_measurement_valid) {
      state.continuous_valid_velocity_duration_s = 0.0;
      state.last_valid_velocity_stamp = ros::Time();
      for (Eigen::MatrixXd& information : state.current_observable_information)
        information = Eigen::MatrixXd::Zero(
            state.coefficient_count, state.coefficient_count);
      if (state.consecutive_source_missing >=
          config_.resource.source_missing_consecutive_limit)
        local_reasons->push_back(ReasonCode::SOURCE_MISSING_LIMIT);
      return;
    }
    if (!AdvancePoseNuisance(&state, nuisance_by_gdu[g].pose_covariance,
                             stamp, local_reasons)) {
      result.valid = false;
      return;
    }
    if (!BuildCurrentObservableInformation(&state, measurements_by_gdu[g],
                                           nuisance_by_gdu[g], stamp,
                                           local_reasons)) {
      result.valid = false;
      return;
    }
    state.nuisance_correlation_id =
        nuisance_by_gdu[g].nuisance_correlation_id;
    for (GaussianComponent& component : state.components) {
      component.active_fixed_lag_lineage.push_back(StableId());
      while (component.active_fixed_lag_lineage.size() >
             static_cast<size_t>(config_.resource.fixed_lag_frames)) {
        component.active_fixed_lag_lineage.erase(
            component.active_fixed_lag_lineage.begin());
      }
      RefreshLineageDigest(&component);
    }
    for (const AssociationMeasurementSet& set : measurements_by_gdu[g]) {
      if (!set.selection.valid) {
        state.other_mass = 1.0;
        state.components.clear();
        local_reasons->push_back(ReasonCode::ASSOCIATION_AMBIGUOUS);
        break;
      }
      std::map<StableId, const FieldMeasurement*> measurement_by_candidate;
      for (const FieldMeasurement& measurement : set.candidates)
        measurement_by_candidate[measurement.candidate_id] = &measurement;
      std::vector<GaussianComponent, Eigen::aligned_allocator<GaussianComponent>> updated;
      std::vector<double> component_log_weights;
      double other_unnormalized = state.other_mass;
      for (const GaussianComponent& component : state.components) {
        for (const AssociationCategory& category : set.selection.categories) {
          if (!(category.probability > 0.0)) continue;
          if (category.geometric_candidate) {
            const auto found = measurement_by_candidate.find(category.candidate_id);
            if (found == measurement_by_candidate.end()) {
              // The selected association alternative has no disjoint
              // MEASUREMENT partner.  This is an identity/missing-data branch
              // for the structural state, not persistent global model
              // inadequacy.  Converting it to state.other_mass would make
              // harmless per-source association uncertainty accumulate to
              // one over repeated static observations.
              GaussianComponent child = component;
              child.contamination_scale = 0;
              AppendCurrentFrameLineage(
                  set.source_evidence_id, category.candidate_id,
                  static_cast<uint8_t>(AssociationSemantics::OTHER), &child);
              const double log_weight = std::log(component.weight) +
                  std::log(category.probability);
              child.weight = log_weight;
              child.component_id = ComponentId(child);
              component_log_weights.push_back(log_weight);
              updated.push_back(std::move(child));
              continue;
            }
            double ignored = 0.0;
            bool numerical_failure = false;
            auto branches = RobustMeasurementUpdate(
                component, *found->second, category.probability, &ignored,
                &numerical_failure);
            if (numerical_failure) {
              local_reasons->push_back(ReasonCode::INVALID_NUMERICAL);
              result.valid = false;
              return;
            }
            for (GaussianComponent& branch : branches) {
              component_log_weights.push_back(branch.weight);
              updated.push_back(std::move(branch));
            }
          } else {
            const int state_index = static_cast<int>(component.state);
            // SOURCE_MISSING is an identity/missing-data branch: its
            // likelihood is deliberately state-independent and therefore
            // cannot turn absence into deformation evidence.
            const double log_likelihood =
                category.semantics == AssociationSemantics::SOURCE_MISSING
                    ? 0.0
                    : set.other_log_likelihood[static_cast<size_t>(state_index)];
            const double log_weight = std::log(component.weight) +
                std::log(category.probability) + log_likelihood;
            // SOURCE_MISSING and per-source OTHER are both association
            // alternatives with state-independent identity likelihoods.
            // OTHER records that this source was not explained by a retained
            // candidate; it is not the persistent global other_mass used for
            // mixture pruning/model inadequacy.
            GaussianComponent child = component;
            child.contamination_scale = 0;
            AppendCurrentFrameLineage(
                set.source_evidence_id, StableId(),
                static_cast<uint8_t>(category.semantics), &child);
            child.weight = log_weight;
            child.component_id = ComponentId(child);
            component_log_weights.push_back(log_weight);
            updated.push_back(std::move(child));
          }
        }
      }
      std::vector<double> all_log_weights = component_log_weights;
      if (other_unnormalized > 0.0) all_log_weights.push_back(std::log(other_unnormalized));
      const double normalizer = LogSumExp(all_log_weights);
      if (!std::isfinite(normalizer)) {
        local_reasons->push_back(ReasonCode::INVALID_NUMERICAL);
        result.valid = false;
        return;
      }
      for (GaussianComponent& component : updated)
        component.weight = std::exp(component.weight - normalizer);
      state.other_mass = other_unnormalized > 0.0
                             ? std::exp(std::log(other_unnormalized) - normalizer) : 0.0;
      state.components.swap(updated);
      MergePruneNormalize(&state, &result.approximation_flags);
      // Keep the historical per-measurement numerical checkpoint.  Although
      // sqrt_information is derived, removing this checkpoint could allow a
      // later moment merge to hide an intermediate factorization failure and
      // would therefore change failure semantics.
      if (!RefreshDerived(&state, local_reasons)) {
        result.valid = false;
        return;
      }
    }
    for (GaussianComponent& component : state.components) {
      const int persistent_dimension = StateDimension(component.state,
                                                      state.coefficient_count);
      component.mean = component.mean.head(persistent_dimension).eval();
      component.covariance = component.covariance.topLeftCorner(
          persistent_dimension, persistent_dimension).eval();
      if (!component.mean.allFinite() || !std::isfinite(component.weight) ||
          component.weight < 0.0 ||
          !ValidateSymmetricPsd(
              component.covariance,
              config_.numerical.psd_eigenvalue_tolerance)) {
        local_reasons->push_back(ReasonCode::INVALID_NUMERICAL);
        result.valid = false;
        return;
      }
    }
    double current_velocity_probability = 0.0;
    for (const GaussianComponent& component : state.components) {
      if (component.state != StructuralState::A) continue;
      current_velocity_probability += component.weight *
          ComponentTaskScaleProbability(g, component).current_velocity;
    }
    const bool velocity_condition = current_velocity_probability >=
        config_.calibration.detection_probability_on;
    if (velocity_condition) {
      if (!state.last_valid_velocity_stamp.isZero() &&
          stamp > state.last_valid_velocity_stamp)
        state.continuous_valid_velocity_duration_s +=
            (stamp - state.last_valid_velocity_stamp).toSec();
      else
        state.continuous_valid_velocity_duration_s = 0.0;
      state.last_valid_velocity_stamp = stamp;
    } else {
      state.continuous_valid_velocity_duration_s = 0.0;
      state.last_valid_velocity_stamp = ros::Time();
    }
  };

  if (!parallelize_independent_gdus || states_.size() < 2U) {
    for (uint32_t g = 0U; g < states_.size(); ++g) update_one_gdu(g);
  } else {
    // Update algebra, fixed-lag histories and all mixture reductions are
    // strictly GDU-local.  Cross-GDU interfaces are rejected at reference
    // freeze, and shared-interface products are rebuilt only after this
    // block.  A fixed upper bound avoids oversubscribing Gazebo/FAST-LIO on
    // large references; index-owned results make scheduling irrelevant.
    std::atomic<size_t> next_gdu(0U);
    const size_t worker_count = std::min<size_t>(states_.size(), 8U);
    const auto worker = [&]() {
      while (true) {
        const size_t index = next_gdu.fetch_add(1U);
        if (index >= states_.size()) return;
        update_one_gdu(static_cast<uint32_t>(index));
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(worker_count > 0U ? worker_count - 1U : 0U);
    for (size_t worker_index = 1U; worker_index < worker_count;
         ++worker_index) {
      try {
        workers.emplace_back(worker);
      } catch (const std::system_error&) {
        // Existing workers plus the calling thread continue consuming the
        // same atomic queue.  Thread-resource pressure changes latency only.
        break;
      }
    }
    worker();
    for (std::thread& thread : workers) thread.join();
  }

  // Reasons and approximation flags retain the exact canonical GDU order,
  // irrespective of worker completion order.  The final derived refresh is
  // deliberately serial: it provides the same per-GDU failure boundary as
  // the reference path before any state is committed.
  for (uint32_t g = 0U; g < results.size(); ++g) {
    GduUpdateResult& result = results[g];
    approximation_flags_ |= result.approximation_flags;
    if (reasons)
      reasons->insert(reasons->end(), result.reasons.begin(),
                      result.reasons.end());
    if (!result.valid) return false;
    std::vector<ReasonCode> derived_reasons;
    if (!RefreshDerived(result.state.get(), &derived_reasons)) {
      if (reasons)
        reasons->insert(reasons->end(), derived_reasons.begin(),
                        derived_reasons.end());
      return false;
    }
    if (reasons)
      reasons->insert(reasons->end(), derived_reasons.begin(),
                      derived_reasons.end());
  }

  for (uint32_t g = 0U; g < results.size(); ++g)
    commit_update_state(results[g].state.get(), &states_[g]);
  if (!SynchronizeSharedInterfaces(reasons)) return false;
  return true;
}

std::string UacGaussianSumFilter::DiscreteKey(const GaussianComponent& component) const {
  std::ostringstream stream;
  // Association lineage is provenance, not a physical discrete state.  If it
  // participates in the merge key, every independent source doubles the
  // mixture and a static GDU is driven into other_mass solely by the 24
  // component resource cap.  Moment matching is performed within the same
  // structural/geometry/interface mode; the lineages are combined below into
  // a deterministic audit digest.
  stream << static_cast<int>(component.state) << ':'
         << static_cast<int>(component.geometry);
  for (InterfaceJointState mode : component.interface_modes)
    stream << ':' << static_cast<int>(mode);
  return stream.str();
}

StableId UacGaussianSumFilter::ComponentId(const GaussianComponent& component) const {
  CanonicalBytes fields;
  fields.AddU8(static_cast<uint8_t>(component.state));
  fields.AddU8(static_cast<uint8_t>(component.geometry));
  for (InterfaceJointState mode : component.interface_modes)
    fields.AddU8(static_cast<uint8_t>(mode));
  fields.AddId(component.association_lineage);
  fields.AddI64(component.contamination_scale);
  fields.AddId(component.parent_id);
  return MakeStableId("gaussian-component-v1", fields);
}

void UacGaussianSumFilter::MergePruneNormalize(
    GduState* state, uint32_t* deferred_approximation_flags) {
  if (!state) return;
  std::map<std::string, std::vector<GaussianComponent, Eigen::aligned_allocator<GaussianComponent>>> groups;
  for (const GaussianComponent& component : state->components)
    if (component.weight > 0.0 && std::isfinite(component.weight))
      groups[DiscreteKey(component)].push_back(component);
  std::vector<GaussianComponent, Eigen::aligned_allocator<GaussianComponent>> merged;
  for (auto& group : groups) {
    double mass = 0.0;
    for (const GaussianComponent& component : group.second) mass += component.weight;
    if (!(mass > 0.0)) continue;
    GaussianComponent output = group.second.front();
    output.weight = mass;
    // Student-t quadrature scale is a measurement-local nuisance.  Its
    // responsibilities are moment-marginalized here and do not survive in the
    // persistent discrete key (DEC-013).
    output.contamination_scale = -1;
    if (output.mean.size() > 0) {
      output.mean.setZero();
      for (const GaussianComponent& component : group.second)
        output.mean += (component.weight / mass) * component.mean;
      output.covariance.setZero();
      for (const GaussianComponent& component : group.second) {
        const Eigen::VectorXd delta = component.mean - output.mean;
        output.covariance += (component.weight / mass) *
            (component.covariance + delta * delta.transpose());
      }
      output.covariance = 0.5 * (output.covariance + output.covariance.transpose());
    }
    size_t lineage_depth = 0U;
    for (const GaussianComponent& component : group.second)
      lineage_depth = std::max(lineage_depth,
                               component.active_fixed_lag_lineage.size());
    output.active_fixed_lag_lineage.assign(lineage_depth, StableId());
    for (size_t lag = 0; lag < lineage_depth; ++lag) {
      std::vector<StableId> ids;
      for (const GaussianComponent& component : group.second)
        if (lag < component.active_fixed_lag_lineage.size() &&
            !component.active_fixed_lag_lineage[lag].empty())
          ids.push_back(component.active_fixed_lag_lineage[lag]);
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      if (!ids.empty()) {
        CanonicalBytes lineage_fields;
        for (const StableId& id : ids) lineage_fields.AddId(id);
        output.active_fixed_lag_lineage[lag] = MakeStableId(
            "association-lineage-moment-merge-v1", lineage_fields);
      }
    }
    RefreshLineageDigest(&output);
    output.component_id = ComponentId(output);
    merged.push_back(output);
  }
  std::sort(merged.begin(), merged.end(), [](const GaussianComponent& lhs,
                                             const GaussianComponent& rhs) {
    if (lhs.weight != rhs.weight) return lhs.weight > rhs.weight;
    if (lhs.state != rhs.state) return lhs.state < rhs.state;
    if (lhs.geometry != rhs.geometry) return lhs.geometry < rhs.geometry;
    return lhs.component_id < rhs.component_id;
  });
  if (merged.size() > static_cast<size_t>(config_.resource.mixture_max_components)) {
    for (size_t i = static_cast<size_t>(config_.resource.mixture_max_components);
         i < merged.size(); ++i) state->other_mass += merged[i].weight;
    merged.resize(static_cast<size_t>(config_.resource.mixture_max_components));
    if (deferred_approximation_flags)
      *deferred_approximation_flags |= APPROX_MIXTURE_PRUNED_TO_OTHER;
    else
      approximation_flags_ |= APPROX_MIXTURE_PRUNED_TO_OTHER;
  }
  double total = state->other_mass;
  for (const GaussianComponent& component : merged) total += component.weight;
  if (total > 0.0) {
    state->other_mass /= total;
    for (GaussianComponent& component : merged) component.weight /= total;
  } else state->other_mass = 1.0;
  state->components.swap(merged);
}

bool UacGaussianSumFilter::RefreshDerived(
    GduState* state, std::vector<ReasonCode>* reasons) const {
  if (!state) {
    if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    return false;
  }
  for (GaussianComponent& component : state->components) {
    if (component.covariance.rows() == 0) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
    if (!DeterministicPsdInformationFactor(
            component.covariance, config_.numerical,
            &component.sqrt_information)) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
    if (component.sqrt_information.cols() != component.mean.size() ||
        !component.sqrt_information.allFinite()) {
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      return false;
    }
  }
  return true;
}

std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>>
UacGaussianSumFilter::Posterior() const {
  return PosteriorImpl(true);
}

std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>>
UacGaussianSumFilter::PosteriorImpl(
    bool parallelize_independent_gdus,
    size_t successful_worker_creation_limit) const {
  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>> output(
      states_.size());
  // A byte (not vector<bool>) and an exception slot are owned by exactly one
  // GDU worker.  No worker appends to a shared container, so scheduling cannot
  // alter product order or any floating-point reduction.
  std::vector<uint8_t> completed(states_.size(), 0U);
  std::vector<std::exception_ptr> exceptions(states_.size());
  const auto build_one_gdu = [&](uint32_t g) {
    try {
    GduPosterior posterior;
    double task_union_without_velocity = 0.0;
    posterior.gdu_id = reference_.gdus[g].id;
    posterior.nuisance_correlation_id = states_[g].nuisance_correlation_id;
    posterior.state_probability = {{0.0, 0.0, 0.0}};
    posterior.other_mass = states_[g].other_mass;
    const int n = states_[g].coefficient_count;
    // The task amplitude is the equal seed-cell quadrature of physical field
    // RMS on the frozen surfels.  ReferenceBuilder creates at most one surfel
    // per fixed seed voxel, so equal weights are density compensation rather
    // than raw-return-count weights.  MDE must be expressed in this metric;
    // the Euclidean norm of a control coefficient vector is not a physical
    // displacement and changes under control-grid refinement.
    if (!states_[g].frozen_reference_cache) return;
    const Eigen::MatrixXd& task_metric =
        states_[g].frozen_reference_cache->task_metric;
    const size_t task_quadrature_count =
        states_[g].frozen_reference_cache->task_quadrature_count;
    struct ObservableSummary {
      int rank = 0;
      int task_rank = 0;
      Eigen::MatrixXd projector;
      Eigen::MatrixXd nullspace;
      double mde = std::numeric_limits<double>::infinity();
      bool mde_defined = false;
      double velocity_mde = std::numeric_limits<double>::infinity();
      bool velocity_mde_defined = false;
      double jump_mde = std::numeric_limits<double>::infinity();
      bool jump_mde_defined = false;
      bool jump_required = false;
      bool task_power_complete = false;
      bool any_task_direction_at_power = false;
      double task_power_ratio = std::numeric_limits<double>::infinity();
      double best_task_power_ratio = std::numeric_limits<double>::infinity();
    };
    std::array<ObservableSummary, 4> summaries;
    for (int h = 0; h < 4; ++h) {
      ObservableSummary& summary = summaries[static_cast<size_t>(h)];
      summary.projector = Eigen::MatrixXd::Zero(n, n);
      summary.nullspace = Eigen::MatrixXd::Identity(n, n);
      const Eigen::MatrixXd& information =
          states_[g].current_observable_information[static_cast<size_t>(h)];
      if (information.rows() != n || information.cols() != n || n == 0) continue;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> information_eigen(information);
      if (information_eigen.info() != Eigen::Success) continue;
      summary.rank = static_cast<int>((information_eigen.eigenvalues().array() >
          config_.numerical.psd_eigenvalue_tolerance).count());
      std::vector<int> null_modes;
      for (int mode = 0; mode < n; ++mode)
        if (information_eigen.eigenvalues()[mode] >
            config_.numerical.psd_eigenvalue_tolerance)
          summary.projector += information_eigen.eigenvectors().col(mode) *
                               information_eigen.eigenvectors().col(mode).transpose();
        else
          null_modes.push_back(mode);
      summary.nullspace = Eigen::MatrixXd::Zero(
          n, static_cast<int>(null_modes.size()));
      for (size_t column = 0; column < null_modes.size(); ++column)
        summary.nullspace.col(static_cast<int>(column)) =
            information_eigen.eigenvectors().col(null_modes[column]);
      if (summary.rank > 0 && task_quadrature_count > 0U) {
        double weakest_information = 0.0;
        double strongest_information = 0.0;
        int task_effective_rank = 0;
        if (!MinimumTaskNormalizedInformation(
                information, task_metric,
                config_.numerical.psd_eigenvalue_tolerance,
                &weakest_information, &task_effective_rank,
                &summary.task_rank, &strongest_information)) continue;
        summary.rank = task_effective_rank;
        summary.mde = MinimumDetectableEffectFromInformation(
            weakest_information, summary.rank,
            config_.calibration.event_false_alarm_probability,
            config_.calibration.target_power);
        summary.mde *= config_.calibration.mde_bootstrap_inflation;
        summary.mde_defined = std::isfinite(summary.mde);
        const double best_mde = MinimumDetectableEffectFromInformation(
            strongest_information, 1,
            config_.calibration.event_false_alarm_probability,
            config_.calibration.target_power) *
            config_.calibration.mde_bootstrap_inflation;
        if (std::isfinite(best_mde))
          summary.best_task_power_ratio = std::min(
              summary.best_task_power_ratio,
              best_mde /
                  config_.calibration.task_displacement_scale_m);
      }

      // Every retained non-CUT interface is a separate registered task
      // direction. Its MDE cannot be averaged into the smooth-field RMS.
      bool all_required_jumps_defined = true;
      double worst_jump_mde = 0.0;
      for (size_t local_interface = 0;
           local_interface < reference_.gdus[g].interface_indices.size();
           ++local_interface) {
        bool required = false;
        for (const GaussianComponent& component : states_[g].components) {
          if (component.state == StructuralState::U ||
              static_cast<int>(component.geometry) != h ||
              component.weight <= config_.numerical.probability_floor ||
              local_interface >= component.interface_modes.size()) continue;
          if (component.interface_modes[local_interface] !=
              InterfaceJointState::CUT_NOT_APPLICABLE) {
            required = true;
            break;
          }
        }
        if (!required) continue;
        summary.jump_required = true;
        Eigen::MatrixXd jump_metric = Eigen::MatrixXd::Zero(n, n);
        const int coordinate_rank = reference_.gdus[g].coordinate_rank;
        const int offset = coordinate_rank * static_cast<int>(
            reference_.gdus[g].control_surfel_indices.size() +
            local_interface);
        if (coordinate_rank <= 0 || offset + coordinate_rank > n) {
          all_required_jumps_defined = false;
          continue;
        }
        const Eigen::MatrixXd axes =
            reference_.gdus[g].coordinate_axes_R.leftCols(coordinate_rank);
        jump_metric.block(offset, offset, coordinate_rank, coordinate_rank) =
            axes.transpose() * config_.calibration.task_interface_metric * axes;
        double jump_information = 0.0;
        double strongest_jump_information = 0.0;
        int jump_effective_rank = 0;
        int jump_task_rank = 0;
        if (!MinimumTaskNormalizedInformation(
                information, jump_metric,
                config_.numerical.psd_eigenvalue_tolerance,
                &jump_information, &jump_effective_rank, &jump_task_rank,
                &strongest_jump_information) ||
            jump_effective_rank != jump_task_rank) {
          all_required_jumps_defined = false;
        } else {
          double interface_mde = MinimumDetectableEffectFromInformation(
              jump_information, jump_effective_rank,
              config_.calibration.event_false_alarm_probability,
              config_.calibration.target_power);
          interface_mde *= config_.calibration.mde_bootstrap_inflation;
          if (!std::isfinite(interface_mde)) {
            all_required_jumps_defined = false;
          } else {
            worst_jump_mde = std::max(worst_jump_mde, interface_mde);
          }
        }
        const double best_interface_mde =
            MinimumDetectableEffectFromInformation(
            strongest_jump_information, 1,
            config_.calibration.event_false_alarm_probability,
            config_.calibration.target_power) *
            config_.calibration.mde_bootstrap_inflation;
        if (std::isfinite(best_interface_mde))
          summary.best_task_power_ratio = std::min(
              summary.best_task_power_ratio,
              best_interface_mde /
                  config_.calibration.task_interface_jump_scale_m);
      }
      if (summary.jump_required && all_required_jumps_defined) {
        summary.jump_mde = worst_jump_mde;
        summary.jump_mde_defined = true;
      }

      // Fixed-lag active-motion information uses d(t_i)=d(t)-age_i v(t).
      // All frame rows are whitened once with their joint covariance: the
      // epoch-static reference and extrinsic blocks are shared, pose blocks
      // follow the calibrated OU kernel, and the integrated
      // white-acceleration displacement error is shared
      // over overlapping look-back intervals. No independent-frame
      // information multiplication is used here.
      Eigen::MatrixXd active_information = Eigen::MatrixXd::Zero(2 * n, 2 * n);
      const auto& frame_history =
          states_[g].observable_frame_history[static_cast<size_t>(h)];
      std::vector<const ObservableFrameBlock*> usable_blocks;
      std::vector<double> block_ages_s;
      std::vector<int> block_row_offsets;
      std::vector<Eigen::MatrixXd> pose_lowers;
      int joint_rows = 0;
      bool temporal_model_valid = true;
      for (const ObservableFrameBlock& block : frame_history) {
        const int block_rows = block.design.rows();
        const double age_s = (last_stamp_ - block.stamp).toSec();
        if (block_rows <= 0) continue;
        if (age_s < -config_.numerical.tie_tolerance ||
            block.design.cols() != n ||
            block.independent_covariance.rows() != block_rows ||
            block.independent_covariance.cols() != block_rows ||
            block.reference_bias_design.rows() != block_rows ||
            block.reference_bias_design.cols() != 3 ||
            block.extrinsic_design.rows() != block_rows ||
            block.extrinsic_design.cols() != 6 ||
            block.pose_design.rows() != block_rows ||
            block.pose_design.cols() != 6) {
          temporal_model_valid = false;
          break;
        }
        Eigen::MatrixXd pose_lower;
        if (!StableCholesky(block.pose_covariance, config_.numerical,
                            &pose_lower) || pose_lower.rows() != 6) {
          temporal_model_valid = false;
          break;
        }
        usable_blocks.push_back(&block);
        block_ages_s.push_back(std::max(0.0, age_s));
        block_row_offsets.push_back(joint_rows);
        pose_lowers.push_back(pose_lower);
        joint_rows += block_rows;
      }
      if (temporal_model_valid && joint_rows > 0) {
        Eigen::MatrixXd active_design =
            Eigen::MatrixXd::Zero(joint_rows, 2 * n);
        Eigen::MatrixXd joint_covariance =
            Eigen::MatrixXd::Zero(joint_rows, joint_rows);
        for (size_t left = 0; left < usable_blocks.size(); ++left) {
          const ObservableFrameBlock& lhs = *usable_blocks[left];
          const int left_rows = lhs.design.rows();
          const int left_offset = block_row_offsets[left];
          active_design.block(left_offset, 0, left_rows, n) = lhs.design;
          active_design.block(left_offset, n, left_rows, n) =
              -block_ages_s[left] * lhs.design;
          for (size_t right = 0; right < usable_blocks.size(); ++right) {
            const ObservableFrameBlock& rhs = *usable_blocks[right];
            const int right_rows = rhs.design.rows();
            const int right_offset = block_row_offsets[right];
            Eigen::MatrixXd covariance =
                lhs.reference_bias_design *
                    states_[g].reference_bias_prior_covariance *
                    rhs.reference_bias_design.transpose() +
                lhs.extrinsic_design * config_.resource.extrinsic.covariance *
                    rhs.extrinsic_design.transpose();
            const double separation_s =
                std::abs((lhs.stamp - rhs.stamp).toSec());
            const Eigen::Matrix<double, 6, 6> pose_cross =
                pose_lowers[left] *
                std::exp(-separation_s /
                         config_.calibration.pose_ou_tau_s) *
                pose_lowers[right].transpose();
            covariance += lhs.pose_design * pose_cross *
                          rhs.pose_design.transpose();
            if (left == right)
              covariance += lhs.independent_covariance;
            const double minimum_age = std::min(
                block_ages_s[left], block_ages_s[right]);
            const double integrated_acceleration_covariance =
                config_.calibration.active_acceleration_psd *
                (minimum_age * block_ages_s[left] * block_ages_s[right] -
                 0.5 * (block_ages_s[left] + block_ages_s[right]) *
                     minimum_age * minimum_age +
                 minimum_age * minimum_age * minimum_age / 3.0);
            if (integrated_acceleration_covariance > 0.0)
              covariance += integrated_acceleration_covariance *
                  lhs.design * rhs.design.transpose();
            joint_covariance.block(left_offset, right_offset,
                                   left_rows, right_rows) = covariance;
          }
        }
        joint_covariance = 0.5 *
            (joint_covariance + joint_covariance.transpose());
        Eigen::MatrixXd joint_lower;
        if (StableCholesky(joint_covariance, config_.numerical,
                           &joint_lower)) {
          const Eigen::MatrixXd active_whitened =
              joint_lower.triangularView<Eigen::Lower>().solve(active_design);
          active_information =
              active_whitened.transpose() * active_whitened;
        }
      }
      Eigen::MatrixXd velocity_metric = Eigen::MatrixXd::Zero(2 * n, 2 * n);
      velocity_metric.bottomRightCorner(n, n) = task_metric;
      double velocity_information = 0.0;
      double strongest_velocity_information = 0.0;
      int velocity_effective_rank = 0;
      int velocity_task_rank = 0;
      if (MinimumTaskNormalizedInformation(
              active_information, velocity_metric,
              config_.numerical.psd_eigenvalue_tolerance,
              &velocity_information, &velocity_effective_rank,
              &velocity_task_rank, &strongest_velocity_information) &&
          velocity_effective_rank == velocity_task_rank) {
        summary.velocity_mde = MinimumDetectableEffectFromInformation(
            velocity_information, velocity_effective_rank,
            config_.calibration.event_false_alarm_probability,
            config_.calibration.target_power);
        summary.velocity_mde *= config_.calibration.mde_bootstrap_inflation;
        summary.velocity_mde_defined =
            std::isfinite(summary.velocity_mde);
      }
      const double best_velocity_mde =
          MinimumDetectableEffectFromInformation(
              strongest_velocity_information, 1,
              config_.calibration.event_false_alarm_probability,
              config_.calibration.target_power) *
          config_.calibration.mde_bootstrap_inflation;
      if (std::isfinite(best_velocity_mde))
        summary.best_task_power_ratio = std::min(
            summary.best_task_power_ratio,
            best_velocity_mde /
                config_.calibration.task_velocity_scale_mps);
      const bool smooth_complete = summary.mde_defined &&
          summary.rank == summary.task_rank;
      const bool jump_complete = !summary.jump_required ||
          summary.jump_mde_defined;
      summary.task_power_complete = smooth_complete && jump_complete &&
          summary.velocity_mde_defined;
      if (summary.task_power_complete) {
        summary.task_power_ratio = std::max(
            summary.mde / config_.calibration.task_displacement_scale_m,
            summary.velocity_mde /
                config_.calibration.task_velocity_scale_mps);
        if (summary.jump_required)
          summary.task_power_ratio = std::max(
              summary.task_power_ratio,
              summary.jump_mde /
                  config_.calibration.task_interface_jump_scale_m);
      }
      summary.any_task_direction_at_power =
          std::isfinite(summary.best_task_power_ratio) &&
          summary.best_task_power_ratio <=
              1.0 + config_.numerical.tie_tolerance;
    }

    // Build the GDU-level guaranteed observable subspace from a deterministic
    // credible set of the three physical geometry hypotheses.  Geometry
    // OTHER is an explicit unusable/model-inadequacy hypothesis and therefore
    // degrades admission separately; intersecting its zero projector would
    // incorrectly make every GDU rank zero regardless of the measurements.
    std::array<double, 4> geometry_mass{{0.0, 0.0, 0.0, 0.0}};
    for (const GaussianComponent& component : states_[g].components)
      geometry_mass[static_cast<size_t>(component.geometry)] += component.weight;
    std::array<int, 3> physical_order{{0, 1, 2}};
    std::sort(physical_order.begin(), physical_order.end(),
              [&geometry_mass](int lhs, int rhs) {
                if (geometry_mass[static_cast<size_t>(lhs)] !=
                    geometry_mass[static_cast<size_t>(rhs)])
                  return geometry_mass[static_cast<size_t>(lhs)] >
                         geometry_mass[static_cast<size_t>(rhs)];
                return lhs < rhs;
              });
    const double physical_mass = geometry_mass[0] + geometry_mass[1] + geometry_mass[2];
    const double credible_target =
        (1.0 - config_.resource.other_mass_alarm) * physical_mass;
    std::array<bool, 3> selected{{false, false, false}};
    double selected_mass = 0.0;
    for (int hypothesis : physical_order) {
      if (geometry_mass[static_cast<size_t>(hypothesis)] <=
          config_.numerical.probability_floor) continue;
      selected[static_cast<size_t>(hypothesis)] = true;
      selected_mass += geometry_mass[static_cast<size_t>(hypothesis)];
      if (selected_mass + config_.numerical.normalization_tolerance >= credible_target)
        break;
    }
    Eigen::MatrixXd common_observable = Eigen::MatrixXd::Identity(n, n);
    bool has_selected_geometry = false;
    double worst_mde = 0.0;
    double worst_reference_information_ratio = 1.0;
    double worst_effective_spatial_support =
        std::numeric_limits<double>::infinity();
    double worst_effective_temporal_support =
        std::numeric_limits<double>::infinity();
    double worst_velocity_mde = 0.0;
    double worst_jump_mde = 0.0;
    bool any_jump_required = false;
    bool all_velocity_mde_defined = true;
    bool all_required_jump_mde_defined = true;
    bool task_power_complete = true;
    bool any_task_direction_at_power = true;
    double worst_task_power_ratio = 0.0;
    bool reference_confounding = false;
    for (int h = 0; h < 3; ++h) {
      if (!selected[static_cast<size_t>(h)]) continue;
      has_selected_geometry = true;
      worst_reference_information_ratio = std::min(
          worst_reference_information_ratio,
          states_[g].reference_information_ratio[static_cast<size_t>(h)]);
      worst_effective_spatial_support = std::min(
          worst_effective_spatial_support,
          states_[g].effective_spatial_support[static_cast<size_t>(h)]);
      worst_effective_temporal_support = std::min(
          worst_effective_temporal_support,
          states_[g].effective_temporal_support[static_cast<size_t>(h)]);
      reference_confounding = reference_confounding ||
          states_[g].reference_confounding[static_cast<size_t>(h)];
      task_power_complete = task_power_complete &&
          summaries[static_cast<size_t>(h)].task_power_complete;
      any_task_direction_at_power = any_task_direction_at_power &&
          summaries[static_cast<size_t>(h)].any_task_direction_at_power;
      all_velocity_mde_defined = all_velocity_mde_defined &&
          summaries[static_cast<size_t>(h)].velocity_mde_defined;
      if (summaries[static_cast<size_t>(h)].velocity_mde_defined)
        worst_velocity_mde = std::max(
            worst_velocity_mde,
            summaries[static_cast<size_t>(h)].velocity_mde);
      if (summaries[static_cast<size_t>(h)].jump_required) {
        any_jump_required = true;
        all_required_jump_mde_defined = all_required_jump_mde_defined &&
            summaries[static_cast<size_t>(h)].jump_mde_defined;
        if (summaries[static_cast<size_t>(h)].jump_mde_defined)
          worst_jump_mde = std::max(
              worst_jump_mde, summaries[static_cast<size_t>(h)].jump_mde);
      }
      if (summaries[static_cast<size_t>(h)].task_power_complete)
        worst_task_power_ratio = std::max(
            worst_task_power_ratio,
            summaries[static_cast<size_t>(h)].task_power_ratio);
      const Eigen::MatrixXd& geometry_projector =
          summaries[static_cast<size_t>(h)].projector;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> intersection(
          0.5 * (common_observable * geometry_projector * common_observable +
                 common_observable * geometry_projector.transpose() * common_observable));
      if (intersection.info() != Eigen::Success) {
        common_observable.setZero();
      } else {
        common_observable.setZero();
        for (int mode = 0; mode < n; ++mode)
          if (intersection.eigenvalues()[mode] >
              1.0 - 10.0 * config_.numerical.psd_eigenvalue_tolerance)
            common_observable += intersection.eigenvectors().col(mode) *
                                 intersection.eigenvectors().col(mode).transpose();
      }
      if (!summaries[static_cast<size_t>(h)].mde_defined)
        worst_mde = std::numeric_limits<double>::infinity();
      else if (std::isfinite(worst_mde))
        worst_mde = std::max(worst_mde, summaries[static_cast<size_t>(h)].mde);
    }
    if (!has_selected_geometry) {
      common_observable.setZero();
      worst_mde = std::numeric_limits<double>::infinity();
      worst_reference_information_ratio = 0.0;
      worst_effective_spatial_support = 0.0;
      worst_effective_temporal_support = 0.0;
      task_power_complete = false;
      any_task_direction_at_power = false;
      all_velocity_mde_defined = false;
      all_required_jump_mde_defined = false;
      worst_task_power_ratio = std::numeric_limits<double>::infinity();
    }
    int rank = 0;
    if (n > 0) {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> observable_eigen(
          common_observable);
      if (observable_eigen.info() != Eigen::Success) {
        posterior.observable_projector = Eigen::MatrixXd::Zero(n, n);
        posterior.nullspace = Eigen::MatrixXd::Identity(n, n);
        rank = 0;
      } else {
      std::vector<int> null_modes;
      for (int mode = 0; mode < n; ++mode)
        if (observable_eigen.eigenvalues()[mode] < 0.5) null_modes.push_back(mode);
      posterior.observable_projector = common_observable;
      posterior.nullspace = Eigen::MatrixXd::Zero(n, static_cast<int>(null_modes.size()));
      for (size_t column = 0; column < null_modes.size(); ++column)
        posterior.nullspace.col(static_cast<int>(column)) =
          observable_eigen.eigenvectors().col(null_modes[column]);
      rank = n - static_cast<int>(null_modes.size());
      }
    } else {
      // OTHER/non-monitorable GDUs intentionally have no physical
      // coefficient coordinates.  Eigen's dynamic self-adjoint solver is not
      // defined on a 0x0 matrix, so preserve the explicit empty observable and
      // null spaces without invoking a decomposition.
      posterior.observable_projector = Eigen::MatrixXd::Zero(0, 0);
      posterior.nullspace = Eigen::MatrixXd::Zero(0, 0);
      rank = 0;
    }
    double unresolved_interface_mass = 0.0;
    for (const GaussianComponent& component : states_[g].components) {
      posterior.state_probability[static_cast<size_t>(component.state)] += component.weight;
      if (std::find(component.interface_modes.begin(),
                    component.interface_modes.end(),
                    InterfaceJointState::LINK_UNRESOLVED) !=
          component.interface_modes.end())
        unresolved_interface_mass += component.weight;
      if (component.state != StructuralState::U) {
        const TaskEventProbability probability =
            ComponentTaskScaleProbability(g, component);
        posterior.task_displacement_probability +=
            component.weight * probability.displacement;
        posterior.current_velocity_exceedance_probability +=
            component.weight * probability.current_velocity;
        posterior.task_interface_jump_probability +=
            component.weight * probability.interface_jump;
        task_union_without_velocity +=
            component.weight * probability.event_union_without_velocity;
        posterior.task_scale_deformation_probability +=
            component.weight * probability.event_union;
      }
    }
    for (double& probability : posterior.state_probability)
      probability = ClampUnitProbability(probability);
    posterior.other_mass = ClampUnitProbability(posterior.other_mass);
    posterior.task_displacement_probability = ClampUnitProbability(
        posterior.task_displacement_probability);
    posterior.current_velocity_exceedance_probability = ClampUnitProbability(
        posterior.current_velocity_exceedance_probability);
    posterior.task_interface_jump_probability = ClampUnitProbability(
        posterior.task_interface_jump_probability);
    task_union_without_velocity = ClampUnitProbability(
        task_union_without_velocity);
    posterior.task_scale_deformation_probability = ClampUnitProbability(
        posterior.task_scale_deformation_probability);
    posterior.continuous_valid_observation_duration_s =
        states_[g].continuous_valid_velocity_duration_s;
    posterior.velocity_duration_requirement_met =
        posterior.continuous_valid_observation_duration_s +
            config_.numerical.tie_tolerance >=
        config_.calibration.task_velocity_required_duration_s;
    if (!posterior.velocity_duration_requirement_met)
      posterior.task_scale_deformation_probability =
          task_union_without_velocity;
    posterior.task_scale_deformation_lower = std::max(
        0.0, posterior.task_scale_deformation_probability -
                 config_.calibration.qmc_error_upper);
    posterior.task_scale_deformation_upper = std::min(
        1.0, posterior.task_scale_deformation_probability +
                 config_.calibration.qmc_error_upper + states_[g].other_mass);
    posterior.task_scale_deformation_mean_identified =
        states_[g].other_mass <= config_.numerical.probability_floor;
    posterior.minimum_detectable_effect_m = worst_mde;
    posterior.minimum_detectable_effect_defined = std::isfinite(worst_mde);
    posterior.velocity_minimum_detectable_effect_mps =
        worst_velocity_mde;
    posterior.velocity_minimum_detectable_effect_defined =
        all_velocity_mde_defined && std::isfinite(worst_velocity_mde);
    posterior.interface_jump_minimum_detectable_effect_m = worst_jump_mde;
    posterior.interface_jump_minimum_detectable_effect_defined =
        any_jump_required && all_required_jump_mde_defined &&
        std::isfinite(worst_jump_mde);
    posterior.task_power_complete = task_power_complete;
    posterior.task_power_worst_normalized_mde =
        task_power_complete ? worst_task_power_ratio
                            : std::numeric_limits<double>::infinity();
    posterior.observable_rank = rank;
    posterior.any_task_direction_detectable = rank > 0 &&
        any_task_direction_at_power;
    posterior.all_task_directions_detectable = task_power_complete &&
        worst_task_power_ratio <=
            1.0 + config_.numerical.tie_tolerance;
    posterior.reference_information_ratio =
        worst_reference_information_ratio;
    posterior.reference_confounding = reference_confounding;
    posterior.effective_spatial_support =
        std::isfinite(worst_effective_spatial_support)
            ? worst_effective_spatial_support : 0.0;
    posterior.effective_temporal_support =
        std::isfinite(worst_effective_temporal_support)
            ? worst_effective_temporal_support : 0.0;
    posterior.components = states_[g].components;
    for (GaussianComponent& component : posterior.components) {
      const ObservableSummary& summary =
          summaries[static_cast<size_t>(component.geometry)];
      component.observable_rank = summary.rank;
      component.observable_projector = summary.projector;
      component.observable_nullspace = summary.nullspace;
      component.minimum_detectable_effect_m = summary.mde;
      component.minimum_detectable_effect_defined = summary.mde_defined;
    }
    const bool source_available = states_[g].consecutive_source_missing <
        config_.resource.source_missing_consecutive_limit;
    const bool common_detection_gate = config_.calibration.validated &&
        states_[g].current_measurement_valid && source_available &&
        states_[g].other_mass <= config_.resource.other_mass_alarm &&
        !posterior.reference_confounding &&
        posterior.effective_spatial_support + config_.numerical.tie_tolerance >=
            config_.calibration.detection_min_effective_spatial_support &&
        posterior.effective_temporal_support + config_.numerical.tie_tolerance >=
            config_.calibration.detection_min_effective_temporal_support;
    posterior.detection_allowed = common_detection_gate &&
        posterior.any_task_direction_detectable;
    posterior.clearing_allowed = common_detection_gate &&
        posterior.all_task_directions_detectable &&
        unresolved_interface_mass <=
            config_.calibration.clearing_probability_on +
                config_.numerical.tie_tolerance &&
        posterior.task_power_worst_normalized_mde <=
            1.0 + config_.numerical.tie_tolerance;
    if (!config_.calibration.validated)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_UNCALIBRATED;
    else if (!states_[g].current_measurement_valid || !source_available)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_ASSOCIATION;
    else if (states_[g].other_mass > config_.resource.other_mass_alarm)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_ASSOCIATION;
    else if (geometry_mass[3] > config_.resource.other_mass_alarm)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_REFERENCE;
    else if (posterior.reference_confounding)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_REFERENCE;
    else if (posterior.effective_spatial_support +
                 config_.numerical.tie_tolerance <
             config_.calibration.detection_min_effective_spatial_support ||
             posterior.effective_temporal_support +
                 config_.numerical.tie_tolerance <
             config_.calibration.detection_min_effective_temporal_support)
      posterior.admission = DetectionAdmission::INCONCLUSIVE_LOW_POWER;
    else if (posterior.detection_allowed &&
             posterior.task_scale_deformation_lower >=
             config_.calibration.detection_probability_on)
      posterior.admission = DetectionAdmission::DETECTED_AT_SCALE;
    else if (posterior.clearing_allowed &&
             posterior.task_scale_deformation_upper <=
             config_.calibration.clearing_probability_on)
      posterior.admission = DetectionAdmission::CLEARED_AT_SCALE;
    else posterior.admission = DetectionAdmission::INCONCLUSIVE_LOW_POWER;
    output[g] = std::move(posterior);
    completed[g] = 1U;
    } catch (...) {
      // Defer propagation until every successfully-created thread has been
      // joined.  Posterior is const, so a failed worker has no filter state to
      // roll back and no partial result is returned.
      exceptions[g] = std::current_exception();
    }
  };

  if (!parallelize_independent_gdus || states_.size() < 2U) {
    for (uint32_t g = 0U; g < states_.size(); ++g)
      build_one_gdu(g);
  } else {
    std::atomic<size_t> next_gdu(0U);
    const size_t worker_count = std::min<size_t>(states_.size(), 8U);
    const auto worker = [&]() {
      while (true) {
        const size_t index = next_gdu.fetch_add(1U);
        if (index >= states_.size()) return;
        build_one_gdu(static_cast<uint32_t>(index));
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(worker_count > 0U ? worker_count - 1U : 0U);
    for (size_t worker_index = 1U; worker_index < worker_count;
         ++worker_index) {
      // The finite limit is a regression hook for the exact state reached
      // after a std::thread constructor fails.  Production leaves it at
      // size_t::max and therefore attempts the complete bounded worker set.
      if (workers.size() >= successful_worker_creation_limit) break;
      try {
        workers.emplace_back(worker);
      } catch (const std::system_error&) {
        // Resource pressure changes only latency: already-created workers and
        // the calling thread still drain the same complete atomic index set.
        break;
      }
    }
    worker();
    for (std::thread& thread : workers) thread.join();
  }

  // Reproduce the serial path's first failing GDU semantics.  Work completed
  // for later GDUs is private and discarded when an earlier cache is absent
  // or an exception is rethrown.
  for (size_t g = 0U; g < states_.size(); ++g) {
    if (completed[g] != 0U) continue;
    if (exceptions[g]) std::rethrow_exception(exceptions[g]);
    return {};
  }
  return output;
}

std::vector<InterfacePosterior, Eigen::aligned_allocator<InterfacePosterior>>
UacGaussianSumFilter::InterfacePosteriors() const {
  return InterfacePosteriors(Posterior());
}

std::vector<InterfacePosterior, Eigen::aligned_allocator<InterfacePosterior>>
UacGaussianSumFilter::InterfacePosteriors(
    const std::vector<GduPosterior,
        Eigen::aligned_allocator<GduPosterior>>& frozen_posterior) const {
  return InterfacePosteriorsImpl(frozen_posterior, true);
}

std::vector<InterfacePosterior, Eigen::aligned_allocator<InterfacePosterior>>
UacGaussianSumFilter::InterfacePosteriorsImpl(
    const std::vector<GduPosterior,
        Eigen::aligned_allocator<GduPosterior>>& frozen_posterior,
    bool use_frozen_surfel_cache) const {
  std::vector<InterfacePosterior, Eigen::aligned_allocator<InterfacePosterior>> output;
  for (uint32_t interface_index = 0; interface_index < reference_.interfaces.size();
       ++interface_index) {
    const InterfaceDefinition& definition = reference_.interfaces[interface_index];
    InterfacePosterior posterior; posterior.interface_id = definition.id;
    if (interface_index >= shared_interfaces_.size()) {
      posterior.validity = Validity::INVALID; output.push_back(posterior); continue;
    }
    const SharedInterfaceState& shared = shared_interfaces_[interface_index];
    posterior.joint_probability = shared.probability;
    for (double& probability : posterior.joint_probability)
      probability = ClampUnitProbability(probability);
    posterior.other_mass = ClampUnitProbability(shared.other_mass);
    double noncut_mass = 0.0;
    for (size_t mode = 1U; mode < 4U; ++mode) {
      posterior.jump_mean_R += shared.probability[mode] * shared.mean[mode];
      noncut_mass += shared.probability[mode];
    }
    if (noncut_mass > 0.0) posterior.jump_mean_R /= noncut_mass;
    for (size_t mode = 1U; mode < 4U; ++mode) {
      const Eigen::Vector3d delta = shared.mean[mode] - posterior.jump_mean_R;
      posterior.jump_covariance_R += shared.probability[mode] *
          (shared.covariance[mode] + delta * delta.transpose());
    }
    if (noncut_mass > 0.0) posterior.jump_covariance_R /= noncut_mass;
    posterior.connectivity_coherence_confounding =
        posterior.joint_probability[0] > config_.numerical.probability_floor &&
        posterior.joint_probability[2] > config_.numerical.probability_floor;
    posterior.validity = config_.calibration.validated &&
        posterior.other_mass <= config_.resource.other_mass_alarm
            ? Validity::VALID : Validity::INCONCLUSIVE;
    if (definition.left_surfel < reference_.surfels.size() &&
        definition.right_surfel < reference_.surfels.size()) {
      posterior.left_query_position_R =
          reference_.surfels[definition.left_surfel].center_R;
      posterior.right_query_position_R =
          reference_.surfels[definition.right_surfel].center_R;
      const GduPosterior* left_posterior =
          definition.left_gdu < frozen_posterior.size()
              ? &frozen_posterior[definition.left_gdu] : nullptr;
      const GduPosterior* right_posterior =
          definition.right_gdu < frozen_posterior.size()
              ? &frozen_posterior[definition.right_gdu] : nullptr;
      const auto query_endpoint = [&](uint32_t gdu_index,
                                      uint32_t surfel_index,
                                      const Eigen::Vector3d& position_R,
                                      const GduPosterior* endpoint_posterior) {
        if (use_frozen_surfel_cache && endpoint_posterior &&
            gdu_index < states_.size()) {
          const GduState& state = states_[gdu_index];
          // A missing/corrupted cache must preserve the former generic-query
          // behavior rather than introduce a new interface failure mode.
          if (state.frozen_reference_cache) {
            const auto cached =
                state.frozen_reference_cache->surfel_basis_offsets.find(
                    surfel_index);
            if (cached !=
                    state.frozen_reference_cache->surfel_basis_offsets.end() &&
                cached->second <
                    state.frozen_reference_cache->surfel_bases.size() &&
                surfel_index < reference_.surfels.size() &&
                reference_.surfels[surfel_index].gdu_index == gdu_index) {
              return QueryReferenceSurfelWithPosterior(
                  gdu_index, surfel_index, *endpoint_posterior);
            }
          }
        }
        return QueryCurrent(gdu_index, position_R, endpoint_posterior);
      };
      posterior.left_field = query_endpoint(
          definition.left_gdu, definition.left_surfel,
          posterior.left_query_position_R, left_posterior);
      posterior.right_field = query_endpoint(
          definition.right_gdu, definition.right_surfel,
          posterior.right_query_position_R, right_posterior);
    } else {
      posterior.validity = Validity::INVALID;
    }
    output.push_back(posterior);
  }
  return output;
}

FieldQueryResult UacGaussianSumFilter::Query(uint32_t gdu_index,
                                             const Eigen::Vector3d& position_R) const {
  return QueryAtHorizon(gdu_index, position_R, 0.0);
}

FieldQueryResult UacGaussianSumFilter::QueryWithPosterior(
    uint32_t gdu_index, const Eigen::Vector3d& position_R,
    const GduPosterior& posterior) const {
  return QueryCurrent(gdu_index, position_R, &posterior);
}

FieldQueryResult UacGaussianSumFilter::QueryReferenceSurfelWithPosterior(
    uint32_t gdu_index, uint32_t surfel_index,
    const GduPosterior& posterior) const {
  FieldQueryResult result;
  if (!initialized_ || gdu_index >= states_.size()) {
    result.reasons.push_back(ReasonCode::REFERENCE_NOT_FROZEN);
    return result;
  }
  const GduState& state = states_[gdu_index];
  if (!state.frozen_reference_cache) {
    result.validity = Validity::INVALID;
    result.reasons.push_back(ReasonCode::INVALID_NUMERICAL);
    return result;
  }
  const auto found =
      state.frozen_reference_cache->surfel_basis_offsets.find(surfel_index);
  if (found == state.frozen_reference_cache->surfel_basis_offsets.end() ||
      found->second >= state.frozen_reference_cache->surfel_bases.size() ||
      surfel_index >= reference_.surfels.size() ||
      reference_.surfels[surfel_index].gdu_index != gdu_index) {
    result.validity = Validity::INVALID;
    result.reasons.push_back(ReasonCode::INVALID_NUMERICAL);
    return result;
  }
  const GduState::FrozenReferenceSurfelBasis& cached =
      state.frozen_reference_cache->surfel_bases[found->second];
  return QueryCurrentWithBasis(
      gdu_index, cached.basis, cached.validity, cached.reasons,
      cached.attachment_distance_m, &posterior);
}

bool UacGaussianSumFilter::ReferenceSurfelBasis(
    uint32_t gdu_index, uint32_t surfel_index, Eigen::MatrixXd* basis,
    Validity* validity, std::vector<ReasonCode>* reasons,
    double* attachment_distance_m) const {
  if (basis) basis->resize(0, 0);
  if (validity) *validity = Validity::UNKNOWN;
  if (attachment_distance_m)
    *attachment_distance_m = std::numeric_limits<double>::infinity();
  if (!basis || !initialized_ || gdu_index >= states_.size()) {
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_NOT_FROZEN);
    return false;
  }
  const GduState& state = states_[gdu_index];
  if (!state.frozen_reference_cache) {
    if (validity) *validity = Validity::INVALID;
    if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    return false;
  }
  const auto found =
      state.frozen_reference_cache->surfel_basis_offsets.find(surfel_index);
  if (found == state.frozen_reference_cache->surfel_basis_offsets.end() ||
      found->second >= state.frozen_reference_cache->surfel_bases.size() ||
      surfel_index >= reference_.surfels.size() ||
      reference_.surfels[surfel_index].gdu_index != gdu_index) {
    if (validity) *validity = Validity::INVALID;
    if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    return false;
  }
  const GduState::FrozenReferenceSurfelBasis& cached =
      state.frozen_reference_cache->surfel_bases[found->second];
  *basis = cached.basis;
  if (validity) *validity = cached.validity;
  if (reasons) reasons->insert(reasons->end(), cached.reasons.begin(),
                               cached.reasons.end());
  if (attachment_distance_m)
    *attachment_distance_m = cached.attachment_distance_m;
  return true;
}

FieldQueryResult UacGaussianSumFilter::QueryAtHorizon(
    uint32_t gdu_index, const Eigen::Vector3d& position_R, double horizon_s) const {
  FieldQueryResult result;
  if (!initialized_ || gdu_index >= states_.size() || !(horizon_s >= 0.0) ||
      horizon_s > config_.resource.prediction_horizons_s.back()) {
    if (initialized_ && gdu_index < states_.size() &&
        (!(horizon_s >= 0.0) ||
         horizon_s > config_.resource.prediction_horizons_s.back()))
      result.reasons.push_back(ReasonCode::UNKNOWN_PREDICTION);
    else
      result.reasons.push_back(ReasonCode::REFERENCE_NOT_FROZEN);
    return result;
  }
  // A future field query is a query of the complete future switching
  // posterior, not merely an A/C kinematic extrapolation with today's state
  // weights.  Work on a copy so the live monitoring state remains frozen.
  if (horizon_s > config_.numerical.tie_tolerance) {
    UacGaussianSumFilter predicted(*this);
    if (!predicted.PredictInterval(
            last_stamp_ + ros::Duration(horizon_s), false,
            &result.reasons)) {
      result.validity = Validity::UNKNOWN;
      return result;
    }
    return predicted.QueryAtHorizon(gdu_index, position_R, 0.0);
  }
  return QueryCurrent(gdu_index, position_R, nullptr);
}

FieldQueryResult UacGaussianSumFilter::QueryCurrent(
    uint32_t gdu_index, const Eigen::Vector3d& position_R,
    const GduPosterior* frozen_posterior) const {
  FieldQueryResult result;
  if (!initialized_ || gdu_index >= states_.size()) {
    result.reasons.push_back(ReasonCode::REFERENCE_NOT_FROZEN);
    return result;
  }
  PouField field(&reference_, &config_);
  Validity basis_validity = Validity::UNKNOWN;
  std::vector<ReasonCode> basis_reasons;
  double attachment_distance_m = std::numeric_limits<double>::infinity();
  Eigen::MatrixXd basis = field.VectorBasis(gdu_index, position_R,
                                            &basis_validity, &basis_reasons,
                                            &attachment_distance_m);
  return QueryCurrentWithBasis(
      gdu_index, basis, basis_validity, basis_reasons,
      attachment_distance_m, frozen_posterior);
}

FieldQueryResult UacGaussianSumFilter::QueryCurrentWithBasis(
    uint32_t gdu_index, const Eigen::MatrixXd& basis,
    Validity basis_validity, const std::vector<ReasonCode>& basis_reasons,
    double attachment_distance_m,
    const GduPosterior* frozen_posterior) const {
  FieldQueryResult result;
  if (!initialized_ || gdu_index >= states_.size()) {
    result.reasons.push_back(ReasonCode::REFERENCE_NOT_FROZEN);
    return result;
  }
  if (basis.size() == 0) {
    result.validity = basis_validity; result.reasons = basis_reasons; return result;
  }
  result.interpolation_error_upper_m = attachment_distance_m +
      config_.calibration.prediction_model_error_m +
      config_.calibration.ctmc_halving_error_upper_m;
  const int n = states_[gdu_index].coefficient_count;
  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>>
      owned_posterior;
  if (!frozen_posterior) {
    owned_posterior = Posterior();
    if (gdu_index < owned_posterior.size())
      frozen_posterior = &owned_posterior[gdu_index];
  }
  if (!frozen_posterior || frozen_posterior->gdu_id !=
          reference_.gdus[gdu_index].id ||
      frozen_posterior->components.size() !=
          states_[gdu_index].components.size()) {
    result.validity = Validity::INVALID;
    result.reasons.push_back(ReasonCode::INVALID_NUMERICAL);
    return result;
  }
  if (frozen_posterior->observable_projector.rows() == n && n > 0) {
    const Eigen::Matrix3d local_range = basis *
        frozen_posterior->observable_projector * basis.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> local_eigen(
        0.5 * (local_range + local_range.transpose()));
    if (local_eigen.info() == Eigen::Success) {
      const double scale = std::max(1.0, local_eigen.eigenvalues().maxCoeff());
      for (int mode = 0; mode < 3; ++mode)
        if (local_eigen.eigenvalues()[mode] >
            config_.numerical.psd_eigenvalue_tolerance * scale)
          result.observable_projector_R += local_eigen.eigenvectors().col(mode) *
              local_eigen.eigenvectors().col(mode).transpose();
    }
  }
  double resolved_mass = 0.0;
  for (size_t component_index = 0;
       component_index < states_[gdu_index].components.size(); ++component_index) {
    const GaussianComponent& component =
        states_[gdu_index].components[component_index];
    const GaussianComponent& observable_component =
        frozen_posterior->components[component_index];
    if (observable_component.component_id != component.component_id) {
      result.validity = Validity::INVALID;
      result.reasons.push_back(ReasonCode::INVALID_NUMERICAL);
      result.components.clear();
      return result;
    }
    FieldComponentQuery query_component;
    query_component.component_id = component.component_id;
    query_component.weight = component.weight;
    query_component.state = component.state;
    query_component.geometry = component.geometry;
    Eigen::MatrixXd component_basis = basis;
    const GduDefinition& gdu = reference_.gdus[gdu_index];
    for (size_t i = 0; i < component.interface_modes.size(); ++i) {
      if (component.interface_modes[i] == InterfaceJointState::CUT_NOT_APPLICABLE) {
        const int column = gdu.coordinate_rank * static_cast<int>(
            gdu.control_surfel_indices.size() + i);
        component_basis.block(0, column, 3, gdu.coordinate_rank).setZero();
      }
    }
    if (observable_component.observable_projector.rows() == n && n > 0) {
      const Eigen::Matrix3d local_information_range = component_basis *
          observable_component.observable_projector * component_basis.transpose();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> local_eigen(
          0.5 * (local_information_range + local_information_range.transpose()));
      if (local_eigen.info() == Eigen::Success) {
        const double scale = std::max(1.0, local_eigen.eigenvalues().maxCoeff());
        for (int mode = 0; mode < 3; ++mode)
          if (local_eigen.eigenvalues()[mode] >
              config_.numerical.psd_eigenvalue_tolerance * scale)
            query_component.observable_projector_R +=
                local_eigen.eigenvectors().col(mode) *
                local_eigen.eigenvectors().col(mode).transpose();
      }
    }
    query_component.minimum_detectable_effect_m =
        observable_component.minimum_detectable_effect_m;
    query_component.minimum_detectable_effect_defined =
        observable_component.minimum_detectable_effect_defined;
    if (component.state != StructuralState::U) {
      Eigen::VectorXd position = component.mean.head(n);
      Eigen::MatrixXd position_covariance =
          component.covariance.block(0, 0, n, n);
      if (component.state == StructuralState::A) {
        const Eigen::VectorXd velocity = component.mean.segment(n, n);
        query_component.mean_velocity_R = component_basis * velocity;
        query_component.velocity_covariance_R = component_basis *
            component.covariance.block(n, n, n, n) *
            component_basis.transpose();
      }
      query_component.mean_displacement_R = component_basis * position;
      query_component.displacement_covariance_R = component_basis *
          position_covariance * component_basis.transpose();
    }
    result.components.push_back(query_component);
    resolved_mass += component.weight;
  }
  if (resolved_mass > config_.numerical.probability_floor) {
    for (const FieldComponentQuery& component : result.components) {
      const double weight = component.weight / resolved_mass;
      result.mean_displacement_R += weight * component.mean_displacement_R;
      result.mean_velocity_R += weight * component.mean_velocity_R;
    }
    for (const FieldComponentQuery& component : result.components) {
      const double weight = component.weight / resolved_mass;
      const Eigen::Vector3d displacement_delta =
          component.mean_displacement_R - result.mean_displacement_R;
      const Eigen::Vector3d velocity_delta =
          component.mean_velocity_R - result.mean_velocity_R;
      result.displacement_covariance_R += weight *
          (component.displacement_covariance_R +
           displacement_delta * displacement_delta.transpose());
      result.velocity_covariance_R += weight *
          (component.velocity_covariance_R +
           velocity_delta * velocity_delta.transpose());
    }
  }
  result.active_probability.mean = 0.0;
  for (const GaussianComponent& component : states_[gdu_index].components)
    if (component.state != StructuralState::U) result.active_probability.mean += component.weight;
  result.active_probability.lower = result.active_probability.mean;
  result.active_probability.upper = std::min(1.0, result.active_probability.mean +
                                                    states_[gdu_index].other_mass);
  result.active_probability.calibrated = config_.calibration.validated;
  result.active_probability.mean_identified =
      states_[gdu_index].other_mass <= config_.numerical.probability_floor;
  result.other_mass = states_[gdu_index].other_mass;
  result.validity = config_.calibration.validated &&
      result.other_mass <= config_.resource.other_mass_alarm
          ? Validity::VALID : Validity::INCONCLUSIVE;
  if (!config_.calibration.validated) result.reasons.push_back(ReasonCode::CALIBRATION_UNSET);
  if (result.other_mass > config_.resource.other_mass_alarm)
    result.reasons.push_back(ReasonCode::ASSOCIATION_AMBIGUOUS);
  return result;
}

double UacGaussianSumFilter::other_mass() const {
  double maximum = 0.0;
  for (const GduState& state : states_) maximum = std::max(maximum, state.other_mass);
  return maximum;
}

}  // namespace v1
}  // namespace deform_monitor_v3
