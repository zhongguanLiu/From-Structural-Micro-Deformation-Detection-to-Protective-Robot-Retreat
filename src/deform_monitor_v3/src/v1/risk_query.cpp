#include "deform_monitor_v3/v1/risk_query.hpp"

#include "deform_monitor_v3/v1/deterministic.hpp"
#include "deform_monitor_v3/v1/math.hpp"

#include <Eigen/Eigenvalues>
#include <boost/random/sobol.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <map>
#include <set>

namespace deform_monitor_v3 {
namespace v1 {
namespace {

Eigen::Vector3d CellCenter(const VoxelKey& key, const V1Config& config) {
  return config.resource.grid_origin_m + config.resource.occupancy_voxel_m *
      Eigen::Vector3d(static_cast<double>(key.x) + 0.5,
                      static_cast<double>(key.y) + 0.5,
                      static_cast<double>(key.z) + 0.5);
}

bool SphereIntersectsBox(const Eigen::Vector3d& center, double radius,
                         const Eigen::Vector3d& box_center, double half_extent) {
  const Eigen::Vector3d closest = center.cwiseMax(
      box_center - Eigen::Vector3d::Constant(half_extent)).cwiseMin(
      box_center + Eigen::Vector3d::Constant(half_extent));
  return (center - closest).squaredNorm() <= radius * radius;
}

bool ValidRigidPose(const Eigen::Isometry3d& pose,
                    double tolerance) {
  if (!pose.matrix().allFinite()) return false;
  const Eigen::Matrix3d rotation = pose.linear();
  return (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
                 .norm() <= tolerance &&
         std::abs(rotation.determinant() - 1.0) <= tolerance;
}

struct BodySamplingGrid {
  Eigen::Vector3d minimum = Eigen::Vector3d::Zero();
  Eigen::Array3i count = Eigen::Array3i::Zero();
  double spacing_m = 0.0;
};

bool BuildBodySamplingGrid(const RobotGeometry& robot, double occupancy_voxel_m,
                           size_t maximum_samples, BodySamplingGrid* grid,
                           bool* resource_exceeded) {
  if (resource_exceeded) *resource_exceeded = false;
  if (!grid || robot.body_frame.empty() || robot.spheres.empty() ||
      !(occupancy_voxel_m > 0.0) || !std::isfinite(occupancy_voxel_m) ||
      maximum_samples == 0U) {
    return false;
  }
  if (robot.spheres.size() > maximum_samples) {
    if (resource_exceeded) *resource_exceeded = true;
    return false;
  }
  Eigen::Vector3d body_min = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::infinity());
  Eigen::Vector3d body_max = Eigen::Vector3d::Constant(
      -std::numeric_limits<double>::infinity());
  for (const RobotSphere& sphere : robot.spheres) {
    if (!sphere.center_body.allFinite() || !std::isfinite(sphere.radius_m) ||
        !(sphere.radius_m > 0.0)) {
      return false;
    }
    body_min = body_min.cwiseMin(
        sphere.center_body - Eigen::Vector3d::Constant(sphere.radius_m));
    body_max = body_max.cwiseMax(
        sphere.center_body + Eigen::Vector3d::Constant(sphere.radius_m));
  }
  const long double spacing =
      0.5L * static_cast<long double>(occupancy_voxel_m);
  long double sample_count = 1.0L;
  Eigen::Array3i count = Eigen::Array3i::Zero();
  for (int axis = 0; axis < 3; ++axis) {
    const long double extent = static_cast<long double>(
        body_max[axis] - body_min[axis]);
    const long double axis_count = std::max(
        1.0L, std::ceil(extent / spacing));
    if (!std::isfinite(axis_count) ||
        axis_count > static_cast<long double>(std::numeric_limits<int>::max()) ||
        sample_count > static_cast<long double>(maximum_samples) / axis_count) {
      if (resource_exceeded) *resource_exceeded = true;
      return false;
    }
    count[axis] = static_cast<int>(axis_count);
    sample_count *= axis_count;
  }
  grid->minimum = body_min;
  grid->count = count;
  grid->spacing_m = static_cast<double>(spacing);
  return true;
}

Eigen::Matrix3d RotationExponential(const Eigen::Vector3d& rotation_vector) {
  const double angle = rotation_vector.norm();
  if (angle <= 1.0e-15) return Eigen::Matrix3d::Identity() + Skew(rotation_vector);
  return Eigen::AngleAxisd(angle, rotation_vector / angle).toRotationMatrix();
}

Eigen::Isometry3d PerturbPose(const Eigen::Isometry3d& nominal,
                             const Eigen::Matrix<double, 6, 1>& perturbation) {
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  // Query covariance is [world-left rotation, world-additive body-origin
  // translation], not an unqualified SE(3) twist (API-020/DEC-029).
  result.linear() = RotationExponential(perturbation.head<3>()) * nominal.linear();
  result.translation() = nominal.translation() + perturbation.tail<3>();
  return result;
}

bool SymmetricSquareRoot(const Eigen::MatrixXd& covariance,
                         const NumericalConfig& numerical,
                         Eigen::MatrixXd* square_root) {
  return DeterministicPsdSquareRoot(covariance, numerical, square_root);
}

std::vector<uint8_t> SobolSeedBytes(const StableId& snapshot_id,
                                    const StableId& cause_id,
                                    const std::string& sample_family) {
  std::vector<uint8_t> bytes;
  bytes.insert(bytes.end(), snapshot_id.bytes.begin(), snapshot_id.bytes.end());
  bytes.insert(bytes.end(), cause_id.bytes.begin(), cause_id.bytes.end());
  bytes.insert(bytes.end(), sample_family.begin(), sample_family.end());
  static const char label[] = "sobol-scramble-v1";
  bytes.insert(bytes.end(), label, label + sizeof(label) - 1U);
  return bytes;
}

uint64_t DigestWord(const Sha256Digest& digest, size_t offset) {
  uint64_t value = 0U;
  for (size_t i = 0; i < 8U; ++i)
    value = (value << 8U) | digest[(offset + i) % digest.size()];
  return value;
}

std::vector<uint64_t> SobolDigitalShifts(const StableId& snapshot_id,
                                         const StableId& cause_id,
                                         const std::string& sample_family,
                                         int dimension,
                                         uint64_t engine_maximum) {
  const Sha256Digest base = Sha256(
      SobolSeedBytes(snapshot_id, cause_id, sample_family));
  std::vector<uint64_t> shifts(static_cast<size_t>(dimension), 0U);
  for (int d = 0; d < dimension; ++d) {
    std::vector<uint8_t> bytes(base.begin(), base.end());
    const uint32_t index = static_cast<uint32_t>(d);
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes.push_back(static_cast<uint8_t>((index >> shift) & 0xffU));
    const Sha256Digest per_dimension = Sha256(bytes);
    shifts[static_cast<size_t>(d)] = DigestWord(per_dimension, 0U) & engine_maximum;
  }
  return shifts;
}

double ScrambledUniform(boost::random::sobol* engine, uint64_t digital_shift) {
  const uint64_t maximum = static_cast<uint64_t>(engine->max());
  const uint64_t raw = static_cast<uint64_t>((*engine)()) ^ (digital_shift & maximum);
  const long double denominator = static_cast<long double>(maximum) + 1.0L;
  return static_cast<double>((static_cast<long double>(raw) + 0.5L) / denominator);
}

bool PredictedFieldLatent(const GaussianComponent& component, int coefficient_count,
                          double horizon_s, const V1Config& config,
                          Eigen::VectorXd* mean, Eigen::MatrixXd* covariance) {
  if (!mean || !covariance || coefficient_count <= 0) return false;
  const int physical_dimension = component.state == StructuralState::A
      ? 2 * coefficient_count
      : (component.state == StructuralState::C ? coefficient_count : 0);
  if (component.mean.size() < physical_dimension + 3 ||
      component.covariance.rows() != component.mean.size() ||
      component.covariance.cols() != component.mean.size()) return false;
  Eigen::MatrixXd projection = Eigen::MatrixXd::Zero(
      coefficient_count + 3, component.mean.size());
  if (component.state != StructuralState::U)
    projection.block(0, 0, coefficient_count, coefficient_count).setIdentity();
  if (component.state == StructuralState::A)
    projection.block(0, coefficient_count, coefficient_count,
                     coefficient_count) =
        horizon_s * Eigen::MatrixXd::Identity(coefficient_count, coefficient_count);
  projection.block(coefficient_count, physical_dimension, 3, 3).setIdentity();
  *mean = projection * component.mean;
  *covariance = projection * component.covariance * projection.transpose();
  if (component.state == StructuralState::A) {
    covariance->topLeftCorner(coefficient_count, coefficient_count).diagonal().array() +=
        config.calibration.active_acceleration_psd * horizon_s * horizon_s * horizon_s / 3.0;
  } else if (component.state == StructuralState::C) {
    covariance->topLeftCorner(coefficient_count, coefficient_count).diagonal().array() +=
        config.calibration.changed_random_walk_psd * horizon_s;
  }
  *covariance = 0.5 * (*covariance + covariance->transpose());
  return mean->allFinite() && covariance->allFinite();
}

Eigen::MatrixXd ComponentBasis(const PouField& field,
                               const ReferenceSnapshot& reference,
                               uint32_t gdu_index,
                               const GaussianComponent& component,
                               const Eigen::Vector3d& position_R) {
  Validity validity = Validity::UNKNOWN;
  std::vector<ReasonCode> reasons;
  Eigen::MatrixXd basis = field.VectorBasis(gdu_index, position_R, &validity, &reasons);
  if (validity != Validity::VALID || basis.rows() != 3 ||
      basis.size() == 0 || !basis.allFinite() ||
      gdu_index >= reference.gdus.size())
    return Eigen::MatrixXd();
  const GduDefinition& gdu = reference.gdus[gdu_index];
  for (size_t i = 0; i < component.interface_modes.size() &&
                     i < gdu.interface_indices.size(); ++i) {
    if (component.interface_modes[i] == InterfaceJointState::CUT_NOT_APPLICABLE) {
      const int column = gdu.coordinate_rank * static_cast<int>(
          gdu.control_surfel_indices.size() + i);
      if (column + gdu.coordinate_rank <= basis.cols())
        basis.block(0, column, 3, gdu.coordinate_rank).setZero();
    }
  }
  return basis;
}

void NormalizeReasons(std::vector<ReasonCode>* reasons) {
  if (!reasons) return;
  std::sort(reasons->begin(), reasons->end(), [](ReasonCode lhs, ReasonCode rhs) {
    return static_cast<uint16_t>(lhs) < static_cast<uint16_t>(rhs);
  });
  reasons->erase(std::unique(reasons->begin(), reasons->end()), reasons->end());
}

ProbabilityInterval JointCausePathEvent(
    const ProbabilityInterval& accumulated,
    const ProbabilityInterval& current, bool initialized) {
  if (!initialized) return current;
  // Canonical common-rank coupling: sample index k denotes the same cause
  // latent over every fixed prediction node.  The path event is therefore an
  // OR over that sample's intersections, represented by the maximum CDF-rank
  // threshold, never a sum of repeated time/voxel marginals.
  ProbabilityInterval output;
  output.lower = std::max(accumulated.lower, current.lower);
  output.upper = std::max(accumulated.upper, current.upper);
  output.calibrated = accumulated.calibrated && current.calibrated;
  output.mean_identified = accumulated.mean_identified &&
                           current.mean_identified;
  output.mean = output.mean_identified
      ? std::max(accumulated.mean, current.mean)
      : output.upper;
  return output;
}

void StoreViolationConfiguration(const TimedRobotState& state,
                                 const std::string& pose_frame,
                                 const std::string& covariance_origin_frame,
                                 RiskAssessment* assessment) {
  if (!assessment) return;
  assessment->first_violation_time = state.stamp;
  assessment->first_violation_configuration_defined = true;
  assessment->first_violation_pose_frame = pose_frame;
  assessment->first_violation_covariance_origin_frame =
      covariance_origin_frame;
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < 4; ++column)
      assessment->first_violation_pose_row_major[
          static_cast<size_t>(4 * row + column)] =
          state.T_RB.matrix()(row, column);
  for (int row = 0; row < 6; ++row)
    for (int column = 0; column < 6; ++column)
      assessment->first_violation_covariance_row_major[
          static_cast<size_t>(6 * row + column)] =
          state.covariance(row, column);
}

}  // namespace

StableId RiskQueryEngine::GduCauseId(const StableId& gdu_id) const {
  CanonicalBytes fields;
  fields.AddString(config_.resource.session_id); fields.AddId(gdu_id);
  return MakeStableId("hazard-cause-v1", fields);
}

size_t RiskQueryEngine::EstimateStateQueryOperations(
    const ReferenceSnapshot& reference,
    const std::vector<GduPosterior,
        Eigen::aligned_allocator<GduPosterior>>& posterior,
    const OccupancyEventGraph& occupancy,
    const RobotGeometry& robot) const {
  std::set<StableId> structural_causes;
  long double primitive_tests = 0.0L;
  for (uint32_t gdu_index = 0U;
       gdu_index < reference.gdus.size() && gdu_index < posterior.size();
       ++gdu_index) {
    if (!reference.gdus[gdu_index].monitorable ||
        reference.gdus[gdu_index].coordinate_rank == 0U)
      continue;
    structural_causes.insert(GduCauseId(reference.gdus[gdu_index].id));
    size_t positive_components = 0U;
    for (const GaussianComponent& component : posterior[gdu_index].components)
      if (component.weight > 0.0) ++positive_components;
    primitive_tests += static_cast<long double>(positive_components) *
                       static_cast<long double>(
                           reference.gdus[gdu_index].surfel_indices.size());
  }
  for (const OccupancyCell& cell : occupancy.cells())
    for (const StableId& cause_id : cell.cause_ids)
      if (structural_causes.count(cause_id) == 0U)
        primitive_tests += 1.0L;
  const long double operations = primitive_tests *
      static_cast<long double>(robot.spheres.size()) *
      static_cast<long double>(config_.resource.prediction_samples) +
      static_cast<long double>(config_.resource.max_body_query_samples);
  if (!std::isfinite(operations) ||
      operations >= static_cast<long double>(
          std::numeric_limits<size_t>::max()))
    return std::numeric_limits<size_t>::max();
  return static_cast<size_t>(std::ceil(operations));
}

bool RiskQueryEngine::WithinOperationBudget(
    const ReferenceSnapshot& reference,
    const std::vector<GduPosterior,
        Eigen::aligned_allocator<GduPosterior>>& posterior,
    const OccupancyEventGraph& occupancy,
    const RobotGeometry& robot,
    size_t temporal_sample_count) const {
  if (temporal_sample_count == 0U) return false;
  const size_t per_state = EstimateStateQueryOperations(
      reference, posterior, occupancy, robot);
  return per_state <= config_.resource.max_risk_query_operations /
                          temporal_sample_count;
}

bool RiskQueryEngine::BuildHazards(
    const CommonHeader&, const ReferenceSnapshot& reference,
    const UacGaussianSumFilter& filter,
    size_t maximum_hazards,
    std::vector<HazardGeometry,
        Eigen::aligned_allocator<HazardGeometry>>* hazards,
    std::vector<ReasonCode>* reasons) const {
  if (!hazards) {
    if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    return false;
  }
  hazards->clear();
  if (!StableIdRegistryHealthy()) {
    if (reasons) reasons->push_back(StableIdRegistryResourceExhausted()
                                        ? ReasonCode::RESOURCE_LIMIT
                                        : ReasonCode::INVALID_ID_COLLISION);
    return false;
  }
  if (config_.resource.prediction_horizons_s.empty()) {
    if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    return false;
  }
  PouField field(&reference, &config_);
  using PosteriorVector = std::vector<
      GduPosterior, Eigen::aligned_allocator<GduPosterior>>;
  std::vector<PosteriorVector> predicted_by_horizon;
  predicted_by_horizon.reserve(config_.resource.prediction_horizons_s.size());
  for (double horizon_s : config_.resource.prediction_horizons_s) {
    PosteriorVector predicted;
    if (!filter.PredictiveMixture(horizon_s, &predicted, reasons) ||
        predicted.size() != reference.gdus.size()) {
      if (reasons && (reasons->empty() ||
                      reasons->back() == ReasonCode::NONE))
        reasons->push_back(ReasonCode::UNKNOWN_PREDICTION);
      hazards->clear();
      return false;
    }
    predicted_by_horizon.push_back(std::move(predicted));
  }
  size_t prospective_hazards = 0U;
  for (const PosteriorVector& predicted : predicted_by_horizon) {
    for (uint32_t g = 0; g < reference.gdus.size(); ++g) {
      // OTHER/non-monitorable carriers have no deformation coordinates. They
      // remain available through the occupancy layer, but cannot generate a
      // field prediction or consume the structural-hazard primitive budget.
      if (!reference.gdus[g].monitorable ||
          reference.gdus[g].coordinate_rank == 0U)
        continue;
      size_t positive_components = 0U;
      for (const GaussianComponent& component : predicted[g].components)
        if (component.weight > 0.0) ++positive_components;
      const size_t surfels = reference.gdus[g].surfel_indices.size();
      if (positive_components > 0U && surfels > 0U &&
          positive_components > maximum_hazards / surfels) {
        std::cerr << "[deform_v1] resource stage=gdu_hazards gdu=" << g
                  << " value=" << positive_components * surfels
                  << " limit=" << maximum_hazards << std::endl;
        if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
        return false;
      }
      const size_t contribution = positive_components * surfels;
      if (contribution > maximum_hazards - prospective_hazards) {
        std::cerr << "[deform_v1] resource stage=total_hazards value="
                  << prospective_hazards + contribution << " limit="
                  << maximum_hazards << std::endl;
        if (reasons) reasons->push_back(ReasonCode::RESOURCE_LIMIT);
        return false;
      }
      prospective_hazards += contribution;
    }
  }
  hazards->reserve(prospective_hazards);
  for (uint32_t g = 0; g < reference.gdus.size(); ++g) {
    if (!reference.gdus[g].monitorable ||
        reference.gdus[g].coordinate_rank == 0U)
      continue;
    const int coefficient_count = field.CoefficientCount(g);
    if (coefficient_count <= 0) {
      std::cerr << "[deform_v1] hazard invalid stage=coefficient_count gdu="
                << g << " value=" << coefficient_count << std::endl;
      if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
      hazards->clear();
      return false;
    }
    for (uint32_t horizon_index = 0;
         horizon_index < config_.resource.prediction_horizons_s.size(); ++horizon_index) {
      const double horizon = config_.resource.prediction_horizons_s[horizon_index];
      const GduPosterior& predicted =
          predicted_by_horizon[horizon_index][g];
      for (const GaussianComponent& component : predicted.components) {
        if (!(component.weight > 0.0)) continue;
        Eigen::VectorXd latent_mean;
        Eigen::MatrixXd latent_covariance;
        if (!PredictedFieldLatent(component, coefficient_count, 0.0, config_,
                                  &latent_mean, &latent_covariance)) {
          std::cerr << "[deform_v1] hazard invalid stage=latent gdu=" << g
                    << " coeff=" << coefficient_count << " state="
                    << static_cast<int>(component.state) << " mean="
                    << component.mean.size() << " covariance="
                    << component.covariance.rows() << "x"
                    << component.covariance.cols() << std::endl;
          if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
          hazards->clear();
          return false;
        }
        for (uint32_t surfel_index : reference.gdus[g].surfel_indices) {
          if (surfel_index >= reference.surfels.size()) {
            if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
            hazards->clear();
            return false;
          }
          const Surfel& surfel = reference.surfels[surfel_index];
          const Eigen::MatrixXd basis = ComponentBasis(
              field, reference, g, component, surfel.center_R);
          if (basis.rows() != 3 || basis.cols() != coefficient_count ||
              !basis.allFinite()) {
            std::cerr << "[deform_v1] hazard invalid stage=basis gdu=" << g
                      << " surfel=" << surfel_index << " expected=3x"
                      << coefficient_count << " actual=" << basis.rows()
                      << "x" << basis.cols() << std::endl;
            if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
            hazards->clear();
            return false;
          }
          Eigen::MatrixXd geometry = Eigen::MatrixXd::Zero(3, coefficient_count + 3);
          geometry.leftCols(coefficient_count) = basis;
          geometry.rightCols<3>().setIdentity();
          HazardGeometry hazard;
          hazard.cause_id = GduCauseId(reference.gdus[g].id);
          hazard.component_id = component.component_id;
          hazard.source_ids.push_back(reference.gdus[g].id);
          hazard.semantics = HazardSemantics::STRUCTURAL_DEFORMATION;
          hazard.structural_state = component.state;
          hazard.geometry_hypothesis = component.geometry;
          hazard.gdu_index = g; hazard.horizon_s = horizon;
          hazard.center_R = surfel.center_R + geometry * latent_mean;
          hazard.covariance_R = geometry * latent_covariance * geometry.transpose();
          hazard.radius_m = surfel.support_radius_m;
          hazard.existence = ProbabilityInterval{component.weight, component.weight,
                                                 component.weight,
                                                 config_.calibration.validated,
                                                 true};
          hazard.correlation_group = g;
          hazard.model_envelope_m =
              config_.calibration.prediction_model_error_m +
              config_.calibration.ctmc_halving_error_upper_m;
          hazard.other_mass = predicted.other_mass;
          hazards->push_back(hazard);
        }
      }
    }
  }
  std::sort(hazards->begin(), hazards->end(), [](const HazardGeometry& lhs,
                                                 const HazardGeometry& rhs) {
    if (lhs.cause_id != rhs.cause_id) return lhs.cause_id < rhs.cause_id;
    if (lhs.horizon_s != rhs.horizon_s) return lhs.horizon_s < rhs.horizon_s;
    if (lhs.component_id != rhs.component_id) return lhs.component_id < rhs.component_id;
    if (lhs.center_R.x() != rhs.center_R.x()) return lhs.center_R.x() < rhs.center_R.x();
    if (lhs.center_R.y() != rhs.center_R.y()) return lhs.center_R.y() < rhs.center_R.y();
    return lhs.center_R.z() < rhs.center_R.z();
  });
  if (hazards->size() != prospective_hazards) {
    if (reasons) reasons->push_back(ReasonCode::INVALID_NUMERICAL);
    hazards->clear();
    return false;
  }
  if (!StableIdRegistryHealthy()) {
    if (reasons) reasons->push_back(StableIdRegistryResourceExhausted()
                                        ? ReasonCode::RESOURCE_LIMIT
                                        : ReasonCode::INVALID_ID_COLLISION);
    hazards->clear();
    return false;
  }
  return true;
}

RiskQueryEngine::CauseRisk RiskQueryEngine::StructuralCauseRisk(
    const CommonHeader& header, const ReferenceSnapshot& reference,
    const GduDefinition& gdu, uint32_t gdu_index,
    const GduPosterior& posterior, const RobotGeometry& robot,
    const TimedRobotState& state) const {
  CauseRisk result;
  const StableId cause_id = GduCauseId(gdu.id);
  PouField field(&reference, &config_);
  const int coefficient_count = field.CoefficientCount(gdu_index);
  double collision_estimate = 0.0;
  double encounter_estimate = 0.0;
  double failed_mass = 0.0;
  for (const GaussianComponent& component : posterior.components) {
    if (!(component.weight > 0.0)) continue;
    Eigen::VectorXd field_mean;
    Eigen::MatrixXd field_covariance;
    // `posterior` is already the complete U/A/C prediction at this state's
    // absolute ETA.  A second kinematic propagation here would double-count
    // process noise and would still miss switching/reset semantics.
    if (!PredictedFieldLatent(component, coefficient_count, 0.0, config_,
                              &field_mean, &field_covariance)) {
      failed_mass += component.weight; continue;
    }
    const int field_dimension = coefficient_count + 3;
    Eigen::MatrixXd field_square_root;
    Eigen::MatrixXd robot_square_root;
    if (!SymmetricSquareRoot(field_covariance, config_.numerical,
                             &field_square_root) ||
        !SymmetricSquareRoot(state.covariance, config_.numerical,
                             &robot_square_root)) {
      failed_mass += component.weight; continue;
    }
    std::vector<Eigen::MatrixXd> bases;
    bases.reserve(gdu.surfel_indices.size());
    bool valid_bases = true;
    for (uint32_t surfel_index : gdu.surfel_indices) {
      bases.push_back(ComponentBasis(field, reference, gdu_index, component,
                                     reference.surfels[surfel_index].center_R));
      valid_bases = valid_bases && bases.back().cols() == coefficient_count;
    }
    if (!valid_bases) { failed_mass += component.weight; continue; }
    boost::random::sobol hazard_engine(static_cast<size_t>(field_dimension));
    boost::random::sobol robot_engine(6U);
    const auto hazard_shifts = SobolDigitalShifts(
        header.snapshot_id, cause_id, "structural-latent-v1", field_dimension,
        static_cast<uint64_t>(hazard_engine.max()));
    const auto robot_shifts = SobolDigitalShifts(
        header.snapshot_id, cause_id, "robot-pose-v1", 6,
        static_cast<uint64_t>(robot_engine.max()));
    int collision_count = 0;
    for (int sample = 0; sample < config_.resource.prediction_samples; ++sample) {
      Eigen::VectorXd hazard_normal(field_dimension);
      for (int d = 0; d < field_dimension; ++d) {
        const double uniform = ScrambledUniform(
            &hazard_engine, hazard_shifts[static_cast<size_t>(d)]);
        hazard_normal[d] = InverseNormalCdf(std::max(1.0e-15,
            std::min(1.0 - 1.0e-15, uniform)));
      }
      Eigen::Matrix<double, 6, 1> robot_normal;
      for (int d = 0; d < 6; ++d) {
        const double uniform = ScrambledUniform(
            &robot_engine, robot_shifts[static_cast<size_t>(d)]);
        robot_normal[d] = InverseNormalCdf(std::max(
            1.0e-15, std::min(1.0 - 1.0e-15, uniform)));
      }
      const Eigen::VectorXd field_draw =
          field_mean + field_square_root * hazard_normal;
      const Eigen::VectorXd coefficients =
          field_draw.head(coefficient_count);
      const Eigen::Vector3d reference_bias =
          field_draw.segment<3>(coefficient_count);
      const Eigen::Matrix<double, 6, 1> robot_perturbation =
          robot_square_root * robot_normal;
      const Eigen::Isometry3d robot_pose = PerturbPose(state.T_RB, robot_perturbation);
      bool collided = false;
      for (size_t s = 0; s < gdu.surfel_indices.size() && !collided; ++s) {
        const Surfel& surfel = reference.surfels[gdu.surfel_indices[s]];
        const Eigen::Vector3d hazard_center = surfel.center_R + reference_bias +
                                             bases[s] * coefficients;
        for (const RobotSphere& sphere : robot.spheres) {
          const Eigen::Vector3d robot_center = robot_pose * sphere.center_body;
          const double radius = sphere.radius_m +
              surfel.support_radius_m +
              config_.policy.hard_clearance_m +
              config_.calibration.prediction_model_error_m +
              config_.calibration.ctmc_halving_error_upper_m;
          if ((hazard_center - robot_center).squaredNorm() <= radius * radius) {
            collided = true; break;
          }
        }
      }
      if (collided) ++collision_count;
    }
    const double conditional_collision =
        static_cast<double>(collision_count) /
        static_cast<double>(config_.resource.prediction_samples);
    collision_estimate += component.weight * conditional_collision;
    if (component.state != StructuralState::U)
      encounter_estimate += component.weight * conditional_collision;
  }
  result.collision.mean = std::max(0.0, std::min(1.0, collision_estimate));
  result.collision.lower = std::max(
      0.0, collision_estimate - config_.calibration.qmc_error_upper);
  result.collision.upper = std::min(1.0, collision_estimate +
      config_.calibration.qmc_error_upper + posterior.other_mass + failed_mass);
  result.collision.calibrated =
      config_.calibration.validated && failed_mass == 0.0;
  result.collision.mean_identified =
      failed_mass <= config_.numerical.probability_floor &&
      posterior.other_mass <= config_.numerical.probability_floor;
  result.encounter.mean = std::max(0.0, std::min(1.0, encounter_estimate));
  result.encounter.lower = std::max(
      0.0, encounter_estimate - config_.calibration.qmc_error_upper);
  result.encounter.upper = std::min(1.0, encounter_estimate +
      config_.calibration.qmc_error_upper + posterior.other_mass + failed_mass);
  result.encounter.calibrated = config_.calibration.validated && failed_mass == 0.0;
  result.encounter.mean_identified = result.collision.mean_identified;
  // The same component draws define both events. Encounter is the A/C subset
  // of collision, so its three endpoints cannot exceed collision apart from
  // numerical tolerance.
  result.encounter.lower = std::min(result.encounter.lower,
                                    result.collision.lower);
  result.encounter.mean = std::min(result.encounter.mean,
                                   result.collision.mean);
  result.encounter.upper = std::min(result.encounter.upper,
                                    result.collision.upper);
  return result;
}

ProbabilityInterval RiskQueryEngine::OccupancyCauseRisk(
    const CommonHeader& header, const StableId& cause_id,
    const std::vector<std::pair<const OccupancyCell*, ProbabilityInterval>>& cells,
    HazardSemantics semantics,
    const RobotGeometry& robot, const TimedRobotState& state) const {
  ProbabilityInterval result;
  if (cells.empty()) {
    result.upper = 0.0;
    result.calibrated = true;
    result.mean_identified = true;
    return result;
  }
  const int dimension = 6;
  Eigen::MatrixXd robot_square_root;
  if (!SymmetricSquareRoot(state.covariance, config_.numerical, &robot_square_root)) {
    result.upper = 1.0; return result;
  }
  boost::random::sobol engine(static_cast<size_t>(dimension));
  const auto shifts = SobolDigitalShifts(header.snapshot_id, cause_id,
                                         "robot-pose-v1", dimension,
                                         static_cast<uint64_t>(engine.max()));
  double lower_sum = 0.0, mean_sum = 0.0, upper_sum = 0.0;
  bool calibrated = config_.calibration.validated;
  bool mean_identified = cells.size() == 1U;
  const double horizon_s = std::max(
      0.0, (state.stamp - header.measurement_stamp).toSec());
  // A reference-persistence cause is static in the frozen world frame.
  // Every other untracked occupancy cause receives the pre-registered
  // isotropic reachable envelope v_max * dt plus the prediction residual.
  // Structural causes are excluded before this function and are propagated
  // by their U/A/C continuous state instead.
  const double reachable_envelope_m =
      semantics == HazardSemantics::REFERENCE_PERSISTENCE
          ? 0.0
          : config_.calibration.occupancy_max_speed_mps * horizon_s +
                config_.calibration.prediction_model_error_m;
  for (int sample = 0; sample < config_.resource.prediction_samples; ++sample) {
    Eigen::Matrix<double, 6, 1> normal;
    for (int d = 0; d < 6; ++d) {
      const double uniform = ScrambledUniform(&engine, shifts[static_cast<size_t>(d)]);
      normal[d] = InverseNormalCdf(std::max(1.0e-15,
          std::min(1.0 - 1.0e-15, uniform)));
    }
    const Eigen::Isometry3d pose = PerturbPose(state.T_RB, robot_square_root * normal);
    double conditional_lower = 0.0;
    double conditional_mean = 0.0;
    double conditional_upper = 0.0;
    for (size_t c = 0; c < cells.size(); ++c) {
      const OccupancyCell& cell = *cells[c].first;
      const ProbabilityInterval& occupancy = cells[c].second;
      calibrated = calibrated && occupancy.calibrated;
      mean_identified = mean_identified && occupancy.mean_identified;
      bool intersects = false;
      for (const RobotSphere& sphere : robot.spheres) {
        if (SphereIntersectsBox(pose * sphere.center_body,
                                sphere.radius_m + config_.policy.hard_clearance_m,
                                CellCenter(cell.key, config_),
                                0.5 * config_.resource.occupancy_voxel_m +
                                    reachable_envelope_m)) {
          intersects = true; break;
        }
      }
      if (!intersects) continue;
      // All projected cells carrying one HazardCauseId are alternative
      // spatial representations of the same event, not independent Bernoulli
      // variables. For a deterministic conservative footprint, collision is
      // conditioned on that single cause and the largest cell marginal is the
      // correct non-duplicating envelope.
      conditional_lower = std::max(conditional_lower, occupancy.lower);
      conditional_mean = std::max(conditional_mean, occupancy.mean);
      conditional_upper = std::max(conditional_upper, occupancy.upper);
    }
    // Accumulate probabilities rather than drawing an artificial per-cell
    // occupancy state. Robot-pose QMC is the only integral in this function.
    lower_sum += conditional_lower;
    mean_sum += conditional_mean;
    upper_sum += conditional_upper;
  }
  const double denominator = static_cast<double>(config_.resource.prediction_samples);
  result.lower = std::max(0.0, lower_sum / denominator -
                                  config_.calibration.qmc_error_upper);
  result.mean = mean_sum / denominator;
  result.upper = std::min(1.0, upper_sum / denominator +
                                  config_.calibration.qmc_error_upper);
  result.calibrated = calibrated;
  result.mean_identified = mean_identified;
  if (!result.mean_identified) result.mean = result.upper;
  return result;
}

RiskAssessment RiskQueryEngine::AssessState(
    const CommonHeader& header, const ReferenceSnapshot& reference,
    const UacGaussianSumFilter& filter,
    const OccupancyEventGraph& occupancy,
    const RobotGeometry& robot, const TimedRobotState& state,
    const ros::Time& evaluation_stamp) const {
  RiskAssessment result;
  result.snapshot_id = header.snapshot_id;
  if (!StableIdRegistryHealthy()) {
    result.validity = Validity::UNKNOWN;
    result.hard = HardState::UNKNOWN;
    result.reasons.push_back(StableIdRegistryResourceExhausted()
                                 ? ReasonCode::RESOURCE_LIMIT
                                 : ReasonCode::INVALID_ID_COLLISION);
    result.soft.unknown_cost = config_.policy.unknown_weight;
    result.soft.total = result.soft.unknown_cost;
    return result;
  }
  const ros::Time effective_evaluation_stamp = !evaluation_stamp.isZero()
      ? evaluation_stamp
      : (!header.publish_stamp.isZero() ? header.publish_stamp
                                        : header.measurement_stamp);
  const bool snapshot_stale = !header.valid_until.isZero() &&
      effective_evaluation_stamp > header.valid_until;
  if (snapshot_stale) {
    result.validity = CombineValidity(header.validity, Validity::STALE);
    result.hard = HardState::UNKNOWN;
    result.reasons.push_back(ReasonCode::SNAPSHOT_STALE);
    result.reasons.insert(result.reasons.end(), header.reasons.begin(),
                          header.reasons.end());
    result.soft.unknown_cost = config_.policy.unknown_weight;
    result.soft.stale_cost = config_.policy.stale_weight;
    result.soft.total = std::min(
        config_.policy.maximum_soft_cost,
        result.soft.unknown_cost + result.soft.stale_cost);
    NormalizeReasons(&result.reasons);
    return result;
  }
  const bool contract_valid = !header.snapshot_id.empty() &&
      !header.valid_until.isZero() &&
      robot.revision == config_.resource.robot_geometry_revision &&
      !robot.body_frame.empty() && !robot.spheres.empty() &&
      effective_evaluation_stamp >= header.measurement_stamp &&
      state.stamp >= header.measurement_stamp &&
      state.stamp <= header.prediction_valid_until &&
      ValidRigidPose(state.T_RB, 1.0e-8) &&
      ValidateSymmetricPsd(state.covariance,
                           config_.numerical.psd_eigenvalue_tolerance);
  if (!contract_valid) {
    result.validity = state.stamp > header.prediction_valid_until
                          ? Validity::UNKNOWN : Validity::INVALID;
    result.reasons.push_back(state.stamp > header.prediction_valid_until
                                 ? ReasonCode::UNKNOWN_PREDICTION
                                 : ReasonCode::QUERY_CONTRACT_INVALID);
    result.hard = HardState::UNKNOWN;
    result.soft.unknown_cost = config_.policy.unknown_weight;
    const double snapshot_age_s = effective_evaluation_stamp >=
            header.measurement_stamp
        ? (effective_evaluation_stamp - header.measurement_stamp).toSec()
        : config_.resource.map_ttl_s;
    result.soft.stale_cost = config_.policy.stale_weight * std::min(
        1.0, std::max(0.0, snapshot_age_s) /
                 config_.resource.map_ttl_s);
    result.soft.total = std::min(config_.policy.maximum_soft_cost,
                                 result.soft.unknown_cost + result.soft.stale_cost);
    return result;
  }
  BodySamplingGrid body_grid;
  bool body_resource_exceeded = false;
  if (!BuildBodySamplingGrid(robot, config_.resource.occupancy_voxel_m,
                             config_.resource.max_body_query_samples,
                             &body_grid, &body_resource_exceeded)) {
    result.validity = body_resource_exceeded ? Validity::UNKNOWN : Validity::INVALID;
    result.reasons.push_back(body_resource_exceeded
                                 ? ReasonCode::RESOURCE_LIMIT
                                 : ReasonCode::QUERY_CONTRACT_INVALID);
    result.hard = HardState::UNKNOWN;
    result.soft.unknown_cost = config_.policy.unknown_weight;
    result.soft.total = std::min(config_.policy.maximum_soft_cost,
                                 result.soft.unknown_cost);
    return result;
  }
  const double horizon = (state.stamp - header.measurement_stamp).toSec();
  if (horizon < 0.0 ||
      horizon > config_.resource.prediction_horizons_s.back() +
                    config_.numerical.tie_tolerance) {
    result.reasons.push_back(ReasonCode::UNKNOWN_PREDICTION);
    result.hard = HardState::UNKNOWN; result.validity = Validity::UNKNOWN;
    result.soft.unknown_cost = config_.policy.unknown_weight;
    result.soft.total = result.soft.unknown_cost;
    return result;
  }
  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>> posterior;
  if (!filter.PredictiveMixture(horizon, &posterior, &result.reasons) ||
      posterior.size() != reference.gdus.size()) {
    result.validity = Validity::UNKNOWN;
    result.hard = HardState::UNKNOWN;
    if (result.reasons.empty())
      result.reasons.push_back(ReasonCode::UNKNOWN_PREDICTION);
    result.soft.unknown_cost = config_.policy.unknown_weight;
    result.soft.total = result.soft.unknown_cost;
    return result;
  }
  if (!WithinOperationBudget(reference, posterior, occupancy, robot)) {
    result.validity = Validity::UNKNOWN;
    result.hard = HardState::UNKNOWN;
    result.reasons.push_back(ReasonCode::RESOURCE_LIMIT);
    result.soft.unknown_cost = config_.policy.unknown_weight;
    result.soft.total = result.soft.unknown_cost;
    return result;
  }

  std::map<StableId, CauseRisk> by_cause;
  std::set<StableId> structural_causes;
  for (uint32_t g = 0; g < reference.gdus.size() && g < posterior.size(); ++g) {
    if (!reference.gdus[g].monitorable ||
        reference.gdus[g].coordinate_rank == 0U)
      continue;
    const StableId cause = GduCauseId(reference.gdus[g].id);
    structural_causes.insert(cause);
    by_cause[cause] = StructuralCauseRisk(header, reference, reference.gdus[g], g,
                                          posterior[g], robot, state);
  }

  std::map<StableId,
      std::vector<std::pair<const OccupancyCell*, ProbabilityInterval>>> occupancy_by_cause;
  std::map<StableId, HazardSemantics> occupancy_semantics;
  for (const OccupancyCell& cell : occupancy.cells()) {
    for (size_t cause_index = 0; cause_index < cell.cause_ids.size();
         ++cause_index) {
      const StableId& cause = cell.cause_ids[cause_index];
      // A structural projection is a second representation of the same GDU
      // event and must not become an independent occupancy likelihood.
      if (structural_causes.count(cause) > 0U) continue;
      // The cell stores the measurement-time posterior. Presence/absence is a
      // semantic-specific CTMC and must be propagated to this exact ETA before
      // collision integration; expanding only the spatial reachable set would
      // freeze event existence incorrectly.
      const ProbabilityInterval probability =
          occupancy.PredictCauseProbability(cell.key, cause, state.stamp);
      occupancy_by_cause[cause].push_back(
          std::make_pair(&cell, probability));
      const HazardSemantics semantics =
          cause_index < cell.cause_semantics.size()
              ? cell.cause_semantics[cause_index]
              : HazardSemantics::OTHER_ENVELOPE;
      const auto existing = occupancy_semantics.find(cause);
      if (existing == occupancy_semantics.end()) {
        occupancy_semantics[cause] = semantics;
      } else if (existing->second != semantics) {
        // Conflicting semantics for one cause cannot safely be treated as
        // static.  The dynamic OTHER envelope is the deterministic protective
        // fallback and keeps the physical cause de-duplicated.
        existing->second = HazardSemantics::OTHER_ENVELOPE;
      }
    }
  }
  for (const auto& cause : occupancy_by_cause) {
    CauseRisk risk;
    risk.collision = OccupancyCauseRisk(header, cause.first, cause.second,
                                        occupancy_semantics[cause.first],
                                        robot, state);
    risk.encounter = ProbabilityInterval{0.0, 0.0, 0.0,
                                         risk.collision.calibrated, true};
    by_cause[cause.first] = risk;
  }

  std::vector<ProbabilityInterval> collision_causes;
  std::vector<ProbabilityInterval> encounter_causes;
  bool all_calibrated = true;
  for (const auto& cause : by_cause) {
    collision_causes.push_back(cause.second.collision);
    encounter_causes.push_back(cause.second.encounter);
    result.cause_ids.push_back(cause.first);
    result.cause_probabilities.push_back(cause.second.collision);
    result.cause_encounter_probabilities.push_back(cause.second.encounter);
    all_calibrated = all_calibrated && cause.second.collision.calibrated &&
                     cause.second.encounter.calibrated;
  }
  result.collision = ConservativeUnion(collision_causes, 0.0);
  result.encounter = ConservativeUnion(encounter_causes, 0.0);
  result.path_risk = result.collision;
  if (result.encounter.upper > result.collision.upper +
      config_.numerical.normalization_tolerance) {
    result.validity = Validity::INVALID;
    result.hard = HardState::UNKNOWN;
    result.reasons.push_back(ReasonCode::INVALID_NUMERICAL);
    return result;
  }
  result.risk_budget_margin = config_.policy.path_risk_upper_bound - result.collision.upper;
  result.soft.collision_nats = std::min(config_.policy.risk_saturation_nats,
      -std::log(std::max(config_.policy.probability_epsilon,
                         1.0 - (result.collision.mean_identified
                                    ? result.collision.mean
                                    : result.collision.upper))));

  // Deterministic body-volume sampling in the body frame. This estimates the
  // UNKNOWN fraction over the complete configured sphere union, not only the
  // sphere centers.
  uint64_t body_samples = 0U, unknown_samples = 0U;
  bool spatial_query_in_range = true;
  for (int ix = 0; ix < body_grid.count.x(); ++ix)
    for (int iy = 0; iy < body_grid.count.y(); ++iy)
      for (int iz = 0; iz < body_grid.count.z(); ++iz) {
        const Eigen::Vector3d point_body = body_grid.minimum + body_grid.spacing_m *
            Eigen::Vector3d(ix + 0.5, iy + 0.5, iz + 0.5);
        bool inside = false;
        for (const RobotSphere& sphere : robot.spheres)
          inside = inside || (point_body - sphere.center_body).squaredNorm() <=
                               sphere.radius_m * sphere.radius_m;
        if (!inside) continue;
        ++body_samples;
        VoxelKey body_key;
        if (!TryGridKey(state.T_RB * point_body,
                        config_.resource.grid_origin_m,
                        config_.resource.occupancy_voxel_m, &body_key)) {
          spatial_query_in_range = false;
          ++unknown_samples;
          continue;
        }
        const OccupancyCell* cell = occupancy.Find(body_key);
        if (!cell || cell->knowledge == KnowledgeState::UNKNOWN) ++unknown_samples;
      }
  const double unknown_fraction = body_samples > 0U
      ? static_cast<double>(unknown_samples) / static_cast<double>(body_samples) : 1.0;
  result.soft.unknown_cost = config_.policy.unknown_weight * unknown_fraction;
  // ETA is prediction time, not snapshot age.  The caller supplies one
  // evaluation timestamp for the complete query transaction; replay callers
  // that omit it use the recorded publish stamp.  Thus a distant but valid ETA
  // does not itself make the map stale.
  const double snapshot_age_s = std::max(
      0.0, (effective_evaluation_stamp - header.measurement_stamp).toSec());
  result.soft.stale_cost = config_.policy.stale_weight * std::min(
      1.0, snapshot_age_s / config_.resource.map_ttl_s);
  result.soft.total = std::min(config_.policy.maximum_soft_cost,
      result.soft.collision_nats + result.soft.unknown_cost + result.soft.stale_cost);

  const bool authoritative = spatial_query_in_range &&
      config_.calibration.validated && all_calibrated &&
      header.validity == Validity::VALID && header.probabilities_calibrated;
  result.validity = !spatial_query_in_range ? Validity::UNKNOWN :
      (authoritative ? Validity::VALID :
      (header.validity == Validity::VALID ? Validity::INCONCLUSIVE :
                                           header.validity));
  if (!spatial_query_in_range) {
    result.hard = HardState::UNKNOWN;
    result.reasons.push_back(ReasonCode::RESOURCE_LIMIT);
  } else
  if (!authoritative) {
    result.hard = HardState::UNKNOWN;
    if (!config_.calibration.validated)
      result.reasons.push_back(ReasonCode::CALIBRATION_UNSET);
    result.reasons.insert(result.reasons.end(), header.reasons.begin(), header.reasons.end());
  } else if (config_.policy.unknown_is_blocked && unknown_fraction > 0.0) {
    result.hard = HardState::BLOCK;
    result.reasons.push_back(ReasonCode::UNKNOWN_PREDICTION);
  } else if (result.collision.upper >= config_.policy.hard_collision_probability ||
             result.encounter.upper >= config_.policy.hard_encounter_probability) {
    result.hard = HardState::BLOCK;
  } else {
    result.hard = HardState::PASS;
  }
  if (result.hard == HardState::BLOCK) {
    StoreViolationConfiguration(state, config_.resource.world_frame,
                                robot.body_frame, &result);
  }
  if (!StableIdRegistryHealthy()) {
    result.validity = Validity::UNKNOWN;
    result.hard = HardState::UNKNOWN;
    result.cause_ids.clear();
    result.cause_probabilities.clear();
    result.cause_encounter_probabilities.clear();
    result.reasons.push_back(StableIdRegistryResourceExhausted()
                                 ? ReasonCode::RESOURCE_LIMIT
                                 : ReasonCode::INVALID_ID_COLLISION);
    result.soft.unknown_cost = config_.policy.unknown_weight;
    result.soft.total = std::min(config_.policy.maximum_soft_cost,
                                 result.soft.collision_nats +
                                     result.soft.unknown_cost +
                                     result.soft.stale_cost);
  }
  NormalizeReasons(&result.reasons);
  return result;
}

std::vector<TimedRobotState, Eigen::aligned_allocator<TimedRobotState>>
RiskQueryEngine::CanonicalMotionSamples(
    const RobotGeometry& robot,
    const std::vector<TimedRobotState, Eigen::aligned_allocator<TimedRobotState>>& motion,
    double body_sweep_tolerance_m,
    bool* resource_exceeded) const {
  std::vector<TimedRobotState, Eigen::aligned_allocator<TimedRobotState>> output;
  if (resource_exceeded) *resource_exceeded = false;
  if (motion.empty() || !(body_sweep_tolerance_m > 0.0) ||
      !std::isfinite(body_sweep_tolerance_m) || robot.body_frame.empty() ||
      robot.spheres.empty())
    return output;
  for (const RobotSphere& sphere : robot.spheres)
    if (!sphere.center_body.allFinite() || !(sphere.radius_m > 0.0) ||
        !std::isfinite(sphere.radius_m))
      return {};
  for (size_t index = 0U; index < motion.size(); ++index) {
    if (motion[index].stamp.isZero() ||
        !ValidRigidPose(motion[index].T_RB, 1.0e-8) ||
        !ValidateSymmetricPsd(
            motion[index].covariance,
            config_.numerical.psd_eigenvalue_tolerance) ||
        (index > 0U && !(motion[index].stamp > motion[index - 1U].stamp)))
      return {};
  }
  if (motion.size() == 1U) { output.push_back(motion.front()); return output; }
  double maximum_body_radius = 0.0;
  for (const RobotSphere& sphere : robot.spheres)
    maximum_body_radius = std::max(maximum_body_radius,
        sphere.center_body.norm() + sphere.radius_m);
  double maximum_body_speed = 0.0;
  for (size_t i = 1; i < motion.size(); ++i) {
    const double dt = (motion[i].stamp - motion[i - 1U].stamp).toSec();
    if (!(dt > 0.0)) return {};
    const double linear_speed = (motion[i].T_RB.translation() -
                                 motion[i - 1U].T_RB.translation()).norm() / dt;
    const Eigen::Matrix3d relative = motion[i - 1U].T_RB.linear().transpose() *
                                     motion[i].T_RB.linear();
    const double cosine = std::max(-1.0, std::min(1.0,
        0.5 * (relative.trace() - 1.0)));
    const double angular_speed = std::acos(cosine) / dt;
    maximum_body_speed = std::max(maximum_body_speed,
        linear_speed + maximum_body_radius * angular_speed);
  }
  const double duration = (motion.back().stamp - motion.front().stamp).toSec();
  double step_s = config_.policy.motion_sample_interval_s;
  // The contract bounds relative, not robot-only, swept displacement.  The
  // hazard speed upper bound is calibrated for every carrier admitted by V1.
  const double maximum_relative_speed = maximum_body_speed +
      config_.calibration.hazard_sweep_speed_upper_mps;
  if (!std::isfinite(maximum_relative_speed)) {
    if (resource_exceeded) *resource_exceeded = true;
    return {};
  }
  if (maximum_relative_speed > 0.0)
    step_s = std::min(step_s,
                      body_sweep_tolerance_m / maximum_relative_speed);
  if (!(duration > 0.0) || !(step_s > 0.0)) return {};
  // One additional interval protects the closed inequality against ROS time
  // quantization (1 ns) and floating-point division at an exact boundary.
  const long double interval_count = std::floor(
      static_cast<long double>(duration) /
          static_cast<long double>(step_s) +
      static_cast<long double>(config_.numerical.tie_tolerance));
  if (!std::isfinite(interval_count) || interval_count < 0.0L ||
      interval_count + 2.0L > static_cast<long double>(
          config_.resource.max_motion_query_samples)) {
    if (resource_exceeded) *resource_exceeded = true;
    return {};
  }
  const size_t sample_count = std::max<size_t>(
      2U, static_cast<size_t>(interval_count) + 2U);
  size_t segment = 1U;
  for (size_t sample = 0; sample < sample_count; ++sample) {
    const double fraction = static_cast<double>(sample) /
                            static_cast<double>(sample_count - 1U);
    const ros::Time stamp = motion.front().stamp + ros::Duration(fraction * duration);
    while (segment + 1U < motion.size() && stamp > motion[segment].stamp) ++segment;
    const TimedRobotState& first = motion[segment - 1U];
    const TimedRobotState& second = motion[segment];
    const double segment_duration = (second.stamp - first.stamp).toSec();
    const double local = segment_duration > 0.0
        ? (stamp - first.stamp).toSec() / segment_duration : 0.0;
    TimedRobotState state; state.stamp = stamp;
    state.T_RB = Eigen::Isometry3d::Identity();
    state.T_RB.translation() = (1.0 - local) * first.T_RB.translation() +
                               local * second.T_RB.translation();
    Eigen::Quaterniond q0(first.T_RB.linear()), q1(second.T_RB.linear());
    state.T_RB.linear() = q0.slerp(local, q1).normalized().toRotationMatrix();
    if (local <= config_.numerical.tie_tolerance) {
      state.covariance = first.covariance;
    } else if (local >= 1.0 - config_.numerical.tie_tolerance) {
      state.covariance = second.covariance;
    } else {
      // The endpoint cross-covariance is not part of API-020.  For any unit
      // direction u, std((1-a)e0+a e1) is bounded by the weighted sum of the
      // two endpoint maximum standard deviations.  The resulting isotropic
      // matrix is therefore a Loewner upper envelope for every unknown
      // endpoint correlation, unlike a convex covariance interpolation.
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> first_eigen(
          0.5 * (first.covariance + first.covariance.transpose()),
          Eigen::EigenvaluesOnly);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> second_eigen(
          0.5 * (second.covariance + second.covariance.transpose()),
          Eigen::EigenvaluesOnly);
      if (first_eigen.info() != Eigen::Success ||
          second_eigen.info() != Eigen::Success)
        return {};
      const double standard_deviation =
          (1.0 - local) * std::sqrt(std::max(
              0.0, first_eigen.eigenvalues().maxCoeff())) +
          local * std::sqrt(std::max(
              0.0, second_eigen.eigenvalues().maxCoeff()));
      state.covariance = standard_deviation * standard_deviation *
                         Eigen::Matrix<double, 6, 6>::Identity();
    }
    if (!output.empty() && !(state.stamp > output.back().stamp))
      return {};
    output.push_back(state);
  }
  return output;
}

RiskAssessment RiskQueryEngine::AssessMotion(
    const CommonHeader& header, const ReferenceSnapshot& reference,
    const UacGaussianSumFilter& filter,
    const OccupancyEventGraph& occupancy,
    const RobotGeometry& robot,
    const std::vector<TimedRobotState, Eigen::aligned_allocator<TimedRobotState>>& motion,
    double body_sweep_tolerance_m,
    std::vector<TimedRobotState,
        Eigen::aligned_allocator<TimedRobotState>>* canonical_states,
    std::vector<RiskAssessment>* state_assessments,
    std::vector<double>* path_upper_contributions,
    const ros::Time& evaluation_stamp) const {
  RiskAssessment result; result.snapshot_id = header.snapshot_id;
  if (canonical_states) canonical_states->clear();
  if (state_assessments) state_assessments->clear();
  if (path_upper_contributions) path_upper_contributions->clear();
  bool resource_exceeded = false;
  const auto path = CanonicalMotionSamples(
      robot, motion, body_sweep_tolerance_m, &resource_exceeded);
  if (path.empty()) {
    result.validity = resource_exceeded ? Validity::UNKNOWN : Validity::INVALID;
    result.hard = HardState::UNKNOWN;
    result.reasons.push_back(resource_exceeded
                                 ? ReasonCode::RESOURCE_LIMIT
                                 : ReasonCode::QUERY_CONTRACT_INVALID);
    result.soft.unknown_cost = config_.policy.unknown_weight;
    result.soft.total = result.soft.unknown_cost;
    return result;
  }
  // Account against the actual mixture at every ETA.  Future CTMC/coherence
  // branching can have more components than the current snapshot, so a
  // current-mixture estimate is not a safe resource bound.
  size_t estimated_operations = 0U;
  bool operation_budget_exceeded = false;
  for (const TimedRobotState& sample_state : path) {
    const double horizon_s =
        (sample_state.stamp - header.measurement_stamp).toSec();
    std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>> predicted;
    std::vector<ReasonCode> prediction_reasons;
    if (!filter.PredictiveMixture(horizon_s, &predicted,
                                  &prediction_reasons))
      continue;  // AssessState will return the precise validity/reason.
    const size_t state_operations = EstimateStateQueryOperations(
        reference, predicted, occupancy, robot);
    if (state_operations > config_.resource.max_risk_query_operations -
                               std::min(estimated_operations,
                                        config_.resource.max_risk_query_operations)) {
      operation_budget_exceeded = true;
      break;
    }
    estimated_operations += state_operations;
  }
  if (operation_budget_exceeded) {
    result.validity = Validity::UNKNOWN;
    result.hard = HardState::UNKNOWN;
    result.reasons.push_back(ReasonCode::RESOURCE_LIMIT);
    result.soft.unknown_cost = config_.policy.unknown_weight;
    result.soft.total = result.soft.unknown_cost;
    return result;
  }
  if (canonical_states) *canonical_states = path;
  std::map<StableId, ProbabilityInterval> collision_by_cause;
  std::map<StableId, ProbabilityInterval> encounter_by_cause;
  Validity worst_state_validity = Validity::VALID;
  double previous_running_path_upper = 0.0;
  double maximum_unknown_cost = 0.0, maximum_stale_cost = 0.0;
  RobotGeometry swept_robot = robot;
  // Every time between two canonical samples is at most half a sample gap
  // from one endpoint. Inflating the full body by half the permitted relative
  // swept displacement closes that continuous-time gap without pretending
  // that the input contains a denser trajectory covariance process.
  if (path.size() > 1U)
    for (RobotSphere& sphere : swept_robot.spheres)
      sphere.radius_m += 0.5 * body_sweep_tolerance_m;
  for (size_t i = 0; i < path.size(); ++i) {
    if (i > 0 && !(path[i].stamp > path[i - 1].stamp)) {
      result.validity = Validity::INVALID; result.hard = HardState::UNKNOWN;
      result.reasons.push_back(ReasonCode::QUERY_CONTRACT_INVALID); return result;
    }
    const RiskAssessment state = AssessState(header, reference, filter, occupancy,
                                             swept_robot, path[i],
                                             evaluation_stamp);
    if (state_assessments) state_assessments->push_back(state);
    worst_state_validity = CombineValidity(worst_state_validity,
                                           state.validity);
    result.reasons.insert(result.reasons.end(), state.reasons.begin(), state.reasons.end());
    maximum_unknown_cost = std::max(maximum_unknown_cost, state.soft.unknown_cost);
    maximum_stale_cost = std::max(maximum_stale_cost, state.soft.stale_cost);
    for (size_t cause_index = 0; cause_index < state.cause_ids.size(); ++cause_index) {
      const ProbabilityInterval probability = cause_index < state.cause_probabilities.size()
          ? state.cause_probabilities[cause_index] : state.collision;
      const StableId cause_id = state.cause_ids[cause_index];
      const bool collision_initialized =
          collision_by_cause.find(cause_id) != collision_by_cause.end();
      collision_by_cause[cause_id] = JointCausePathEvent(
          collision_by_cause[cause_id], probability,
          collision_initialized);
      const ProbabilityInterval encounter =
          cause_index < state.cause_encounter_probabilities.size()
              ? state.cause_encounter_probabilities[cause_index]
              : ProbabilityInterval{0.0, 0.0, probability.upper,
                                    probability.calibrated, false};
      const bool encounter_initialized =
          encounter_by_cause.find(cause_id) != encounter_by_cause.end();
      encounter_by_cause[cause_id] = JointCausePathEvent(
          encounter_by_cause[cause_id], encounter,
          encounter_initialized);
    }
    double running_path_upper = 0.0;
    for (const auto& cause : collision_by_cause)
      running_path_upper += cause.second.upper;
    running_path_upper = std::min(1.0, running_path_upper);
    if (path_upper_contributions)
      path_upper_contributions->push_back(
          std::max(0.0, running_path_upper - previous_running_path_upper));
    previous_running_path_upper = running_path_upper;
    if ((state.hard == HardState::BLOCK ||
         running_path_upper >= config_.policy.path_risk_upper_bound) &&
        result.first_violation_time.isZero()) {
      StoreViolationConfiguration(path[i], config_.resource.world_frame,
                                  robot.body_frame, &result);
    }
  }
  std::vector<ProbabilityInterval> cause_probability;
  std::vector<ProbabilityInterval> cause_encounter_probability;
  for (const auto& cause : collision_by_cause) {
    const ProbabilityInterval probability = cause.second;
    const ProbabilityInterval encounter = encounter_by_cause[cause.first];
    cause_probability.push_back(probability);
    cause_encounter_probability.push_back(encounter);
    result.cause_ids.push_back(cause.first);
    result.cause_probabilities.push_back(probability);
    result.cause_encounter_probabilities.push_back(encounter);
  }
  result.path_risk = ConservativeUnion(cause_probability, 0.0);
  result.collision = result.path_risk;
  result.encounter = ConservativeUnion(cause_encounter_probability, 0.0);
  result.risk_budget_margin = config_.policy.path_risk_upper_bound - result.path_risk.upper;
  result.soft.collision_nats = std::min(config_.policy.risk_saturation_nats,
      -std::log(std::max(config_.policy.probability_epsilon,
                         1.0 - (result.path_risk.mean_identified
                                    ? result.path_risk.mean
                                    : result.path_risk.upper))));
  result.soft.unknown_cost = maximum_unknown_cost;
  result.soft.stale_cost = maximum_stale_cost;
  result.soft.total = std::min(config_.policy.maximum_soft_cost,
      result.soft.collision_nats + result.soft.unknown_cost + result.soft.stale_cost);
  if (worst_state_validity != Validity::VALID ||
      !config_.calibration.validated ||
      header.validity != Validity::VALID) {
    result.validity = CombineValidity(header.validity,
                                      worst_state_validity);
    if (result.validity == Validity::VALID)
      result.validity = Validity::INCONCLUSIVE;
    result.hard = HardState::UNKNOWN;
    if (!config_.calibration.validated)
      result.reasons.push_back(ReasonCode::CALIBRATION_UNSET);
    result.reasons.insert(result.reasons.end(), header.reasons.begin(), header.reasons.end());
  } else if (result.path_risk.upper >= config_.policy.path_risk_upper_bound ||
             !result.first_violation_time.isZero()) {
    result.validity = Validity::VALID; result.hard = HardState::BLOCK;
  } else {
    result.validity = Validity::VALID; result.hard = HardState::PASS;
  }
  NormalizeReasons(&result.reasons);
  return result;
}

RiskAssessment RiskQueryEngine::AssessPath(
    const CommonHeader& header, const ReferenceSnapshot& reference,
    const UacGaussianSumFilter& filter,
    const OccupancyEventGraph& occupancy,
    const RobotGeometry& robot,
    const std::vector<TimedRobotState, Eigen::aligned_allocator<TimedRobotState>>& path,
    std::vector<TimedRobotState,
        Eigen::aligned_allocator<TimedRobotState>>* canonical_states,
    std::vector<RiskAssessment>* state_assessments,
    std::vector<double>* path_upper_contributions,
    const ros::Time& evaluation_stamp) const {
  return AssessMotion(header, reference, filter, occupancy, robot, path,
                      config_.policy.motion_sample_spacing_m,
                      canonical_states, state_assessments,
                      path_upper_contributions, evaluation_stamp);
}

}  // namespace v1
}  // namespace deform_monitor_v3
