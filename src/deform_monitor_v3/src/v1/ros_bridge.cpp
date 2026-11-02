#include "deform_monitor_v3/v1/ros_bridge.hpp"

#include "deform_monitor_v3/v1/deterministic.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <system_error>
#include <thread>

namespace deform_monitor_v3 {
namespace v1 {
namespace {

Validity ConservativeValidity(Validity local, Validity common) {
  return CombineValidity(local, common);
}

Eigen::Vector3d OccupancyCellCenter(const VoxelKey& key,
                                    const V1Config& config) {
  return config.resource.grid_origin_m + config.resource.occupancy_voxel_m *
      Eigen::Vector3d(static_cast<double>(key.x) + 0.5,
                      static_cast<double>(key.y) + 0.5,
                      static_cast<double>(key.z) + 0.5);
}

template <typename MatrixT, typename ContainerT>
void FlattenRowMajor(const MatrixT& matrix, ContainerT* output) {
  output->resize(static_cast<size_t>(matrix.rows() * matrix.cols()));
  for (int row = 0; row < matrix.rows(); ++row)
    for (int column = 0; column < matrix.cols(); ++column)
      (*output)[static_cast<size_t>(row * matrix.cols() + column)] = matrix(row, column);
}

geometry_msgs::Point PointMessage(const Eigen::Vector3d& value) {
  geometry_msgs::Point point; point.x = value.x(); point.y = value.y(); point.z = value.z();
  return point;
}

geometry_msgs::Vector3 VectorMessage(const Eigen::Vector3d& value) {
  geometry_msgs::Vector3 vector; vector.x = value.x(); vector.y = value.y(); vector.z = value.z();
  return vector;
}

deform_monitor_v3::V1StableId128 IdMessage(const StableId& value) {
  deform_monitor_v3::V1StableId128 output;
  for (int i = 0; i < 8; ++i) {
    output.high = (output.high << 8U) | value.bytes[static_cast<size_t>(i)];
    output.low = (output.low << 8U) | value.bytes[static_cast<size_t>(i + 8)];
  }
  return output;
}

deform_monitor_v3::V1DenseMatrix64 DenseMessage(const Eigen::MatrixXd& matrix) {
  deform_monitor_v3::V1DenseMatrix64 output;
  output.rows = static_cast<uint32_t>(matrix.rows());
  output.cols = static_cast<uint32_t>(matrix.cols());
  FlattenRowMajor(matrix, &output.data_row_major);
  return output;
}

deform_monitor_v3::V1SparseCsrMatrix64 SparseMessage(const Eigen::MatrixXd& matrix) {
  deform_monitor_v3::V1SparseCsrMatrix64 output;
  output.rows = static_cast<uint32_t>(matrix.rows());
  output.cols = static_cast<uint32_t>(matrix.cols());
  output.row_ptr.reserve(static_cast<size_t>(matrix.rows() + 1));
  output.row_ptr.push_back(0U);
  for (int row = 0; row < matrix.rows(); ++row) {
    for (int column = 0; column < matrix.cols(); ++column) {
      const double value = matrix(row, column);
      if (value == 0.0) continue;
      output.col_idx.push_back(static_cast<uint32_t>(column));
      output.data.push_back(value);
    }
    output.row_ptr.push_back(static_cast<uint64_t>(output.data.size()));
  }
  return output;
}

deform_monitor_v3::V1HazardGeometry HazardMessage(
    const HazardGeometry& source) {
  deform_monitor_v3::V1HazardGeometry target;
  target.cause_id = IdMessage(source.cause_id);
  target.component_id = IdMessage(source.component_id);
  target.gdu_index = source.gdu_index;
  for (const StableId& id : source.source_ids)
    target.source_ids.push_back(IdMessage(id));
  target.semantics = static_cast<uint8_t>(source.semantics);
  target.structural_state = static_cast<uint8_t>(source.structural_state);
  target.geometry_hypothesis =
      static_cast<uint8_t>(source.geometry_hypothesis);
  target.horizon_s = source.horizon_s;
  target.center = PointMessage(source.center_R);
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      target.covariance_row_major[static_cast<size_t>(3 * row + column)] =
          source.covariance_R(row, column);
  target.radius_m = source.radius_m;
  target.probability_lower = source.existence.lower;
  target.probability_mean = source.existence.mean;
  target.probability_upper = source.existence.upper;
  target.probability_mean_identified = source.existence.mean_identified;
  target.correlation_group = source.correlation_group;
  target.model_envelope_m = source.model_envelope_m;
  target.other_mass = source.other_mass;
  return target;
}

deform_monitor_v3::V1InterfacePosterior InterfaceMessage(
    const V1Snapshot& snapshot, size_t interface_index) {
  const InterfacePosterior& source = snapshot.interfaces[interface_index];
  const InterfaceDefinition& definition =
      snapshot.reference.interfaces[interface_index];
  deform_monitor_v3::V1InterfacePosterior target;
  target.interface_id = IdMessage(source.interface_id);
  target.left_gdu_id =
      IdMessage(snapshot.reference.gdus[definition.left_gdu].id);
  target.right_gdu_id =
      IdMessage(snapshot.reference.gdus[definition.right_gdu].id);
  target.reference_center = PointMessage(definition.center_R);
  target.reference_normal = VectorMessage(definition.normal_R);
  target.scale_m = definition.scale_m;
  for (size_t i = 0; i < 4U; ++i)
    target.joint_probability[i] = source.joint_probability[i];
  target.jump_mean = VectorMessage(source.jump_mean_R);
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      target.jump_covariance_row_major[static_cast<size_t>(3 * row + column)] =
          source.jump_covariance_R(row, column);
  target.other_mass = source.other_mass;
  target.connectivity_coherence_confounding =
      source.connectivity_coherence_confounding;
  target.validity = static_cast<uint8_t>(ConservativeValidity(
      source.validity, snapshot.common.validity));
  target.left_query_validity = static_cast<uint8_t>(ConservativeValidity(
      source.left_field.validity, snapshot.common.validity));
  target.right_query_validity = static_cast<uint8_t>(ConservativeValidity(
      source.right_field.validity, snapshot.common.validity));
  target.left_query_position = PointMessage(source.left_query_position_R);
  target.right_query_position = PointMessage(source.right_query_position_R);
  target.left_displacement_mean =
      VectorMessage(source.left_field.mean_displacement_R);
  target.right_displacement_mean =
      VectorMessage(source.right_field.mean_displacement_R);
  target.left_velocity_mean = VectorMessage(source.left_field.mean_velocity_R);
  target.right_velocity_mean = VectorMessage(source.right_field.mean_velocity_R);
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const size_t index = static_cast<size_t>(3 * row + column);
      target.left_displacement_covariance_row_major[index] =
          source.left_field.displacement_covariance_R(row, column);
      target.right_displacement_covariance_row_major[index] =
          source.right_field.displacement_covariance_R(row, column);
    }
  }
  target.left_other_mass = source.left_field.other_mass;
  target.right_other_mass = source.right_field.other_mass;
  target.localization_lower_m = 0.0;
  target.localization_upper_m = definition.scale_m;
  return target;
}

deform_monitor_v3::V1DeformationComponentQuery ComponentQueryMessage(
    const FieldComponentQuery& component) {
  deform_monitor_v3::V1DeformationComponentQuery output;
  output.component_id = IdMessage(component.component_id);
  output.weight = component.weight;
  output.structural_state = static_cast<uint8_t>(component.state);
  output.geometry_hypothesis = static_cast<uint8_t>(component.geometry);
  output.displacement_mean = VectorMessage(component.mean_displacement_R);
  output.velocity_mean = VectorMessage(component.mean_velocity_R);
  output.displacement_covariance =
      DenseMessage(component.displacement_covariance_R);
  output.velocity_covariance = DenseMessage(component.velocity_covariance_R);
  output.observable_projector =
      DenseMessage(component.observable_projector_R);
  output.minimum_detectable_effect_defined =
      component.minimum_detectable_effect_defined;
  output.minimum_detectable_effect_m =
      component.minimum_detectable_effect_defined
          ? component.minimum_detectable_effect_m : 0.0;
  return output;
}

deform_monitor_v3::V1TimedRobotPose TimedPoseMessage(
    const ros::Time& stamp, const std::array<double, 16>& pose_row_major,
    const std::array<double, 36>& covariance_row_major,
    const std::string& pose_frame,
    const std::string& covariance_origin_frame) {
  deform_monitor_v3::V1TimedRobotPose output;
  output.eta = stamp;
  output.pose_frame = pose_frame;
  output.covariance_origin_frame = covariance_origin_frame;
  Eigen::Matrix3d rotation;
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      rotation(row, column) = pose_row_major[
          static_cast<size_t>(4 * row + column)];
  output.pose.position.x = pose_row_major[3U];
  output.pose.position.y = pose_row_major[7U];
  output.pose.position.z = pose_row_major[11U];
  Eigen::Quaterniond quaternion(rotation);
  quaternion.normalize();
  if (quaternion.w() < 0.0 ||
      (quaternion.w() == 0.0 &&
       (quaternion.x() < 0.0 ||
        (quaternion.x() == 0.0 &&
         (quaternion.y() < 0.0 ||
          (quaternion.y() == 0.0 && quaternion.z() < 0.0))))))
    quaternion.coeffs() *= -1.0;
  output.pose.orientation.x = quaternion.x();
  output.pose.orientation.y = quaternion.y();
  output.pose.orientation.z = quaternion.z();
  output.pose.orientation.w = quaternion.w();
  output.pose_covariance.rows = 6U;
  output.pose_covariance.cols = 6U;
  output.pose_covariance.data_row_major.assign(
      covariance_row_major.begin(), covariance_row_major.end());
  return output;
}

}  // namespace

deform_monitor_v3::V1StableId128 BuildStableIdMessage(const StableId& id) {
  return IdMessage(id);
}

bool ParseStableIdMessage(const deform_monitor_v3::V1StableId128& message,
                          StableId* id) {
  if (!id) return false;
  for (int i = 0; i < 8; ++i) {
    id->bytes[static_cast<size_t>(i)] = static_cast<uint8_t>(
        (message.high >> static_cast<unsigned int>(8 * (7 - i))) & 0xffU);
    id->bytes[static_cast<size_t>(i + 8)] = static_cast<uint8_t>(
        (message.low >> static_cast<unsigned int>(8 * (7 - i))) & 0xffU);
  }
  return true;
}

deform_monitor_v3::V1Common BuildCommonMessage(const CommonHeader& common) {
  deform_monitor_v3::V1Common message;
  message.header.stamp = common.measurement_stamp; message.header.frame_id = common.world_frame;
  message.publish_stamp = common.publish_stamp; message.schema_id = common.schema_id;
  message.schema_version = common.schema_version; message.session_id = common.session_id;
  message.snapshot_id = IdMessage(common.snapshot_id); message.reference_epoch = common.reference_epoch;
  message.accepted_frame_index = common.accepted_frame_index;
  message.software_revision = common.software_revision; message.design_version = common.design_version;
  message.implementation_freeze_version = common.implementation_freeze_version;
  message.config_digest_sha256 = common.config_digest;
  message.calibration_version = common.calibration_version;
  message.robot_geometry_revision = common.robot_geometry_revision;
  message.input_capability_profile = common.input_capability_profile;
  message.sensor_scope = common.sensor_scope;
  message.hazard_coverage_mask = common.hazard_coverage_mask;
  message.uncovered_hazards = common.uncovered_hazards;
  message.probabilities_calibrated = common.probabilities_calibrated;
  message.validity = static_cast<uint8_t>(common.validity);
  for (ReasonCode reason : common.reasons)
    message.validity_reason_codes.push_back(static_cast<uint16_t>(reason));
  message.valid_from = common.valid_from; message.valid_until = common.valid_until;
  message.prediction_valid_until = common.prediction_valid_until;
  message.dropped_input_count = common.dropped_input_count;
  message.dropped_output_count = common.dropped_output_count;
  message.approximation_flags = common.approximation_flags;
  return message;
}

deform_monitor_v3::V1DetectionCommon BuildDetectionCommonMessage(
    const CommonHeader& common) {
  deform_monitor_v3::V1DetectionCommon message;
  message.header.stamp = common.measurement_stamp;
  message.header.frame_id = common.world_frame;
  message.publish_stamp = common.publish_stamp;
  message.schema_id = "deform-detection-v1";
  message.schema_version = common.schema_version;
  message.session_id = common.session_id;
  message.snapshot_id = IdMessage(common.snapshot_id);
  message.reference_epoch = common.reference_epoch;
  message.accepted_frame_index = common.accepted_frame_index;
  message.software_revision = common.software_revision;
  message.config_digest_sha256 = common.config_digest;
  message.calibration_version = common.calibration_version;
  message.input_capability_profile = common.input_capability_profile;
  message.sensor_scope = common.sensor_scope;
  message.probabilities_calibrated = common.probabilities_calibrated;
  message.validity = static_cast<uint8_t>(common.validity);
  for (ReasonCode reason : common.reasons)
    message.validity_reason_codes.push_back(static_cast<uint16_t>(reason));
  message.valid_from = common.valid_from;
  message.valid_until = common.valid_until;
  message.dropped_input_count = common.dropped_input_count;
  message.dropped_output_count = common.dropped_output_count;
  message.approximation_flags = common.approximation_flags;
  return message;
}

deform_monitor_v3::V1DeformationField BuildDeformationMessage(const V1Snapshot& snapshot) {
  deform_monitor_v3::V1DeformationField message;
  message.common = BuildCommonMessage(snapshot.common);
  for (size_t gdu_index = 0; gdu_index < snapshot.deformation.size(); ++gdu_index) {
    const GduPosterior& source = snapshot.deformation[gdu_index];
    deform_monitor_v3::V1GduPosterior target;
    target.gdu_id = IdMessage(source.gdu_id);
    target.nuisance_correlation_id =
        IdMessage(source.nuisance_correlation_id);
    target.support_revision = HexId(snapshot.reference.id);
    target.basis_revision = HexId(snapshot.reference.gdus[gdu_index].id);
    target.p_u = source.state_probability[0];
    target.p_a = source.state_probability[1]; target.p_c = source.state_probability[2];
    target.p_other = source.other_mass;
    target.p_task_scale_deformation = source.task_scale_deformation_probability;
    target.p_task_scale_deformation_lower = source.task_scale_deformation_lower;
    target.p_task_scale_deformation_upper = source.task_scale_deformation_upper;
    target.p_task_scale_deformation_mean_identified =
        source.task_scale_deformation_mean_identified;
    target.p_task_displacement = source.task_displacement_probability;
    target.p_current_velocity_exceedance =
        source.current_velocity_exceedance_probability;
    target.continuous_valid_observation_duration_s =
        source.continuous_valid_observation_duration_s;
    target.velocity_duration_requirement_met =
        source.velocity_duration_requirement_met;
    target.p_task_interface_jump = source.task_interface_jump_probability;
    target.minimum_detectable_effect_defined =
        source.minimum_detectable_effect_defined;
    target.minimum_detectable_effect_m = source.minimum_detectable_effect_defined
        ? source.minimum_detectable_effect_m : 0.0;
    target.velocity_minimum_detectable_effect_defined =
        source.velocity_minimum_detectable_effect_defined;
    target.velocity_minimum_detectable_effect_mps =
        source.velocity_minimum_detectable_effect_defined
            ? source.velocity_minimum_detectable_effect_mps : 0.0;
    target.interface_jump_minimum_detectable_effect_defined =
        source.interface_jump_minimum_detectable_effect_defined;
    target.interface_jump_minimum_detectable_effect_m =
        source.interface_jump_minimum_detectable_effect_defined
            ? source.interface_jump_minimum_detectable_effect_m : 0.0;
    target.task_power_complete = source.task_power_complete;
    target.task_power_worst_normalized_mde = source.task_power_complete
        ? source.task_power_worst_normalized_mde : 0.0;
    target.any_task_direction_detectable =
        source.any_task_direction_detectable;
    target.all_task_directions_detectable =
        source.all_task_directions_detectable;
    target.detection_allowed = source.detection_allowed;
    target.clearing_allowed = source.clearing_allowed;
    target.observable_rank = source.observable_rank;
    target.observable_projector = DenseMessage(source.observable_projector);
    target.nullspace = DenseMessage(source.nullspace);
    target.reference_information_ratio = source.reference_information_ratio;
    target.reference_confounding = source.reference_confounding;
    target.effective_spatial_support = source.effective_spatial_support;
    target.effective_temporal_support = source.effective_temporal_support;
    target.detection_admission = static_cast<uint8_t>(source.admission);
    for (const GaussianComponent& component : source.components) {
      deform_monitor_v3::V1GaussianComponent output;
      output.component_id = IdMessage(component.component_id);
      output.parent_id = IdMessage(component.parent_id);
      output.association_lineage = IdMessage(component.association_lineage);
      output.weight = component.weight; output.structural_state = static_cast<uint8_t>(component.state);
      output.geometry_hypothesis = static_cast<uint8_t>(component.geometry);
      for (InterfaceJointState mode : component.interface_modes)
        output.interface_joint_modes.push_back(static_cast<uint8_t>(mode));
      output.state_dimension = static_cast<uint32_t>(component.mean.size());
      const uint32_t coefficient_count = static_cast<uint32_t>(
          snapshot.reference.gdus[gdu_index].coordinate_rank) *
          static_cast<uint32_t>(
          snapshot.reference.gdus[gdu_index].control_surfel_indices.size() +
          snapshot.reference.gdus[gdu_index].interface_indices.size());
      output.physical_dimension = component.state == StructuralState::U ? 0U :
          (component.state == StructuralState::A ? 2U * coefficient_count : coefficient_count);
      output.velocity_offset = component.state == StructuralState::A
          ? static_cast<int32_t>(coefficient_count) : -1;
      output.reference_bias_offset = output.physical_dimension;
      output.extrinsic_nuisance_offset = output.physical_dimension + 3U;
      output.pose_nuisance_offset = output.physical_dimension + 9U;
      output.mean.assign(component.mean.data(), component.mean.data() + component.mean.size());
      output.sqrt_information = SparseMessage(component.sqrt_information);
      output.observable_rank = component.observable_rank;
      output.observable_projector = DenseMessage(component.observable_projector);
      output.observable_nullspace = DenseMessage(component.observable_nullspace);
      output.minimum_detectable_effect_defined =
          component.minimum_detectable_effect_defined;
      output.minimum_detectable_effect_m = component.minimum_detectable_effect_defined
          ? component.minimum_detectable_effect_m : 0.0;
      target.components.push_back(output);
    }
    message.gdus.push_back(target);
  }
  return message;
}

deform_monitor_v3::V1InterfaceMotionFrame BuildInterfaceMessage(const V1Snapshot& snapshot) {
  deform_monitor_v3::V1InterfaceMotionFrame message;
  message.common = BuildCommonMessage(snapshot.common);
  for (size_t interface_index = 0; interface_index < snapshot.interfaces.size(); ++interface_index) {
    message.interfaces.push_back(InterfaceMessage(snapshot, interface_index));
  }
  return message;
}

deform_monitor_v3::V1StructuralEventFrame BuildStructuralEventMessage(const V1Snapshot& snapshot) {
  deform_monitor_v3::V1StructuralEventFrame message;
  message.common = BuildCommonMessage(snapshot.common);
  for (const HazardGeometry& source : snapshot.hazards) {
    message.events.push_back(HazardMessage(source));
  }
  for (const ObservationExplanation& source : snapshot.explanations) {
    deform_monitor_v3::V1ObservationExplanation target;
    target.cause_id = IdMessage(source.cause_id);
    target.source_evidence_id = IdMessage(source.source_evidence_id);
    target.semantics = static_cast<uint8_t>(source.semantics);
    target.probability_lower = source.probability.lower;
    target.probability_mean = source.probability.mean;
    target.probability_upper = source.probability.upper;
    target.probability_mean_identified =
        source.probability.mean_identified;
    target.committed_to_posterior = source.committed_to_posterior;
    target.moved_available = false;
    message.explanations.push_back(target);
  }
  return message;
}

deform_monitor_v3::V1OccupancyKnowledgeFrame BuildOccupancyMessage(
    const V1Snapshot& snapshot, const V1Config& config) {
  deform_monitor_v3::V1OccupancyKnowledgeFrame message;
  message.common = BuildCommonMessage(snapshot.common);
  message.grid_origin = PointMessage(config.resource.grid_origin_m);
  message.voxel_size_m = config.resource.occupancy_voxel_m;

  struct LineageAccumulator {
    HazardSemantics semantics = HazardSemantics::OTHER_ENVELOPE;
    uint32_t correlation_group = 0U;
    std::set<StableId> source_ids;
    std::map<StableId, ProbabilityInterval> component_probability;
    ProbabilityInterval direct_probability;
    bool has_direct_probability = false;
    double other_mass = 0.0;
  };
  std::map<StableId, LineageAccumulator> lineage;
  for (const HazardGeometry& hazard : snapshot.hazards) {
    if (std::abs(hazard.horizon_s) > config.numerical.tie_tolerance) continue;
    LineageAccumulator& entry = lineage[hazard.cause_id];
    entry.semantics = hazard.semantics;
    entry.correlation_group = hazard.correlation_group;
    entry.source_ids.insert(hazard.source_ids.begin(), hazard.source_ids.end());
    auto inserted_component = entry.component_probability.emplace(
        hazard.component_id,
        ProbabilityInterval{0.0, 0.0, 0.0, true, true});
    ProbabilityInterval& component = inserted_component.first->second;
    component.lower = std::max(component.lower, hazard.existence.lower);
    component.mean = std::max(component.mean, hazard.existence.mean);
    component.upper = std::max(component.upper, hazard.existence.upper);
    component.calibrated = component.calibrated &&
                           hazard.existence.calibrated;
    component.mean_identified = component.mean_identified &&
                                hazard.existence.mean_identified;
    entry.other_mass = std::max(entry.other_mass, hazard.other_mass);
  }
  // Add singleton reference-persistence and independent-current occupancy
  // events that are not structural hazard components.
  for (const OccupancyCell& cell : snapshot.occupancy) {
    for (size_t cause_index = 0; cause_index < cell.cause_ids.size();
         ++cause_index) {
      const StableId& cause = cell.cause_ids[cause_index];
      if (lineage.count(cause) > 0U) continue;
      LineageAccumulator& entry = lineage[cause];
      entry.correlation_group = 0U;
      for (size_t byte = 0; byte < 4U; ++byte)
        entry.correlation_group =
            (entry.correlation_group << 8U) | cause.bytes[byte];
      entry.has_direct_probability = true;
      const ProbabilityInterval cause_probability =
          cause_index < cell.cause_probabilities.size()
              ? cell.cause_probabilities[cause_index]
              : cell.total_occupancy;
      entry.semantics = cause_index < cell.cause_semantics.size()
          ? cell.cause_semantics[cause_index]
          : HazardSemantics::OTHER_ENVELOPE;
      entry.direct_probability = cause_probability;
      if (cause_index < cell.cause_source_ids.size())
        entry.source_ids.insert(
            cell.cause_source_ids[cause_index].begin(),
            cell.cause_source_ids[cause_index].end());
    }
  }

  std::map<StableId, uint32_t> lineage_offset;
  for (const auto& item : lineage) {
    deform_monitor_v3::V1OccupancyEventLineage output;
    output.cause_id = IdMessage(item.first);
    output.semantics = static_cast<uint8_t>(item.second.semantics);
    output.correlation_group = item.second.correlation_group;
    for (const StableId& source : item.second.source_ids)
      output.source_ids.push_back(IdMessage(source));
    ProbabilityInterval probability{0.0, 0.0, 0.0, true, true};
    if (item.second.has_direct_probability) {
      probability = item.second.direct_probability;
    } else {
      probability.calibrated = true;
      probability.mean_identified = true;
      for (const auto& component : item.second.component_probability) {
        probability.lower += component.second.lower;
        probability.mean += component.second.mean;
        probability.upper += component.second.upper;
        probability.calibrated = probability.calibrated &&
                                 component.second.calibrated;
        probability.mean_identified = probability.mean_identified &&
                                      component.second.mean_identified;
      }
      probability.lower = std::min(1.0, probability.lower);
      probability.mean = std::min(1.0, probability.mean);
      probability.upper = std::min(
          1.0, probability.upper + item.second.other_mass);
      if (item.second.other_mass > config.numerical.probability_floor)
        probability.mean_identified = false;
      if (!probability.mean_identified) probability.mean = probability.upper;
    }
    output.probability_lower = probability.lower;
    output.probability_mean = probability.mean;
    output.probability_upper = probability.upper;
    output.probability_mean_identified = probability.mean_identified;
    output.other_mass = item.second.other_mass;
    lineage_offset[item.first] =
        static_cast<uint32_t>(message.event_lineage.size());
    message.event_lineage.push_back(output);
  }
  for (const OccupancyCell& source : snapshot.occupancy) {
    deform_monitor_v3::V1OccupancyCell target;
    target.key_x = source.key.x; target.key_y = source.key.y; target.key_z = source.key.z;
    target.reference_lower = source.reference_occupancy.lower;
    target.reference_mean = source.reference_occupancy.mean;
    target.reference_upper = source.reference_occupancy.upper;
    target.reference_mean_identified =
        source.reference_occupancy.mean_identified;
    target.total_lower = source.total_occupancy.lower; target.total_mean = source.total_occupancy.mean;
    target.total_upper = source.total_occupancy.upper;
    target.total_mean_identified = source.total_occupancy.mean_identified;
    target.new_lower = source.new_occupancy.lower; target.new_mean = source.new_occupancy.mean;
    target.new_upper = source.new_occupancy.upper;
    target.new_mean_identified = source.new_occupancy.mean_identified;
    target.new_occupancy_applicable = source.new_occupancy_applicable;
    target.knowledge = static_cast<uint8_t>(source.knowledge);
    target.visibility = source.visibility; target.exposure_count = source.exposure_count;
    target.last_observed = source.last_observed;
    target.reference_cause_id = IdMessage(source.reference_cause_id);
    for (const StableId& cause : source.cause_ids) {
      target.cause_ids.push_back(IdMessage(cause));
      const auto offset = lineage_offset.find(cause);
      if (offset != lineage_offset.end())
        target.cause_projection_offsets.push_back(offset->second);
    }
    message.cells.push_back(target);
  }
  return message;
}

std::vector<deform_monitor_v3::V1DeformationSpatialTile> BuildSpatialTileMessages(
    const V1Snapshot& snapshot, const UacGaussianSumFilter& filter,
    const OccupancyEventGraph& occupancy, const V1Config& config) {
  std::vector<deform_monitor_v3::V1DeformationSpatialTile> messages(3U);
  for (uint32_t lod = 0; lod < messages.size(); ++lod) {
    messages[lod].common = BuildCommonMessage(snapshot.common);
    messages[lod].tile_key = "roi-root/lod-" + std::to_string(lod);
    messages[lod].lod = lod;
    messages[lod].source_snapshot_id = IdMessage(snapshot.common.snapshot_id);
  }

  // Global LOD: event influence envelopes and interface bands. Regional LOD:
  // frozen control support and one detection sample per GDU. Local LOD: every
  // reference support point and the full retained-component field query.
  for (const HazardGeometry& hazard : snapshot.hazards) {
    messages[0].spatial_influence.push_back(HazardMessage(hazard));
    if (std::abs(hazard.horizon_s) <= 1.0e-12)
      messages[1].spatial_influence.push_back(HazardMessage(hazard));
    messages[2].spatial_influence.push_back(HazardMessage(hazard));
  }
  std::set<StableId> geometry_causes;
  for (const HazardGeometry& hazard : snapshot.hazards)
    geometry_causes.insert(hazard.cause_id);
  // Independent occupancy carriers are not structural events, but they are
  // part of the same navigation-facing SpatialInfluence layer. Render their
  // frozen singleton support and future reachable envelope with the same
  // cause id used by risk queries; never create a per-horizon probability.
  for (const OccupancyCell& cell : snapshot.occupancy) {
    for (size_t cause_index = 0; cause_index < cell.cause_ids.size();
         ++cause_index) {
      const StableId& cause = cell.cause_ids[cause_index];
      if (geometry_causes.count(cause) > 0U) continue;
      const HazardSemantics semantics =
          cause_index < cell.cause_semantics.size()
              ? cell.cause_semantics[cause_index]
              : HazardSemantics::OTHER_ENVELOPE;
      for (double horizon : config.resource.prediction_horizons_s) {
        // Match the navigation query exactly: singleton event existence and
        // its spatial reachable envelope are both evaluated at this ETA.
        // Protective carriers without a persistent EventState deliberately
        // retain their published conservative interval.
        const ProbabilityInterval probability =
            occupancy.PredictCauseProbability(
                cell.key, cause,
                snapshot.common.measurement_stamp + ros::Duration(horizon));
        HazardGeometry influence;
        influence.cause_id = cause;
        // This is a spatial projection of an already registered singleton
        // event, not a new stochastic component.  Reuse the cause identity so
        // message construction after snapshot commit cannot create new IDs or
        // turn cell/horizon rendering density into extra causal identities.
        influence.component_id = cause;
        if (cause_index < cell.cause_source_ids.size())
          influence.source_ids = cell.cause_source_ids[cause_index];
        influence.semantics = semantics;
        influence.structural_state = StructuralState::U;
        influence.geometry_hypothesis = GeometryHypothesis::OTHER;
        influence.gdu_index = std::numeric_limits<uint32_t>::max();
        influence.horizon_s = horizon;
        influence.center_R = OccupancyCellCenter(cell.key, config);
        influence.radius_m = 0.5 * std::sqrt(3.0) *
                             config.resource.occupancy_voxel_m;
        influence.existence = probability;
        for (size_t byte = 0; byte < 4U; ++byte)
          influence.correlation_group =
              (influence.correlation_group << 8U) | cause.bytes[byte];
        influence.model_envelope_m =
            semantics == HazardSemantics::REFERENCE_PERSISTENCE
                ? 0.0
                : config.calibration.occupancy_max_speed_mps * horizon +
                      config.calibration.prediction_model_error_m;
        const auto output = HazardMessage(influence);
        messages[0].spatial_influence.push_back(output);
        messages[2].spatial_influence.push_back(output);
        if (std::abs(horizon) <= config.numerical.tie_tolerance)
          messages[1].spatial_influence.push_back(output);
      }
    }
  }
  for (size_t interface_index = 0;
       interface_index < snapshot.interfaces.size(); ++interface_index) {
    const auto output = InterfaceMessage(snapshot, interface_index);
    messages[0].interface_motion.push_back(output);
    messages[1].interface_motion.push_back(output);
    messages[2].interface_motion.push_back(output);
  }

  for (uint32_t g = 0; g < snapshot.reference.gdus.size() &&
                       g < snapshot.deformation.size(); ++g) {
    const GduDefinition& gdu = snapshot.reference.gdus[g];
    const GduPosterior& posterior = snapshot.deformation[g];
    std::set<uint32_t> control_indices(gdu.control_surfel_indices.begin(),
                                       gdu.control_surfel_indices.end());
    bool regional_detection_written = false;
    for (uint32_t surfel_index : gdu.surfel_indices) {
      const Surfel& surfel = snapshot.reference.surfels[surfel_index];
      const FieldQueryResult field =
          filter.QueryReferenceSurfelWithPosterior(
              g, surfel_index, posterior);

      deform_monitor_v3::V1ReferenceSupportSample support;
      support.gdu_id = IdMessage(gdu.id);
      support.surfel_id = IdMessage(surfel.id);
      support.position = PointMessage(surfel.center_R);
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          const size_t index = static_cast<size_t>(3 * row + column);
          support.reference_covariance_row_major[index] =
              surfel.covariance_R(row, column);
          support.observable_projector_row_major[index] =
              surfel.observable_projector_R(row, column);
        }
      }
      for (size_t hypothesis = 0; hypothesis < 4U; ++hypothesis)
        support.geometry_probability[hypothesis] =
            surfel.geometry_probability[static_cast<int>(hypothesis)];
      support.support_radius_m = surfel.support_radius_m;
      support.exposure_count = surfel.exposure_count;
      support.point_count = surfel.point_count;
      support.validity = static_cast<uint8_t>(ConservativeValidity(
          snapshot.reference.validity, snapshot.common.validity));
      messages[2].reference_support.push_back(support);
      if (control_indices.count(surfel_index) > 0U)
        messages[1].reference_support.push_back(support);

      deform_monitor_v3::V1DeformationSample sample;
      sample.gdu_id = IdMessage(gdu.id);
      sample.position = PointMessage(surfel.center_R);
      sample.displacement_mean = VectorMessage(field.mean_displacement_R);
      sample.velocity_mean = VectorMessage(field.mean_velocity_R);
      for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column)
      {
        const size_t matrix_index =
            static_cast<size_t>(3 * row + column);
        sample.displacement_covariance_row_major[matrix_index] =
            field.displacement_covariance_R(row, column);
        sample.velocity_covariance_row_major[matrix_index] =
            field.velocity_covariance_R(row, column);
        sample.observable_projector_row_major[matrix_index] =
            field.observable_projector_R(row, column);
      }
      sample.active_probability_lower = field.active_probability.lower;
      sample.active_probability_mean = field.active_probability.mean;
      sample.active_probability_upper = field.active_probability.upper;
      sample.active_probability_mean_identified =
          field.active_probability.mean_identified;
      sample.validity = static_cast<uint8_t>(ConservativeValidity(
          field.validity, snapshot.common.validity));
      sample.minimum_detectable_effect_defined =
          snapshot.deformation[g].minimum_detectable_effect_defined;
      sample.minimum_detectable_effect_m =
          posterior.minimum_detectable_effect_defined
              ? posterior.minimum_detectable_effect_m : 0.0;
      sample.detection_admission = static_cast<uint8_t>(posterior.admission);
      sample.other_mass = field.other_mass;
      sample.interpolation_error_upper_m =
          std::isfinite(field.interpolation_error_upper_m)
              ? field.interpolation_error_upper_m
              : config.resource.surfel_seed_spacing_m +
                    config.calibration.prediction_model_error_m +
                    config.calibration.ctmc_halving_error_upper_m;
      for (const FieldComponentQuery& component : field.components)
        sample.components.push_back(ComponentQueryMessage(component));
      messages[2].physical_deformation.push_back(sample);

      if (!regional_detection_written ||
          control_indices.count(surfel_index) > 0U) {
        deform_monitor_v3::V1SpatialDetectionEvidence detection;
        detection.gdu_id = IdMessage(gdu.id);
        detection.position = PointMessage(surfel.center_R);
        detection.p_u = posterior.state_probability[0];
        detection.p_a = posterior.state_probability[1];
        detection.p_c = posterior.state_probability[2];
        detection.p_other = posterior.other_mass;
        detection.task_scale_probability_lower =
            posterior.task_scale_deformation_lower;
        detection.task_scale_probability_mean =
            posterior.task_scale_deformation_probability;
        detection.task_scale_probability_upper =
            posterior.task_scale_deformation_upper;
        detection.task_scale_probability_mean_identified =
            posterior.task_scale_deformation_mean_identified;
        detection.task_displacement_probability =
            posterior.task_displacement_probability;
        detection.current_velocity_exceedance_probability =
            posterior.current_velocity_exceedance_probability;
        detection.continuous_valid_observation_duration_s =
            posterior.continuous_valid_observation_duration_s;
        detection.velocity_duration_requirement_met =
            posterior.velocity_duration_requirement_met;
        detection.task_interface_jump_probability =
            posterior.task_interface_jump_probability;
        detection.observable_rank = posterior.observable_rank;
        detection.minimum_detectable_effect_defined =
            posterior.minimum_detectable_effect_defined;
        detection.minimum_detectable_effect_m =
            posterior.minimum_detectable_effect_defined
                ? posterior.minimum_detectable_effect_m : 0.0;
        detection.velocity_minimum_detectable_effect_defined =
            posterior.velocity_minimum_detectable_effect_defined;
        detection.velocity_minimum_detectable_effect_mps =
            posterior.velocity_minimum_detectable_effect_defined
                ? posterior.velocity_minimum_detectable_effect_mps : 0.0;
        detection.interface_jump_minimum_detectable_effect_defined =
            posterior.interface_jump_minimum_detectable_effect_defined;
        detection.interface_jump_minimum_detectable_effect_m =
            posterior.interface_jump_minimum_detectable_effect_defined
                ? posterior.interface_jump_minimum_detectable_effect_m : 0.0;
        detection.task_power_complete = posterior.task_power_complete;
        detection.task_power_worst_normalized_mde =
            posterior.task_power_complete
                ? posterior.task_power_worst_normalized_mde : 0.0;
        detection.any_task_direction_detectable =
            posterior.any_task_direction_detectable;
        detection.all_task_directions_detectable =
            posterior.all_task_directions_detectable;
        detection.detection_allowed = posterior.detection_allowed;
        detection.clearing_allowed = posterior.clearing_allowed;
        detection.effective_spatial_support =
            posterior.effective_spatial_support;
        detection.effective_temporal_support =
            posterior.effective_temporal_support;
        detection.detection_admission =
            static_cast<uint8_t>(posterior.admission);
        detection.validity = static_cast<uint8_t>(ConservativeValidity(
            field.validity, snapshot.common.validity));
        messages[2].detection_evidence.push_back(detection);
        if (!regional_detection_written) {
          messages[1].detection_evidence.push_back(detection);
          regional_detection_written = true;
        }
      }
    }
  }
  return messages;
}

namespace {

constexpr size_t kMaximumDetectionTileWorkerCount = 8U;

struct GduDetectionTileContent {
  std::vector<deform_monitor_v3::V1ReferenceSupportSample>
      regional_reference_support;
  std::vector<deform_monitor_v3::V1ReferenceSupportSample>
      dense_reference_support;
  std::vector<deform_monitor_v3::V1DeformationSample> dense_deformation;
  std::vector<deform_monitor_v3::V1SpatialDetectionEvidence>
      regional_detection_evidence;
  std::vector<deform_monitor_v3::V1SpatialDetectionEvidence>
      dense_detection_evidence;
  std::exception_ptr failure;
};

deform_monitor_v3::V1ReferenceSupportSample ReferenceSupportMessage(
    const V1Snapshot& snapshot, const GduDefinition& gdu,
    const Surfel& surfel) {
  deform_monitor_v3::V1ReferenceSupportSample support;
  support.gdu_id = IdMessage(gdu.id);
  support.surfel_id = IdMessage(surfel.id);
  support.position = PointMessage(surfel.center_R);
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const size_t index = static_cast<size_t>(3 * row + column);
      support.reference_covariance_row_major[index] =
          surfel.covariance_R(row, column);
      support.observable_projector_row_major[index] =
          surfel.observable_projector_R(row, column);
    }
  }
  for (size_t hypothesis = 0; hypothesis < 4U; ++hypothesis)
    support.geometry_probability[hypothesis] =
        surfel.geometry_probability[static_cast<int>(hypothesis)];
  support.support_radius_m = surfel.support_radius_m;
  support.exposure_count = surfel.exposure_count;
  support.point_count = surfel.point_count;
  support.validity = static_cast<uint8_t>(ConservativeValidity(
      snapshot.reference.validity, snapshot.common.validity));
  return support;
}

deform_monitor_v3::V1DeformationSample DeformationSampleMessage(
    const V1Snapshot& snapshot, const GduDefinition& gdu,
    const GduPosterior& posterior, const Surfel& surfel,
    const FieldQueryResult& field, const V1Config& config) {
  deform_monitor_v3::V1DeformationSample sample;
  sample.gdu_id = IdMessage(gdu.id);
  sample.position = PointMessage(surfel.center_R);
  sample.displacement_mean = VectorMessage(field.mean_displacement_R);
  sample.velocity_mean = VectorMessage(field.mean_velocity_R);
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const size_t matrix_index = static_cast<size_t>(3 * row + column);
      sample.displacement_covariance_row_major[matrix_index] =
          field.displacement_covariance_R(row, column);
      sample.velocity_covariance_row_major[matrix_index] =
          field.velocity_covariance_R(row, column);
      sample.observable_projector_row_major[matrix_index] =
          field.observable_projector_R(row, column);
    }
  }
  sample.active_probability_lower = field.active_probability.lower;
  sample.active_probability_mean = field.active_probability.mean;
  sample.active_probability_upper = field.active_probability.upper;
  sample.active_probability_mean_identified =
      field.active_probability.mean_identified;
  sample.validity = static_cast<uint8_t>(ConservativeValidity(
      field.validity, snapshot.common.validity));
  sample.minimum_detectable_effect_defined =
      posterior.minimum_detectable_effect_defined;
  sample.minimum_detectable_effect_m =
      posterior.minimum_detectable_effect_defined
          ? posterior.minimum_detectable_effect_m : 0.0;
  sample.detection_admission = static_cast<uint8_t>(posterior.admission);
  sample.other_mass = field.other_mass;
  sample.interpolation_error_upper_m =
      std::isfinite(field.interpolation_error_upper_m)
          ? field.interpolation_error_upper_m
          : config.resource.surfel_seed_spacing_m +
                config.calibration.prediction_model_error_m +
                config.calibration.ctmc_halving_error_upper_m;
  sample.components.reserve(field.components.size());
  for (const FieldComponentQuery& component : field.components)
    sample.components.push_back(ComponentQueryMessage(component));
  return sample;
}

deform_monitor_v3::V1SpatialDetectionEvidence DetectionEvidenceMessage(
    const V1Snapshot& snapshot, const GduDefinition& gdu,
    const GduPosterior& posterior, const Surfel& surfel,
    const FieldQueryResult& field) {
  deform_monitor_v3::V1SpatialDetectionEvidence detection;
  detection.gdu_id = IdMessage(gdu.id);
  detection.position = PointMessage(surfel.center_R);
  detection.p_u = posterior.state_probability[0];
  detection.p_a = posterior.state_probability[1];
  detection.p_c = posterior.state_probability[2];
  detection.p_other = posterior.other_mass;
  detection.task_scale_probability_lower =
      posterior.task_scale_deformation_lower;
  detection.task_scale_probability_mean =
      posterior.task_scale_deformation_probability;
  detection.task_scale_probability_upper =
      posterior.task_scale_deformation_upper;
  detection.task_scale_probability_mean_identified =
      posterior.task_scale_deformation_mean_identified;
  detection.task_displacement_probability =
      posterior.task_displacement_probability;
  detection.current_velocity_exceedance_probability =
      posterior.current_velocity_exceedance_probability;
  detection.continuous_valid_observation_duration_s =
      posterior.continuous_valid_observation_duration_s;
  detection.velocity_duration_requirement_met =
      posterior.velocity_duration_requirement_met;
  detection.task_interface_jump_probability =
      posterior.task_interface_jump_probability;
  detection.observable_rank = posterior.observable_rank;
  detection.minimum_detectable_effect_defined =
      posterior.minimum_detectable_effect_defined;
  detection.minimum_detectable_effect_m =
      posterior.minimum_detectable_effect_defined
          ? posterior.minimum_detectable_effect_m : 0.0;
  detection.velocity_minimum_detectable_effect_defined =
      posterior.velocity_minimum_detectable_effect_defined;
  detection.velocity_minimum_detectable_effect_mps =
      posterior.velocity_minimum_detectable_effect_defined
          ? posterior.velocity_minimum_detectable_effect_mps : 0.0;
  detection.interface_jump_minimum_detectable_effect_defined =
      posterior.interface_jump_minimum_detectable_effect_defined;
  detection.interface_jump_minimum_detectable_effect_m =
      posterior.interface_jump_minimum_detectable_effect_defined
          ? posterior.interface_jump_minimum_detectable_effect_m : 0.0;
  detection.task_power_complete = posterior.task_power_complete;
  detection.task_power_worst_normalized_mde =
      posterior.task_power_complete
          ? posterior.task_power_worst_normalized_mde : 0.0;
  detection.any_task_direction_detectable =
      posterior.any_task_direction_detectable;
  detection.all_task_directions_detectable =
      posterior.all_task_directions_detectable;
  detection.detection_allowed = posterior.detection_allowed;
  detection.clearing_allowed = posterior.clearing_allowed;
  detection.effective_spatial_support =
      posterior.effective_spatial_support;
  detection.effective_temporal_support =
      posterior.effective_temporal_support;
  detection.detection_admission = static_cast<uint8_t>(posterior.admission);
  detection.validity = static_cast<uint8_t>(ConservativeValidity(
      gdu.monitorable ? field.validity : Validity::INCONCLUSIVE,
      snapshot.common.validity));
  return detection;
}

GduDetectionTileContent BuildGduDetectionTileContent(
    const V1Snapshot& snapshot, const UacGaussianSumFilter& filter,
    const V1Config& config, uint32_t gdu_index) {
  GduDetectionTileContent content;
  const GduDefinition& gdu = snapshot.reference.gdus[gdu_index];
  const GduPosterior& posterior = snapshot.deformation[gdu_index];
  const std::set<uint32_t> control_indices(
      gdu.control_surfel_indices.begin(),
      gdu.control_surfel_indices.end());
  content.regional_reference_support.reserve(control_indices.size());
  content.dense_reference_support.reserve(gdu.surfel_indices.size());
  if (gdu.monitorable)
    content.dense_deformation.reserve(gdu.surfel_indices.size());
  content.regional_detection_evidence.reserve(
      control_indices.size() + 1U);
  content.dense_detection_evidence.reserve(control_indices.size() + 1U);

  bool regional_detection_written = false;
  for (uint32_t surfel_index : gdu.surfel_indices) {
    const Surfel& surfel = snapshot.reference.surfels[surfel_index];
    const bool control = control_indices.count(surfel_index) > 0U;
    FieldQueryResult field;
    if (gdu.monitorable) {
      field = filter.QueryReferenceSurfelWithPosterior(
          gdu_index, surfel_index, posterior);
    }

    deform_monitor_v3::V1ReferenceSupportSample support =
        ReferenceSupportMessage(snapshot, gdu, surfel);
    if (control)
      content.regional_reference_support.push_back(support);
    content.dense_reference_support.push_back(std::move(support));

    // OTHER is an explicit coverage carrier for disconnected/unretained
    // reference fragments.  It has no deformation coordinate system and
    // therefore cannot truthfully produce a dynamic physical field sample.
    if (gdu.monitorable) {
      content.dense_deformation.push_back(DeformationSampleMessage(
          snapshot, gdu, posterior, surfel, field, config));
    }

    if (!regional_detection_written || control) {
      deform_monitor_v3::V1SpatialDetectionEvidence detection =
          DetectionEvidenceMessage(snapshot, gdu, posterior, surfel, field);
      content.dense_detection_evidence.push_back(detection);
      if (!regional_detection_written) {
        content.regional_detection_evidence.push_back(std::move(detection));
        regional_detection_written = true;
      }
    }
  }
  return content;
}

std::vector<deform_monitor_v3::V1DetectionTile>
BuildDetectionTileMessagesImpl(
    const V1Snapshot& snapshot, const UacGaussianSumFilter& filter,
    const V1Config& config, size_t requested_worker_count) {
  // Build the detector product directly.  In particular, this function does
  // not instantiate the archived hazard/occupancy spatial-tile contract and
  // cannot call live-occupancy prediction as a side effect of serialization.
  std::vector<deform_monitor_v3::V1DetectionTile> output(3U);
  for (uint32_t lod = 0; lod < output.size(); ++lod) {
    output[lod].common = BuildDetectionCommonMessage(snapshot.common);
    output[lod].tile_key = "roi-root/lod-" + std::to_string(lod);
    output[lod].lod = lod;
    output[lod].source_snapshot_id = IdMessage(snapshot.common.snapshot_id);
  }
  for (size_t interface_index = 0;
       interface_index < snapshot.interfaces.size(); ++interface_index) {
    const auto message = InterfaceMessage(snapshot, interface_index);
    output[0].interface_motion.push_back(message);
    output[1].interface_motion.push_back(message);
    output[2].interface_motion.push_back(message);
  }

  const size_t gdu_count = std::min(snapshot.reference.gdus.size(),
                                    snapshot.deformation.size());
  std::vector<GduDetectionTileContent> contents(gdu_count);
  const size_t worker_count = std::min(
      gdu_count, std::min(kMaximumDetectionTileWorkerCount,
                          std::max<size_t>(1U, requested_worker_count)));
  if (worker_count <= 1U) {
    for (uint32_t g = 0U; g < gdu_count; ++g)
      contents[g] = BuildGduDetectionTileContent(
          snapshot, filter, config, g);
  } else {
    // All query state is immutable and every worker owns exactly one local
    // result at a time.  Scheduling can therefore affect latency only.  A
    // fixed bound prevents serialization from oversubscribing Gazebo and
    // FAST-LIO.  If a worker cannot be created, already-created workers plus
    // this calling thread drain the same queue, giving an exact safe fallback.
    std::atomic<size_t> next_gdu(0U);
    const auto worker = [&]() {
      while (true) {
        const size_t index = next_gdu.fetch_add(1U);
        if (index >= gdu_count) return;
        try {
          contents[index] = BuildGduDetectionTileContent(
              snapshot, filter, config, static_cast<uint32_t>(index));
        } catch (...) {
          // Preserve an all-or-nothing product boundary: no local result is
          // merged and no caller can publish a prefix if one GDU fails.
          contents[index].failure = std::current_exception();
        }
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(worker_count - 1U);
    for (size_t worker_index = 1U; worker_index < worker_count;
         ++worker_index) {
      try {
        workers.emplace_back(worker);
      } catch (const std::system_error&) {
        break;
      }
    }
    worker();
    for (std::thread& thread : workers) thread.join();
  }

  // Inspect all failures before moving even one GDU into the return product.
  // Rethrowing is the same fail-closed allocation/numerical exception contract
  // as the former serial builder, while the node's pre-publish construction
  // barrier prevents cross-topic partial publication.
  for (const GduDetectionTileContent& content : contents)
    if (content.failure) std::rethrow_exception(content.failure);

  size_t regional_reference_count = 0U;
  size_t dense_reference_count = 0U;
  size_t dense_deformation_count = 0U;
  size_t regional_detection_count = 0U;
  size_t dense_detection_count = 0U;
  for (const GduDetectionTileContent& content : contents) {
    regional_reference_count += content.regional_reference_support.size();
    dense_reference_count += content.dense_reference_support.size();
    dense_deformation_count += content.dense_deformation.size();
    regional_detection_count += content.regional_detection_evidence.size();
    dense_detection_count += content.dense_detection_evidence.size();
  }
  output[1].reference_support.reserve(regional_reference_count);
  output[2].reference_support.reserve(dense_reference_count);
  output[2].physical_deformation.reserve(dense_deformation_count);
  output[1].detection_evidence.reserve(regional_detection_count);
  output[2].detection_evidence.reserve(dense_detection_count);

  // Canonical merge is deliberately serial: GDU index first, then the exact
  // original surfel order retained by each local builder.  Completion order
  // can never change ROS array order or serialized bytes.
  const auto append = [](auto* destination, auto* source) {
    destination->insert(destination->end(),
                        std::make_move_iterator(source->begin()),
                        std::make_move_iterator(source->end()));
  };
  for (GduDetectionTileContent& content : contents) {
    append(&output[1].reference_support,
           &content.regional_reference_support);
    append(&output[2].reference_support, &content.dense_reference_support);
    append(&output[2].physical_deformation, &content.dense_deformation);
    append(&output[1].detection_evidence,
           &content.regional_detection_evidence);
    append(&output[2].detection_evidence,
           &content.dense_detection_evidence);
  }
  return output;
}

}  // namespace

std::vector<deform_monitor_v3::V1DetectionTile> BuildDetectionTileMessages(
    const V1Snapshot& snapshot, const UacGaussianSumFilter& filter,
    const V1Config& config) {
  return BuildDetectionTileMessagesImpl(
      snapshot, filter, config, kMaximumDetectionTileWorkerCount);
}

std::vector<deform_monitor_v3::V1DetectionTile>
BuildDetectionTileMessagesSerialForTesting(
    const V1Snapshot& snapshot, const UacGaussianSumFilter& filter,
    const V1Config& config) {
  return BuildDetectionTileMessagesImpl(snapshot, filter, config, 1U);
}

deform_monitor_v3::V1RiskAssessment BuildRiskAssessmentMessage(
    const RiskAssessment& source) {
  deform_monitor_v3::V1RiskAssessment target;
  target.snapshot_id = IdMessage(source.snapshot_id);
  target.assessment_scope = source.assessment_scope;
  target.geometric_support_evaluated = source.geometric_support_evaluated;
  target.geometric_support_validity =
      static_cast<uint8_t>(source.geometric_support_validity);
  target.hard_state = static_cast<uint8_t>(source.hard);
  target.first_violation_time = source.first_violation_time;
  target.first_violation_configuration_defined =
      source.first_violation_configuration_defined;
  if (source.first_violation_configuration_defined) {
    target.first_violation_configuration = TimedPoseMessage(
        source.first_violation_time,
        source.first_violation_pose_row_major,
        source.first_violation_covariance_row_major,
        source.first_violation_pose_frame,
        source.first_violation_covariance_origin_frame);
  }
  target.collision_lower = source.collision.lower; target.collision_mean = source.collision.mean;
  target.collision_upper = source.collision.upper;
  target.collision_mean_identified = source.collision.mean_identified;
  target.encounter_lower = source.encounter.lower;
  target.encounter_mean = source.encounter.mean;
  target.encounter_upper = source.encounter.upper;
  target.encounter_mean_identified = source.encounter.mean_identified;
  target.path_risk_lower = source.path_risk.lower;
  target.path_risk_mean = source.path_risk.mean; target.path_risk_upper = source.path_risk.upper;
  target.path_risk_mean_identified = source.path_risk.mean_identified;
  target.risk_budget_margin = source.risk_budget_margin;
  target.soft_collision_nats = source.soft.collision_nats;
  target.soft_unknown_cost = source.soft.unknown_cost; target.soft_stale_cost = source.soft.stale_cost;
  target.soft_total = source.soft.total;
  for (size_t i = 0; i < source.cause_ids.size(); ++i) {
    target.cause_ids.push_back(IdMessage(source.cause_ids[i]));
    const ProbabilityInterval probability = i < source.cause_probabilities.size()
        ? source.cause_probabilities[i] : ProbabilityInterval();
    target.cause_probability_lower.push_back(probability.lower);
    target.cause_probability_mean.push_back(probability.mean);
    target.cause_probability_upper.push_back(probability.upper);
    target.cause_probability_mean_identified.push_back(
        probability.mean_identified);
    const ProbabilityInterval encounter =
        i < source.cause_encounter_probabilities.size()
            ? source.cause_encounter_probabilities[i]
            : ProbabilityInterval{0.0, 0.0, probability.upper,
                                  probability.calibrated, false};
    target.cause_encounter_probability_lower.push_back(encounter.lower);
    target.cause_encounter_probability_mean.push_back(encounter.mean);
    target.cause_encounter_probability_upper.push_back(encounter.upper);
    target.cause_encounter_probability_mean_identified.push_back(
        encounter.mean_identified);
  }
  target.validity = static_cast<uint8_t>(source.validity);
  for (ReasonCode reason : source.reasons)
    target.validity_reason_codes.push_back(static_cast<uint16_t>(reason));
  return target;
}

deform_monitor_v3::V1SystemStatus BuildStatusMessage(
    const V1Snapshot& snapshot, MonitoringPhase phase,
    uint64_t deadline_failure_count, uint64_t resource_failure_count) {
  deform_monitor_v3::V1SystemStatus message;
  message.common = BuildCommonMessage(snapshot.common); message.phase = static_cast<uint8_t>(phase);
  message.reference_snapshot_id = IdMessage(snapshot.reference.id);
  message.reference_frozen = snapshot.reference.frozen;
  message.gdu_count = snapshot.deformation.size(); message.interface_count = snapshot.interfaces.size();
  message.occupancy_cell_count = snapshot.occupancy.size(); message.hazard_count = snapshot.hazards.size();
  // These are the algorithmic contracts active in the current run.
  message.active_gates = {
      "STRICT_SCHEMA", "ATOMIC_SNAPSHOT_IDENTITY", "UNKNOWN_NOT_PASS",
      "INPUT_QUALITY_PROTECTIVE_ENVELOPE", "FIXED_LAG_JOINT_WHITENING",
      "TASK_POWER_ADMISSION", "CAUSE_DEDUPLICATION",
      "FULL_BODY_SPHERE_UNION", "BODY_QUERY_RESOURCE_BOUND"};
  message.deadline_failure_count = deadline_failure_count;
  message.resource_failure_count = resource_failure_count;
  message.reason_catalog_version = "reason-catalog-v1";
  return message;
}

deform_monitor_v3::V1DetectionStatus BuildDetectionStatusMessage(
    const V1Snapshot& snapshot, MonitoringPhase phase,
    uint64_t deadline_failure_count, uint64_t resource_failure_count) {
  deform_monitor_v3::V1DetectionStatus message;
  message.common = BuildDetectionCommonMessage(snapshot.common);
  message.phase = static_cast<uint8_t>(phase);
  message.reference_snapshot_id = IdMessage(snapshot.reference.id);
  message.reference_frozen = snapshot.reference.frozen;
  message.gdu_count = snapshot.deformation.size();
  message.interface_count = snapshot.interfaces.size();
  message.active_gates = {
      "STRICT_SCHEMA", "ATOMIC_SNAPSHOT_IDENTITY",
      "DISJOINT_POINT_ROLES", "REFERENCE_FROZEN",
      "FIXED_LAG_JOINT_WHITENING", "TASK_POWER_ADMISSION"};
  message.deadline_failure_count = deadline_failure_count;
  message.resource_failure_count = resource_failure_count;
  message.reason_catalog_version = "reason-catalog-v1";
  return message;
}

}  // namespace v1
}  // namespace deform_monitor_v3
