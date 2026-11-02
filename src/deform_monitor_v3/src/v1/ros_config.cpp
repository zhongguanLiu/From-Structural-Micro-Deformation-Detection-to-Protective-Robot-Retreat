#include "deform_monitor_v3/v1/ros_config.hpp"

#include "deform_monitor_v3/v1/config.hpp"
#include "deform_monitor_v3/v1/deterministic.hpp"

#include <XmlRpcValue.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <map>
#include <set>

namespace deform_monitor_v3 {
namespace v1 {
namespace {

template <typename T>
bool Required(const ros::NodeHandle& node, const std::string& key, T* value,
              std::vector<std::string>* errors) {
  if (node.getParam(key, *value)) return true;
  errors->push_back("missing required parameter: v1/" + key);
  return false;
}

bool RequiredDoubleVector(const ros::NodeHandle& node, const std::string& key,
                          size_t size, std::vector<double>* value,
                          std::vector<std::string>* errors) {
  if (!Required(node, key, value, errors)) return false;
  if (value->size() != size) {
    errors->push_back("parameter v1/" + key + " has wrong vector length");
    return false;
  }
  return true;
}

void CollectLeaves(const XmlRpc::XmlRpcValue& value, const std::string& prefix,
                   std::set<std::string>* leaves) {
  if (value.getType() != XmlRpc::XmlRpcValue::TypeStruct) {
    leaves->insert(prefix); return;
  }
  for (auto iterator = value.begin(); iterator != value.end(); ++iterator) {
    const std::string path = prefix.empty() ? iterator->first : prefix + "/" + iterator->first;
    CollectLeaves(iterator->second, path, leaves);
  }
}

bool ValidateYamlMappingKeys(const YAML::Node& node, const std::string& path,
                             std::vector<std::string>* errors) {
  if (node.IsMap()) {
    std::set<std::string> keys;
    bool valid = true;
    for (const auto& item : node) {
      if (!item.first.IsScalar()) {
        errors->push_back("non-scalar YAML key at " + path);
        valid = false;
        continue;
      }
      const std::string key = item.first.Scalar();
      if (!keys.insert(key).second) {
        errors->push_back("duplicate YAML key: " + path + "/" + key);
        valid = false;
      }
      valid = ValidateYamlMappingKeys(
          item.second, path.empty() ? key : path + "/" + key, errors) && valid;
    }
    return valid;
  }
  if (node.IsSequence()) {
    bool valid = true;
    for (size_t index = 0; index < node.size(); ++index)
      valid = ValidateYamlMappingKeys(
          node[index], path + "[" + std::to_string(index) + "]", errors) && valid;
    return valid;
  }
  return true;
}

void CollectYamlLeaves(const YAML::Node& node, const std::string& prefix,
                       std::set<std::string>* leaves) {
  if (!node.IsMap()) {
    leaves->insert(prefix);
    return;
  }
  for (const auto& item : node) {
    if (!item.first.IsScalar()) continue;
    const std::string key = item.first.Scalar();
    const std::string path = prefix.empty() ? key : prefix + "/" + key;
    CollectYamlLeaves(item.second, path, leaves);
  }
}

bool ValidateYamlSources(const ros::NodeHandle& private_node,
                         std::set<std::string>* merged_leaves,
                         std::vector<std::string>* errors) {
  const std::array<std::string, 3> names{{
      "algorithm", "resource", "calibration"}};
  std::map<std::string, std::string> owner_by_leaf;
  bool valid = true;
  for (const std::string& name : names) {
    std::string path;
    if (!private_node.getParam("config_sources/" + name, path) || path.empty()) {
      errors->push_back("missing required config source path: config_sources/" + name);
      valid = false;
      continue;
    }
    try {
      const YAML::Node root = YAML::LoadFile(path);
      valid = ValidateYamlMappingKeys(root, path, errors) && valid;
      if (!root.IsMap() || root.size() != 1U || !root["v1"] ||
          !root["v1"].IsMap()) {
        errors->push_back(
            "configuration source must contain exactly one mapping root named v1: " +
            path);
        valid = false;
        continue;
      }
      std::set<std::string> source_leaves;
      CollectYamlLeaves(root["v1"], "", &source_leaves);
      if (source_leaves.empty()) {
        errors->push_back("configuration source has no v1 leaves: " + path);
        valid = false;
      }
      for (const std::string& leaf : source_leaves) {
        const auto inserted = owner_by_leaf.emplace(leaf, path);
        if (!inserted.second) {
          errors->push_back("configuration leaf is defined by multiple sources: v1/" +
                            leaf + " in " + inserted.first->second + " and " + path);
          valid = false;
        } else if (merged_leaves) {
          merged_leaves->insert(leaf);
        }
      }
    } catch (const std::exception& exception) {
      errors->push_back("cannot parse YAML source " + path + ": " + exception.what());
      valid = false;
    }
  }
  return valid;
}

bool VerifyAlgorithmConstants(const ros::NodeHandle& node,
                              std::vector<std::string>* errors) {
  const std::map<std::string, std::string> expected = {
      {"algorithm/schema_id", "deform-monitor-v1"},
      {"algorithm/input_profile", "FRAME_XYZI_REGISTERED"},
      {"algorithm/cloud_coordinate_contract", "WORLD_REGISTERED_HEADER_MUST_MATCH"},
      {"algorithm/pose_covariance_source_contract", "FASTLIO23_POS_WORLD_ROT_IMU_LOCAL_RIGHT"},
      {"algorithm/pose_covariance_contract", "LIDAR_WORLD_LEFT_ROTATION_WORLD_ADDITIVE_ORIGIN"},
      {"algorithm/pose_covariance_order", "ROTATION_THEN_SENSOR_ORIGIN"},
      {"algorithm/fastlio_state_order", "pos,rot,offset_R_L_I,offset_T_L_I,vel,bg,ba,grav"},
      {"algorithm/pose_covariance_invalid_action", "REJECT_FRAME_NO_FALLBACK"},
      {"algorithm/sensor_extrinsic_mode", "FIXED_DIGESTED_ONLINE_ESTIMATION_REJECTED"},
      {"algorithm/pose_temporal_model", "SPATIALLY_EXCLUDED_CONTROL_ONLY_SE3_PLUS_FIXED_LAG_OU"},
      {"algorithm/semantic_scene_inference", "DISABLED"},
      {"algorithm/object_identity_input", "DISABLED"},
      {"algorithm/intensity_likelihood_usage", "DISABLED"},
      {"algorithm/reference_policy", "SAME_SESSION_LATEST_PRETRIGGER_ELIGIBLE_ELSE_DWELL_FROZEN"},
      {"algorithm/reference_uncertainty_representation", "EXPOSURE_SCATTER_PLUS_NONSHRINKING_EPOCH_STATIC_STATE"},
      {"algorithm/geometry_carrier", "NONSEMANTIC_SURFEL_GRAPH"},
      {"algorithm/reference_connectivity", "LINK_CORE_WITH_AMBIGUOUS_STATIC_MIXTURE"},
      {"algorithm/field_basis", "GRAPH_POU_WENDLAND_C2_MINIMAL_OBSERVABLE_COORDINATES"},
      {"algorithm/state_time_model", "CTMC_UAC_FIXED_SPARSITY"},
      {"algorithm/ctmc_continuous_coupling", "EQUAL_SUBSTEP_PARENT_PROPAGATE_THEN_ENDPOINT_RESET"},
      {"algorithm/update_granularity", "ONE_ACCEPTED_FRAME"},
      {"algorithm/active_dynamics", "CONSTANT_VELOCITY_WHITE_ACCELERATION"},
      {"algorithm/changed_dynamics", "PERSISTENT_OFFSET_RANDOM_WALK"},
      {"algorithm/robust_likelihood", "GAUSSIAN_STUDENT_T_CONTAMINATION_MIXTURE"},
      {"algorithm/selection_model", "SPLIT_SUPPORT_WITH_EXPLICIT_SELECTION"},
      {"algorithm/association_factor", "CALIBRATED_MULTINOMIAL_LOGIT_TOP3_TAIL_TO_OTHER_SINGLE_MIXTURE_UPDATE"},
      {"algorithm/source_missing_action", "DIAGNOSTIC_SKIP_UPDATE_COUNT_TO_INCONCLUSIVE"},
      {"algorithm/moved_semantics", "DIAGNOSTIC_ONLY"}};
  bool valid = true;
  for (const auto& item : expected) {
    std::string actual;
    if (!node.getParam(item.first, actual) || actual != item.second) {
      errors->push_back("frozen algorithm key mismatch: v1/" + item.first);
      valid = false;
    }
  }
  int schema_version = 0, state_dim = 0, quadrature = 0;
  valid = Required(node, "algorithm/schema_version", &schema_version, errors) && valid;
  valid = Required(node, "algorithm/fastlio_state_dim", &state_dim, errors) && valid;
  valid = Required(node, "algorithm/student_scale_quadrature_order", &quadrature, errors) && valid;
  if (schema_version != 3 || state_dim != 23 || quadrature != 8) {
    errors->push_back("frozen integer algorithm key mismatch"); valid = false;
  }
  bool reference_update = true;
  if (!Required(node, "algorithm/reference_update_during_monitoring",
                &reference_update, errors) || reference_update) {
    errors->push_back("frozen boolean algorithm key mismatch"); valid = false;
  }
  const std::map<std::string, std::vector<std::string>> expected_string_vectors = {
      {"algorithm/observable_geometry_hypotheses",
       {"SURFACE_LIKE", "CURVE_LIKE", "ISOTROPIC", "OTHER"}},
      {"algorithm/support_partition_roles",
       {"POSE_CONTROL", "ASSOCIATION", "MEASUREMENT", "OCCUPANCY"}},
      {"algorithm/structural_states", {"U", "A", "C"}},
      {"algorithm/field_mode_set",
       {"SURFACE_NORMAL_1D", "CURVE_NORMAL_PLANE_2D",
        "MULTIVIEW_FULL_3D", "OTHER_UNMONITORED"}},
      {"algorithm/association_semantics",
       {"SOURCE_MISSING", "NOVEL", "CLUTTER", "OTHER"}}};
  for (const auto& item : expected_string_vectors) {
    std::vector<std::string> actual;
    if (!node.getParam(item.first, actual) || actual != item.second) {
      errors->push_back("frozen algorithm vector mismatch: v1/" + item.first);
      valid = false;
    }
  }
  std::vector<int> partition;
  if (!node.getParam("algorithm/support_partition_percent", partition) ||
      partition != std::vector<int>({20, 20, 40, 20})) {
    errors->push_back("frozen support partition mismatch"); valid = false;
  }
  return valid;
}

std::set<std::string> AllowedLeaves() {
  return {
    "algorithm/schema_id", "algorithm/schema_version", "algorithm/input_profile",
    "algorithm/cloud_coordinate_contract", "algorithm/pose_covariance_source_contract",
    "algorithm/pose_covariance_contract", "algorithm/pose_covariance_order",
    "algorithm/fastlio_state_dim", "algorithm/fastlio_state_order",
    "algorithm/pose_covariance_invalid_action", "algorithm/sensor_extrinsic_mode",
    "algorithm/pose_temporal_model", "algorithm/semantic_scene_inference",
    "algorithm/object_identity_input", "algorithm/intensity_likelihood_usage",
    "algorithm/reference_policy", "algorithm/reference_update_during_monitoring",
    "algorithm/reference_uncertainty_representation", "algorithm/geometry_carrier",
    "algorithm/reference_connectivity", "algorithm/field_basis",
    "algorithm/observable_geometry_hypotheses", "algorithm/support_partition_roles",
    "algorithm/support_partition_percent",
    "algorithm/structural_states", "algorithm/state_time_model",
    "algorithm/ctmc_continuous_coupling", "algorithm/field_mode_set",
    "algorithm/update_granularity", "algorithm/active_dynamics",
    "algorithm/changed_dynamics", "algorithm/robust_likelihood",
    "algorithm/student_scale_quadrature_order", "algorithm/selection_model",
    "algorithm/association_factor", "algorithm/source_missing_action",
    "algorithm/association_semantics", "algorithm/moved_semantics",
    "resource/session_id", "resource/world_frame", "resource/grid_origin_m",
    "resource/frame_sync_tolerance_s", "resource/upstream_registered_cloud_deskew_verified",
    "resource/raw_frame_buffer_s", "resource/max_points_per_frame",
    "resource/max_raw_buffer_points", "resource/reference_min_frames",
    "resource/reference_min_surfel_exposures",
    "resource/reference_min_duration_s", "resource/bounded_reference_search", "resource/max_update_gap_s",
    "resource/fixed_lag_frames", "resource/mixture_max_components",
    "resource/other_mass_alarm", "resource/surfel_seed_spacing_m",
    "resource/graph_neighbor_radius_m", "resource/graph_max_neighbors",
    "resource/control_spacing_m", "resource/pou_support_radius_m",
    "resource/interface_scales_m", "resource/occupancy_voxel_m",
    "resource/prediction_horizons_s", "resource/prediction_samples",
    "resource/deadline_ms", "resource/max_gdu", "resource/min_controls_per_gdu",
    "resource/max_controls_per_gdu", "resource/max_interfaces_per_gdu",
    "resource/max_interfaces", "resource/max_events", "resource/max_tiles",
    "resource/max_stable_ids",
    "resource/max_evidence_links", "resource/map_ttl_s", "resource/query_ttl_s",
    "resource/association_max_candidates", "resource/min_patches_per_gdu",
    "resource/max_patches_per_gdu", "resource/pose_control_min_patches",
    "resource/pose_control_exclusion_radius_m",
    "resource/upstream_stationary_pose_freeze_verified",
    "resource/source_missing_consecutive_limit",
    "resource/ctmc_max_substep_s",
    "resource/extrinsic/revision", "resource/extrinsic/digest",
    "resource/extrinsic/online_estimation_enabled",
    "resource/extrinsic/rotation_row_major", "resource/extrinsic/translation_m",
    "resource/extrinsic/covariance_row_major",
    "calibration/validated", "calibration/version", "calibration/data_digest",
    "calibration/independent_point_sigma_m", "calibration/unobservable_residual_sigma_m",
    "calibration/contamination_probability",
    "calibration/student_t_dof", "calibration/student_t_scale",
    "calibration/pose_ou_tau_s", "calibration/pose_cross_covariance_bound_m2",
    "calibration/ctmc_rates", "calibration/active_acceleration_psd",
    "calibration/changed_random_walk_psd", "calibration/active_birth_velocity_variance",
    "calibration/active_birth_displacement_variance",
    "calibration/changed_birth_displacement_variance",
    "calibration/ctmc_halving_error_upper_m", "calibration/reference_stability_sigma",
    "calibration/reference_epoch_common_bias_sigma_m",
    "calibration/geometry_min_eigengap_ratio",
    "calibration/pose_control_max_condition_number",
    "calibration/pose_control_huber_scale_m",
    "calibration/reference_confounding_min_information_ratio",
    "calibration/reference_link_logit_bias", "calibration/reference_link_distance_weight",
    "calibration/reference_link_projector_weight", "calibration/link_probability_on",
    "calibration/cut_probability_on", "calibration/interface_initial_link_probability",
    "calibration/interface_coherence_transition_row_major",
    "calibration/interface_coherent_rho", "calibration/interface_relative_rho",
    "calibration/interface_coherent_process_variance",
    "calibration/interface_relative_process_variance",
    "calibration/interface_unresolved_process_variance",
    "calibration/geometry_softmax_weights_row_major", "calibration/geometry_softmax_bias",
    "calibration/association_weights", "calibration/association_candidate_bias",
    "calibration/association_missing_bias", "calibration/association_other_bias",
    "calibration/association_entropy_max",
    "calibration/visibility_min_range_m", "calibration/visibility_max_range_m",
    "calibration/occupancy_interval_alpha",
    "calibration/reference_beta_tempering", "calibration/reference_free_threshold",
    "calibration/reference_occupied_threshold", "calibration/task_displacement_scale_m",
    "calibration/task_velocity_scale_mps",
    "calibration/task_velocity_required_duration_s",
    "calibration/task_interface_jump_scale_m",
    "calibration/task_interface_metric_row_major",
    "calibration/detection_probability_on", "calibration/clearing_probability_on",
    "calibration/event_false_alarm_probability", "calibration/target_power",
    "calibration/mde_bootstrap_inflation",
    "calibration/detection_min_effective_spatial_support",
    "calibration/detection_min_effective_temporal_support",
    "calibration/qmc_error_upper",
    "calibration/prediction_model_error_m",
    "numerical/probability_floor", "numerical/pou_denominator_min",
    "numerical/psd_eigenvalue_tolerance", "numerical/cholesky_jitter_initial",
    "numerical/cholesky_jitter_multiplier", "numerical/cholesky_jitter_attempts",
    "numerical/solve_residual_tolerance", "numerical/normalization_tolerance",
    "numerical/tie_tolerance",
    "software_revision", "io/cloud_topic", "io/covariance_topic"
  };
}

}  // namespace

bool LoadStrictRosConfig(const ros::NodeHandle& private_node, V1Config* config,
                         RobotGeometry* robot, std::vector<std::string>* errors) {
  if (!config || !robot || !errors) return false;
  *config = V1Config{};
  *robot = RobotGeometry{};
  errors->clear();
  std::set<std::string> source_leaves;
  ValidateYamlSources(private_node, &source_leaves, errors);
  ros::NodeHandle node(private_node, "v1");
  XmlRpc::XmlRpcValue root;
  if (!private_node.getParam("v1", root) ||
      root.getType() != XmlRpc::XmlRpcValue::TypeStruct) {
    errors->push_back("missing v1 configuration root");
    return false;
  }
  std::set<std::string> observed;
  CollectLeaves(root, "", &observed);
  const std::set<std::string> allowed = AllowedLeaves();
  for (const std::string& leaf : observed) {
    if (allowed.count(leaf) == 0U)
      errors->push_back("unknown parameter: v1/" + leaf);
    if (source_leaves.count(leaf) == 0U)
      errors->push_back("parameter has no unique declared YAML source: v1/" + leaf);
  }
  for (const std::string& leaf : source_leaves) {
    if (allowed.count(leaf) == 0U)
      errors->push_back("configuration source contains unknown parameter: v1/" + leaf);
    if (observed.count(leaf) == 0U)
      errors->push_back("declared configuration parameter was not loaded: v1/" + leaf);
  }
  VerifyAlgorithmConstants(node, errors);
  ResourceConfig& r = config->resource;
  CalibrationConfig& c = config->calibration;
  NumericalConfig& n = config->numerical;
  std::vector<double> values;
  Required(node, "resource/session_id", &r.session_id, errors);
  Required(node, "resource/world_frame", &r.world_frame, errors);
  if (RequiredDoubleVector(node, "resource/grid_origin_m", 3U, &values, errors))
    r.grid_origin_m = Eigen::Vector3d(values[0], values[1], values[2]);
  Required(node, "resource/frame_sync_tolerance_s", &r.frame_sync_tolerance_s, errors);
  Required(node, "resource/upstream_registered_cloud_deskew_verified",
           &r.upstream_registered_cloud_deskew_verified, errors);
  Required(node, "resource/raw_frame_buffer_s", &r.raw_frame_buffer_s, errors);
  int max_points_per_frame = 0, max_raw_buffer_points = 0;
  Required(node, "resource/max_points_per_frame", &max_points_per_frame, errors);
  Required(node, "resource/max_raw_buffer_points", &max_raw_buffer_points, errors);
  r.max_points_per_frame = static_cast<size_t>(
      std::max(0, max_points_per_frame));
  r.max_raw_buffer_points = static_cast<size_t>(
      std::max(0, max_raw_buffer_points));
  Required(node, "resource/reference_min_frames", &r.reference_min_frames, errors);
  Required(node, "resource/reference_min_surfel_exposures",
           &r.reference_min_surfel_exposures, errors);
  Required(node, "resource/reference_min_duration_s", &r.reference_min_duration_s, errors);
  Required(node, "resource/bounded_reference_search", &r.bounded_reference_search, errors);
  Required(node, "resource/max_update_gap_s", &r.max_update_gap_s, errors);
  Required(node, "resource/fixed_lag_frames", &r.fixed_lag_frames, errors);
  Required(node, "resource/mixture_max_components", &r.mixture_max_components, errors);
  Required(node, "resource/other_mass_alarm", &r.other_mass_alarm, errors);
  Required(node, "resource/surfel_seed_spacing_m", &r.surfel_seed_spacing_m, errors);
  Required(node, "resource/graph_neighbor_radius_m", &r.graph_neighbor_radius_m, errors);
  Required(node, "resource/graph_max_neighbors", &r.graph_max_neighbors, errors);
  Required(node, "resource/control_spacing_m", &r.control_spacing_m, errors);
  Required(node, "resource/pou_support_radius_m", &r.pou_support_radius_m, errors);
  Required(node, "resource/interface_scales_m", &r.interface_scales_m, errors);
  Required(node, "resource/occupancy_voxel_m", &r.occupancy_voxel_m, errors);
  Required(node, "resource/prediction_horizons_s", &r.prediction_horizons_s, errors);
  Required(node, "resource/prediction_samples", &r.prediction_samples, errors);
  Required(node, "resource/deadline_ms", &r.deadline_ms, errors);
  int max_gdu = 0, min_controls = 0, max_controls = 0;
  int max_interfaces_per_gdu = 0, max_interfaces = 0;
  int max_events = 0, max_tiles = 0;
  int max_stable_ids = 0;
  int max_evidence_links = 0;
  Required(node, "resource/max_gdu", &max_gdu, errors);
  Required(node, "resource/min_controls_per_gdu", &min_controls, errors);
  Required(node, "resource/max_controls_per_gdu", &max_controls, errors);
  Required(node, "resource/max_interfaces_per_gdu",
           &max_interfaces_per_gdu, errors);
  Required(node, "resource/max_interfaces", &max_interfaces, errors);
  Required(node, "resource/max_events", &max_events, errors);
  Required(node, "resource/max_tiles", &max_tiles, errors);
  Required(node, "resource/max_stable_ids", &max_stable_ids, errors);
  Required(node, "resource/max_evidence_links", &max_evidence_links, errors);
  r.max_gdu = static_cast<size_t>(std::max(0, max_gdu));
  r.min_controls_per_gdu = static_cast<size_t>(std::max(0, min_controls));
  r.max_controls_per_gdu = static_cast<size_t>(std::max(0, max_controls));
  r.max_interfaces_per_gdu = static_cast<size_t>(
      std::max(0, max_interfaces_per_gdu));
  r.max_interfaces = static_cast<size_t>(std::max(0, max_interfaces));
  r.max_events = static_cast<size_t>(std::max(0, max_events));
  r.max_tiles = static_cast<size_t>(std::max(0, max_tiles));
  r.max_stable_ids = static_cast<size_t>(std::max(0, max_stable_ids));
  r.max_evidence_links = static_cast<size_t>(
      std::max(0, max_evidence_links));
  Required(node, "resource/map_ttl_s", &r.map_ttl_s, errors);
  Required(node, "resource/query_ttl_s", &r.query_ttl_s, errors);
  Required(node, "resource/association_max_candidates", &r.association_max_candidates, errors);
  Required(node, "resource/min_patches_per_gdu", &r.min_patches_per_gdu, errors);
  Required(node, "resource/max_patches_per_gdu", &r.max_patches_per_gdu, errors);
  Required(node, "resource/pose_control_min_patches",
           &r.pose_control_min_patches, errors);
  Required(node, "resource/pose_control_exclusion_radius_m",
           &r.pose_control_exclusion_radius_m, errors);
  Required(node, "resource/upstream_stationary_pose_freeze_verified",
           &r.upstream_stationary_pose_freeze_verified, errors);
  Required(node, "resource/source_missing_consecutive_limit",
           &r.source_missing_consecutive_limit, errors);
  Required(node, "resource/ctmc_max_substep_s", &r.ctmc_max_substep_s, errors);
  Required(node, "resource/extrinsic/revision", &r.extrinsic.revision, errors);
  Required(node, "resource/extrinsic/digest", &r.extrinsic.digest, errors);
  Required(node, "resource/extrinsic/online_estimation_enabled",
           &r.extrinsic.online_estimation_enabled, errors);
  if (RequiredDoubleVector(node, "resource/extrinsic/rotation_row_major", 9U, &values, errors))
    for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column)
      r.extrinsic.R_IL(row, column) = values[static_cast<size_t>(3 * row + column)];
  if (RequiredDoubleVector(node, "resource/extrinsic/translation_m", 3U, &values, errors))
    r.extrinsic.t_IL = Eigen::Vector3d(values[0], values[1], values[2]);
  if (RequiredDoubleVector(node, "resource/extrinsic/covariance_row_major", 36U, &values, errors))
    for (int row = 0; row < 6; ++row) for (int column = 0; column < 6; ++column)
      r.extrinsic.covariance(row, column) = values[static_cast<size_t>(6 * row + column)];

  // Real entry: FAST-LIO must be launched first. Read its loaded calibration,
  // not a second, easily stale copy in the detector YAML. Never write upstream.
  bool use_fastlio_extrinsic = false;
  private_node.param("use_fastlio_extrinsic", use_fastlio_extrinsic, false);
  if (use_fastlio_extrinsic) {
    ros::NodeHandle mapping("/mapping");
    std::vector<double> rotation, translation;
    bool online = true;
    if (!mapping.getParam("extrinsic_R", rotation) || rotation.size() != 9U ||
        !mapping.getParam("extrinsic_T", translation) || translation.size() != 3U ||
        !mapping.getParam("extrinsic_est_en", online) || online) {
      errors->push_back("real input requires FAST-LIO /mapping fixed extrinsic_R, "
                        "extrinsic_T and extrinsic_est_en=false; launch FAST-LIO first");
    } else {
      CanonicalBytes extrinsic_bytes;
      for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column) {
        const double value = rotation[static_cast<size_t>(3 * row + column)];
        r.extrinsic.R_IL(row, column) = value;
        extrinsic_bytes.AddDouble(value);
      }
      r.extrinsic.t_IL = Eigen::Vector3d(translation[0], translation[1], translation[2]);
      for (double value : translation) extrinsic_bytes.AddDouble(value);
      r.extrinsic.online_estimation_enabled = false;
      r.extrinsic.revision = "fastlio-loaded-fixed-real-v1";
      r.extrinsic.digest = HexDigest(Sha256(extrinsic_bytes.bytes()));
      // Mirror effective values for rosparam inspection as well as the config digest.
      node.setParam("resource/extrinsic/rotation_row_major", rotation);
      node.setParam("resource/extrinsic/translation_m", translation);
      node.setParam("resource/extrinsic/online_estimation_enabled", false);
      node.setParam("resource/extrinsic/revision", r.extrinsic.revision);
      node.setParam("resource/extrinsic/digest", r.extrinsic.digest);
      ROS_INFO_STREAM("[deform_v1] using FAST-LIO fixed extrinsic, translation ["
                      << r.extrinsic.t_IL.transpose() << "] m; digest " << r.extrinsic.digest);
    }
  }

  Required(node, "calibration/validated", &c.validated, errors);
  Required(node, "calibration/version", &c.version, errors);
  Required(node, "calibration/data_digest", &c.data_digest, errors);
  Required(node, "calibration/independent_point_sigma_m", &c.independent_point_sigma_m, errors);
  Required(node, "calibration/unobservable_residual_sigma_m",
           &c.unobservable_residual_sigma_m, errors);
  Required(node, "calibration/contamination_probability", &c.contamination_probability, errors);
  Required(node, "calibration/student_t_dof", &c.student_t_dof, errors);
  Required(node, "calibration/student_t_scale", &c.student_t_scale, errors);
  Required(node, "calibration/pose_ou_tau_s", &c.pose_ou_tau_s, errors);
  Required(node, "calibration/pose_cross_covariance_bound_m2",
           &c.pose_cross_covariance_bound_m2, errors);
  if (RequiredDoubleVector(node, "calibration/ctmc_rates", 5U, &values, errors))
    for (int i = 0; i < 5; ++i) c.ctmc_rates[static_cast<size_t>(i)] = values[static_cast<size_t>(i)];
  Required(node, "calibration/active_acceleration_psd", &c.active_acceleration_psd, errors);
  Required(node, "calibration/changed_random_walk_psd", &c.changed_random_walk_psd, errors);
  Required(node, "calibration/active_birth_velocity_variance",
           &c.active_birth_velocity_variance, errors);
  Required(node, "calibration/active_birth_displacement_variance",
           &c.active_birth_displacement_variance, errors);
  Required(node, "calibration/changed_birth_displacement_variance",
           &c.changed_birth_displacement_variance, errors);
  Required(node, "calibration/ctmc_halving_error_upper_m",
           &c.ctmc_halving_error_upper_m, errors);
  Required(node, "calibration/reference_stability_sigma", &c.reference_stability_sigma, errors);
  Required(node, "calibration/reference_epoch_common_bias_sigma_m",
           &c.reference_epoch_common_bias_sigma_m, errors);
  Required(node, "calibration/geometry_min_eigengap_ratio",
           &c.geometry_min_eigengap_ratio, errors);
  Required(node, "calibration/pose_control_max_condition_number",
           &c.pose_control_max_condition_number, errors);
  Required(node, "calibration/pose_control_huber_scale_m",
           &c.pose_control_huber_scale_m, errors);
  Required(node, "calibration/reference_confounding_min_information_ratio",
           &c.reference_confounding_min_information_ratio, errors);
  Required(node, "calibration/reference_link_logit_bias", &c.reference_link_logit_bias, errors);
  Required(node, "calibration/reference_link_distance_weight",
           &c.reference_link_distance_weight, errors);
  Required(node, "calibration/reference_link_projector_weight",
           &c.reference_link_projector_weight, errors);
  Required(node, "calibration/link_probability_on", &c.link_probability_on, errors);
  Required(node, "calibration/cut_probability_on", &c.cut_probability_on, errors);
  if (RequiredDoubleVector(node, "calibration/interface_initial_link_probability", 3U,
                           &values, errors))
    for (int i = 0; i < 3; ++i) c.interface_initial_link_probability[static_cast<size_t>(i)] = values[i];
  if (RequiredDoubleVector(node, "calibration/interface_coherence_transition_row_major", 9U,
                           &values, errors))
    for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column)
      c.interface_coherence_transition(row, column) = values[static_cast<size_t>(3 * row + column)];
  Required(node, "calibration/interface_coherent_rho", &c.interface_coherent_rho, errors);
  Required(node, "calibration/interface_relative_rho", &c.interface_relative_rho, errors);
  Required(node, "calibration/interface_coherent_process_variance",
           &c.interface_coherent_process_variance, errors);
  Required(node, "calibration/interface_relative_process_variance",
           &c.interface_relative_process_variance, errors);
  Required(node, "calibration/interface_unresolved_process_variance",
           &c.interface_unresolved_process_variance, errors);
  if (RequiredDoubleVector(node, "calibration/geometry_softmax_weights_row_major", 16U,
                           &values, errors))
    for (int row = 0; row < 4; ++row) for (int column = 0; column < 4; ++column)
      c.geometry_softmax_weights(row, column) = values[static_cast<size_t>(4 * row + column)];
  if (RequiredDoubleVector(node, "calibration/geometry_softmax_bias", 4U, &values, errors))
    for (int i = 0; i < 4; ++i) c.geometry_softmax_bias[i] = values[i];
  if (RequiredDoubleVector(node, "calibration/association_weights", 7U, &values, errors))
    for (int i = 0; i < 7; ++i) c.association_weights[i] = values[i];
  Required(node, "calibration/association_candidate_bias", &c.association_candidate_bias, errors);
  Required(node, "calibration/association_missing_bias", &c.association_missing_bias, errors);
  Required(node, "calibration/association_other_bias", &c.association_other_bias, errors);
  Required(node, "calibration/association_entropy_max", &c.association_entropy_max, errors);
  Required(node, "calibration/visibility_min_range_m", &c.visibility_min_range_m, errors);
  Required(node, "calibration/visibility_max_range_m", &c.visibility_max_range_m, errors);
  Required(node, "calibration/occupancy_interval_alpha", &c.occupancy_interval_alpha, errors);
  Required(node, "calibration/reference_beta_tempering", &c.reference_beta_tempering, errors);
  Required(node, "calibration/reference_free_threshold", &c.reference_free_threshold, errors);
  Required(node, "calibration/reference_occupied_threshold", &c.reference_occupied_threshold, errors);
  Required(node, "calibration/task_displacement_scale_m", &c.task_displacement_scale_m, errors);
  Required(node, "calibration/task_velocity_scale_mps",
           &c.task_velocity_scale_mps, errors);
  Required(node, "calibration/task_velocity_required_duration_s",
           &c.task_velocity_required_duration_s, errors);
  Required(node, "calibration/task_interface_jump_scale_m",
           &c.task_interface_jump_scale_m, errors);
  if (RequiredDoubleVector(node,
                           "calibration/task_interface_metric_row_major",
                           9U, &values, errors))
    for (int row = 0; row < 3; ++row)
      for (int column = 0; column < 3; ++column)
        c.task_interface_metric(row, column) =
            values[static_cast<size_t>(3 * row + column)];
  Required(node, "calibration/detection_probability_on",
           &c.detection_probability_on, errors);
  Required(node, "calibration/clearing_probability_on",
           &c.clearing_probability_on, errors);
  Required(node, "calibration/event_false_alarm_probability",
           &c.event_false_alarm_probability, errors);
  Required(node, "calibration/target_power", &c.target_power, errors);
  Required(node, "calibration/mde_bootstrap_inflation",
           &c.mde_bootstrap_inflation, errors);
  Required(node, "calibration/detection_min_effective_spatial_support",
           &c.detection_min_effective_spatial_support, errors);
  Required(node, "calibration/detection_min_effective_temporal_support",
           &c.detection_min_effective_temporal_support, errors);
  Required(node, "calibration/qmc_error_upper", &c.qmc_error_upper, errors);
  Required(node, "calibration/prediction_model_error_m", &c.prediction_model_error_m, errors);

  Required(node, "numerical/probability_floor", &n.probability_floor, errors);
  Required(node, "numerical/pou_denominator_min", &n.pou_denominator_min, errors);
  Required(node, "numerical/psd_eigenvalue_tolerance", &n.psd_eigenvalue_tolerance, errors);
  Required(node, "numerical/cholesky_jitter_initial", &n.cholesky_jitter_initial, errors);
  Required(node, "numerical/cholesky_jitter_multiplier", &n.cholesky_jitter_multiplier, errors);
  Required(node, "numerical/cholesky_jitter_attempts", &n.cholesky_jitter_attempts, errors);
  Required(node, "numerical/solve_residual_tolerance", &n.solve_residual_tolerance, errors);
  Required(node, "numerical/normalization_tolerance", &n.normalization_tolerance, errors);
  Required(node, "numerical/tie_tolerance", &n.tie_tolerance, errors);
  Required(node, "software_revision", &config->software_revision, errors);

  // Robot geometry is deliberately outside the detector configuration.  A
  // private compatibility sentinel keeps the frozen legacy V1Config validator
  // usable while no robot body, path or collision state enters the runtime
  // estimator or its public output contract.
  r.robot_geometry_revision = "detector-only-compatibility-sentinel";
  robot->revision = r.robot_geometry_revision;
  robot->body_frame = "detector-only-unused-body";
  RobotSphere compatibility_sphere;
  compatibility_sphere.radius_m = r.occupancy_voxel_m;
  robot->spheres.push_back(compatibility_sphere);
  config->robot_geometry = *robot;
  const ConfigValidation validation = ValidateConfig(*config);
  errors->insert(errors->end(), validation.errors.begin(), validation.errors.end());
  const ConfigValidation frozen = ValidateFrozenV11Profile(*config);
  errors->insert(errors->end(), frozen.errors.begin(), frozen.errors.end());
  config->config_digest = CanonicalConfigDigest(*config);
  return errors->empty();
}

}  // namespace v1
}  // namespace deform_monitor_v3
