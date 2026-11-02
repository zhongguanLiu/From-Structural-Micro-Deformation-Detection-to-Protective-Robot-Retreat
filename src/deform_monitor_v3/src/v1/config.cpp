#include "deform_monitor_v3/v1/config.hpp"

#include "deform_monitor_v3/v1/deterministic.hpp"
#include "deform_monitor_v3/v1/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace deform_monitor_v3 {
namespace v1 {
namespace {

void Require(bool condition, const std::string& message, ConfigValidation* validation) {
  if (!condition) validation->errors.push_back(message);
}

bool StrictlyIncreasingPositive(const std::vector<double>& values) {
  if (values.empty()) return false;
  for (size_t i = 0; i < values.size(); ++i) {
    if (!(values[i] > 0.0) || !std::isfinite(values[i])) return false;
    if (i > 0 && !(values[i] > values[i - 1])) return false;
  }
  return true;
}

bool IsPowerOfTwo(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

void AddVector(const std::vector<double>& values, CanonicalBytes* bytes) {
  bytes->AddU64(values.size());
  for (double value : values) bytes->AddDouble(value);
}

}  // namespace

ConfigValidation ValidateConfig(const V1Config& config) {
  ConfigValidation validation;
  const ResourceConfig& r = config.resource;
  const CalibrationConfig& c = config.calibration;
  const NumericalConfig& n = config.numerical;
  const PolicyConfig& p = config.policy;
  const auto all_finite = [](std::initializer_list<double> values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  Require(!config.software_revision.empty(),
          "software_revision must be non-empty", &validation);
  Require(all_finite({
              r.frame_sync_tolerance_s, r.raw_frame_buffer_s,
              r.reference_min_duration_s, r.max_update_gap_s,
              r.other_mass_alarm, r.surfel_seed_spacing_m,
              r.graph_neighbor_radius_m, r.control_spacing_m,
              r.pou_support_radius_m, r.occupancy_voxel_m, r.deadline_ms,
              r.map_ttl_s, r.query_ttl_s, r.ctmc_max_substep_s}),
          "PAR-RES scalar values must be finite", &validation);
  for (double horizon : r.prediction_horizons_s)
    Require(std::isfinite(horizon),
            "PAR-RES-020 horizons must be finite", &validation);
  Require(all_finite({
              n.probability_floor, n.pou_denominator_min,
              n.psd_eigenvalue_tolerance, n.cholesky_jitter_initial,
              n.cholesky_jitter_multiplier, n.solve_residual_tolerance,
              n.normalization_tolerance, n.tie_tolerance}),
          "PAR-NUM scalar values must be finite", &validation);
  Require(all_finite({
              p.hard_clearance_m, p.hard_collision_probability,
              p.hard_encounter_probability, p.path_risk_upper_bound,
              p.unknown_weight, p.stale_weight, p.probability_epsilon,
              p.risk_saturation_nats, p.maximum_soft_cost,
              p.motion_sample_spacing_m, p.motion_sample_interval_s}),
          "policy scalar values must be finite", &validation);
  Require(all_finite({
              c.independent_point_sigma_m,
              c.unobservable_residual_sigma_m,
              c.contamination_probability, c.student_t_dof,
              c.student_t_scale, c.pose_ou_tau_s,
              c.pose_cross_covariance_bound_m2, c.active_acceleration_psd,
              c.changed_random_walk_psd,
              c.active_birth_velocity_variance,
              c.active_birth_displacement_variance,
              c.changed_birth_displacement_variance,
              c.ctmc_halving_error_upper_m, c.reference_stability_sigma,
              c.reference_epoch_common_bias_sigma_m,
              c.geometry_min_eigengap_ratio,
              c.pose_control_max_condition_number,
              c.pose_control_huber_scale_m,
              c.reference_confounding_min_information_ratio,
              c.reference_link_logit_bias,
              c.reference_link_distance_weight,
              c.reference_link_projector_weight,
              c.interface_coherent_rho, c.interface_relative_rho,
              c.interface_coherent_process_variance,
              c.interface_relative_process_variance,
              c.interface_unresolved_process_variance,
              c.link_probability_on, c.cut_probability_on,
              c.association_candidate_bias, c.association_missing_bias,
              c.association_other_bias, c.association_entropy_max,
              c.visibility_min_range_m, c.visibility_max_range_m,
              c.nb_dispersion, c.nb_clutter_mean,
              c.nb_occupied_mean_scale, c.new_occupancy_prior,
              c.novel_occupancy_birth_rate_hz,
              c.novel_occupancy_death_rate_hz,
              c.reference_persistence_birth_rate_hz,
              c.reference_persistence_death_rate_hz,
              c.unknown_occupancy_birth_rate_hz,
              c.unknown_occupancy_death_rate_hz,
              c.visibility_exposure_scale, c.occupancy_interval_alpha,
              c.reference_beta_tempering, c.reference_free_threshold,
              c.reference_occupied_threshold,
              c.task_displacement_scale_m, c.task_velocity_scale_mps,
              c.task_velocity_required_duration_s,
              c.task_interface_jump_scale_m,
              c.detection_probability_on, c.clearing_probability_on,
              c.event_false_alarm_probability, c.target_power,
              c.mde_bootstrap_inflation,
              c.detection_min_effective_spatial_support,
              c.detection_min_effective_temporal_support,
              c.qmc_error_upper,
              c.prediction_model_error_m, c.occupancy_max_speed_mps,
              c.hazard_sweep_speed_upper_mps}),
          "PAR-CAL scalar values must be finite", &validation);
  Require(!r.session_id.empty(), "PAR-RES-001 session_id must be non-empty", &validation);
  Require(!r.world_frame.empty(), "PAR-RES-002 world_frame must be non-empty", &validation);
  Require(r.grid_origin_m.allFinite(), "PAR-RES-003 grid origin must be finite", &validation);
  Require(r.frame_sync_tolerance_s > 0.0, "PAR-RES-005 sync tolerance must be positive", &validation);
  Require(r.raw_frame_buffer_s > 0.0, "PAR-RES-006 raw buffer must be positive", &validation);
  Require(r.raw_frame_buffer_s >= r.reference_min_duration_s,
          "PAR-RES-006 raw buffer must cover the minimum reference duration",
          &validation);
  Require(r.max_points_per_frame > 0U &&
              r.max_raw_buffer_points >= r.max_points_per_frame &&
              r.max_points_per_frame <=
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()) &&
              r.max_raw_buffer_points <=
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
          "PAR-RES-006 point and raw-buffer resource limits invalid",
          &validation);
  Require(r.reference_min_frames >= 6,
          "PAR-RES-007 requires at least six reference frames", &validation);
  Require(r.reference_min_surfel_exposures >= 5 &&
              r.reference_min_surfel_exposures <= r.reference_min_frames,
          "PAR-RES-007 surfel exposures must be in [5,reference_min_frames]",
          &validation);
  Require(r.reference_min_duration_s > 0.0, "PAR-RES-008 duration must be positive", &validation);
  Require(r.max_update_gap_s > 0.0, "PAR-RES-009 update gap must be positive", &validation);
  Require(r.fixed_lag_frames >= 1, "PAR-RES-010 lag must be at least one", &validation);
  Require(r.mixture_max_components >= 3, "PAR-RES-011 mixture cap must be at least three", &validation);
  Require(r.other_mass_alarm > 0.0 && r.other_mass_alarm < 1.0,
          "PAR-RES-012 OTHER alarm must be in (0,1)", &validation);
  Require(r.surfel_seed_spacing_m > 0.0, "PAR-RES-013 seed spacing must be positive", &validation);
  Require(r.graph_neighbor_radius_m > r.surfel_seed_spacing_m,
          "PAR-RES-014 graph radius must exceed seed spacing", &validation);
  Require(r.graph_max_neighbors >= 3, "PAR-RES-015 graph degree must be at least three", &validation);
  Require(r.control_spacing_m > 0.0, "PAR-RES-016 control spacing must be positive", &validation);
  Require(r.pou_support_radius_m > r.control_spacing_m,
          "PAR-RES-017 PoU radius must exceed control spacing", &validation);
  Require(StrictlyIncreasingPositive(r.interface_scales_m),
          "PAR-RES-018 interface scales must be positive and strictly increasing", &validation);
  Require(r.occupancy_voxel_m > 0.0, "PAR-RES-019 occupancy voxel must be positive", &validation);
  Require(r.prediction_horizons_s.size() >= 2U &&
              r.prediction_horizons_s.front() == 0.0 &&
              r.prediction_horizons_s.back() > 0.0,
          "PAR-RES-020 horizons must start at zero and contain a positive future horizon",
          &validation);
  for (size_t i = 1; i < r.prediction_horizons_s.size(); ++i)
    Require(r.prediction_horizons_s[i] > r.prediction_horizons_s[i - 1],
            "PAR-RES-020 horizons must be strictly increasing", &validation);
  Require(IsPowerOfTwo(r.prediction_samples) && r.prediction_samples >= 64,
          "PAR-RES-021 Sobol samples must be a power of two >=64", &validation);
  Require(r.deadline_ms > 0.0, "PAR-RES-022 deadline must be positive", &validation);
  Require(r.max_gdu > 0U && r.max_gdu <= 64U &&
              r.min_controls_per_gdu >= 6U &&
              r.max_controls_per_gdu >= r.min_controls_per_gdu &&
              r.max_controls_per_gdu <= 16U &&
              r.max_interfaces_per_gdu == 1U && r.max_interfaces > 0U &&
              r.max_events > 0U && r.max_tiles > 0U &&
              r.max_hazard_primitives > 0U &&
              r.max_hazard_projection_entries > 0U &&
              r.max_body_query_samples > 0U &&
              r.max_motion_query_samples >= 2U &&
              r.max_risk_query_operations > 0U &&
              r.max_stable_ids > 0U && r.max_evidence_links > 0U,
          "PAR-RES-023 resource limits must be positive", &validation);
  const bool frame_identity_budget_safe =
      r.max_points_per_frame <=
          (std::numeric_limits<size_t>::max() - 1U) / 2U &&
      r.max_stable_ids >= 1U + 2U * r.max_points_per_frame;
  Require(frame_identity_budget_safe,
          "PAR-RES-023 stable-id budget cannot atomically admit one maximum input frame",
          &validation);
  Require(r.max_evidence_links >= r.max_points_per_frame,
          "PAR-RES-023 evidence-link budget is smaller than one maximum input frame",
          &validation);
  Require(r.max_hazard_primitives >= r.max_events,
          "PAR-RES-023 hazard-primitive budget is smaller than the event ledger",
          &validation);
  Require(r.max_hazard_projection_entries >= r.max_tiles,
          "PAR-RES-023 hazard-projection budget is smaller than the voxel ledger",
          &validation);
  Require(r.mixture_max_components > 0 &&
              static_cast<size_t>(r.mixture_max_components) <= r.max_events,
          "PAR-RES-023 Gaussian-mixture cap exceeds the event resource envelope",
          &validation);
  constexpr size_t kBoostSobolMaximumDimension = 3667U;
  const bool sobol_dimension_safe =
      r.max_controls_per_gdu <= kBoostSobolMaximumDimension / 6U &&
      r.max_interfaces <= kBoostSobolMaximumDimension / 6U &&
      r.max_controls_per_gdu + r.max_interfaces <=
          kBoostSobolMaximumDimension / 6U;
  Require(sobol_dimension_safe,
          "PAR-RES-023 worst-case active field exceeds the Sobol dimension limit",
          &validation);
  Require(!r.robot_geometry_revision.empty(), "PAR-RES-024 robot revision is required", &validation);
  const RobotGeometry& robot = config.robot_geometry;
  Require(robot.revision == r.robot_geometry_revision,
          "PAR-RES-024 robot geometry revision mismatch", &validation);
  Require(!robot.body_frame.empty() && !robot.spheres.empty(),
          "PAR-RES-024 robot geometry requires a body frame and sphere union",
          &validation);
  Require(robot.spheres.size() <= r.max_body_query_samples,
          "PAR-RES-024 robot sphere count exceeds its resource limit",
          &validation);
  Eigen::Vector3d body_min = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::infinity());
  Eigen::Vector3d body_max = Eigen::Vector3d::Constant(
      -std::numeric_limits<double>::infinity());
  bool robot_geometry_finite = !robot.spheres.empty();
  for (const RobotSphere& sphere : robot.spheres) {
    const bool sphere_valid = sphere.center_body.allFinite() &&
        std::isfinite(sphere.radius_m) && sphere.radius_m > 0.0;
    robot_geometry_finite = robot_geometry_finite && sphere_valid;
    if (!sphere_valid) continue;
    body_min = body_min.cwiseMin(
        sphere.center_body - Eigen::Vector3d::Constant(sphere.radius_m));
    body_max = body_max.cwiseMax(
        sphere.center_body + Eigen::Vector3d::Constant(sphere.radius_m));
  }
  Require(robot_geometry_finite,
          "PAR-RES-024 robot sphere union contains invalid values", &validation);
  if (robot_geometry_finite && r.occupancy_voxel_m > 0.0 &&
      r.max_body_query_samples > 0U) {
    const long double spacing =
        0.5L * static_cast<long double>(r.occupancy_voxel_m);
    long double bounding_samples = 1.0L;
    for (int axis = 0; axis < 3; ++axis) {
      const long double extent = static_cast<long double>(
          body_max[axis] - body_min[axis]);
      bounding_samples *= std::max(1.0L, std::ceil(extent / spacing));
    }
    Require(std::isfinite(bounding_samples) &&
                bounding_samples <=
                    static_cast<long double>(r.max_body_query_samples),
            "PAR-RES-024 robot body query grid exceeds its resource limit",
            &validation);
  }
  Require(r.query_ttl_s > 0.0 && r.query_ttl_s <= r.map_ttl_s,
          "PAR-RES-025 requires 0 < query TTL <= map TTL", &validation);
  Require(r.association_max_candidates >= 1 &&
              r.association_max_candidates <= 3,
          "PAR-RES-026 candidate limit must be in [1,3]", &validation);
  Require(r.min_patches_per_gdu >= 3 &&
              r.max_patches_per_gdu >= r.min_patches_per_gdu &&
              r.max_patches_per_gdu <= 100,
          "PAR-RES-029 patch count must stay in [3,100]", &validation);
  Require(r.pose_control_min_patches >= 6 &&
              r.pose_control_min_patches <= r.max_patches_per_gdu,
          "PAR-RES-030 pose-control support is invalid", &validation);
  Require(std::isfinite(r.pose_control_exclusion_radius_m) &&
              r.pose_control_exclusion_radius_m >=
                  r.graph_neighbor_radius_m,
          "PAR-RES-030 GDU-local pose-control exclusion radius is invalid",
          &validation);
  Require(r.source_missing_consecutive_limit >= 1,
          "PAR-RES-031 SOURCE_MISSING limit must be positive", &validation);
  Require(!r.extrinsic.revision.empty() && !r.extrinsic.digest.empty() &&
              !r.extrinsic.online_estimation_enabled,
          "PAR-RES-027 requires fixed revisioned extrinsic and disabled online estimation", &validation);
  Require(r.extrinsic.R_IL.allFinite() && r.extrinsic.t_IL.allFinite() &&
              r.extrinsic.covariance.allFinite() &&
              (r.extrinsic.R_IL.transpose() * r.extrinsic.R_IL -
               Eigen::Matrix3d::Identity()).norm() <= 1.0e-8 &&
              std::abs(r.extrinsic.R_IL.determinant() - 1.0) <= 1.0e-8 &&
              ValidateSymmetricPsd(r.extrinsic.covariance,
                                   n.psd_eigenvalue_tolerance),
          "PAR-RES-027 extrinsic transform/covariance invalid", &validation);
  Require(r.ctmc_max_substep_s > 0.0 && r.ctmc_max_substep_s <= r.max_update_gap_s,
          "PAR-RES-028 CTMC substep must lie in (0,max_update_gap]", &validation);

  Require(n.probability_floor > 0.0 && n.probability_floor < 0.1,
          "PAR-NUM probability floor invalid", &validation);
  Require(n.pou_denominator_min > 0.0 &&
              n.psd_eigenvalue_tolerance > 0.0 &&
              n.cholesky_jitter_initial > 0.0 &&
              n.cholesky_jitter_multiplier > 1.0 &&
              n.cholesky_jitter_attempts >= 0 &&
              n.cholesky_jitter_attempts <= 32 &&
              n.solve_residual_tolerance > 0.0 &&
              n.normalization_tolerance > 0.0 &&
              n.tie_tolerance > 0.0,
          "PAR-NUM numerical stability values invalid", &validation);
  Require(p.hard_clearance_m >= 0.0 && p.hard_collision_probability > 0.0 &&
              p.hard_collision_probability < 1.0 &&
              p.hard_encounter_probability > 0.0 &&
              p.hard_encounter_probability <= p.hard_collision_probability &&
              p.path_risk_upper_bound > 0.0 &&
              p.path_risk_upper_bound < 1.0 && p.unknown_weight >= 0.0 &&
              p.stale_weight >= 0.0 && p.probability_epsilon > 0.0 &&
              p.probability_epsilon < 1.0 && p.risk_saturation_nats > 0.0 &&
              p.maximum_soft_cost > 0.0 && p.motion_sample_spacing_m > 0.0 &&
              p.motion_sample_interval_s > 0.0 && !p.revision.empty(),
          "policy values invalid", &validation);

  Require(c.independent_point_sigma_m > 0.0 &&
                c.unobservable_residual_sigma_m > c.independent_point_sigma_m &&
                std::isfinite(c.unobservable_residual_sigma_m) &&
                c.contamination_probability >= 0.0 &&
                c.contamination_probability < 1.0 && c.student_t_dof > 2.0 &&
                c.student_t_scale >= 1.0 && c.pose_ou_tau_s > 0.0,
            "PAR-CAL-SENSOR/SLAM invalid", &validation);
  for (double rate : c.ctmc_rates) Require(std::isfinite(rate) && rate > 0.0,
      "all five PAR-CAL-DYN rates must be positive", &validation);
  Require(c.active_acceleration_psd >= 0.0 && c.changed_random_walk_psd >= 0.0 &&
                c.active_birth_velocity_variance > 0.0 &&
                c.active_birth_displacement_variance > 0.0 &&
                c.changed_birth_displacement_variance > 0.0,
            "PAR-CAL-DYN continuous values invalid", &validation);
  Require(c.link_probability_on > c.cut_probability_on && c.link_probability_on < 1.0 &&
                c.cut_probability_on > 0.0,
            "PAR-CAL-REF LINK/CUT thresholds invalid", &validation);
  Require(c.reference_stability_sigma > 0.0 &&
              std::isfinite(c.reference_stability_sigma) &&
              c.reference_epoch_common_bias_sigma_m > 0.0 &&
              c.geometry_min_eigengap_ratio > 0.0 &&
              c.geometry_min_eigengap_ratio < 1.0 &&
              c.pose_control_max_condition_number > 1.0 &&
              c.pose_control_huber_scale_m > 0.0 &&
              c.geometry_softmax_weights.allFinite() &&
              c.geometry_softmax_bias.allFinite() &&
              c.association_weights.allFinite() &&
              std::isfinite(c.association_candidate_bias) &&
              std::isfinite(c.association_missing_bias) &&
              std::isfinite(c.association_other_bias) &&
              c.association_entropy_max > 0.0 &&
              std::isfinite(c.association_entropy_max) &&
              std::isfinite(c.reference_link_logit_bias) &&
              std::isfinite(c.reference_link_distance_weight) &&
              std::isfinite(c.reference_link_projector_weight),
          "PAR-CAL-REF/ASSOC finite-model parameters invalid", &validation);
  Require(c.reference_confounding_min_information_ratio > 0.0 &&
              c.reference_confounding_min_information_ratio <= 1.0 &&
              std::isfinite(c.reference_confounding_min_information_ratio),
          "PAR-CAL-REF confounding information ratio must lie in (0,1]",
          &validation);
    const bool interface_initial_valid = std::all_of(
        c.interface_initial_link_probability.begin(),
        c.interface_initial_link_probability.end(),
        [](double probability) {
          return std::isfinite(probability) && probability >= 0.0;
        });
    const double interface_initial_sum = c.interface_initial_link_probability[0] +
        c.interface_initial_link_probability[1] + c.interface_initial_link_probability[2];
    Require(interface_initial_valid &&
                std::abs(interface_initial_sum - 1.0) <= n.normalization_tolerance,
            "initial interface coherence probabilities must sum to one", &validation);
  for (int row = 0; row < 3; ++row) {
      Require((c.interface_coherence_transition.row(row).array() >= 0.0).all() &&
                  std::abs(c.interface_coherence_transition.row(row).sum() - 1.0) <=
                      n.normalization_tolerance,
              "interface coherence transition rows must be stochastic", &validation);
  }
  Require(c.interface_coherent_rho >= 0.0 && c.interface_coherent_rho < 1.0 &&
                c.interface_relative_rho >= 0.0 &&
                c.interface_relative_rho <= 1.0 &&
                c.interface_coherent_process_variance > 0.0 &&
                c.interface_relative_process_variance > 0.0 &&
                c.interface_unresolved_process_variance > 0.0,
            "interface coherence dynamics invalid", &validation);
  Require(c.visibility_min_range_m >= 0.0 &&
                c.visibility_max_range_m > c.visibility_min_range_m,
            "visibility range calibration invalid", &validation);
  Require(c.nb_dispersion > 0.0 && c.nb_clutter_mean >= 0.0 &&
                c.nb_occupied_mean_scale > 0.0 && c.new_occupancy_prior > 0.0 &&
                c.new_occupancy_prior < 1.0 && c.visibility_exposure_scale > 0.0 &&
                c.occupancy_interval_alpha > 0.0 && c.occupancy_interval_alpha < 1.0 &&
                c.novel_occupancy_birth_rate_hz > 0.0 &&
                c.novel_occupancy_death_rate_hz > 0.0 &&
                c.reference_persistence_birth_rate_hz > 0.0 &&
                c.reference_persistence_death_rate_hz > 0.0 &&
                c.unknown_occupancy_birth_rate_hz > 0.0 &&
                c.unknown_occupancy_death_rate_hz > 0.0 &&
                std::isfinite(c.novel_occupancy_birth_rate_hz) &&
                std::isfinite(c.novel_occupancy_death_rate_hz) &&
                std::isfinite(c.reference_persistence_birth_rate_hz) &&
                std::isfinite(c.reference_persistence_death_rate_hz) &&
                std::isfinite(c.unknown_occupancy_birth_rate_hz) &&
                std::isfinite(c.unknown_occupancy_death_rate_hz) &&
                c.occupancy_max_speed_mps > 0.0 &&
                std::isfinite(c.occupancy_max_speed_mps) &&
                c.hazard_sweep_speed_upper_mps >= c.occupancy_max_speed_mps &&
                std::isfinite(c.hazard_sweep_speed_upper_mps),
            "PAR-CAL-OCC invalid", &validation);
  Require(c.reference_beta_tempering > 0.0 &&
              c.reference_beta_tempering <= 1.0 &&
              c.reference_free_threshold > 0.0 &&
              c.reference_free_threshold < c.reference_occupied_threshold &&
              c.reference_occupied_threshold < 1.0,
          "PAR-CAL-OCC reference knowledge thresholds invalid", &validation);
  Require(c.task_displacement_scale_m > 0.0 &&
                c.task_velocity_scale_mps > 0.0 &&
                c.task_velocity_required_duration_s > 0.0 &&
                c.task_interface_jump_scale_m > 0.0 &&
                c.task_interface_metric.allFinite() &&
                ValidateSymmetricPsd(c.task_interface_metric,
                                     n.psd_eigenvalue_tolerance) &&
                c.task_interface_metric.trace() > n.psd_eigenvalue_tolerance &&
                c.detection_probability_on > 0.0 &&
                c.detection_probability_on < 1.0 &&
                c.clearing_probability_on >= 0.0 &&
                c.clearing_probability_on < c.detection_probability_on &&
                c.event_false_alarm_probability > 0.0 &&
                c.event_false_alarm_probability < 1.0 &&
                c.target_power > c.event_false_alarm_probability &&
                c.target_power < 1.0 &&
                c.mde_bootstrap_inflation >= 1.0 &&
                std::isfinite(c.mde_bootstrap_inflation) &&
                c.detection_min_effective_spatial_support >= 1.0 &&
                c.detection_min_effective_temporal_support >= 1.0 &&
                c.qmc_error_upper >= 0.0 &&
                c.qmc_error_upper < 1.0 &&
                c.prediction_model_error_m >= 0.0 &&
                c.pose_cross_covariance_bound_m2 >= 0.0 &&
                c.ctmc_halving_error_upper_m >= 0.0,
            "PAR-CAL-DETECT invalid", &validation);
  double maximum_temporal_effective_support = 0.0;
  if (r.fixed_lag_frames >= 1 && r.max_update_gap_s > 0.0 &&
      c.pose_ou_tau_s > 0.0) {
    double correlated_sum = 0.0;
    for (int left = 0; left < r.fixed_lag_frames; ++left)
      for (int right = 0; right < r.fixed_lag_frames; ++right)
        correlated_sum += std::exp(
            -std::abs(left - right) * r.max_update_gap_s /
            c.pose_ou_tau_s);
    maximum_temporal_effective_support =
        static_cast<double>(r.fixed_lag_frames * r.fixed_lag_frames) /
        correlated_sum;
  }
  Require(c.detection_min_effective_temporal_support <=
              maximum_temporal_effective_support + n.tie_tolerance,
          "PAR-CAL-DETECT temporal support threshold is unreachable within the fixed lag",
          &validation);
  if (c.validated) {
    Require(c.version != "UNSET" && !c.version.empty() && !c.data_digest.empty() &&
                c.data_digest != "UNSET",
            "validated calibration needs a version and data digest", &validation);
  } else {
    Require(c.version == "UNSET", "unvalidated calibration version must be UNSET", &validation);
  }
  validation.valid = validation.errors.empty();
  return validation;
}

ConfigValidation ValidateFrozenV11Profile(const V1Config& config) {
  ConfigValidation validation;
  const ResourceConfig& r = config.resource;
  const CalibrationConfig& c = config.calibration;
  const NumericalConfig& n = config.numerical;
  // These are minimum evidence requirements, not fixed scan counts. Real
  // sensors may need a longer reference; ValidateConfig checks consistency.
  Require(r.reference_min_frames >= 6U &&
              r.reference_min_surfel_exposures >= 6U &&
              r.reference_min_duration_s >= 0.5,
          "FREEZE-V1.1 reference below minimum exposure requirements", &validation);
  Require(r.fixed_lag_frames == 3 && r.mixture_max_components == 24,
          "FREEZE-V1.1 posterior resource profile mismatch", &validation);
  Require(r.surfel_seed_spacing_m == 0.10 &&
              r.graph_neighbor_radius_m == 0.25 &&
              r.graph_max_neighbors == 36 &&
              r.control_spacing_m == 0.20 &&
              r.pou_support_radius_m == 0.30 &&
              r.interface_scales_m == std::vector<double>({0.20, 0.40}),
          "FREEZE-V1.1 reference geometry profile mismatch", &validation);
  Require(r.occupancy_voxel_m == 0.10 &&
              r.prediction_horizons_s ==
                  std::vector<double>({0.0, 0.5, 1.0, 2.0}) &&
              r.prediction_samples == 256,
          "FREEZE-V1.1 prediction profile mismatch", &validation);
  Require(r.max_gdu == 64U && r.min_controls_per_gdu == 6U &&
              r.max_controls_per_gdu == 16U &&
              r.max_interfaces_per_gdu == 1U && r.max_interfaces == 64U,
          "FREEZE-V1.1 field resource profile mismatch", &validation);
  Require(r.association_max_candidates == 3 &&
              r.min_patches_per_gdu == 3 &&
              r.max_patches_per_gdu == 100 &&
              r.pose_control_min_patches == 20 &&
              r.pose_control_exclusion_radius_m == 5.0 &&
              r.source_missing_consecutive_limit == 3,
          "FREEZE-V1.1 measurement resource profile mismatch", &validation);
  Require(r.max_update_gap_s == 15.0 &&
              r.ctmc_max_substep_s == 0.05 && r.map_ttl_s == 15.0 &&
              r.query_ttl_s == 0.5 && r.deadline_ms == 15000.0,
          "FREEZE-V1.1 time profile mismatch", &validation);
  Require(n.probability_floor == 1.0e-12 &&
              n.pou_denominator_min == 1.0e-12 &&
              n.psd_eigenvalue_tolerance == 1.0e-10 &&
              n.cholesky_jitter_initial == 1.0e-12 &&
              n.cholesky_jitter_multiplier == 10.0 &&
              n.cholesky_jitter_attempts == 6 &&
              n.solve_residual_tolerance == 1.0e-8 &&
              n.normalization_tolerance == 1.0e-10 &&
              n.tie_tolerance == 1.0e-12,
          "FREEZE-V1.1 numerical profile mismatch", &validation);
  Require(c.pose_control_max_condition_number == 1.0e8 &&
              c.pose_control_huber_scale_m == 0.02,
          "FREEZE-V1.1 pose-control numerical profile mismatch", &validation);
  validation.valid = validation.errors.empty();
  return validation;
}

std::string CanonicalConfigDigest(const V1Config& config) {
  CanonicalBytes bytes;
  // This tag represents the exact algorithm_v1.yaml constant set verified by
  // VerifyAlgorithmConstants.  It must change with any wire or algorithm
  // constant change so the typed canonical digest covers that frozen source.
  bytes.AddString("algorithm-v1-observable-schema3-detector-only-v3");
  const ResourceConfig& r = config.resource;
  bytes.AddString(r.session_id); bytes.AddString(r.world_frame);
  for (int i = 0; i < 3; ++i) bytes.AddDouble(r.grid_origin_m[i]);
  bytes.AddDouble(r.frame_sync_tolerance_s); bytes.AddDouble(r.raw_frame_buffer_s);
  bytes.AddU8(r.upstream_registered_cloud_deskew_verified ? 1U : 0U);
  bytes.AddU64(r.max_points_per_frame);
  bytes.AddU64(r.max_raw_buffer_points);
  bytes.AddI64(r.reference_min_frames);
  bytes.AddI64(r.reference_min_surfel_exposures);
  bytes.AddDouble(r.reference_min_duration_s);
  bytes.AddU8(r.bounded_reference_search ? 1U : 0U);
  bytes.AddDouble(r.max_update_gap_s); bytes.AddI64(r.fixed_lag_frames);
  bytes.AddI64(r.mixture_max_components); bytes.AddDouble(r.other_mass_alarm);
  bytes.AddDouble(r.surfel_seed_spacing_m); bytes.AddDouble(r.graph_neighbor_radius_m);
  bytes.AddI64(r.graph_max_neighbors); bytes.AddDouble(r.control_spacing_m);
  bytes.AddDouble(r.pou_support_radius_m); AddVector(r.interface_scales_m, &bytes);
  bytes.AddDouble(r.occupancy_voxel_m); AddVector(r.prediction_horizons_s, &bytes);
  bytes.AddI64(r.prediction_samples);
  bytes.AddDouble(r.deadline_ms);
  bytes.AddU64(r.max_gdu); bytes.AddU64(r.min_controls_per_gdu);
  bytes.AddU64(r.max_controls_per_gdu);
  bytes.AddU64(r.max_interfaces_per_gdu);
  bytes.AddU64(r.max_interfaces); bytes.AddU64(r.max_events); bytes.AddU64(r.max_tiles);
  bytes.AddU64(r.max_hazard_primitives);
  bytes.AddU64(r.max_hazard_projection_entries);
  bytes.AddU64(r.max_body_query_samples);
  bytes.AddU64(r.max_motion_query_samples);
  bytes.AddU64(r.max_risk_query_operations);
  bytes.AddU64(r.max_stable_ids);
  bytes.AddU64(r.max_evidence_links);
  bytes.AddString(r.robot_geometry_revision); bytes.AddDouble(r.map_ttl_s);
  bytes.AddDouble(r.query_ttl_s); bytes.AddI64(r.association_max_candidates);
  bytes.AddI64(r.min_patches_per_gdu); bytes.AddI64(r.max_patches_per_gdu);
  bytes.AddI64(r.pose_control_min_patches);
  bytes.AddDouble(r.pose_control_exclusion_radius_m);
  bytes.AddU8(r.upstream_stationary_pose_freeze_verified ? 1U : 0U);
  bytes.AddI64(r.source_missing_consecutive_limit);
  bytes.AddString(r.extrinsic.revision); bytes.AddString(r.extrinsic.digest);
  for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column)
    bytes.AddDouble(r.extrinsic.R_IL(row, column));
  for (int i = 0; i < 3; ++i) bytes.AddDouble(r.extrinsic.t_IL[i]);
  for (int row = 0; row < 6; ++row) for (int column = 0; column < 6; ++column)
    bytes.AddDouble(r.extrinsic.covariance(row, column));
  bytes.AddU8(r.extrinsic.online_estimation_enabled ? 1U : 0U);
  bytes.AddDouble(r.ctmc_max_substep_s);

  const CalibrationConfig& c = config.calibration;
  bytes.AddU8(c.validated ? 1U : 0U); bytes.AddString(c.version); bytes.AddString(c.data_digest);
  bytes.AddDouble(c.independent_point_sigma_m);
  bytes.AddDouble(c.unobservable_residual_sigma_m);
  bytes.AddDouble(c.contamination_probability);
  bytes.AddDouble(c.student_t_dof); bytes.AddDouble(c.student_t_scale);
  bytes.AddDouble(c.pose_ou_tau_s); bytes.AddDouble(c.pose_cross_covariance_bound_m2);
  for (double rate : c.ctmc_rates) bytes.AddDouble(rate);
  bytes.AddDouble(c.active_acceleration_psd); bytes.AddDouble(c.changed_random_walk_psd);
  bytes.AddDouble(c.active_birth_velocity_variance);
  bytes.AddDouble(c.active_birth_displacement_variance);
  bytes.AddDouble(c.changed_birth_displacement_variance);
  bytes.AddDouble(c.ctmc_halving_error_upper_m); bytes.AddDouble(c.reference_stability_sigma);
  bytes.AddDouble(c.reference_epoch_common_bias_sigma_m);
  bytes.AddDouble(c.geometry_min_eigengap_ratio);
  bytes.AddDouble(c.pose_control_max_condition_number);
  bytes.AddDouble(c.pose_control_huber_scale_m);
  bytes.AddDouble(c.reference_confounding_min_information_ratio);
  bytes.AddDouble(c.reference_link_logit_bias);
  bytes.AddDouble(c.reference_link_distance_weight);
  bytes.AddDouble(c.reference_link_projector_weight);
  for (double probability : c.interface_initial_link_probability) bytes.AddDouble(probability);
  for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column)
    bytes.AddDouble(c.interface_coherence_transition(row, column));
  bytes.AddDouble(c.interface_coherent_rho); bytes.AddDouble(c.interface_relative_rho);
  bytes.AddDouble(c.interface_coherent_process_variance);
  bytes.AddDouble(c.interface_relative_process_variance);
  bytes.AddDouble(c.interface_unresolved_process_variance);
  bytes.AddDouble(c.link_probability_on); bytes.AddDouble(c.cut_probability_on);
  for (int row = 0; row < 4; ++row) for (int column = 0; column < 4; ++column)
    bytes.AddDouble(c.geometry_softmax_weights(row, column));
  for (int i = 0; i < 4; ++i) bytes.AddDouble(c.geometry_softmax_bias[i]);
  for (int i = 0; i < 7; ++i) bytes.AddDouble(c.association_weights[i]);
  bytes.AddDouble(c.association_candidate_bias); bytes.AddDouble(c.association_missing_bias);
  bytes.AddDouble(c.association_other_bias); bytes.AddDouble(c.association_entropy_max);
  bytes.AddDouble(c.visibility_min_range_m); bytes.AddDouble(c.visibility_max_range_m);
  bytes.AddDouble(c.nb_dispersion); bytes.AddDouble(c.nb_clutter_mean);
  bytes.AddDouble(c.nb_occupied_mean_scale); bytes.AddDouble(c.new_occupancy_prior);
  bytes.AddDouble(c.novel_occupancy_birth_rate_hz);
  bytes.AddDouble(c.novel_occupancy_death_rate_hz);
  bytes.AddDouble(c.reference_persistence_birth_rate_hz);
  bytes.AddDouble(c.reference_persistence_death_rate_hz);
  bytes.AddDouble(c.unknown_occupancy_birth_rate_hz);
  bytes.AddDouble(c.unknown_occupancy_death_rate_hz);
  bytes.AddDouble(c.visibility_exposure_scale); bytes.AddDouble(c.occupancy_interval_alpha);
  bytes.AddDouble(c.reference_beta_tempering);
  bytes.AddDouble(c.reference_free_threshold); bytes.AddDouble(c.reference_occupied_threshold);
  bytes.AddDouble(c.task_displacement_scale_m);
  bytes.AddDouble(c.task_velocity_scale_mps);
  bytes.AddDouble(c.task_velocity_required_duration_s);
  bytes.AddDouble(c.task_interface_jump_scale_m);
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      bytes.AddDouble(c.task_interface_metric(row, column));
  bytes.AddDouble(c.detection_probability_on);
  bytes.AddDouble(c.clearing_probability_on);
  bytes.AddDouble(c.event_false_alarm_probability);
  bytes.AddDouble(c.target_power);
  bytes.AddDouble(c.mde_bootstrap_inflation);
  bytes.AddDouble(c.detection_min_effective_spatial_support);
  bytes.AddDouble(c.detection_min_effective_temporal_support);
  bytes.AddDouble(c.qmc_error_upper);
  bytes.AddDouble(c.prediction_model_error_m);
  bytes.AddDouble(c.occupancy_max_speed_mps);
  bytes.AddDouble(c.hazard_sweep_speed_upper_mps);

  const NumericalConfig& n = config.numerical;
  bytes.AddDouble(n.probability_floor); bytes.AddDouble(n.pou_denominator_min);
  bytes.AddDouble(n.psd_eigenvalue_tolerance); bytes.AddDouble(n.cholesky_jitter_initial);
  bytes.AddDouble(n.cholesky_jitter_multiplier); bytes.AddI64(n.cholesky_jitter_attempts);
  bytes.AddDouble(n.solve_residual_tolerance); bytes.AddDouble(n.normalization_tolerance);
  bytes.AddDouble(n.tie_tolerance);
  const PolicyConfig& p = config.policy;
  bytes.AddString(p.revision); bytes.AddDouble(p.hard_clearance_m);
  bytes.AddDouble(p.hard_collision_probability);
  bytes.AddDouble(p.hard_encounter_probability);
  bytes.AddDouble(p.path_risk_upper_bound);
  bytes.AddU8(p.unknown_is_blocked ? 1U : 0U); bytes.AddDouble(p.unknown_weight);
  bytes.AddDouble(p.stale_weight); bytes.AddDouble(p.probability_epsilon);
  bytes.AddDouble(p.risk_saturation_nats); bytes.AddDouble(p.maximum_soft_cost);
  bytes.AddDouble(p.motion_sample_spacing_m); bytes.AddDouble(p.motion_sample_interval_s);
  bytes.AddString(config.robot_geometry.revision);
  bytes.AddString(config.robot_geometry.body_frame);
  bytes.AddU64(config.robot_geometry.spheres.size());
  for (const RobotSphere& sphere : config.robot_geometry.spheres) {
    for (int i = 0; i < 3; ++i) bytes.AddDouble(sphere.center_body[i]);
    bytes.AddDouble(sphere.radius_m);
  }
  bytes.AddString(config.software_revision);
  return HexDigest(Sha256(bytes.bytes()));
}

V1Config SyntheticKnownConfig(const std::string& session_id) {
  V1Config config;
  config.resource.session_id = session_id;
  config.resource.robot_geometry_revision = "synthetic-sphere-r020-v1";
  config.resource.extrinsic.revision = "synthetic-identity-v1";
  config.resource.extrinsic.digest = "synthetic-identity-v1";
  config.robot_geometry.revision = "synthetic-sphere-r020-v1";
  config.robot_geometry.body_frame = "synthetic_body";
  RobotSphere sphere; sphere.radius_m = 0.20;
  config.robot_geometry.spheres.push_back(sphere);
  config.calibration.validated = true;
  config.calibration.version = "synthetic-known-v1";
  config.calibration.data_digest = "analytic-fixture-manifest-required";
  config.software_revision = "test-build";
  config.config_digest = CanonicalConfigDigest(config);
  return config;
}

}  // namespace v1
}  // namespace deform_monitor_v3
