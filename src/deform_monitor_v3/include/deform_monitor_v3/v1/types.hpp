#ifndef DEFORM_MONITOR_V3_V1_TYPES_HPP
#define DEFORM_MONITOR_V3_V1_TYPES_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/time.h>

#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace deform_monitor_v3 {
namespace v1 {

// Numeric values are frozen by IMPLEMENTATION_FREEZE.md section 6.3.
enum class MonitoringPhase : uint8_t {
  MAPPING_CONTEXT = 0,
  MONITOR_REFERENCE = 1,
  MONITOR_UPDATE = 2,
  CLOSED = 3
};

enum class Validity : uint8_t {
  VALID = 0,
  INCONCLUSIVE = 1,
  UNKNOWN = 2,
  STALE = 3,
  INVALID = 4
};

// Wire enum values are schema identifiers, not an ordering contract.  All
// conservative joins therefore name the semantic precedence explicitly.
inline int ValiditySeverity(Validity validity) {
  switch (validity) {
    case Validity::VALID: return 0;
    case Validity::INCONCLUSIVE: return 1;
    case Validity::UNKNOWN: return 2;
    case Validity::STALE: return 3;
    case Validity::INVALID: return 4;
  }
  return 4;
}

inline Validity CombineValidity(Validity first, Validity second) {
  return ValiditySeverity(first) >= ValiditySeverity(second) ? first : second;
}

enum class StructuralState : uint8_t { U = 0, A = 1, C = 2 };
enum class ReferenceEdgeState : uint8_t { LINK = 0, CUT = 1, AMBIGUOUS = 2 };
enum class FieldCoherence : uint8_t {
  COHERENT = 0,
  RELATIVE_MOTION = 1,
  UNRESOLVED = 2
};
enum class InterfaceJointState : uint8_t {
  CUT_NOT_APPLICABLE = 0,
  LINK_COHERENT = 1,
  LINK_RELATIVE_MOTION = 2,
  LINK_UNRESOLVED = 3
};
enum class KnowledgeState : uint8_t { UNKNOWN = 0, FREE = 1, OCCUPIED = 2 };
enum class AssociationSemantics : uint8_t {
  SOURCE_MISSING = 0,
  NOVEL = 1,
  CLUTTER = 2,
  OTHER = 3
};
enum class HardState : uint8_t { PASS = 0, BLOCK = 1, UNKNOWN = 2 };
enum class DetectionAdmission : uint8_t {
  DETECTED_AT_SCALE = 0,
  CLEARED_AT_SCALE = 1,
  INCONCLUSIVE_LOW_POWER = 2,
  INCONCLUSIVE_ASSOCIATION = 3,
  INCONCLUSIVE_REFERENCE = 4,
  INCONCLUSIVE_SLAM = 5,
  INCONCLUSIVE_UNCALIBRATED = 6,
  INCONCLUSIVE_SELECTION_TRIGGERED = 7,
  INVALID_INPUT_OR_MODEL = 8
};
// The four supports are a deterministic partition of one accepted frame.
// A point/evidence id belongs to exactly one role; all four roles retain the
// frame nuisance correlation id because they still share the same pose error.
enum class PointRole : uint8_t {
  POSE_CONTROL = 0,
  ASSOCIATION = 1,
  MEASUREMENT = 2,
  OCCUPANCY = 3
};
enum class TriggerSource : uint8_t {
  HUMAN_OR_EXTERNAL_TASK = 0,
  ALGORITHM_RESIDUAL = 1,
  // A task begins a new stationary observation dwell and explicitly forbids
  // pre-trigger mapping context from becoming deformation-reference evidence.
  EXTERNAL_TASK_CURRENT_DWELL = 2
};
enum class GeometryHypothesis : uint8_t {
  SURFACE_LIKE = 0,
  CURVE_LIKE = 1,
  ISOTROPIC = 2,
  OTHER = 3
};
enum class HazardSemantics : uint8_t {
  STRUCTURAL_DEFORMATION = 0,
  SOURCE_MISSING = 1,
  NOVEL_OCCUPANCY = 2,
  CLUTTER_PROTECTIVE = 3,
  OTHER_ENVELOPE = 4,
  REFERENCE_PERSISTENCE = 5
};

enum ApproximationFlag : uint32_t {
  APPROX_NONE = 0U,
  APPROX_POSE_CROSS_COVARIANCE_BOUND = 1U << 0U,
  APPROX_CTMC_ENDPOINT_RESET = 1U << 1U,
  APPROX_CTMC_COLLAPSED_MULTIJUMP = 1U << 2U,
  APPROX_QMC_INTERVAL = 1U << 3U,
  APPROX_MIXTURE_PRUNED_TO_OTHER = 1U << 4U,
  APPROX_CONNECTIVITY_COHERENCE_CONFOUNDING = 1U << 5U,
  // Bit 6 is reserved.  V1.1 rejects cross-GDU jump coordinates instead of
  // applying the removed shared-interface covariance-intersection model.
  APPROX_EXTRINSIC_CROSS_GDU_COVARIANCE_OMITTED = 1U << 7U,
  APPROX_CROSS_GDU_ASSOCIATION_TO_ENVELOPE = 1U << 8U,
  // Canonical per-cause QMC ranks are shared over fixed prediction nodes.
  // The path event uses the frozen comonotone/common-rank maximum envelope,
  // not a time-marginal sum and not an unimplemented sample-wise OR.
  APPROX_PATH_CAUSE_COMMON_RANK = 1U << 9U
};

enum HazardCoverageFlag : uint32_t {
  HAZARD_COVERAGE_DEFORMATION = 1U << 0U,
  HAZARD_COVERAGE_INTERFACE_MOTION = 1U << 1U,
  HAZARD_COVERAGE_GEOMETRIC_OCCUPANCY = 1U << 2U,
  HAZARD_COVERAGE_BODY_COLLISION_OVERLAY = 1U << 3U
};

constexpr uint32_t kV1HazardCoverageMask =
    HAZARD_COVERAGE_DEFORMATION |
    HAZARD_COVERAGE_INTERFACE_MOTION |
    HAZARD_COVERAGE_GEOMETRIC_OCCUPANCY |
    HAZARD_COVERAGE_BODY_COLLISION_OVERLAY;

enum class ReasonCode : uint16_t {
  NONE = 0,
  INPUT_SCHEMA_INVALID = 1,
  INPUT_POSE_COVARIANCE_INVALID = 2,
  INPUT_EXTRINSIC_STATE_UNAVAILABLE = 3,
  INPUT_FRAME_MISMATCH = 4,
  INPUT_TIME_INVALID = 5,
  INPUT_DUPLICATE = 6,
  INPUT_DESKEW_UNRESOLVED = 7,
  REFERENCE_NOT_FROZEN = 20,
  REFERENCE_INSUFFICIENT_EXPOSURE = 21,
  REFERENCE_RESOURCE_LIMIT = 22,
  PRE_REFERENCE_CHANGE_UNOBSERVABLE = 23,
  REFERENCE_SELF_CONSISTENCY_FAILED = 24,
  OUTSIDE_SUPPORT = 40,
  INSUFFICIENT_SUPPORT = 41,
  ASSOCIATION_AMBIGUOUS = 42,
  DEFORMATION_REFERENCE_AMBIGUOUS = 43,
  CALIBRATION_UNSET = 60,
  OUTSIDE_CALIBRATED_RANGE = 61,
  INVALID_NUMERICAL = 62,
  RESOURCE_LIMIT = 63,
  DEADLINE_EXCEEDED = 64,
  UNKNOWN_PREDICTION = 65,
  SNAPSHOT_STALE = 66,
  QUERY_CONTRACT_INVALID = 67,
  INVALID_ID_COLLISION = 68,
  INCONCLUSIVE_SELECTION_TRIGGERED = 69,
  // Value 70 is reserved after removal of the cross-GDU interface fusion.
  INCONCLUSIVE_EXTRINSIC_COVARIANCE = 71,
  COMMON_MODE_AMBIGUOUS = 72,
  CROSS_COVARIANCE_UNIDENTIFIED = 73,
  POSE_CONTROL_INSUFFICIENT = 74,
  SOURCE_MISSING_LIMIT = 75,
  INTERFACE_UNIDENTIFIABLE = 76
};

struct StableId {
  std::array<uint8_t, 16> bytes{};

  bool operator==(const StableId& rhs) const { return bytes == rhs.bytes; }
  bool operator!=(const StableId& rhs) const { return !(*this == rhs); }
  bool operator<(const StableId& rhs) const { return bytes < rhs.bytes; }
  bool empty() const {
    for (uint8_t value : bytes) if (value != 0U) return false;
    return true;
  }
};

struct StableIdHash {
  size_t operator()(const StableId& id) const {
    size_t value = 1469598103934665603ULL;
    for (uint8_t byte : id.bytes) {
      value ^= static_cast<size_t>(byte);
      value *= 1099511628211ULL;
    }
    return value;
  }
};

struct VoxelKey {
  int32_t x = 0;
  int32_t y = 0;
  int32_t z = 0;
  bool operator==(const VoxelKey& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }
  bool operator<(const VoxelKey& rhs) const {
    if (x != rhs.x) return x < rhs.x;
    if (y != rhs.y) return y < rhs.y;
    return z < rhs.z;
  }
};

struct ProbabilityInterval {
  double lower = 0.0;
  double mean = 0.0;
  double upper = 1.0;
  bool calibrated = false;
  // True only when `mean` is an identified point estimate for this exact
  // event.  In particular, a union of marginals with unknown dependence has
  // valid Frechet/Boole bounds but no identifiable union mean.  In that case
  // `mean` is a finite legacy/wire placeholder and decision consumers must
  // use `upper`.
  bool mean_identified = false;
};

struct CommonHeader {
  std::string schema_id = "deform-monitor-v1";
  uint32_t schema_version = 3;
  std::string session_id;
  StableId snapshot_id;
  uint32_t reference_epoch = 0;
  uint64_t accepted_frame_index = 0;
  ros::Time measurement_stamp;
  ros::Time publish_stamp;
  std::string world_frame;
  std::string software_revision;
  std::string design_version = "design-v9-lidar-v1-observable";
  std::string implementation_freeze_version = "implementation-freeze-v1.1-research";
  std::string config_digest;
  std::string calibration_version = "UNSET";
  std::string robot_geometry_revision;
  uint8_t input_capability_profile = 0;
  std::string sensor_scope = "LIDAR_ONLY_NO_SEMANTICS_NO_OBJECT_ID";
  uint32_t hazard_coverage_mask = kV1HazardCoverageMask;
  std::vector<std::string> uncovered_hazards;
  bool probabilities_calibrated = false;
  Validity validity = Validity::UNKNOWN;
  std::vector<ReasonCode> reasons;
  ros::Time valid_from;
  ros::Time valid_until;
  ros::Time prediction_valid_until;
  uint64_t dropped_input_count = 0;
  uint64_t dropped_output_count = 0;
  uint32_t approximation_flags = APPROX_NONE;
};

struct ExtrinsicCalibration {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix3d R_IL = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_IL = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 6, 6> covariance =
      Eigen::Matrix<double, 6, 6>::Zero();
  std::string revision;
  std::string digest;
  bool online_estimation_enabled = false;
};

struct PoseEstimate {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix3d R_RL = Eigen::Matrix3d::Identity();
  Eigen::Vector3d origin_L_R = Eigen::Vector3d::Zero();
  // [world-left rotation; world-additive LiDAR-origin translation].
  Eigen::Matrix<double, 6, 6> covariance =
      Eigen::Matrix<double, 6, 6>::Zero();
};

struct CanonicalPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d position_R = Eigen::Vector3d::Zero();
  float intensity = 0.0F;
  uint64_t original_index = 0;
  uint64_t canonical_rank = 0;
  StableId point_id;
  StableId evidence_id;
  StableId correlation_group_id;
  PointRole role = PointRole::MEASUREMENT;
};

using CanonicalPointVector =
    std::vector<CanonicalPoint, Eigen::aligned_allocator<CanonicalPoint>>;

struct FrameBundle {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::string session_id;
  StableId frame_id;
  uint64_t accepted_frame_index = 0;
  ros::Time stamp;
  std::string world_frame;
  PoseEstimate pose;
  std::string extrinsic_revision;
  std::string extrinsic_digest;
  CanonicalPointVector points;
  Validity validity = Validity::INVALID;
  std::vector<ReasonCode> reasons;
};

struct Surfel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StableId id;
  Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance_R = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d observable_projector_R = Eigen::Matrix3d::Zero();
  std::array<Eigen::Matrix3d, 4> hypothesis_projector_R{{
      Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(),
      Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero()}};
  Eigen::Vector4d geometry_probability = Eigen::Vector4d::Zero();
  // Maximum distance from the frozen centroid to every raw reference return
  // assigned to this carrier. covariance_R is centroid uncertainty and must
  // not be reused as a geometric collision radius.
  double support_radius_m = 0.0;
  uint32_t exposure_count = 0;
  uint32_t point_count = 0;
  uint32_t gdu_index = 0;
};

struct GraphEdge {
  StableId id;
  uint32_t first = 0;
  uint32_t second = 0;
  double distance_m = 0.0;
  ReferenceEdgeState state = ReferenceEdgeState::AMBIGUOUS;
  double p_link = 0.5;
};

struct InterfaceDefinition {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StableId id;
  uint32_t left_gdu = 0;
  uint32_t right_gdu = 0;
  uint32_t left_surfel = 0;
  uint32_t right_surfel = 0;
  bool cross_gdu = true;
  // Frozen side membership is required for intra-GDU Voronoi interfaces.
  // Cross-GDU interfaces use the GDU ids as their side partition but retain
  // these vectors for provenance and deterministic visualization.
  std::vector<uint32_t> left_region_surfels;
  std::vector<uint32_t> right_region_surfels;
  std::vector<StableId> boundary_edge_ids;
  Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_R = Eigen::Vector3d::UnitX();
  double scale_m = 0.0;
  double p_link = 0.5;
  // Reference-only multiscale candidates in one competition group are
  // alternative locations/scales of one latent discontinuity opportunity.
  // Their weights sum to one within the group, so library densification
  // redistributes a finite prior instead of creating independent prior mass.
  StableId competition_group_id;
  double competition_weight = 1.0;
  // False when the frozen continuous basis and jump column are rank
  // indistinguishable.  Such a candidate is retained for provenance and a
  // protective UNRESOLVED envelope, but no jump coordinate is created.
  bool identifiable = false;
};

struct GduDefinition {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StableId id;
  std::vector<uint32_t> surfel_indices;
  std::vector<uint32_t> control_surfel_indices;
  std::vector<uint32_t> interface_indices;
  // Minimal frozen physical coordinate frame.  Columns [0,coordinate_rank)
  // are the only displacement directions represented by the filter.  A
  // surface GDU has rank 1 (normal), a curve GDU rank 2 (normal plane), and a
  // repeatably full-rank multi-view carrier rank 3. OTHER has rank 0.
  Eigen::Matrix3d coordinate_axes_R = Eigen::Matrix3d::Zero();
  int coordinate_rank = 0;
  GeometryHypothesis coordinate_geometry = GeometryHypothesis::OTHER;
  bool monitorable = false;
  // Covariance of the epoch-static common reference translation, estimated
  // from leave-one-frame/common-frame residuals after the GDU is frozen.
  // Per-surfel covariance is not added again to measurement rows.
  Eigen::Matrix3d reference_bias_covariance_R = Eigen::Matrix3d::Zero();
};

struct ReferenceSnapshot {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StableId id;
  uint32_t epoch = 0;
  ros::Time begin_stamp;
  ros::Time end_stamp;
  std::string world_frame;
  std::vector<Surfel, Eigen::aligned_allocator<Surfel>> surfels;
  std::vector<GraphEdge> edges;
  std::vector<GduDefinition, Eigen::aligned_allocator<GduDefinition>> gdus;
  std::vector<InterfaceDefinition, Eigen::aligned_allocator<InterfaceDefinition>> interfaces;
  bool frozen = false;
  Validity validity = Validity::INVALID;
  std::vector<ReasonCode> reasons;
};

struct AssociationFeature {
  StableId candidate_id;
  double distance_lower_bound = 0.0;
  double continuous_projection = 0.0;
  double projector_chordal_distance = 0.0;
  double log_support_count_ratio = 0.0;
  double overlap = 0.0;
  double independent_visibility = 0.0;
  double prediction_continuity = 0.0;
};

struct AssociationCategory {
  StableId candidate_id;
  AssociationSemantics semantics = AssociationSemantics::OTHER;
  bool geometric_candidate = false;
  double probability = 0.0;
  double log_score = -std::numeric_limits<double>::infinity();
};

struct AssociationResult {
  std::vector<AssociationCategory> categories;
  double entropy = std::numeric_limits<double>::infinity();
  double other_mass = 1.0;
  bool valid = false;
};

struct HypothesisMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::VectorXd residual_m;
  Eigen::MatrixXd design;
  Eigen::MatrixXd reference_bias_design;
  Eigen::MatrixXd extrinsic_design;
  Eigen::MatrixXd pose_design;
  Eigen::MatrixXd independent_covariance_m2;
  int observable_rank = 0;
};

struct FrameNuisance {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix<double, 6, 6> pose_covariance =
      Eigen::Matrix<double, 6, 6>::Zero();
  // Control-only local correction about the FAST-LIO initial pose, ordered as
  // [world-left rotation, world-additive translation].  The correction mean
  // is applied once to target-GDU measurements; pose_covariance is then used
  // once as the shared residual nuisance.
  Eigen::Matrix<double, 6, 1> pose_correction =
      Eigen::Matrix<double, 6, 1>::Zero();
  StableId nuisance_correlation_id;
  bool pose_temporal_bound_applied = false;
  bool control_registration_valid = false;
  bool deformation_measurement_valid = false;
  uint32_t control_patch_count = 0;
};

struct FieldMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StableId evidence_id;
  StableId candidate_id;
  uint32_t gdu_index = 0;
  Eigen::Vector3d position_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d raw_residual_R = Eigen::Vector3d::Zero();
  double association_probability = 0.0;
  std::array<HypothesisMeasurement, 4> hypothesis;
};

struct GaussianComponent {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StructuralState state = StructuralState::U;
  GeometryHypothesis geometry = GeometryHypothesis::OTHER;
  std::vector<InterfaceJointState> interface_modes;
  StableId association_lineage;
  // One branch digest per accepted frame.  Entries older than the configured
  // fixed lag are removed before component-key comparison (DEC-013).
  std::vector<StableId> active_fixed_lag_lineage;
  int contamination_scale = 0;
  StableId parent_id;
  StableId component_id;
  double weight = 0.0;
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;
  Eigen::MatrixXd sqrt_information;
  // Observability is conditional on this component's frozen geometry
  // hypothesis.  It is deliberately not copied from a mixture-averaged GDU
  // projector: different geometry hypotheses can expose different physical
  // coefficient subspaces.
  int observable_rank = 0;
  Eigen::MatrixXd observable_projector;
  Eigen::MatrixXd observable_nullspace;
  double minimum_detectable_effect_m =
      std::numeric_limits<double>::infinity();
  bool minimum_detectable_effect_defined = false;
};

struct AssociationMeasurementSet {
  StableId source_evidence_id;
  AssociationResult selection;
  std::vector<FieldMeasurement, Eigen::aligned_allocator<FieldMeasurement>> candidates;
  std::array<double, 3> other_log_likelihood{{0.0, 0.0, 0.0}};
};

struct GduPosterior {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StableId gdu_id;
  StableId nuisance_correlation_id;
  std::array<double, 3> state_probability{{1.0, 0.0, 0.0}};
  double other_mass = 0.0;
  double task_scale_deformation_probability = 0.0;
  double task_scale_deformation_lower = 0.0;
  double task_scale_deformation_upper = 1.0;
  bool task_scale_deformation_mean_identified = false;
  double task_displacement_probability = 0.0;
  double current_velocity_exceedance_probability = 0.0;
  double continuous_valid_observation_duration_s = 0.0;
  bool velocity_duration_requirement_met = false;
  double task_interface_jump_probability = 0.0;
  double minimum_detectable_effect_m = std::numeric_limits<double>::infinity();
  bool minimum_detectable_effect_defined = false;
  double velocity_minimum_detectable_effect_mps =
      std::numeric_limits<double>::infinity();
  bool velocity_minimum_detectable_effect_defined = false;
  double interface_jump_minimum_detectable_effect_m =
      std::numeric_limits<double>::infinity();
  bool interface_jump_minimum_detectable_effect_defined = false;
  double task_power_worst_normalized_mde =
      std::numeric_limits<double>::infinity();
  bool any_task_direction_detectable = false;
  bool all_task_directions_detectable = false;
  bool detection_allowed = false;
  bool clearing_allowed = false;
  bool task_power_complete = false;
  int observable_rank = 0;
  Eigen::MatrixXd observable_projector;
  Eigen::MatrixXd nullspace;
  // Smallest retained generalized-information ratio after marginalizing the
  // epoch-static reference bias.  A value below the calibrated threshold
  // means that deformation and reference bias are not distinguishable on at
  // least one otherwise observable physical mode.
  double reference_information_ratio = 0.0;
  bool reference_confounding = false;
  // Kish support on the current reference graph and on the fixed-lag frame
  // history after the calibrated OU correlation penalty.
  double effective_spatial_support = 0.0;
  double effective_temporal_support = 0.0;
  DetectionAdmission admission = DetectionAdmission::INCONCLUSIVE_UNCALIBRATED;
  std::vector<GaussianComponent, Eigen::aligned_allocator<GaussianComponent>> components;
};

struct FieldComponentQuery {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StableId component_id;
  double weight = 0.0;
  StructuralState state = StructuralState::U;
  GeometryHypothesis geometry = GeometryHypothesis::OTHER;
  Eigen::Vector3d mean_displacement_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_velocity_R = Eigen::Vector3d::Zero();
  Eigen::Matrix3d displacement_covariance_R = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d velocity_covariance_R = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d observable_projector_R = Eigen::Matrix3d::Zero();
  double minimum_detectable_effect_m =
      std::numeric_limits<double>::infinity();
  bool minimum_detectable_effect_defined = false;
};

struct FieldQueryResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d mean_displacement_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_velocity_R = Eigen::Vector3d::Zero();
  Eigen::Matrix3d displacement_covariance_R = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d velocity_covariance_R = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d observable_projector_R = Eigen::Matrix3d::Zero();
  ProbabilityInterval active_probability;
  double other_mass = 1.0;
  // Frozen support-graph attachment distance plus the calibrated prediction
  // envelope. This is an error bound for spatial rendering/query consumers,
  // not an extra deformation displacement.
  double interpolation_error_upper_m =
      std::numeric_limits<double>::infinity();
  std::vector<FieldComponentQuery,
      Eigen::aligned_allocator<FieldComponentQuery>> components;
  Validity validity = Validity::UNKNOWN;
  std::vector<ReasonCode> reasons;
};

struct InterfacePosterior {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StableId interface_id;
  std::array<double, 4> joint_probability{{1.0, 0.0, 0.0, 0.0}};
  Eigen::Vector3d jump_mean_R = Eigen::Vector3d::Zero();
  Eigen::Matrix3d jump_covariance_R = Eigen::Matrix3d::Zero();
  double other_mass = 0.0;
  bool connectivity_coherence_confounding = false;
  Validity validity = Validity::UNKNOWN;
  FieldQueryResult left_field;
  FieldQueryResult right_field;
  Eigen::Vector3d left_query_position_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_query_position_R = Eigen::Vector3d::Zero();
};

struct OccupancyCell {
  VoxelKey key;
  ProbabilityInterval reference_occupancy;
  ProbabilityInterval total_occupancy;
  ProbabilityInterval new_occupancy;
  // New occupancy is defined only on frozen reference FREE support. UNKNOWN
  // must not be serialized as a numerical zero that a planner could mistake
  // for evidence of low collision risk.
  bool new_occupancy_applicable = false;
  KnowledgeState knowledge = KnowledgeState::UNKNOWN;
  double visibility = 0.0;
  uint32_t exposure_count = 0;
  ros::Time last_observed;
  // Nonzero when the reference occupancy in this cell is uniquely owned by a
  // frozen GDU.  It shares that GDU's HazardCauseId and must not be counted as
  // an independent reference-persistence cause during risk aggregation.
  StableId reference_cause_id;
  std::vector<StableId> cause_ids;
  // Aligned with cause_ids and stores each cause's cell-level PRESENT
  // probability before the cross-cause union. It is required for exact cause
  // de-duplication in configuration and path risk queries.
  std::vector<ProbabilityInterval> cause_probabilities;
  // Aligned with cause_ids.  Risk prediction uses this field to distinguish
  // static reference persistence from unresolved occupancy events whose
  // future footprint must be enlarged by the calibrated reachable set.
  std::vector<HazardSemantics> cause_semantics;
  // Aligned with cause_ids. Each inner vector is the canonical evidence set
  // used by the most recent event update; historical evidence is summarized
  // recursively by the persistent event posterior and stable cause id.
  std::vector<std::vector<StableId>> cause_source_ids;
};

struct HazardGeometry {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StableId cause_id;
  StableId component_id;
  std::vector<StableId> source_ids;
  HazardSemantics semantics = HazardSemantics::OTHER_ENVELOPE;
  StructuralState structural_state = StructuralState::U;
  GeometryHypothesis geometry_hypothesis = GeometryHypothesis::OTHER;
  uint32_t gdu_index = 0;
  double horizon_s = 0.0;
  Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance_R = Eigen::Matrix3d::Zero();
  double radius_m = 0.0;
  ProbabilityInterval existence;
  uint32_t correlation_group = 0;
  double model_envelope_m = 0.0;
  double other_mass = 0.0;
};

struct ObservationExplanation {
  StableId cause_id;
  StableId source_evidence_id;
  AssociationSemantics semantics = AssociationSemantics::OTHER;
  ProbabilityInterval probability;
  bool committed_to_posterior = false;
  bool moved_available = false;
  bool protective_geometry_available = false;
  // Internal provenance for APPROX_CROSS_GDU_ASSOCIATION_TO_ENVELOPE. Other
  // protective carriers must not be mislabeled as a cross-GDU approximation.
  bool cross_gdu_assignment_envelope = false;
  std::array<double, 3> protective_center_R{{0.0, 0.0, 0.0}};
  std::array<double, 9> protective_covariance_R{{}};
  double protective_radius_m = 0.0;
};

struct RobotSphere {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d center_body = Eigen::Vector3d::Zero();
  double radius_m = 0.0;
};

struct RobotGeometry {
  std::string revision;
  std::string body_frame;
  std::vector<RobotSphere, Eigen::aligned_allocator<RobotSphere>> spheres;
};

struct TimedRobotState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ros::Time stamp;
  Eigen::Isometry3d T_RB = Eigen::Isometry3d::Identity();
  Eigen::Matrix<double, 6, 6> covariance =
      Eigen::Matrix<double, 6, 6>::Zero();
};

struct SoftCostBreakdown {
  double collision_nats = 0.0;
  double unknown_cost = 0.0;
  double stale_cost = 0.0;
  double total = 0.0;
};

struct RiskAssessment {
  StableId snapshot_id;
  std::string assessment_scope =
      "HAZARD_COLLISION_OVERLAY_ONLY";
  bool geometric_support_evaluated = false;
  Validity geometric_support_validity = Validity::UNKNOWN;
  HardState hard = HardState::UNKNOWN;
  ros::Time first_violation_time;
  bool first_violation_configuration_defined = false;
  std::array<double, 16> first_violation_pose_row_major{{}};
  std::array<double, 36> first_violation_covariance_row_major{{}};
  std::string first_violation_pose_frame;
  std::string first_violation_covariance_origin_frame;
  ProbabilityInterval collision;
  ProbabilityInterval encounter;
  ProbabilityInterval path_risk;
  // Meaningful only when validity permits use. Invalid/UNKNOWN responses
  // still serialize a finite scalar; the validity enum carries not-computed
  // because ROS scalar fields cannot be absent.
  double risk_budget_margin = 0.0;
  SoftCostBreakdown soft;
  std::vector<StableId> cause_ids;
  std::vector<ProbabilityInterval> cause_probabilities;
  std::vector<ProbabilityInterval> cause_encounter_probabilities;
  Validity validity = Validity::UNKNOWN;
  std::vector<ReasonCode> reasons;
};

struct V1Snapshot {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CommonHeader common;
  ReferenceSnapshot reference;
  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>> deformation;
  std::vector<InterfacePosterior, Eigen::aligned_allocator<InterfacePosterior>> interfaces;
  std::vector<OccupancyCell> occupancy;
  std::vector<HazardGeometry, Eigen::aligned_allocator<HazardGeometry>> hazards;
  std::vector<ObservationExplanation> explanations;
};

struct AxisAlignedRoi {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d minimum_R = Eigen::Vector3d::Constant(
      -std::numeric_limits<double>::infinity());
  Eigen::Vector3d maximum_R = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::infinity());
  bool Contains(const Eigen::Vector3d& point_R) const {
    return (point_R.array() >= minimum_R.array()).all() &&
           (point_R.array() <= maximum_R.array()).all();
  }
};

struct PhaseRequest {
  std::string session_id;
  MonitoringPhase target = MonitoringPhase::MAPPING_CONTEXT;
  AxisAlignedRoi roi;
  ros::Time effective_stamp;
  TriggerSource trigger_source = TriggerSource::HUMAN_OR_EXTERNAL_TASK;
  ros::Time selection_support_cutoff;
  StableId expected_previous_snapshot;
};

struct PhaseResponse {
  bool accepted = false;
  MonitoringPhase phase = MonitoringPhase::MAPPING_CONTEXT;
  uint32_t reference_epoch = 0;
  StableId reference_snapshot_id;
  std::vector<ReasonCode> reasons;
};

struct ResourceConfig {
  std::string session_id;
  std::string world_frame = "camera_init";
  Eigen::Vector3d grid_origin_m = Eigen::Vector3d::Zero();
  double frame_sync_tolerance_s = 0.005;
  bool upstream_registered_cloud_deskew_verified = false;
  double raw_frame_buffer_s = 120.0;
  size_t max_points_per_frame = 200000;
  size_t max_raw_buffer_points = 2000000;
  int reference_min_frames = 6;
  int reference_min_surfel_exposures = 6;
  double reference_min_duration_s = 0.5;
  bool bounded_reference_search = false;
  double max_update_gap_s = 15.0;
  int fixed_lag_frames = 3;
  int mixture_max_components = 24;
  double other_mass_alarm = 0.25;
  double surfel_seed_spacing_m = 0.10;
  double graph_neighbor_radius_m = 0.25;
  // Six exposure blocks require at least three distinct spatial supports for
  // a leave-one-frame-out plane/curve estimate.  Thirty-six raw neighbours
  // provide six balanced spatial ranks at the frozen six-exposure minimum.
  int graph_max_neighbors = 36;
  double control_spacing_m = 0.20;
  double pou_support_radius_m = 0.30;
  std::vector<double> interface_scales_m{0.20, 0.40};
  double occupancy_voxel_m = 0.10;
  std::vector<double> prediction_horizons_s{0.0, 0.5, 1.0, 2.0};
  int prediction_samples = 256;
  double deadline_ms = 15000.0;
  size_t max_gdu = 64;
  size_t min_controls_per_gdu = 6;
  size_t max_controls_per_gdu = 16;
  size_t max_interfaces_per_gdu = 1;
  size_t max_interfaces = 64;
  size_t max_events = 4096;
  size_t max_tiles = 4096;
  // Expanded (horizon, Gaussian component, reference carrier) geometry
  // records.  This is intentionally distinct from max_events, which counts
  // physical/canonical causes rather than their spatial representations.
  size_t max_hazard_primitives = 65536;
  // Maximum number of unique (voxel, cause, Gaussian-component) entries
  // created while rasterising current-time structural geometry.  This is a
  // different resource dimension from max_hazard_primitives: one geometric
  // primitive can cover several voxels without creating a new causal event.
  size_t max_hazard_projection_entries = 1048576;
  size_t max_body_query_samples = 4096;
  size_t max_motion_query_samples = 4096;
  size_t max_risk_query_operations = 50000000;
  size_t max_stable_ids = 2000000;
  // Maximum number of distinct evidence-to-occupancy-cell provenance links
  // in one update and in the retained event ledger.  A long ray can create
  // many links from one return, so max_tiles alone is not a memory bound.
  size_t max_evidence_links = 2000000;
  std::string robot_geometry_revision;
  double map_ttl_s = 15.0;
  double query_ttl_s = 0.5;
  int association_max_candidates = 3;
  int min_patches_per_gdu = 3;
  int max_patches_per_gdu = 100;
  int pose_control_min_patches = 20;
  // Controls inside this frozen-reference distance from the current GDU are
  // excluded from the GDU-local SE(3) correction.  This prevents adjacent
  // GDU fragments on one moving structure from absorbing one another's
  // deformation without requiring semantic object identity.
  double pose_control_exclusion_radius_m = 5.0;
  // True only when an upstream adapter transforms every reference and
  // monitoring body-frame scan with one frozen pose while the robot is
  // stationary.  The shared rigid transform then cancels from relative
  // deformation, so a second local SE(3) correction would double-correct.
  bool upstream_stationary_pose_freeze_verified = false;
  int source_missing_consecutive_limit = 3;
  ExtrinsicCalibration extrinsic;
  double ctmc_max_substep_s = 0.05;
};

struct CalibrationConfig {
  bool validated = false;
  std::string version = "UNSET";
  std::string data_digest;
  double independent_point_sigma_m = 0.003;
  // Common-measure nuisance scale for residual coordinates that a frozen
  // geometry hypothesis declares unobservable.  Every geometry branch keeps
  // a 3-D density; this term prevents mixing densities of different units.
  double unobservable_residual_sigma_m = 0.10;
  double contamination_probability = 0.05;
  double student_t_dof = 5.0;
  double student_t_scale = 3.0;
  double pose_ou_tau_s = 0.5;
  double pose_cross_covariance_bound_m2 = 1.0e-6;
  std::array<double, 5> ctmc_rates{{0.01, 0.002, 0.10, 0.02, 0.05}};
  double active_acceleration_psd = 1.0e-5;
  double changed_random_walk_psd = 1.0e-7;
  double active_birth_velocity_variance = 1.0e-4;
  double active_birth_displacement_variance = 1.0e-6;
  double changed_birth_displacement_variance = 1.0e-4;
  // Calibrated metric upper bound on the continuous-state endpoint-reset
  // error at ctmc_max_substep_s, obtained by step-size halving.  This is a
  // geometric envelope in metres, not a probability or Gaussian variance.
  double ctmc_halving_error_upper_m = 0.02;
  double reference_stability_sigma = 3.0;
  // Epoch-static floor; unlike independent exposure-center scatter it is not
  // divided by the number of frames or surfels.
  double reference_epoch_common_bias_sigma_m = 0.003;
  double geometry_min_eigengap_ratio = 0.05;
  double pose_control_max_condition_number = 1.0e8;
  double pose_control_huber_scale_m = 0.02;
  double reference_confounding_min_information_ratio = 0.10;
  double reference_link_logit_bias = 4.0;
  double reference_link_distance_weight = -4.0;
  double reference_link_projector_weight = -2.0;
  std::array<double, 3> interface_initial_link_probability{{0.89, 0.01, 0.10}};
  Eigen::Matrix3d interface_coherence_transition =
      (Eigen::Matrix3d() << 0.97, 0.01, 0.02,
                            0.03, 0.94, 0.03,
                            0.20, 0.10, 0.70).finished();
  double interface_coherent_rho = 0.20;
  double interface_relative_rho = 1.0;
  double interface_coherent_process_variance = 1.0e-7;
  double interface_relative_process_variance = 1.0e-5;
  double interface_unresolved_process_variance = 1.0e-3;
  double link_probability_on = 0.90;
  double cut_probability_on = 0.10;
  Eigen::Matrix<double, 4, 4> geometry_softmax_weights =
      Eigen::Matrix<double, 4, 4>::Identity();
  Eigen::Vector4d geometry_softmax_bias = Eigen::Vector4d::Zero();
  Eigen::Matrix<double, 7, 1> association_weights =
      (Eigen::Matrix<double, 7, 1>() << -3.0, 1.0, -1.0, -0.2, 1.0, 1.0, 1.0).finished();
  double association_candidate_bias = 0.0;
  double association_missing_bias = -1.0;
  double association_other_bias = -2.0;
  double association_entropy_max = 1.2;
  double visibility_min_range_m = 0.30;
  double visibility_max_range_m = 40.0;
  double nb_dispersion = 5.0;
  double nb_clutter_mean = 0.05;
  double nb_occupied_mean_scale = 1.0;
  double new_occupancy_prior = 0.01;
  // Continuous-time PRESENT/ABSENT event rates [1/s].  The three pairs are
  // intentionally separate because a novel event, a frozen-reference
  // persistence event, and an UNKNOWN-support current event do not share the
  // same survival prior.
  double novel_occupancy_birth_rate_hz = 0.002;
  double novel_occupancy_death_rate_hz = 0.05;
  double reference_persistence_birth_rate_hz = 0.10;
  double reference_persistence_death_rate_hz = 0.002;
  double unknown_occupancy_birth_rate_hz = 0.01;
  double unknown_occupancy_death_rate_hz = 0.05;
  double visibility_exposure_scale = 2.0;
  double occupancy_interval_alpha = 0.05;
  double reference_beta_tempering = 0.5;
  double reference_free_threshold = 0.15;
  double reference_occupied_threshold = 0.65;
  double task_displacement_scale_m = 0.005;
  double task_velocity_scale_mps = 0.002;
  double task_velocity_required_duration_s = 0.5;
  double task_interface_jump_scale_m = 0.005;
  Eigen::Matrix3d task_interface_metric = Eigen::Matrix3d::Identity();
  // Posterior decision thresholds are not frequentist false-alarm/power
  // parameters.  They are calibrated independently on complete end-to-end
  // runs and must satisfy clear < detect.
  double detection_probability_on = 0.99;
  double clearing_probability_on = 0.01;
  double event_false_alarm_probability = 0.01;
  double target_power = 0.90;
  // Conservative ratio between the full selection/robust/switching injection
  // MDE and the conditional linear-Gaussian noncentral-chi-square oracle.
  // It is >= 1 and prevents the oracle from being presented as operational
  // power in a mixed/selected runtime model.
  double mde_bootstrap_inflation = 1.0;
  double detection_min_effective_spatial_support = 2.0;
  double detection_min_effective_temporal_support = 1.2;
  double qmc_error_upper = 0.02;
  double prediction_model_error_m = 0.01;
  // Speed bound used only for occupancy causes without a tracked velocity.
  // Reference persistence and structurally tracked causes do not use it.
  double occupancy_max_speed_mps = 0.50;
  // Calibrated upper speed used only to close continuous motion-query sweep
  // gaps. It must cover all modeled hazard carriers, including occupancy.
  double hazard_sweep_speed_upper_mps = 0.50;
};

struct NumericalConfig {
  double probability_floor = 1.0e-12;
  double pou_denominator_min = 1.0e-12;
  double psd_eigenvalue_tolerance = 1.0e-10;
  double cholesky_jitter_initial = 1.0e-12;
  double cholesky_jitter_multiplier = 10.0;
  int cholesky_jitter_attempts = 6;
  double solve_residual_tolerance = 1.0e-8;
  double normalization_tolerance = 1.0e-10;
  double tie_tolerance = 1.0e-12;
};

struct PolicyConfig {
  std::string revision = "policy-v1-research";
  double hard_clearance_m = 0.05;
  double hard_collision_probability = 0.05;
  double hard_encounter_probability = 0.02;
  double path_risk_upper_bound = 0.05;
  bool unknown_is_blocked = true;
  double unknown_weight = 1.0;
  double stale_weight = 1.0;
  double probability_epsilon = 1.0e-12;
  double risk_saturation_nats = 20.0;
  double maximum_soft_cost = 1.0e6;
  double motion_sample_spacing_m = 0.05;
  double motion_sample_interval_s = 0.05;
};

struct V1Config {
  ResourceConfig resource;
  CalibrationConfig calibration;
  NumericalConfig numerical;
  PolicyConfig policy;
  RobotGeometry robot_geometry;
  std::string software_revision;
  std::string config_digest;
};

struct ConfigValidation {
  bool valid = false;
  std::vector<std::string> errors;
};

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_TYPES_HPP
