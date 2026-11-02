#ifndef DEFORM_MONITOR_V3_RISK_MAP_RISK_TYPES_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_RISK_TYPES_HPP

#include "deform_monitor_v3/data_types.hpp"

#include <Eigen/Dense>
#include <ros/time.h>

#include <cstdint>
#include <cstddef>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace deform_monitor_v3 {
namespace risk_map {

enum class EvidenceStatus : uint8_t {
  OK = 0,
  EMPTY = 1,
  UNOBSERVABLE_MODE = 2,
  INVALID_COVARIANCE = 3,
  FACTORIZATION_FAILED = 4,
  INCOMPLETE_BLOCK = 5
};

enum class KnowledgeState : uint8_t {
  UNKNOWN = 0,
  FREE = 1,
  OCCUPIED = 2
};

enum class EventSemantics : uint8_t {
  NONE = 0,
  DEFORMATION = 1,
  NOVEL_OCCUPANCY = 2,
  UNKNOWN = 3,
  MIXED = 4
};

enum class UnknownPolicy : uint8_t {
  LETHAL_UNKNOWN = 0,
  FIXED_UNKNOWN_COST = 1,
  EXPLORABLE_UNKNOWN = 2
};

enum class PoseTemporalModel : uint8_t {
  INDEPENDENT_POSE = 0,
  AR1_POSE = 1
};

struct VoxelKey {
  int32_t x = 0;
  int32_t y = 0;
  int32_t z = 0;

  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
  bool operator!=(const VoxelKey& other) const { return !(*this == other); }
  bool operator<(const VoxelKey& other) const {
    if (x != other.x) return x < other.x;
    if (y != other.y) return y < other.y;
    return z < other.z;
  }
};

struct VoxelKeyHash {
  size_t operator()(const VoxelKey& key) const {
    size_t seed = std::hash<int32_t>()(key.x);
    seed ^= std::hash<int32_t>()(key.y) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<int32_t>()(key.z) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

struct ChannelKey {
  int anchor_id = -1;
  uint8_t axis_index = 0;
  uint32_t reference_epoch = 0;

  bool operator==(const ChannelKey& other) const {
    return anchor_id == other.anchor_id && axis_index == other.axis_index &&
           reference_epoch == other.reference_epoch;
  }
  bool operator<(const ChannelKey& other) const {
    if (reference_epoch != other.reference_epoch) return reference_epoch < other.reference_epoch;
    if (anchor_id != other.anchor_id) return anchor_id < other.anchor_id;
    return axis_index < other.axis_index;
  }
};

struct ChannelObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ChannelKey key;
  ros::Time stamp;
  Eigen::Vector3d h_R = Eigen::Vector3d::UnitZ();
  Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
  double z = 0.0;
  double independent_variance = 1.0;
  Eigen::Matrix3d reference_covariance_R = Eigen::Matrix3d::Zero();
  uint8_t reference_covariance_group = 255;
  double psi = 0.0;
  bool matched = false;
};

using ChannelObservationVector = AlignedVector<ChannelObservation>;

struct ChannelFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ros::Time stamp;
  PoseCov6D pose_cov;
  Eigen::Vector3d sensor_origin_R = Eigen::Vector3d::Zero();
  ChannelObservationVector channels;
};

using ChannelFrameDeque = AlignedDeque<ChannelFrame>;

struct FrozenMode {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int anchor_id = -1;
  Eigen::Vector3d direction_R = Eigen::Vector3d::UnitZ();
  double prior_probability = 1.0;
  double direction_confidence = 0.0;
  ros::Time source_stamp;
  bool from_previous_state = false;
};

using FrozenModeVector = AlignedVector<FrozenMode>;

struct EvidenceBlock {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint64_t block_id = 0;
  uint32_t reference_epoch = 0;
  ros::Time begin_stamp;
  ros::Time end_stamp;
  ChannelFrameDeque frames;
  FrozenModeVector modes;
  bool complete = false;
};

struct EvidenceResult {
  EvidenceStatus status = EvidenceStatus::EMPTY;
  bool valid = false;
  double log_bayes_factor = 0.0;
  double gamma = 0.0;
  double xi = 0.0;
  double kappa = 0.0;
  int channel_count = 0;
  double jitter_used = 0.0;
  int factorization_attempts = 0;
  double condition_estimate = std::numeric_limits<double>::infinity();
  PoseTemporalModel covariance_model = PoseTemporalModel::INDEPENDENT_POSE;
  double solve_residual = std::numeric_limits<double>::infinity();
  bool approximate_joint_model = false;
};

struct VisibilityEstimate {
  double probability = 0.0;
  double fov_feature = 0.0;
  double occlusion_feature = 0.0;
  double density_feature = 0.0;
  double range = 0.0;
  double incidence_cos = 0.0;
  bool calibrated = false;
  std::string model_version = "uncalibrated";
};

struct ReferenceVoxel {
  uint32_t traversed_ray_count = 0;
  uint32_t endpoint_count = 0;
  double occupancy_probability = 0.5;
  KnowledgeState state = KnowledgeState::UNKNOWN;
  uint32_t reference_epoch = 0;
};

struct CurrentVoxelObservation {
  uint32_t hit_count = 0;
  uint32_t traversed_count = 0;
  double beam_coverage = 0.0;
  double visibility_probability = 0.0;
  double expected_hit_count = 0.0;
  ros::Time stamp;
};

struct EvidenceEvent {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint64_t event_id = 0;
  VoxelKey cell;
  EventSemantics semantics = EventSemantics::NONE;
  double log_bayes_factor = 0.0;
  double posterior_responsibility = 1.0;
  Eigen::Vector3d direction_R = Eigen::Vector3d::Zero();
  double direction_confidence = 0.0;
  ros::Time stamp;
};

using EvidenceEventVector = AlignedVector<EvidenceEvent>;

struct BeliefCell {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double deformation_log_odds = 0.0;
  double novel_log_odds = 0.0;
  ros::Time deformation_last_update;
  ros::Time novel_last_update;
  uint64_t evidence_count = 0;
  double positive_evidence_sum = 0.0;
  double negative_evidence_sum = 0.0;
  Eigen::Matrix3d direction_second_moment = Eigen::Matrix3d::Zero();
  Eigen::Vector3d signed_direction_sum = Eigen::Vector3d::Zero();
  double direction_weight_sum = 0.0;
  double direction_support_weight_sum = 0.0;
  double visibility_coverage = 0.0;
  uint32_t reference_epoch = 0;
};

struct RiskMapVoxel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VoxelKey key;
  Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
  float p_deformation = std::numeric_limits<float>::quiet_NaN();
  float p_novel_occupancy = std::numeric_limits<float>::quiet_NaN();
  float decision_cost = 0.0F;
  float deformation_log_odds = std::numeric_limits<float>::quiet_NaN();
  float novel_log_odds = std::numeric_limits<float>::quiet_NaN();
  float last_log_bayes_factor = 0.0F;
  float visibility_probability = 0.0F;
  Eigen::Vector3f principal_direction = Eigen::Vector3f::Constant(
      std::numeric_limits<float>::quiet_NaN());
  float direction_confidence = 0.0F;
  KnowledgeState knowledge_state = KnowledgeState::UNKNOWN;
  EventSemantics semantics = EventSemantics::NONE;
  int32_t region_id = -1;
};

using RiskMapVoxelVector = AlignedVector<RiskMapVoxel>;

struct RiskMapRegion {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int32_t id = -1;
  EventSemantics semantics = EventSemantics::NONE;
  Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_min_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_max_R = Eigen::Vector3d::Zero();
  Eigen::Vector3d principal_direction_R = Eigen::Vector3d::Zero();
  double peak_deformation_probability = 0.0;
  double peak_novel_probability = 0.0;
  double peak_decision_cost = 0.0;
  double mean_decision_cost = 0.0;
  double direction_confidence = 0.0;
  double unknown_fraction = 0.0;
  int voxel_count = 0;
};

using RiskMapRegionVector = AlignedVector<RiskMapRegion>;

struct RiskMapFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint64_t frame_index = 0;
  ros::Time stamp;
  std::string world_frame = "camera_init";
  double voxel_size = 0.10;
  Eigen::Vector3d grid_origin_R = Eigen::Vector3d::Zero();
  uint32_t reference_epoch = 0;
  std::string model_version = "risk-map-v3";
  std::string calibration_version = "uncalibrated";
  uint64_t dropped_frame_count = 0;
  bool probabilities_calibrated = false;
  bool approximate_joint_model = true;
  RiskMapVoxelVector voxels;
  RiskMapRegionVector regions;
};

struct RiskMapParams {
  bool enable = true;
  std::string model_version = "risk-map-v3";
  std::string calibration_version = "uncalibrated";
  double voxel_size = 0.10;
  double sigma_mu = 0.005;
  int evidence_block_frames = 5;
  double max_block_duration = 1.0;
  PoseTemporalModel pose_temporal_model = PoseTemporalModel::AR1_POSE;
  double pose_correlation_time_sec = 0.5;
  double independent_variance_floor = 1.0e-8;
  double covariance_jitter_initial = 1.0e-10;
  int covariance_jitter_attempts = 5;
  double kappa_floor = 1.0e-12;
  double probability_floor = 1.0e-6;
  double deformation_hazard_on = 1.0e-4;
  double deformation_hazard_off = 0.02;
  double novel_hazard_on = 1.0e-4;
  double novel_hazard_off = 0.20;
  double reference_beta_a = 1.0;
  double reference_beta_b = 1.0;
  int reference_min_coverage = 5;
  double reference_free_threshold = 0.15;
  double reference_occupied_threshold = 0.65;
  double fov_vertical_min_deg = -7.0;
  double fov_vertical_max_deg = 52.0;
  double fov_soft_width_deg = 2.0;
  double min_range = 0.30;
  double max_range = 40.0;
  double visibility_logit_bias = 0.0;
  double visibility_logit_scale = 1.0;
  double q0_default = 0.95;
  double q0_min = 0.05;
  double q0_range_coefficient = 0.005;
  double q0_incidence_coefficient = 0.25;
  double q0_support_quality_coefficient = 0.15;
  double q0_edge_offset = -0.05;
  double q0_band_offset = -0.03;
  double clutter_rate = 1.0e-3;
  double occupied_rate_scale = 1.0;
  double migration_search_radius = 0.30;
  double migration_distance_scale = 0.15;
  double gone_prior = 0.20;
  double decision_weight_deformation = 0.65;
  double decision_weight_novel = 0.85;
  double decision_weight_unknown = 0.40;
  double robot_inflation_radius = 0.25;
  double hazard_projection_length = 0.75;
  UnknownPolicy unknown_policy = UnknownPolicy::FIXED_UNKNOWN_COST;
  double unknown_fixed_cost = 0.60;
  double unknown_explorable_cost = 0.30;
  double region_cost_threshold = 0.50;
  int region_min_voxels = 2;
  bool export_enable = true;
  std::string export_directory;
  double export_rate_hz = 2.0;
  size_t export_queue_capacity = 8;
};

}  // namespace risk_map
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_RISK_MAP_RISK_TYPES_HPP
