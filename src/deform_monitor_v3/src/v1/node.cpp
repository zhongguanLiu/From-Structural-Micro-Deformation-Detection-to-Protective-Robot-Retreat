#include "deform_monitor_v3/v1/node.hpp"

#include "deform_monitor_v3/v1/deterministic.hpp"
#include "deform_monitor_v3/v1/input_contract.hpp"
#include "deform_monitor_v3/v1/math.hpp"
#include "deform_monitor_v3/v1/pose_adapter.hpp"
#include "deform_monitor_v3/v1/ros_bridge.hpp"
#include "deform_monitor_v3/v1/ros_config.hpp"

#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace deform_monitor_v3 {
namespace v1 {
namespace {

void FillDense3(const Eigen::Matrix3d& matrix,
                deform_monitor_v3::V1DenseMatrix64* output) {
  output->rows = 3U; output->cols = 3U; output->data_row_major.resize(9U);
  for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column)
    output->data_row_major[static_cast<size_t>(3 * row + column)] = matrix(row, column);
}

}  // namespace

V1Node::V1Node() = default;

bool V1Node::Initialize() {
  ResetStableIdRegistry();
  std::vector<std::string> errors;
  RobotGeometry unused_legacy_geometry;
  if (!LoadStrictRosConfig(private_node_, &config_, &unused_legacy_geometry,
                           &errors)) {
    for (const std::string& error : errors) ROS_ERROR_STREAM("[deform_v1] " << error);
    return false;
  }
  if (!ConfigureStableIdRegistryCapacity(config_.resource.max_stable_ids)) {
    ROS_ERROR("[deform_v1] invalid or already exhausted stable-id registry capacity");
    return false;
  }
  pipeline_.reset(new V1Pipeline(config_));
  std::string cloud_topic, covariance_topic;
  if (!private_node_.getParam("v1/io/cloud_topic", cloud_topic) ||
      !private_node_.getParam("v1/io/covariance_topic", covariance_topic)) {
    ROS_ERROR("[deform_v1] v1/io/cloud_topic and covariance_topic are required");
    return false;
  }
  // The estimator is synchronous and intentionally transactional.  A large
  // ROS ingress queue would therefore replay stale scans after an expensive
  // update and make a manual scene change appear frozen in RViz.  Retain only
  // the newest two messages per stream.  The first post-freeze frame has
  // explicit clock re-anchoring in V1Pipeline, so reference construction no
  // longer relies on keeping a ten-second callback backlog.
  cloud_subscriber_ = node_.subscribe(cloud_topic, 2, &V1Node::CloudCallback, this);
  covariance_subscriber_ = node_.subscribe(covariance_topic, 2,
                                           &V1Node::CovarianceCallback, this);
  deformation_publisher_ = node_.advertise<deform_monitor_v3::V1DeformationField>(
      "/deform/v1/deformation_field", 2, true);
  interface_publisher_ = node_.advertise<deform_monitor_v3::V1InterfaceMotionFrame>(
      "/deform/v1/interface_motion", 2, true);
  tile_publisher_ = node_.advertise<deform_monitor_v3::V1DetectionTile>(
      "/deform/v1/detection_tiles", 2, true);
  status_publisher_ = node_.advertise<deform_monitor_v3::V1DetectionStatus>(
      "/deform/v1/status", 2, true);
  phase_service_ = node_.advertiseService("/deform/v1/set_monitoring_phase",
                                          &V1Node::SetPhaseService, this);
  deformation_query_service_ = node_.advertiseService("/deform/v1/query_deformation",
      &V1Node::QueryDeformationService, this);
  ROS_INFO_STREAM("[deform_v1] initialized config_digest=" << config_.config_digest
                  << " calibration=" << config_.calibration.version);
  return true;
}

void V1Node::Spin() { ros::spin(); }

void V1Node::CloudCallback(const sensor_msgs::PointCloud2ConstPtr& message) {
  if (!message) return;
  ReasonCode rejection = ReasonCode::INPUT_SCHEMA_INVALID;
  std::string error;
  if (message->header.stamp.isZero() ||
      message->header.frame_id != config_.resource.world_frame ||
      !ValidateFastLioRegisteredCloudSchema(
          *message, config_.resource.max_points_per_frame,
          &rejection, &error)) {
    if (message->header.stamp.isZero()) rejection = ReasonCode::INPUT_TIME_INVALID;
    else if (message->header.frame_id != config_.resource.world_frame)
      rejection = ReasonCode::INPUT_FRAME_MISMATCH;
    last_attempt_processing_time_ms_ = 0.0;
    last_attempt_input_point_count_ =
        static_cast<uint64_t>(message->width) *
        static_cast<uint64_t>(message->height);
    if (pipeline_) pipeline_->RecordDroppedInput();
    PublishAttemptStatus(message->header.stamp, next_accepted_frame_index_,
                         false, {rejection});
    ROS_ERROR_STREAM_THROTTLE(
        1.0, "[deform_v1] rejected cloud before synchronization queue: "
                 << (error.empty() ? "header contract invalid" : error));
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto position = std::upper_bound(
        cloud_queue_.begin(), cloud_queue_.end(), message->header.stamp,
        [](const ros::Time& stamp, const TimedCloud& queued) {
          return stamp < queued.message.header.stamp;
        });
    cloud_queue_.insert(position, TimedCloud{*message});
    size_t queued_points = 0U;
    for (const TimedCloud& queued : cloud_queue_) {
      const size_t count = static_cast<size_t>(queued.message.width) *
                           static_cast<size_t>(queued.message.height);
      queued_points = count > std::numeric_limits<size_t>::max() - queued_points
          ? std::numeric_limits<size_t>::max()
          : queued_points + count;
    }
    while (cloud_queue_.size() > 4U ||
           queued_points > config_.resource.max_raw_buffer_points) {
      const size_t removed =
          static_cast<size_t>(cloud_queue_.front().message.width) *
          static_cast<size_t>(cloud_queue_.front().message.height);
      cloud_queue_.pop_front();
      queued_points -= std::min(queued_points, removed);
      if (pipeline_) pipeline_->RecordDroppedInput();
    }
  }
  TryProcess();
}

void V1Node::CovarianceCallback(const fast_lio::LioOdomCovConstPtr& message) {
  if (!message) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto position = std::upper_bound(
        covariance_queue_.begin(), covariance_queue_.end(),
        message->odom.header.stamp,
        [](const ros::Time& stamp, const TimedCovariance& queued) {
          return stamp < queued.message.odom.header.stamp;
        });
    covariance_queue_.insert(position, TimedCovariance{*message});
    while (covariance_queue_.size() > 4U) {
      covariance_queue_.pop_front();
      if (pipeline_) pipeline_->RecordDroppedInput();
    }
  }
  TryProcess();
}

bool V1Node::PopSynchronized(sensor_msgs::PointCloud2* cloud,
                             fast_lio::LioOdomCov* covariance) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!cloud || !covariance) return false;
  while (!cloud_queue_.empty() && !covariance_queue_.empty()) {
    const ros::Time cloud_stamp = cloud_queue_.front().message.header.stamp;
    size_t best = covariance_queue_.size();
    const double tolerance = config_.resource.frame_sync_tolerance_s;
    double best_distance = std::numeric_limits<double>::infinity();
    ros::Time best_stamp;
    for (size_t i = 0; i < covariance_queue_.size(); ++i) {
      const ros::Time stamp = covariance_queue_[i].message.odom.header.stamp;
      const double distance = std::abs((stamp - cloud_stamp).toSec());
      if (distance <= tolerance &&
          (best == covariance_queue_.size() ||
           distance < best_distance - config_.numerical.tie_tolerance ||
           (std::abs(distance - best_distance) <=
                config_.numerical.tie_tolerance && stamp < best_stamp))) {
        best = i; best_distance = distance; best_stamp = stamp;
      }
    }
    if (best < covariance_queue_.size()) {
      *cloud = cloud_queue_.front().message;
      *covariance = covariance_queue_[best].message;
      cloud_queue_.pop_front();
      covariance_queue_.erase(covariance_queue_.begin() + static_cast<long>(best));
      return true;
    }
    const ros::Time covariance_stamp = covariance_queue_.front().message.odom.header.stamp;
    if (covariance_stamp + ros::Duration(config_.resource.frame_sync_tolerance_s) < cloud_stamp) {
      covariance_queue_.pop_front();
      if (pipeline_) pipeline_->RecordDroppedInput();
    } else if (cloud_stamp + ros::Duration(config_.resource.frame_sync_tolerance_s) < covariance_stamp) {
      cloud_queue_.pop_front();
      if (pipeline_) pipeline_->RecordDroppedInput();
    } else return false;
  }
  return false;
}

bool V1Node::BuildBundle(const sensor_msgs::PointCloud2& cloud_message,
                         const fast_lio::LioOdomCov& covariance_message,
                         FrameBundle* bundle, ReasonCode* reason,
                         std::string* error) const {
  if (reason) *reason = ReasonCode::INPUT_SCHEMA_INVALID;
  if (!bundle) {
    if (error) *error = "output frame bundle is null";
    return false;
  }
  if (cloud_message.header.stamp.isZero()) {
    if (reason) *reason = ReasonCode::INPUT_TIME_INVALID;
    if (error) *error = "cloud header must carry a nonzero stamp";
    return false;
  }
  if (cloud_message.header.frame_id != config_.resource.world_frame) {
    if (reason) *reason = ReasonCode::INPUT_FRAME_MISMATCH;
    if (error) *error = "cloud header must carry nonzero stamp and exact world frame";
    return false;
  }
  if (covariance_message.odom.header.stamp.isZero() ||
      std::abs((covariance_message.odom.header.stamp -
                cloud_message.header.stamp).toSec()) >
          config_.resource.frame_sync_tolerance_s) {
    if (reason) *reason = ReasonCode::INPUT_TIME_INVALID;
    if (error) *error =
        "LioOdomCov odometry stamp must satisfy synchronization tolerance";
    return false;
  }
  if (covariance_message.odom.header.frame_id != config_.resource.world_frame) {
    if (reason) *reason = ReasonCode::INPUT_FRAME_MISMATCH;
    if (error) *error = "LioOdomCov odometry header has the wrong world frame";
    return false;
  }
  const PoseAdapterResult pose = AdaptFastLioPose(
      covariance_message, config_.resource.extrinsic, config_.numerical);
  if (!pose.valid) {
    if (reason) *reason = pose.reason;
    if (error) *error = pose.detail;
    return false;
  }
  if (!ValidateFastLioRegisteredCloudSchema(
          cloud_message, config_.resource.max_points_per_frame,
          reason, error)) {
    return false;
  }
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::fromROSMsg(cloud_message, cloud);
  if (cloud.size() != static_cast<size_t>(cloud_message.width) *
                          static_cast<size_t>(cloud_message.height)) {
    if (reason) *reason = ReasonCode::INPUT_SCHEMA_INVALID;
    if (error) *error = "PointXYZI conversion changed the declared point count";
    return false;
  }
  // This is part of the input transaction: deterministic identities and their
  // bounded full-digest audit are created only after every downstream int32
  // grid representation can encode the registered cloud and its ray origin.
      if (!PreflightPointCloudSpatialDomain(
          cloud, pose.pose.origin_L_R, config_.resource.grid_origin_m,
          config_.resource.surfel_seed_spacing_m,
          config_.resource.surfel_seed_spacing_m, reason, error)) {
    return false;
  }
  const size_t point_count = cloud.size();
  StableIdRegistryTransaction input_ids;
  if (!input_ids.active() ||
      point_count > (std::numeric_limits<size_t>::max() - 1U) / 2U ||
      !StableIdRegistryCanRegister(1U + 2U * point_count)) {
    if (reason) *reason = ReasonCode::RESOURCE_LIMIT;
    if (error)
      *error = "stable-id budget cannot atomically admit this input frame";
    return false;
  }
  bundle->session_id = config_.resource.session_id;
  bundle->accepted_frame_index = next_accepted_frame_index_;
  bundle->stamp = cloud_message.header.stamp;
  bundle->world_frame = config_.resource.world_frame;
  bundle->pose = pose.pose;
  bundle->extrinsic_revision = config_.resource.extrinsic.revision;
  bundle->extrinsic_digest = config_.resource.extrinsic.digest;
  if (!CanonicalizePointCloud(cloud, bundle->session_id,
                              pipeline_->reference_epoch(),
                              bundle->stamp,
                              bundle->accepted_frame_index, &bundle->points,
                              &bundle->frame_id, error) ||
      !input_ids.Commit()) {
    if (reason) {
      *reason = StableIdRegistryResourceExhausted()
                    ? ReasonCode::RESOURCE_LIMIT
                    : (StableIdRegistryHealthy()
                           ? ReasonCode::INPUT_SCHEMA_INVALID
                           : ReasonCode::INVALID_ID_COLLISION);
    }
    return false;
  }
  bundle->validity = config_.resource.upstream_registered_cloud_deskew_verified
                         ? Validity::VALID : Validity::INCONCLUSIVE;
  if (!config_.resource.upstream_registered_cloud_deskew_verified)
    bundle->reasons.push_back(ReasonCode::INPUT_DESKEW_UNRESOLVED);
  return true;
}

void V1Node::TryProcess() {
  if (!pipeline_) return;
  sensor_msgs::PointCloud2 cloud; fast_lio::LioOdomCov covariance;
  while (PopSynchronized(&cloud, &covariance)) {
    const auto attempt_begin = std::chrono::steady_clock::now();
    last_attempt_input_point_count_ =
        static_cast<uint64_t>(cloud.width) * static_cast<uint64_t>(cloud.height);
    FrameBundle bundle; ReasonCode rejection = ReasonCode::INPUT_SCHEMA_INVALID;
    std::string error;
    if (!BuildBundle(cloud, covariance, &bundle, &rejection, &error)) {
      last_attempt_processing_time_ms_ = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - attempt_begin).count();
      pipeline_->RecordDroppedInput();
      ROS_ERROR_STREAM_THROTTLE(1.0, "[deform_v1] rejected input bundle: " << error);
      PublishAttemptStatus(cloud.header.stamp, next_accepted_frame_index_,
                           false, {rejection});
      continue;
    }
    V1Snapshot snapshot; std::vector<ReasonCode> reasons;
    const bool accepted = pipeline_->AcceptFrame(bundle, &snapshot, &reasons);
    last_attempt_processing_time_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - attempt_begin).count();
    if (accepted) ++next_accepted_frame_index_;
    if (!accepted) {
      ROS_WARN_STREAM_THROTTLE(1.0, "[deform_v1] input rejected, first reason="
          << (reasons.empty() ? 0 : static_cast<int>(reasons.front())));
      PublishAttemptStatus(bundle.stamp, bundle.accepted_frame_index, false,
                           reasons);
      continue;
    }
    if (!snapshot.common.snapshot_id.empty())
      PublishSnapshot(snapshot);
    else
      PublishAttemptStatus(bundle.stamp, bundle.accepted_frame_index, false,
                           reasons);
  }
}

void V1Node::PublishSnapshot(const V1Snapshot& snapshot) {
  // Serialize the complete bounded product set before the first ROS publish.
  // ROS topics are not transport-atomic, but this prevents a local allocation
  // or serialization failure from publishing a prefix of layers that the
  // process itself never finished constructing.  Cross-topic consumers still
  // join exclusively on the common snapshot id.
  const auto product_begin = std::chrono::steady_clock::now();
  const auto deformation = BuildDeformationMessage(snapshot);
  const auto after_deformation = std::chrono::steady_clock::now();
  const auto interfaces = BuildInterfaceMessage(snapshot);
  const auto after_interfaces = std::chrono::steady_clock::now();
  const auto tiles = BuildDetectionTileMessages(
      snapshot, pipeline_->filter(), config_);
  const auto after_tiles = std::chrono::steady_clock::now();
  deform_monitor_v3::V1DetectionStatus status = BuildDetectionStatusMessage(
      snapshot, pipeline_->phase(), pipeline_->deadline_failure_count(),
      pipeline_->resource_failure_count());
  status.last_attempt_stamp = snapshot.common.measurement_stamp;
  status.last_attempt_frame_index = snapshot.common.accepted_frame_index;
  status.last_attempt_published = true;
  status.last_attempt_input_point_count = last_attempt_input_point_count_;
  for (ReasonCode reason : snapshot.common.reasons)
    status.last_attempt_reason_codes.push_back(static_cast<uint16_t>(reason));

  deformation_publisher_.publish(deformation);
  interface_publisher_.publish(interfaces);
  for (const deform_monitor_v3::V1DetectionTile& tile : tiles)
    tile_publisher_.publish(tile);
  const auto after_product_publish = std::chrono::steady_clock::now();
  const double product_ms = std::chrono::duration<double, std::milli>(
      after_product_publish - product_begin).count();
  // The detector callback is not complete until the immutable ROS products
  // have been serialized and handed to roscpp.  Include this cost in the
  // public per-attempt telemetry; otherwise a dense no-prior scene can look
  // inexpensive while the callback actually remains blocked for many
  // seconds constructing LOD-2 samples.
  last_attempt_processing_time_ms_ += product_ms;
  status.last_attempt_processing_time_ms = last_attempt_processing_time_ms_;
  // Product construction longer than one 5 Hz input period is itself enough
  // to reduce detector cadence, so expose its internal split at 200 ms.  This
  // is diagnostic only and does not alter publication or acceptance.
  if (product_ms > 200.0) {
    const double deformation_ms =
        std::chrono::duration<double, std::milli>(
            after_deformation - product_begin).count();
    const double interfaces_ms =
        std::chrono::duration<double, std::milli>(
            after_interfaces - after_deformation).count();
    const double tiles_ms =
        std::chrono::duration<double, std::milli>(
            after_tiles - after_interfaces).count();
    const double publish_ms =
        std::chrono::duration<double, std::milli>(
            after_product_publish - after_tiles).count();
    ROS_WARN_STREAM_THROTTLE(
        1.0, "[deform_v1] slow product serialization elapsed_ms="
                 << product_ms << " stages_ms={deformation:"
                 << deformation_ms << ",interfaces:" << interfaces_ms
                 << ",detection_tiles:" << tiles_ms
                 << ",publish:" << publish_ms
                 << "}");
  }
  status_publisher_.publish(status);
}

void V1Node::PublishAttemptStatus(
    const ros::Time& stamp, uint64_t frame_index, bool published,
    const std::vector<ReasonCode>& reasons) {
  if (!pipeline_) return;
  if (config_.resource.bounded_reference_search &&
      pipeline_->phase() == MonitoringPhase::MONITOR_REFERENCE)
    private_node_.setParam("reference_diagnostics", pipeline_->reference_diagnostics());
  V1Snapshot context = pipeline_->last_snapshot();
  if (context.reference.id.empty()) context.reference = pipeline_->reference();
  if (context.common.session_id.empty()) {
    context.common.session_id = config_.resource.session_id;
    context.common.reference_epoch = pipeline_->reference_epoch();
    context.common.world_frame = config_.resource.world_frame;
    context.common.software_revision = config_.software_revision;
    context.common.config_digest = config_.config_digest;
    context.common.calibration_version = config_.calibration.version;
    context.common.robot_geometry_revision.clear();
    context.common.input_capability_profile = 0U;
    context.common.sensor_scope =
        "LIDAR_ONLY_NO_SEMANTICS_NO_OBJECT_ID";
    context.common.hazard_coverage_mask = 0U;
    context.common.uncovered_hazards.clear();
    // The current FAST-LIO marginal-only input does not identify the
    // SLAM--registered-cloud cross covariance.  A pre-snapshot status must not
    // advertise stronger probability authority than the first real snapshot.
    context.common.probabilities_calibrated = false;
    context.common.validity = Validity::UNKNOWN;
  }
  deform_monitor_v3::V1DetectionStatus status = BuildDetectionStatusMessage(
      context, pipeline_->phase(), pipeline_->deadline_failure_count(),
      pipeline_->resource_failure_count());
  status.last_attempt_stamp = stamp;
  status.last_attempt_frame_index = frame_index;
  status.last_attempt_published = published;
  status.last_attempt_processing_time_ms = last_attempt_processing_time_ms_;
  status.last_attempt_input_point_count = last_attempt_input_point_count_;
  std::vector<ReasonCode> canonical_reasons = reasons;
  std::sort(canonical_reasons.begin(), canonical_reasons.end(),
            [](ReasonCode lhs, ReasonCode rhs) {
              return static_cast<uint16_t>(lhs) <
                     static_cast<uint16_t>(rhs);
            });
  canonical_reasons.erase(
      std::unique(canonical_reasons.begin(), canonical_reasons.end()),
      canonical_reasons.end());
  for (ReasonCode reason : canonical_reasons)
    status.last_attempt_reason_codes.push_back(static_cast<uint16_t>(reason));
  status_publisher_.publish(status);
}

bool V1Node::SetPhaseService(
    deform_monitor_v3::SetV1MonitoringPhase::Request& request,
    deform_monitor_v3::SetV1MonitoringPhase::Response& response) {
  // Phase-transition status is not a synchronized cloud processing cycle.
  // Clear the per-input telemetry so downstream runtime tables cannot count a
  // service acknowledgement as a repeated detector attempt.
  last_attempt_processing_time_ms_ = 0.0;
  last_attempt_input_point_count_ = 0U;
  const auto reject_contract = [&]() {
    response.accepted = false;
    response.phase = static_cast<uint8_t>(pipeline_->phase());
    response.reference_epoch = pipeline_->reference_epoch();
    response.reference_snapshot_id =
        BuildStableIdMessage(pipeline_->reference().id);
    response.reason_codes.push_back(
        static_cast<uint16_t>(ReasonCode::INPUT_SCHEMA_INVALID));
    PublishAttemptStatus(request.effective_stamp, next_accepted_frame_index_,
                         false, {ReasonCode::INPUT_SCHEMA_INVALID});
    return true;
  };
  PhaseRequest phase;
  phase.session_id = request.session_id;
  if (request.target_phase > static_cast<uint8_t>(MonitoringPhase::CLOSED))
    return reject_contract();
  phase.target = static_cast<MonitoringPhase>(request.target_phase);
  phase.roi.minimum_R = Eigen::Vector3d(request.roi_min.x, request.roi_min.y, request.roi_min.z);
  phase.roi.maximum_R = Eigen::Vector3d(request.roi_max.x, request.roi_max.y, request.roi_max.z);
  phase.effective_stamp = request.effective_stamp;
  if (request.trigger_source >
      static_cast<uint8_t>(TriggerSource::EXTERNAL_TASK_CURRENT_DWELL))
    return reject_contract();
  phase.trigger_source = static_cast<TriggerSource>(request.trigger_source);
  phase.selection_support_cutoff = request.selection_support_cutoff;
  ParseStableIdMessage(request.expected_previous_snapshot, &phase.expected_previous_snapshot);
  const PhaseResponse result = pipeline_->SetPhase(phase);
  response.accepted = result.accepted; response.phase = static_cast<uint8_t>(result.phase);
  response.reference_epoch = result.reference_epoch;
  response.reference_snapshot_id = BuildStableIdMessage(result.reference_snapshot_id);
  for (ReasonCode reason : result.reasons) response.reason_codes.push_back(static_cast<uint16_t>(reason));
  PublishAttemptStatus(request.effective_stamp, next_accepted_frame_index_,
                       false, result.reasons);
  return true;
}

bool V1Node::TokenMatches(const deform_monitor_v3::V1StableId128& token,
                          ReasonCode* rejection_reason,
                          ros::Time* evaluation_stamp) const {
  if (rejection_reason)
    *rejection_reason = ReasonCode::QUERY_CONTRACT_INVALID;
  if (!StableIdRegistryHealthy()) {
    if (rejection_reason)
      *rejection_reason = StableIdRegistryResourceExhausted()
          ? ReasonCode::RESOURCE_LIMIT
          : ReasonCode::INVALID_ID_COLLISION;
    return false;
  }
  const ros::Time now = ros::Time::now();
  if (evaluation_stamp) *evaluation_stamp = now;
  StableId parsed;
  if (!ParseStableIdMessage(token, &parsed) || parsed.empty() ||
      parsed != pipeline_->last_snapshot().common.snapshot_id) {
    return false;
  }
  const CommonHeader& common = pipeline_->last_snapshot().common;
  if (common.publish_stamp.isZero() || common.valid_until.isZero() ||
      now < common.publish_stamp || now > common.valid_until ||
      (now - common.publish_stamp).toSec() >
          config_.resource.query_ttl_s) {
    if (rejection_reason)
      *rejection_reason = ReasonCode::SNAPSHOT_STALE;
    return false;
  }
  return true;
}

bool V1Node::QueryDeformationService(
    deform_monitor_v3::QueryV1Deformation::Request& request,
    deform_monitor_v3::QueryV1Deformation::Response& response) {
  StableId gdu_id;
  ReasonCode token_rejection = ReasonCode::QUERY_CONTRACT_INVALID;
  if (!ParseStableIdMessage(request.gdu_id, &gdu_id) || gdu_id.empty() ||
      !TokenMatches(request.snapshot_token, &token_rejection) ||
      request.query_time.isZero() ||
      request.side_id < -1 || request.side_id > 1) {
    response.support_validity = static_cast<uint8_t>(
        token_rejection == ReasonCode::SNAPSHOT_STALE
            ? Validity::STALE : Validity::INVALID);
    response.reason_codes.push_back(
        static_cast<uint16_t>(token_rejection));
    return true;
  }
  const V1Snapshot& snapshot = pipeline_->last_snapshot();
  const double horizon =
      (request.query_time - snapshot.common.measurement_stamp).toSec();
  if (request.query_time < snapshot.common.measurement_stamp ||
      request.query_time > snapshot.common.prediction_valid_until ||
      !(horizon >= 0.0) ||
      horizon > config_.resource.prediction_horizons_s.back()) {
    response.support_validity = static_cast<uint8_t>(Validity::UNKNOWN);
    response.reason_codes.push_back(
        static_cast<uint16_t>(ReasonCode::UNKNOWN_PREDICTION));
    return true;
  }
  const ReferenceSnapshot& reference = pipeline_->reference();
  const auto found = std::find_if(reference.gdus.begin(), reference.gdus.end(),
      [&](const GduDefinition& gdu) { return gdu.id == gdu_id; });
  if (found == reference.gdus.end()) {
    response.support_validity = static_cast<uint8_t>(Validity::INVALID);
    response.reason_codes.push_back(static_cast<uint16_t>(ReasonCode::OUTSIDE_SUPPORT));
    return true;
  }
  const Eigen::Vector3d point(request.point_on_reference.x,
                              request.point_on_reference.y,
                              request.point_on_reference.z);
  if (!point.allFinite()) {
    response.support_validity = static_cast<uint8_t>(Validity::INVALID);
    response.reason_codes.push_back(
        static_cast<uint16_t>(ReasonCode::QUERY_CONTRACT_INVALID));
    return true;
  }
  const uint32_t index = static_cast<uint32_t>(found - reference.gdus.begin());
  if (index >= snapshot.deformation.size()) {
    response.support_validity = static_cast<uint8_t>(Validity::INVALID);
    response.reason_codes.push_back(
        static_cast<uint16_t>(ReasonCode::INVALID_NUMERICAL));
    return true;
  }
  // side_id is 0 away from an interface, -1 for the frozen left side and +1
  // for the frozen right side. At an interface, an omitted or inconsistent
  // side is ambiguous and must not be averaged across the discontinuity.
  const InterfaceDefinition* nearest_interface = nullptr;
  double nearest_normalized_distance = std::numeric_limits<double>::infinity();
  for (uint32_t interface_index : reference.gdus[index].interface_indices) {
    const InterfaceDefinition& interface = reference.interfaces[interface_index];
    const double normalized = (point - interface.center_R).norm() / interface.scale_m;
    if (normalized < 1.0 &&
        (normalized < nearest_normalized_distance - config_.numerical.tie_tolerance ||
         (std::abs(normalized - nearest_normalized_distance) <=
              config_.numerical.tie_tolerance &&
          (!nearest_interface || interface.id < nearest_interface->id)))) {
      nearest_interface = &interface;
      nearest_normalized_distance = normalized;
    }
  }
  if (nearest_interface) {
    const int expected_side = InterfaceSideForPoint(
        reference, *nearest_interface, index, point,
        config_.numerical.tie_tolerance);
    if (expected_side == 0) {
      response.support_validity = static_cast<uint8_t>(Validity::UNKNOWN);
      response.reason_codes.push_back(
          static_cast<uint16_t>(ReasonCode::OUTSIDE_SUPPORT));
      return true;
    }
    if (request.side_id != expected_side) {
      response.support_validity = static_cast<uint8_t>(Validity::UNKNOWN);
      response.reason_codes.push_back(static_cast<uint16_t>(ReasonCode::OUTSIDE_SUPPORT));
      return true;
    }
  } else if (request.side_id != 0) {
    response.support_validity = static_cast<uint8_t>(Validity::INVALID);
    response.reason_codes.push_back(
        static_cast<uint16_t>(ReasonCode::QUERY_CONTRACT_INVALID));
    return true;
  }
  const FieldQueryResult result = pipeline_->filter().QueryAtHorizon(index, point, horizon);
  response.displacement_mean.x = result.mean_displacement_R.x();
  response.displacement_mean.y = result.mean_displacement_R.y();
  response.displacement_mean.z = result.mean_displacement_R.z();
  response.velocity_mean.x = result.mean_velocity_R.x();
  response.velocity_mean.y = result.mean_velocity_R.y();
  response.velocity_mean.z = result.mean_velocity_R.z();
  FillDense3(result.displacement_covariance_R, &response.displacement_covariance);
  FillDense3(result.velocity_covariance_R, &response.velocity_covariance);
  FillDense3(result.observable_projector_R, &response.observable_projector);
  response.minimum_detectable_effect_defined =
      snapshot.deformation[index].minimum_detectable_effect_defined;
  response.minimum_detectable_effect_m = response.minimum_detectable_effect_defined
      ? snapshot.deformation[index].minimum_detectable_effect_m : 0.0;
  const Validity combined_validity = CombineValidity(
      result.validity, snapshot.common.validity);
  response.support_validity = static_cast<uint8_t>(combined_validity);
  response.interpolation_error_upper_m =
      std::isfinite(result.interpolation_error_upper_m)
          ? result.interpolation_error_upper_m
          : config_.resource.surfel_seed_spacing_m +
                config_.calibration.prediction_model_error_m +
                config_.calibration.ctmc_halving_error_upper_m;
  response.other_mass = result.other_mass;
  for (const FieldComponentQuery& component : result.components) {
    deform_monitor_v3::V1DeformationComponentQuery output;
    output.component_id = BuildStableIdMessage(component.component_id);
    output.weight = component.weight;
    output.structural_state = static_cast<uint8_t>(component.state);
    output.geometry_hypothesis = static_cast<uint8_t>(component.geometry);
    output.displacement_mean.x = component.mean_displacement_R.x();
    output.displacement_mean.y = component.mean_displacement_R.y();
    output.displacement_mean.z = component.mean_displacement_R.z();
    output.velocity_mean.x = component.mean_velocity_R.x();
    output.velocity_mean.y = component.mean_velocity_R.y();
    output.velocity_mean.z = component.mean_velocity_R.z();
    FillDense3(component.displacement_covariance_R,
               &output.displacement_covariance);
    FillDense3(component.velocity_covariance_R,
               &output.velocity_covariance);
    FillDense3(component.observable_projector_R,
               &output.observable_projector);
    output.minimum_detectable_effect_defined =
        component.minimum_detectable_effect_defined;
    output.minimum_detectable_effect_m =
        component.minimum_detectable_effect_defined
            ? component.minimum_detectable_effect_m : 0.0;
    response.components.push_back(output);
  }
  for (ReasonCode reason : result.reasons)
    response.reason_codes.push_back(static_cast<uint16_t>(reason));
  for (ReasonCode reason : snapshot.common.reasons)
    response.reason_codes.push_back(static_cast<uint16_t>(reason));
  std::sort(response.reason_codes.begin(), response.reason_codes.end());
  response.reason_codes.erase(
      std::unique(response.reason_codes.begin(), response.reason_codes.end()),
      response.reason_codes.end());
  return true;
}

}  // namespace v1
}  // namespace deform_monitor_v3
