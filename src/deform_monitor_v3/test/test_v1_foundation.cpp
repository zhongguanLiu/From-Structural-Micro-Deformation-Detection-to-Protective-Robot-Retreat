#include <gtest/gtest.h>

#include "deform_monitor_v3/v1/association.hpp"
#include "deform_monitor_v3/v1/config.hpp"
#include "deform_monitor_v3/v1/deterministic.hpp"
#include "deform_monitor_v3/v1/input_contract.hpp"
#include "deform_monitor_v3/v1/math.hpp"
#include "deform_monitor_v3/v1/occupancy.hpp"
#include "deform_monitor_v3/v1/pipeline.hpp"
#include "deform_monitor_v3/v1/pose_adapter.hpp"
#include "deform_monitor_v3/v1/reference_graph.hpp"
#include "deform_monitor_v3/v1/risk_query.hpp"
#include "deform_monitor_v3/v1/ros_bridge.hpp"

#include <ros/serialization.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>

namespace deform_monitor_v3 {
namespace v1 {

struct UacGaussianSumFilterTestAccess {
  static bool PredictUncached(UacGaussianSumFilter* filter,
                              const ros::Time& stamp,
                              std::vector<ReasonCode>* reasons) {
    return filter && filter->PredictIntervalImpl(
        stamp, true, false, false, reasons);
  }

  static bool PredictCachedSerial(UacGaussianSumFilter* filter,
                                  const ros::Time& stamp,
                                  std::vector<ReasonCode>* reasons) {
    return filter && filter->PredictIntervalImpl(
        stamp, true, true, false, reasons);
  }

  static bool UpdateSerial(
      UacGaussianSumFilter* filter,
      const std::vector<std::vector<AssociationMeasurementSet>>& measurements,
      const std::vector<FrameNuisance,
          Eigen::aligned_allocator<FrameNuisance>>& nuisances,
      const ros::Time& stamp, std::vector<ReasonCode>* reasons) {
    return filter && filter->UpdateImpl(
        measurements, nuisances, stamp, false, reasons);
  }

  static std::vector<GduPosterior,
      Eigen::aligned_allocator<GduPosterior>> PosteriorSerial(
          const UacGaussianSumFilter& filter) {
    return filter.PosteriorImpl(false);
  }

  static std::vector<GduPosterior,
      Eigen::aligned_allocator<GduPosterior>>
  PosteriorAfterWorkerCreationFailure(
      const UacGaussianSumFilter& filter,
      size_t successful_worker_creation_count) {
    return filter.PosteriorImpl(
        true, successful_worker_creation_count);
  }

  static std::vector<InterfacePosterior,
      Eigen::aligned_allocator<InterfacePosterior>>
  InterfacePosteriorsGeneric(
      const UacGaussianSumFilter& filter,
      const std::vector<GduPosterior,
          Eigen::aligned_allocator<GduPosterior>>& posterior) {
    return filter.InterfacePosteriorsImpl(posterior, false);
  }

  static std::pair<
      std::vector<InterfacePosterior,
          Eigen::aligned_allocator<InterfacePosterior>>,
      std::vector<InterfacePosterior,
          Eigen::aligned_allocator<InterfacePosterior>>>
  InterfacePosteriorsWithMissingCache(
      UacGaussianSumFilter* filter,
      const std::vector<GduPosterior,
          Eigen::aligned_allocator<GduPosterior>>& posterior,
      size_t gdu_index) {
    if (!filter || gdu_index >= filter->states_.size()) return {};
    auto saved = filter->states_[gdu_index].frozen_reference_cache;
    filter->states_[gdu_index].frozen_reference_cache.reset();
    auto cached_path = filter->InterfacePosteriorsImpl(posterior, true);
    auto generic_path = filter->InterfacePosteriorsImpl(posterior, false);
    filter->states_[gdu_index].frozen_reference_cache = std::move(saved);
    return std::make_pair(std::move(cached_path), std::move(generic_path));
  }

  static std::vector<uint8_t> MutableStateBytes(
      const UacGaussianSumFilter& filter) {
    CanonicalBytes bytes;
    const auto add_time = [&bytes](const ros::Time& stamp) {
      bytes.AddU32(stamp.sec);
      bytes.AddU32(stamp.nsec);
    };
    const auto add_matrix = [&bytes](const auto& matrix) {
      bytes.AddI64(matrix.rows());
      bytes.AddI64(matrix.cols());
      for (Eigen::Index index = 0; index < matrix.size(); ++index)
        bytes.AddDouble(matrix.data()[index]);
    };
    const auto add_reason_vector = [&bytes](
        const std::vector<ReasonCode>& reasons) {
      bytes.AddU64(static_cast<uint64_t>(reasons.size()));
      for (ReasonCode reason : reasons)
        bytes.AddI64(static_cast<int64_t>(reason));
    };
    const auto add_component = [&](const GaussianComponent& component) {
      bytes.AddI64(static_cast<int64_t>(component.state));
      bytes.AddI64(static_cast<int64_t>(component.geometry));
      bytes.AddU64(static_cast<uint64_t>(component.interface_modes.size()));
      for (InterfaceJointState mode : component.interface_modes)
        bytes.AddI64(static_cast<int64_t>(mode));
      bytes.AddId(component.association_lineage);
      bytes.AddU64(
          static_cast<uint64_t>(component.active_fixed_lag_lineage.size()));
      for (const StableId& id : component.active_fixed_lag_lineage)
        bytes.AddId(id);
      bytes.AddI64(component.contamination_scale);
      bytes.AddId(component.parent_id);
      bytes.AddId(component.component_id);
      bytes.AddDouble(component.weight);
      add_matrix(component.mean);
      add_matrix(component.covariance);
      add_matrix(component.sqrt_information);
      bytes.AddI64(component.observable_rank);
      add_matrix(component.observable_projector);
      add_matrix(component.observable_nullspace);
      bytes.AddDouble(component.minimum_detectable_effect_m);
      bytes.AddU8(component.minimum_detectable_effect_defined ? 1U : 0U);
    };

    bytes.AddU8(filter.initialized_ ? 1U : 0U);
    bytes.AddU8(filter.first_observation_reanchor_available_ ? 1U : 0U);
    bytes.AddU32(filter.approximation_flags_);
    add_time(filter.last_stamp_);
    bytes.AddU64(static_cast<uint64_t>(filter.states_.size()));
    for (const UacGaussianSumFilter::GduState& state : filter.states_) {
      bytes.AddI64(state.coefficient_count);
      if (!state.frozen_reference_cache) return {};
      bytes.AddU64(static_cast<uint64_t>(
          state.frozen_reference_cache->task_bases.size()));
      for (const Eigen::MatrixXd& basis :
           state.frozen_reference_cache->task_bases)
        add_matrix(basis);
      add_matrix(state.frozen_reference_cache->task_metric);
      bytes.AddU64(static_cast<uint64_t>(
          state.frozen_reference_cache->task_quadrature_count));
      bytes.AddU64(static_cast<uint64_t>(
          state.frozen_reference_cache->surfel_bases.size()));
      for (const auto& basis : state.frozen_reference_cache->surfel_bases) {
        add_matrix(basis.basis);
        bytes.AddI64(static_cast<int64_t>(basis.validity));
        add_reason_vector(basis.reasons);
        bytes.AddDouble(basis.attachment_distance_m);
      }
      bytes.AddU64(static_cast<uint64_t>(
          state.frozen_reference_cache->surfel_basis_offsets.size()));
      for (const auto& offset :
           state.frozen_reference_cache->surfel_basis_offsets) {
        bytes.AddU32(offset.first);
        bytes.AddU64(static_cast<uint64_t>(offset.second));
      }
      add_matrix(state.reference_bias_prior_covariance);
      add_matrix(state.last_pose_prior_covariance);
      add_time(state.last_pose_stamp);
      bytes.AddU8(state.has_pose_prior ? 1U : 0U);
      bytes.AddId(state.nuisance_correlation_id);
      for (const Eigen::MatrixXd& information :
           state.current_observable_information)
        add_matrix(information);
      for (double value : state.reference_information_ratio)
        bytes.AddDouble(value);
      for (bool value : state.reference_confounding)
        bytes.AddU8(value ? 1U : 0U);
      for (double value : state.effective_spatial_support)
        bytes.AddDouble(value);
      for (double value : state.effective_temporal_support)
        bytes.AddDouble(value);
      for (const auto& history : state.information_weight_history) {
        bytes.AddU64(static_cast<uint64_t>(history.size()));
        for (const auto& sample : history) {
          add_time(sample.first);
          bytes.AddDouble(sample.second);
        }
      }
      for (const auto& history : state.observable_frame_history) {
        bytes.AddU64(static_cast<uint64_t>(history.size()));
        for (const auto& block : history) {
          add_time(block.stamp);
          add_matrix(block.design);
          add_matrix(block.independent_covariance);
          add_matrix(block.reference_bias_design);
          add_matrix(block.extrinsic_design);
          add_matrix(block.pose_design);
          add_matrix(block.pose_covariance);
        }
      }
      bytes.AddDouble(state.other_mass);
      bytes.AddI64(state.consecutive_source_missing);
      bytes.AddU8(state.current_measurement_valid ? 1U : 0U);
      bytes.AddDouble(state.continuous_valid_velocity_duration_s);
      add_time(state.last_valid_velocity_stamp);
      bytes.AddU64(static_cast<uint64_t>(state.components.size()));
      for (const GaussianComponent& component : state.components)
        add_component(component);
    }
    bytes.AddU64(static_cast<uint64_t>(filter.shared_interfaces_.size()));
    for (const auto& shared : filter.shared_interfaces_) {
      for (double probability : shared.probability)
        bytes.AddDouble(probability);
      for (const Eigen::Vector3d& mean : shared.mean) add_matrix(mean);
      for (const Eigen::Matrix3d& covariance : shared.covariance)
        add_matrix(covariance);
      bytes.AddDouble(shared.other_mass);
    }
    return bytes.bytes();
  }

  static std::vector<const void*> FrozenCacheIdentities(
      const UacGaussianSumFilter& filter) {
    std::vector<const void*> identities;
    identities.reserve(filter.states_.size());
    for (const auto& state : filter.states_)
      identities.push_back(state.frozen_reference_cache.get());
    return identities;
  }
};

struct V1PipelineTestAccess {
  static uint64_t ReferenceBuildAttempts(const V1Pipeline& pipeline) {
    return pipeline.reference_build_attempts_;
  }
  static bool InstallFrozenReference(
      V1Pipeline* pipeline, const ReferenceSnapshot& reference,
      const AxisAlignedRoi& roi, std::vector<ReasonCode>* reasons) {
    if (!pipeline) return false;
    pipeline->reference_ = reference;
    pipeline->reference_epoch_ = reference.epoch;
    pipeline->roi_ = roi;
    return pipeline->PrepareMeasurementReferenceCache(reasons);
  }

  static bool BuildMeasurements(
      const V1Pipeline& pipeline, const FrameBundle& frame,
      const UacGaussianSumFilter& predicted_filter,
      bool include_unmonitorable_matching_for_test,
      std::vector<std::vector<AssociationMeasurementSet>>* measurements,
      std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>*
          nuisances,
      std::vector<ObservationExplanation>* explanations,
      std::vector<ReasonCode>* reasons, MeasurementBuildStats* stats) {
    return pipeline.BuildMeasurements(
        frame, predicted_filter, measurements, nuisances, explanations, reasons,
        include_unmonitorable_matching_for_test, stats);
  }
};

}  // namespace v1
}  // namespace deform_monitor_v3

using namespace deform_monitor_v3::v1;

namespace {

StableId TestId(const std::string& label, uint64_t index) {
  CanonicalBytes fields;
  fields.AddString(label);
  fields.AddU64(index);
  return MakeStableId("v1-test-id", fields);
}

CanonicalPoint TestPoint(double x, PointRole role, uint64_t index) {
  CanonicalPoint point;
  point.position_R = Eigen::Vector3d(x, 0.05, 0.05);
  point.role = role;
  point.point_id = TestId("point", index);
  point.evidence_id = TestId("evidence", index);
  point.correlation_group_id = TestId("test-frame-correlation", index / 10000U);
  return point;
}

FrameBundle ReferenceRayFrame(uint64_t index) {
  FrameBundle frame;
  frame.session_id = "occupancy-test";
  frame.frame_id = TestId("reference-frame", index);
  frame.accepted_frame_index = index;
  frame.stamp = ros::Time(1U + static_cast<uint32_t>(index), 0U);
  frame.world_frame = "camera_init";
  frame.pose.origin_L_R = Eigen::Vector3d(0.05, 0.05, 0.05);
  frame.points.push_back(TestPoint(1.05, PointRole::OCCUPANCY, index));
  frame.validity = Validity::VALID;
  return frame;
}

FrameBundle CurrentRayFrame(uint64_t index, double endpoint_x,
                            int repeated_rays) {
  FrameBundle frame;
  frame.session_id = "occupancy-test";
  frame.frame_id = TestId("current-frame", index);
  frame.accepted_frame_index = index;
  frame.stamp = ros::Time(200U + static_cast<uint32_t>(index), 0U);
  frame.world_frame = "camera_init";
  frame.pose.origin_L_R = Eigen::Vector3d(0.05, 0.05, 0.05);
  for (int ray = 0; ray < repeated_rays; ++ray) {
    frame.points.push_back(TestPoint(
        endpoint_x, PointRole::ASSOCIATION, 10000U * index + 2U * ray));
    frame.points.push_back(TestPoint(
        endpoint_x, PointRole::OCCUPANCY,
        10000U * index + 2U * ray + 1U));
  }
  frame.validity = Validity::VALID;
  return frame;
}

fast_lio::LioOdomCov ValidFastLioMessage() {
  fast_lio::LioOdomCov message;
  message.odom.header.stamp = ros::Time(10U, 20U);
  message.odom.header.frame_id = "camera_init";
  message.odom.pose.pose.position.x = 1.0;
  message.odom.pose.pose.position.y = -2.0;
  message.odom.pose.pose.position.z = 0.5;
  const Eigen::Quaterniond quaternion(
      Eigen::AngleAxisd(0.3, Eigen::Vector3d(1.0, -2.0, 0.5).normalized()));
  message.odom.pose.pose.orientation.w = quaternion.w();
  message.odom.pose.pose.orientation.x = quaternion.x();
  message.odom.pose.pose.orientation.y = quaternion.y();
  message.odom.pose.pose.orientation.z = quaternion.z();
  message.state_dim = 23U;
  message.state_order =
      "pos,rot,offset_R_L_I,offset_T_L_I,vel,bg,ba,grav";
  Eigen::Matrix<double, 23, 23> covariance =
      Eigen::Matrix<double, 23, 23>::Zero();
  Eigen::Matrix<double, 6, 6> factor;
  factor << 0.02, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.01, 0.03, 0.00, 0.00, 0.00, 0.00,
           -0.01, 0.01, 0.04, 0.00, 0.00, 0.00,
            0.01, 0.00, 0.01, 0.02, 0.00, 0.00,
            0.00, 0.01, 0.00, 0.01, 0.03, 0.00,
            0.00, 0.00, 0.01, 0.00, 0.01, 0.02;
  covariance.topLeftCorner<6, 6>() = factor * factor.transpose();
  covariance.bottomRightCorner<17, 17>().diagonal().setConstant(1.0e-6);
  message.state_covariance.resize(23U * 23U);
  for (int row = 0; row < 23; ++row)
    for (int column = 0; column < 23; ++column)
      message.state_covariance[static_cast<size_t>(23 * row + column)] =
          covariance(row, column);
  return message;
}

AssociationFeature AssociationCandidate(uint64_t index, double distance) {
  AssociationFeature feature;
  feature.candidate_id = TestId("association-candidate", index);
  feature.distance_lower_bound = distance;
  feature.continuous_projection = 0.7 - 0.02 * distance;
  feature.projector_chordal_distance = 0.1 + 0.01 * distance;
  feature.log_support_count_ratio = 0.05 * distance;
  feature.overlap = 0.8 - 0.03 * distance;
  feature.independent_visibility = 0.9;
  feature.prediction_continuity = 0.6 + 0.01 * distance;
  return feature;
}

ReferenceSnapshot MetricLineReference(bool with_interface) {
  ReferenceSnapshot reference;
  reference.epoch = 1U;
  reference.world_frame = "camera_init";
  reference.frozen = true;
  reference.validity = Validity::VALID;
  for (uint32_t index = 0; index < 6U; ++index) {
    Surfel surfel;
    surfel.id = TestId("metric-line-surfel", index);
    surfel.center_R = Eigen::Vector3d(0.1 * index, 0.0, 0.0);
    surfel.support_radius_m = 0.05;
    surfel.gdu_index = 0U;
    surfel.observable_projector_R = Eigen::Matrix3d::Identity();
    reference.surfels.push_back(surfel);
  }
  for (uint32_t index = 0; index < 5U; ++index) {
    GraphEdge edge;
    edge.id = TestId("metric-line-edge", index);
    edge.first = index; edge.second = index + 1U;
    edge.distance_m = 0.1;
    edge.p_link = 1.0;
    edge.state = ReferenceEdgeState::LINK;
    reference.edges.push_back(edge);
  }
  GduDefinition gdu;
  gdu.id = TestId("metric-line-gdu", 0U);
  gdu.surfel_indices = {0U, 1U, 2U, 3U, 4U, 5U};
  gdu.control_surfel_indices = gdu.surfel_indices;
  gdu.coordinate_axes_R = Eigen::Matrix3d::Identity();
  gdu.coordinate_rank = 3U;
  gdu.coordinate_geometry = GeometryHypothesis::ISOTROPIC;
  gdu.monitorable = true;
  if (with_interface) {
    InterfaceDefinition interface;
    interface.id = TestId("metric-line-interface", 0U);
    interface.cross_gdu = false;
    interface.left_gdu = 0U; interface.right_gdu = 0U;
    interface.left_surfel = 0U; interface.right_surfel = 5U;
    interface.left_region_surfels = {0U, 1U, 2U};
    interface.right_region_surfels = {3U, 4U, 5U};
    interface.center_R = Eigen::Vector3d(0.25, 0.0, 0.0);
    interface.scale_m = 0.1;
    interface.p_link = 1.0;
    interface.identifiable = true;
    reference.interfaces.push_back(interface);
    gdu.interface_indices.push_back(0U);
  }
  reference.gdus.push_back(gdu);
  return reference;
}

ReferenceSnapshot ReplicatedMetricLineReference(uint32_t gdu_count,
                                                bool with_interface) {
  ReferenceSnapshot reference;
  reference.epoch = 1U;
  reference.world_frame = "camera_init";
  reference.frozen = true;
  reference.validity = Validity::VALID;
  for (uint32_t g = 0U; g < gdu_count; ++g) {
    const uint32_t surfel_offset =
        static_cast<uint32_t>(reference.surfels.size());
    const double gdu_offset = 2.0 * static_cast<double>(g);
    for (uint32_t index = 0U; index < 6U; ++index) {
      Surfel surfel;
      surfel.id = TestId("replicated-metric-line-surfel",
                         6U * g + index);
      surfel.center_R = Eigen::Vector3d(
          gdu_offset + 0.1 * static_cast<double>(index), 0.0, 0.0);
      surfel.support_radius_m = 0.05;
      surfel.gdu_index = g;
      surfel.observable_projector_R = Eigen::Matrix3d::Identity();
      reference.surfels.push_back(surfel);
    }
    for (uint32_t index = 0U; index < 5U; ++index) {
      GraphEdge edge;
      edge.id = TestId("replicated-metric-line-edge", 5U * g + index);
      edge.first = surfel_offset + index;
      edge.second = surfel_offset + index + 1U;
      edge.distance_m = 0.1;
      edge.p_link = 1.0;
      edge.state = ReferenceEdgeState::LINK;
      reference.edges.push_back(edge);
    }
    GduDefinition gdu;
    gdu.id = TestId("replicated-metric-line-gdu", g);
    for (uint32_t index = 0U; index < 6U; ++index)
      gdu.surfel_indices.push_back(surfel_offset + index);
    gdu.control_surfel_indices = gdu.surfel_indices;
    gdu.coordinate_axes_R = Eigen::Matrix3d::Identity();
    gdu.coordinate_rank = 3U;
    gdu.coordinate_geometry = GeometryHypothesis::ISOTROPIC;
    gdu.monitorable = true;
    if (with_interface) {
      InterfaceDefinition interface;
      interface.id = TestId("replicated-metric-line-interface", g);
      interface.cross_gdu = false;
      interface.left_gdu = g;
      interface.right_gdu = g;
      interface.left_surfel = surfel_offset;
      interface.right_surfel = surfel_offset + 5U;
      interface.left_region_surfels = {
          surfel_offset, surfel_offset + 1U, surfel_offset + 2U};
      interface.right_region_surfels = {
          surfel_offset + 3U, surfel_offset + 4U, surfel_offset + 5U};
      interface.center_R = Eigen::Vector3d(gdu_offset + 0.25, 0.0, 0.0);
      interface.scale_m = 0.1;
      interface.p_link = 1.0;
      interface.competition_weight = 1.0;
      interface.identifiable = true;
      gdu.interface_indices.push_back(
          static_cast<uint32_t>(reference.interfaces.size()));
      reference.interfaces.push_back(interface);
    }
    reference.gdus.push_back(gdu);
  }
  return reference;
}

FrameBundle PipelineFrame(const V1Config& config, uint64_t index,
                          double stamp_s, bool split_roles) {
  FrameBundle frame;
  frame.session_id = config.resource.session_id;
  frame.frame_id = TestId("pipeline-frame", index);
  frame.accepted_frame_index = index;
  frame.stamp.fromSec(stamp_s);
  frame.world_frame = config.resource.world_frame;
  frame.pose.origin_L_R = Eigen::Vector3d::Zero();
  frame.pose.covariance = 1.0e-8 *
      Eigen::Matrix<double, 6, 6>::Identity();
  frame.extrinsic_revision = config.resource.extrinsic.revision;
  frame.extrinsic_digest = config.resource.extrinsic.digest;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      centers;
  centers.reserve(48U);
  for (int cluster = 0; cluster < 2; ++cluster) {
    const double base_x = cluster == 0 ? 0.40 : 2.00;
    for (int row = 0; row < 4; ++row) {
      for (int column = 0; column < 6; ++column) {
        centers.emplace_back(base_x + 0.10 * column,
                             0.20 + 0.10 * row, 0.02);
      }
    }
  }
  uint64_t local = 0U;
  for (const Eigen::Vector3d& center : centers) {
    const std::array<PointRole, 4> roles{{PointRole::POSE_CONTROL,
        PointRole::ASSOCIATION, PointRole::MEASUREMENT,
        PointRole::OCCUPANCY}};
    const size_t role_count = split_roles ? roles.size() : 1U;
    for (size_t role_index = 0U; role_index < role_count; ++role_index) {
      CanonicalPoint point;
      point.position_R = center;
      point.original_index = local;
      point.canonical_rank = local;
      point.role = split_roles ? roles[role_index] : PointRole::MEASUREMENT;
      point.point_id = TestId("pipeline-point", 1000U * index + local);
      point.evidence_id = TestId("pipeline-evidence", 1000U * index + local);
      point.correlation_group_id = frame.frame_id;
      frame.points.push_back(point);
      ++local;
    }
  }
  frame.validity = Validity::VALID;
  return frame;
}

std::deque<FrameBundle> PlanarExposureFrames(
    const V1Config& config, double alternating_normal_offset_m,
    double alternating_tangential_offset_m, bool reverse_point_order) {
  std::deque<FrameBundle> frames;
  for (uint64_t frame_index = 0U; frame_index < 6U; ++frame_index) {
    FrameBundle frame;
    frame.session_id = config.resource.session_id;
    frame.frame_id = TestId("planar-exposure-frame", frame_index);
    frame.accepted_frame_index = frame_index;
    frame.stamp.fromSec(1.0 + 0.1 * static_cast<double>(frame_index));
    frame.world_frame = config.resource.world_frame;
    frame.pose.origin_L_R = Eigen::Vector3d::Zero();
    frame.pose.covariance = 1.0e-8 *
        Eigen::Matrix<double, 6, 6>::Identity();
    frame.extrinsic_revision = config.resource.extrinsic.revision;
    frame.extrinsic_digest = config.resource.extrinsic.digest;
    const double sign = frame_index % 2U == 0U ? -1.0 : 1.0;
    uint64_t local_index = 0U;
    for (uint64_t row = 0U; row < 4U; ++row) {
      for (uint64_t column = 0U; column < 6U; ++column) {
        CanonicalPoint point;
        // The +/- offsets remain inside the same fixed 10 cm carrier cell.
        // X is tangential to the reference plane and Z is its observable
        // normal, so the two tests below differ only in the physically
        // relevant projection of their exposure-to-exposure variation.
        point.position_R = Eigen::Vector3d(
            0.05 + 0.10 * static_cast<double>(column) +
                sign * alternating_tangential_offset_m,
            0.05 + 0.10 * static_cast<double>(row),
            0.05 + sign * alternating_normal_offset_m);
        point.original_index = local_index;
        point.canonical_rank = local_index;
        point.role = PointRole::MEASUREMENT;
        point.point_id = TestId(
            "planar-exposure-point", 1000U * frame_index + local_index);
        point.evidence_id = TestId(
            "planar-exposure-evidence", 1000U * frame_index + local_index);
        point.correlation_group_id = frame.frame_id;
        frame.points.push_back(point);
        ++local_index;
      }
    }
    if (reverse_point_order) std::reverse(frame.points.begin(), frame.points.end());
    frame.validity = Validity::VALID;
    frames.push_back(frame);
  }
  return frames;
}

}  // namespace

TEST(V1Determinism, Sha256KnownVector) {
  const std::vector<uint8_t> abc{'a', 'b', 'c'};
  EXPECT_EQ(HexDigest(Sha256(abc)),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(V1Determinism, StableIdCollisionRegistryIsExplicitlyResourceBounded) {
  ResetStableIdRegistry();
  Sha256Digest reserved_digest{};
  reserved_digest.back() = 1U;
  EXPECT_FALSE(RegisterStableIdDigest(StableId(), reserved_digest));
  EXPECT_FALSE(StableIdRegistryHealthy());
  EXPECT_EQ(StableIdCollisionCount(), 1U);
  ResetStableIdRegistry();
  ASSERT_TRUE(ConfigureStableIdRegistryCapacity(2U));
  const StableId first = TestId("bounded-registry", 0U);
  const StableId second = TestId("bounded-registry", 1U);
  EXPECT_NE(first, second);
  EXPECT_EQ(StableIdRegistrySize(), 2U);
  EXPECT_TRUE(StableIdRegistryHealthy());
  static_cast<void>(TestId("bounded-registry", 2U));
  EXPECT_EQ(StableIdRegistrySize(), 2U);
  EXPECT_TRUE(StableIdRegistryResourceExhausted());
  EXPECT_FALSE(StableIdRegistryHealthy());
  EXPECT_EQ(StableIdCollisionCount(), 0U);
  ResetStableIdRegistry();
  ASSERT_TRUE(ConfigureStableIdRegistryCapacity(2000000U));
}

TEST(V1Determinism,
     StableIdCandidateTransactionRollsBackResourcesButNotCollisions) {
  ResetStableIdRegistry();
  ASSERT_TRUE(ConfigureStableIdRegistryCapacity(2U));
  {
    StableIdRegistryTransaction failed_candidate;
    ASSERT_TRUE(failed_candidate.active());
    static_cast<void>(TestId("failed-candidate", 0U));
    static_cast<void>(TestId("failed-candidate", 1U));
    static_cast<void>(TestId("failed-candidate", 2U));
    EXPECT_TRUE(StableIdRegistryResourceExhausted());
    EXPECT_FALSE(StableIdRegistryHealthy());
  }
  EXPECT_EQ(StableIdRegistrySize(), 0U);
  EXPECT_TRUE(StableIdRegistryHealthy());
  EXPECT_TRUE(StableIdRegistryCanRegister(2U));

  StableId committed;
  {
    StableIdRegistryTransaction accepted_candidate;
    ASSERT_TRUE(accepted_candidate.active());
    committed = TestId("accepted-candidate", 0U);
    ASSERT_TRUE(accepted_candidate.Commit());
  }
  EXPECT_EQ(StableIdRegistrySize(), 0U);
  EXPECT_TRUE(StableIdRegistryHealthy());

  {
    StableIdRegistryTransaction colliding_candidate;
    ASSERT_TRUE(colliding_candidate.active());
    committed = TestId("accepted-candidate", 0U);
    Sha256Digest different_digest{};
    different_digest.fill(0xa5U);
    EXPECT_FALSE(RegisterStableIdDigest(committed, different_digest));
    EXPECT_FALSE(StableIdRegistryHealthy());
  }
  EXPECT_EQ(StableIdRegistrySize(), 0U);
  EXPECT_EQ(StableIdCollisionCount(), 1U);
  EXPECT_FALSE(StableIdRegistryHealthy());

  ResetStableIdRegistry();
  ASSERT_TRUE(ConfigureStableIdRegistryCapacity(2000000U));
}

TEST(V1Determinism,
     StableIdAtomicAuditCapacityDoesNotAccumulateAcrossCommittedFrames) {
  ResetStableIdRegistry();
  ASSERT_TRUE(ConfigureStableIdRegistryCapacity(3U));
  for (uint64_t frame = 0U; frame < 256U; ++frame) {
    StableIdRegistryTransaction input_frame;
    ASSERT_TRUE(input_frame.active());
    static_cast<void>(TestId("frame-id", frame));
    static_cast<void>(TestId("point-id", frame));
    static_cast<void>(TestId("evidence-id", frame));
    ASSERT_TRUE(StableIdRegistryHealthy());
    ASSERT_TRUE(input_frame.Commit());
    EXPECT_EQ(StableIdRegistrySize(), 0U);
    EXPECT_TRUE(StableIdRegistryCanRegister(3U));
  }
  ResetStableIdRegistry();
  ASSERT_TRUE(ConfigureStableIdRegistryCapacity(2000000U));
}

TEST(V1Determinism,
     SparseSurfaceRoleSplitIsDeterministicAndMatchesFrozenProportions) {
  constexpr uint64_t kSampleCount = 100000U;
  std::array<uint64_t, 4> counts{{0U, 0U, 0U, 0U}};
  for (uint64_t rank = 0U; rank < kSampleCount; ++rank) {
    const PointRole role = RoleForPoint(
        "role-split-contract", 3U, 17U, rank);
    const PointRole repeated = RoleForPoint(
        "role-split-contract", 3U, 17U, rank);
    EXPECT_EQ(role, repeated);
    switch (role) {
      case PointRole::POSE_CONTROL: ++counts[0]; break;
      case PointRole::ASSOCIATION: ++counts[1]; break;
      case PointRole::MEASUREMENT: ++counts[2]; break;
      case PointRole::OCCUPANCY: ++counts[3]; break;
    }
  }
  const double denominator = static_cast<double>(kSampleCount);
  EXPECT_NEAR(static_cast<double>(counts[0]) / denominator, 0.15, 0.005);
  EXPECT_NEAR(static_cast<double>(counts[1]) / denominator, 0.30, 0.005);
  EXPECT_NEAR(static_cast<double>(counts[2]) / denominator, 0.45, 0.005);
  EXPECT_NEAR(static_cast<double>(counts[3]) / denominator, 0.10, 0.005);
  EXPECT_EQ(counts[0] + counts[1] + counts[2] + counts[3], kSampleCount);
}

TEST(V1InputContract,
     StrictFastLioWireSchemaRejectsSilentFallbacksBeforePcl) {
  sensor_msgs::PointCloud2 cloud;
  cloud.width = 1U;
  cloud.height = 1U;
  cloud.is_bigendian = false;
  cloud.point_step = 48U;
  cloud.row_step = 48U;
  cloud.data.resize(48U, 0U);
  const auto field = [](const std::string& name, uint32_t offset) {
    sensor_msgs::PointField output;
    output.name = name;
    output.offset = offset;
    output.datatype = sensor_msgs::PointField::FLOAT32;
    output.count = 1U;
    return output;
  };
  cloud.fields = {field("x", 0U), field("y", 4U), field("z", 8U),
                  field("intensity", 32U), field("normal_x", 16U),
                  field("normal_y", 20U), field("normal_z", 24U),
                  field("curvature", 36U)};
  ReasonCode reason = ReasonCode::INPUT_SCHEMA_INVALID;
  std::string error;
  EXPECT_TRUE(ValidateFastLioRegisteredCloudSchema(
      cloud, 1U, &reason, &error));
  EXPECT_EQ(reason, ReasonCode::NONE);

  sensor_msgs::PointCloud2 missing_intensity = cloud;
  missing_intensity.fields.erase(missing_intensity.fields.begin() + 3);
  EXPECT_FALSE(ValidateFastLioRegisteredCloudSchema(
      missing_intensity, 1U, &reason, &error));
  EXPECT_EQ(reason, ReasonCode::INPUT_SCHEMA_INVALID);

  sensor_msgs::PointCloud2 extra_field = cloud;
  extra_field.fields.push_back(field("ring", 40U));
  EXPECT_FALSE(ValidateFastLioRegisteredCloudSchema(
      extra_field, 1U, &reason, &error));
  EXPECT_EQ(reason, ReasonCode::INPUT_SCHEMA_INVALID);

  sensor_msgs::PointCloud2 oversized = cloud;
  oversized.width = 2U;
  oversized.row_step = 96U;
  oversized.data.resize(96U);
  EXPECT_FALSE(ValidateFastLioRegisteredCloudSchema(
      oversized, 1U, &reason, &error));
  EXPECT_EQ(reason, ReasonCode::RESOURCE_LIMIT);

  sensor_msgs::PointCloud2 big_endian = cloud;
  big_endian.is_bigendian = true;
  EXPECT_FALSE(ValidateFastLioRegisteredCloudSchema(
      big_endian, 1U, &reason, &error));
  EXPECT_EQ(reason, ReasonCode::INPUT_SCHEMA_INVALID);
}

TEST(V1Determinism, CheckedGridKeyAndPsdRootFailClosedWithoutEigenbasisChoice) {
  VoxelKey key;
  EXPECT_TRUE(TryGridKey(Eigen::Vector3d(-0.01, 0.09, 0.10),
                         Eigen::Vector3d::Zero(), 0.1, &key));
  EXPECT_EQ(key, (VoxelKey{-1, 0, 1}));
  EXPECT_FALSE(TryGridKey(Eigen::Vector3d(1.0e300, 0.0, 0.0),
                          Eigen::Vector3d::Zero(), 0.1, &key));

  V1Config config = SyntheticKnownConfig("psd-root-test");
  Eigen::Matrix3d covariance;
  covariance << 2.0, 1.0, 0.0,
                1.0, 2.0, 0.0,
                0.0, 0.0, 0.0;
  Eigen::MatrixXd first;
  Eigen::MatrixXd second;
  ASSERT_TRUE(DeterministicPsdSquareRoot(
      covariance, config.numerical, &first));
  ASSERT_TRUE(DeterministicPsdSquareRoot(
      covariance, config.numerical, &second));
  EXPECT_EQ(first, second);
  EXPECT_NEAR((first * first.transpose() - covariance).norm(), 0.0, 1.0e-12);
  EXPECT_TRUE(first.triangularView<Eigen::StrictlyUpper>()
                  .toDenseMatrix().isZero());
  Eigen::MatrixXd pseudoinverse_first;
  Eigen::MatrixXd pseudoinverse_second;
  Eigen::MatrixXd information_factor;
  ASSERT_TRUE(DeterministicPsdInformationFactor(
      covariance, config.numerical, &information_factor));
  EXPECT_EQ(information_factor.rows(), 2);
  EXPECT_EQ(information_factor.cols(), 3);
  ASSERT_TRUE(DeterministicPsdPseudoInverse(
      covariance, config.numerical, &pseudoinverse_first));
  ASSERT_TRUE(DeterministicPsdPseudoInverse(
      covariance, config.numerical, &pseudoinverse_second));
  EXPECT_EQ(pseudoinverse_first, pseudoinverse_second);
  EXPECT_NEAR((information_factor.transpose() * information_factor -
               pseudoinverse_first).norm(),
              0.0, 1.0e-12);
  EXPECT_NEAR((covariance * pseudoinverse_first * covariance - covariance)
                  .norm(),
              0.0, 1.0e-12);
  EXPECT_NEAR((pseudoinverse_first * covariance * pseudoinverse_first -
               pseudoinverse_first).norm(),
              0.0, 1.0e-12);
}

TEST(V1Determinism,
     SpatialPreflightRejectsBeforeIdentityAuditAllocation) {
  ResetStableIdRegistry();
  ASSERT_TRUE(ConfigureStableIdRegistryCapacity(32U));
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI point;
  point.x = 1.0e30F;
  point.y = 0.0F;
  point.z = 0.0F;
  point.intensity = 1.0F;
  cloud.push_back(point);
  const size_t identities_before = StableIdRegistrySize();
  ReasonCode reason = ReasonCode::NONE;
  std::string error;
  EXPECT_FALSE(PreflightPointCloudSpatialDomain(
      cloud, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.1, 0.1,
      &reason, &error));
  EXPECT_EQ(reason, ReasonCode::RESOURCE_LIMIT);
  EXPECT_EQ(StableIdRegistrySize(), identities_before);
  EXPECT_TRUE(StableIdRegistryHealthy());
  EXPECT_TRUE(StableIdRegistryCanRegister(32U));
  EXPECT_FALSE(StableIdRegistryCanRegister(33U));
  ResetStableIdRegistry();
  ASSERT_TRUE(ConfigureStableIdRegistryCapacity(2000000U));
}

TEST(V1Determinism, MonitoringTransactionOwnsFrameIdentityEpochBeforeFreeze) {
  V1Config config;
  config.resource.session_id = "identity-epoch-test";
  V1Pipeline pipeline(config);
  EXPECT_EQ(pipeline.reference_epoch(), 0U);
  EXPECT_FALSE(pipeline.reference().frozen);

  PhaseRequest request;
  request.session_id = config.resource.session_id;
  request.target = MonitoringPhase::MONITOR_REFERENCE;
  request.roi.minimum_R = Eigen::Vector3d::Constant(-1.0);
  request.roi.maximum_R = Eigen::Vector3d::Constant(1.0);
  request.effective_stamp = ros::Time(10, 0);
  request.selection_support_cutoff = ros::Time(10, 0);
  const PhaseResponse response = pipeline.SetPhase(request);
  ASSERT_TRUE(response.accepted);
  EXPECT_EQ(pipeline.reference_epoch(), 1U);
  EXPECT_EQ(response.reference_epoch, 1U);
  // No fallback window has yet been collected, so reading the empty snapshot
  // would still return its default epoch and is therefore not an identity
  // source.
  EXPECT_FALSE(pipeline.reference().frozen);
  EXPECT_EQ(pipeline.reference().epoch, 0U);
}

TEST(V1Math, CtmcTransitionIsStochasticAndHasSemigroup) {
  const Eigen::Matrix3d q = UacGenerator({{0.03, 0.01, 0.2, 0.04, 0.08}});
  const Eigen::Matrix3d direct = CtmcTransition(q, 0.2);
  const Eigen::Matrix3d split = CtmcTransition(q, 0.1) * CtmcTransition(q, 0.1);
  EXPECT_TRUE(direct.isApprox(split, 1.0e-13));
  for (int row = 0; row < 3; ++row) {
    EXPECT_NEAR(direct.row(row).sum(), 1.0, 1.0e-14);
    EXPECT_GE(direct.row(row).minCoeff(), 0.0);
  }
  EXPECT_DOUBLE_EQ(q(2, 0), 0.0);
}

TEST(V1Math, WendlandAndNegativeBinomialMatchDefinitions) {
  EXPECT_DOUBLE_EQ(WendlandC2(0.0), 1.0);
  EXPECT_DOUBLE_EQ(WendlandC2(1.0), 0.0);
  EXPECT_NEAR(LogNegativeBinomialPmf(0, 2.0, 5.0),
              5.0 * (std::log(5.0) - std::log(7.0)), 1.0e-14);
}

TEST(V1Math, UnknownDependenceUnionDoesNotInventAPosteriorMean) {
  const ProbabilityInterval first{0.20, 0.25, 0.30, true, true};
  const ProbabilityInterval second{0.35, 0.40, 0.45, true, true};
  const ProbabilityInterval singleton = ConservativeUnion({first}, 0.01);
  EXPECT_TRUE(singleton.mean_identified);
  EXPECT_NEAR(singleton.mean, first.mean, 1.0e-15);
  EXPECT_NEAR(singleton.lower, 0.19, 1.0e-15);
  EXPECT_NEAR(singleton.upper, 0.31, 1.0e-15);

  const ProbabilityInterval unknown_joint =
      ConservativeUnion({first, second}, 0.01);
  EXPECT_FALSE(unknown_joint.mean_identified);
  EXPECT_NEAR(unknown_joint.lower, 0.34, 1.0e-15);
  EXPECT_NEAR(unknown_joint.upper, 0.76, 1.0e-15);
  // A legacy consumer sees the conservative endpoint, while a conforming
  // consumer additionally knows that it is not a posterior union mean.
  EXPECT_DOUBLE_EQ(unknown_joint.mean, unknown_joint.upper);
}

TEST(V1Math, NegativeBinomialAndVisibilityMixtureMatchDirectMoments) {
  const double mean = 2.3;
  const double dispersion = 4.5;
  double total = 0.0, first = 0.0, second = 0.0;
  for (int count = 0; count <= 300; ++count) {
    const double probability = std::exp(
        LogNegativeBinomialPmf(count, mean, dispersion));
    total += probability;
    first += static_cast<double>(count) * probability;
    second += static_cast<double>(count * count) * probability;
  }
  EXPECT_NEAR(total, 1.0, 1.0e-13);
  EXPECT_NEAR(first, mean, 1.0e-12);
  EXPECT_NEAR(second - first * first,
              mean + mean * mean / dispersion, 1.0e-11);
  const double visibility = 0.35;
  EXPECT_NEAR(std::exp(LogVisibilityMixedNegativeBinomial(
                  0, mean, dispersion, visibility, 1.0e-12)),
              (1.0 - visibility) + visibility *
                  std::exp(LogNegativeBinomialPmf(0, mean, dispersion)),
              1.0e-15);
  EXPECT_NEAR(std::exp(LogVisibilityMixedNegativeBinomial(
                  3, mean, dispersion, visibility, 1.0e-12)),
              visibility *
                  std::exp(LogNegativeBinomialPmf(3, mean, dispersion)),
              1.0e-15);
}

TEST(V1Math, ReferenceConfoundingUsesKnownInformationPositiveRange) {
  Eigen::Matrix3d known = Eigen::Matrix3d::Zero();
  known.diagonal() << 4.0, 1.0, 0.0;
  Eigen::Matrix3d marginalized = Eigen::Matrix3d::Zero();
  marginalized.diagonal() << 1.0, 0.5, 100.0;
  double ratio = -1.0;
  int rank = 0;
  ASSERT_TRUE(MinimumRetainedInformationRatio(
      known, marginalized, 1.0e-12, &ratio, &rank));
  EXPECT_EQ(rank, 2);
  EXPECT_NEAR(ratio, 0.25, 1.0e-15);
  marginalized.topLeftCorner<2, 2>() = known.topLeftCorner<2, 2>();
  ASSERT_TRUE(MinimumRetainedInformationRatio(
      known, marginalized, 1.0e-12, &ratio, &rank));
  EXPECT_NEAR(ratio, 1.0, 1.0e-15);
}

TEST(V1Math, MdeInformationIsNormalizedByPhysicalFieldMetric) {
  Eigen::Matrix2d information = Eigen::Matrix2d::Zero();
  information.diagonal() << 4.0, 1.0;
  Eigen::Matrix2d task_metric = Eigen::Matrix2d::Zero();
  task_metric.diagonal() << 1.0, 4.0;
  double weakest = -1.0;
  double strongest = -1.0;
  int effective_rank = 0, task_rank = 0;
  ASSERT_TRUE(MinimumTaskNormalizedInformation(
      information, task_metric, 1.0e-12, &weakest,
      &effective_rank, &task_rank, &strongest));
  EXPECT_EQ(effective_rank, 2);
  EXPECT_EQ(task_rank, 2);
  EXPECT_NEAR(weakest, 0.25, 1.0e-15);
  EXPECT_NEAR(strongest, 4.0, 1.0e-15);

  information(1, 1) = 0.0;
  ASSERT_TRUE(MinimumTaskNormalizedInformation(
      information, task_metric, 1.0e-12, &weakest,
      &effective_rank, &task_rank, &strongest));
  EXPECT_EQ(effective_rank, 1);
  EXPECT_EQ(task_rank, 2);
  EXPECT_DOUBLE_EQ(weakest, 0.0);
  EXPECT_NEAR(strongest, 4.0, 1.0e-15);
  EXPECT_FALSE(std::isfinite(MinimumDetectableEffectFromInformation(
      weakest, effective_rank, 0.01, 0.9)));

  // The second coefficient has zero task amplitude and may vary freely.
  // Observing only their sum contains no guaranteed information about the
  // first coefficient after that zero-metric direction is marginalized.
  information << 1.0, 1.0,
                 1.0, 1.0;
  task_metric << 1.0, 0.0,
                 0.0, 0.0;
  ASSERT_TRUE(MinimumTaskNormalizedInformation(
      information, task_metric, 1.0e-12, &weakest,
      &effective_rank, &task_rank, &strongest));
  EXPECT_EQ(task_rank, 1);
  EXPECT_EQ(effective_rank, 0);
  EXPECT_NEAR(weakest, 0.0, 1.0e-12);
  EXPECT_NEAR(strongest, 0.0, 1.0e-12);
}

TEST(V1Math, StudentScaleQuadratureMatchesAnalyticContaminationDensity) {
  CalibrationConfig calibration;
  calibration.contamination_probability = 0.05;
  calibration.student_t_dof = 5.0;
  calibration.student_t_scale = 3.0;
  const double sigma = 0.003;
  const auto nodes = GaussianStudentTScaleMixture(
      calibration.contamination_probability, calibration.student_t_dof, 8);
  double mass = 0.0;
  for (const RobustScaleNode& node : nodes) mass += node.prior_weight;
  EXPECT_NEAR(mass, 1.0, 1.0e-14);
  for (double residual : {0.0, sigma, 3.0 * sigma}) {
    const double scale = calibration.student_t_scale * sigma;
    const double nu = calibration.student_t_dof;
    const double gaussian = std::exp(GaussianLogDensity(
        residual, sigma * sigma));
    const double student = std::tgamma(0.5 * (nu + 1.0)) /
        (std::tgamma(0.5 * nu) * std::sqrt(nu * M_PI) * scale) *
        std::pow(1.0 + residual * residual /
                          (nu * scale * scale),
                 -0.5 * (nu + 1.0));
    const double oracle = (1.0 - calibration.contamination_probability) *
        gaussian + calibration.contamination_probability * student;
    const double quadrature = std::exp(RobustGaussianStudentLogDensity(
        residual, 0.0, sigma * sigma, calibration));
    EXPECT_NEAR(quadrature / oracle, 1.0, 3.0e-4);
  }
}

TEST(V1Config, SyntheticProfileIsClosedAndDigestStable) {
  V1Config first = SyntheticKnownConfig("00000000-0000-0000-0000-000000000001");
  V1Config second = first;
  ASSERT_TRUE(ValidateConfig(first).valid);
  ASSERT_TRUE(ValidateFrozenV11Profile(first).valid);
  EXPECT_EQ(CanonicalConfigDigest(first), CanonicalConfigDigest(second));
  second.resource.control_spacing_m += 0.01;
  EXPECT_NE(CanonicalConfigDigest(first), CanonicalConfigDigest(second));
  EXPECT_FALSE(ValidateFrozenV11Profile(second).valid);
  V1Config pose_numerical = first;
  pose_numerical.calibration.pose_control_huber_scale_m = 0.021;
  EXPECT_FALSE(ValidateFrozenV11Profile(pose_numerical).valid);
}

TEST(V1Config, LongerRealReferenceIsAllowedWithoutWeakeningMinimumEvidence) {
  V1Config config = SyntheticKnownConfig("real-reference-profile");
  const auto original_digest = CanonicalConfigDigest(config);
  config.resource.reference_min_frames = 10;
  config.resource.reference_min_surfel_exposures = 6;
  config.resource.bounded_reference_search = true;
  config.resource.reference_min_duration_s = 2.0;
  config.resource.max_raw_buffer_points = 5000000;
  config.calibration.independent_point_sigma_m = 0.006;
  config.calibration.reference_epoch_common_bias_sigma_m = 0.005;
  config.calibration.reference_stability_sigma = 1.5;
  config.calibration.task_velocity_required_duration_s = 1.0;
  EXPECT_TRUE(ValidateConfig(config).valid);
  EXPECT_TRUE(ValidateFrozenV11Profile(config).valid);
  EXPECT_NE(CanonicalConfigDigest(config), original_digest);
  config.resource.reference_min_surfel_exposures = 11;
  EXPECT_FALSE(ValidateConfig(config).valid);
  config.resource.reference_min_surfel_exposures = 5;
  EXPECT_FALSE(ValidateFrozenV11Profile(config).valid);
  config.resource.reference_min_surfel_exposures = 8;
  config.resource.reference_min_duration_s = 0.1;
  EXPECT_FALSE(ValidateFrozenV11Profile(config).valid);
}

TEST(V1Config, RejectsUnreachableFixedLagTemporalSupportThreshold) {
  V1Config config = SyntheticKnownConfig("unreachable-support-test");
  config.resource.fixed_lag_frames = 3;
  config.resource.max_update_gap_s = 0.2;
  config.calibration.pose_ou_tau_s = 0.5;
  config.calibration.detection_min_effective_temporal_support = 2.0;
  const ConfigValidation validation = ValidateConfig(config);
  EXPECT_FALSE(validation.valid);
  EXPECT_NE(std::find_if(
                validation.errors.begin(), validation.errors.end(),
                [](const std::string& error) {
                  return error.find("temporal support threshold is unreachable") !=
                         std::string::npos;
                }),
            validation.errors.end());
}

TEST(V1Config, RejectsFieldDimensionBeyondDeterministicSobolCapacity) {
  V1Config config = SyntheticKnownConfig("sobol-capacity-test");
  config.resource.max_controls_per_gdu = 128U;
  config.resource.max_interfaces = 512U;
  const ConfigValidation validation = ValidateConfig(config);
  EXPECT_FALSE(validation.valid);
  EXPECT_NE(std::find_if(
                validation.errors.begin(), validation.errors.end(),
                [](const std::string& error) {
                  return error.find("Sobol dimension limit") !=
                         std::string::npos;
                }),
            validation.errors.end());
}

TEST(V1Config, RejectsRobotBodySamplingGridBeyondResourceCapacity) {
  V1Config config = SyntheticKnownConfig("body-grid-capacity-test");
  config.resource.max_body_query_samples = 64U;
  const ConfigValidation validation = ValidateConfig(config);
  EXPECT_FALSE(validation.valid);
  EXPECT_NE(std::find_if(
                validation.errors.begin(), validation.errors.end(),
                [](const std::string& error) {
                  return error.find("body query grid exceeds") !=
                         std::string::npos;
                }),
            validation.errors.end());
}

TEST(V1Config, RejectsDegeneratePredictionHorizonAndInvalidCalibrationSimplex) {
  V1Config horizon = SyntheticKnownConfig("horizon-contract-test");
  horizon.resource.prediction_horizons_s = {0.0};
  EXPECT_FALSE(ValidateConfig(horizon).valid);

  V1Config simplex = SyntheticKnownConfig("simplex-contract-test");
  simplex.calibration.interface_initial_link_probability = {{-0.1, 0.5, 0.6}};
  EXPECT_FALSE(ValidateConfig(simplex).valid);

  V1Config reference = SyntheticKnownConfig("reference-threshold-test");
  reference.calibration.reference_free_threshold = 0.8;
  reference.calibration.reference_occupied_threshold = 0.7;
  EXPECT_FALSE(ValidateConfig(reference).valid);

  V1Config query_budget = SyntheticKnownConfig("query-budget-test");
  query_budget.resource.max_risk_query_operations = 0U;
  EXPECT_FALSE(ValidateConfig(query_budget).valid);

  V1Config id_budget = SyntheticKnownConfig("id-budget-test");
  id_budget.resource.max_stable_ids = 0U;
  EXPECT_FALSE(ValidateConfig(id_budget).valid);

  V1Config non_atomic_id_budget =
      SyntheticKnownConfig("non-atomic-id-budget-test");
  non_atomic_id_budget.resource.max_stable_ids =
      2U * non_atomic_id_budget.resource.max_points_per_frame;
  EXPECT_FALSE(ValidateConfig(non_atomic_id_budget).valid);

  V1Config evidence_budget = SyntheticKnownConfig("evidence-budget-test");
  evidence_budget.resource.max_evidence_links =
      evidence_budget.resource.max_points_per_frame - 1U;
  EXPECT_FALSE(ValidateConfig(evidence_budget).valid);

  V1Config primitive_budget = SyntheticKnownConfig("primitive-budget-test");
  primitive_budget.resource.max_hazard_primitives =
      primitive_budget.resource.max_events - 1U;
  EXPECT_FALSE(ValidateConfig(primitive_budget).valid);

  V1Config mixture_budget = SyntheticKnownConfig("mixture-budget-test");
  mixture_budget.resource.mixture_max_components =
      static_cast<int>(mixture_budget.resource.max_events + 1U);
  EXPECT_FALSE(ValidateConfig(mixture_budget).valid);
}

TEST(V1Config, RejectsNonFiniteScalarsAndMissingSoftwareRevision) {
  V1Config config = SyntheticKnownConfig("nonfinite-config-test");
  config.resource.deadline_ms = std::numeric_limits<double>::infinity();
  config.policy.maximum_soft_cost =
      std::numeric_limits<double>::infinity();
  config.calibration.student_t_scale =
      std::numeric_limits<double>::infinity();
  config.software_revision.clear();
  EXPECT_FALSE(ValidateConfig(config).valid);
}

TEST(V1Pose, FastLioJacobianContainsRotationAndLeverArm) {
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Vector3d lever(0.2, -0.1, 0.05);
  const auto jacobian = FastLioToLidarJacobian(rotation, lever);
  EXPECT_TRUE((jacobian.block<3, 3>(0, 3).isApprox(rotation)));
  EXPECT_TRUE((jacobian.block<3, 3>(3, 0).isApprox(Eigen::Matrix3d::Identity())));
  EXPECT_TRUE((jacobian.block<3, 3>(3, 3).isApprox(-rotation * Skew(lever))));
}

TEST(V1Pose, FastLioJacobianMatchesRightBoxplusFiniteDifference) {
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(0.4, Eigen::Vector3d(1.0, 2.0, -1.0).normalized())
          .toRotationMatrix();
  const Eigen::Vector3d lever(0.2, -0.1, 0.05);
  const Eigen::Matrix<double, 6, 6> jacobian =
      FastLioToLidarJacobian(rotation, lever);
  const double epsilon = 1.0e-7;
  for (int column = 0; column < 6; ++column) {
    Eigen::Matrix<double, 6, 1> numerical =
        Eigen::Matrix<double, 6, 1>::Zero();
    if (column < 3) {
      numerical.tail<3>()[column] = 1.0;
    } else {
      Eigen::Vector3d axis = Eigen::Vector3d::Zero();
      axis[column - 3] = epsilon;
      const Eigen::Matrix3d perturbed = rotation *
          Eigen::AngleAxisd(epsilon, axis.normalized()).toRotationMatrix();
      const Eigen::Matrix3d delta_rotation = perturbed * rotation.transpose();
      numerical.head<3>() = Eigen::Vector3d(
          delta_rotation(2, 1) - delta_rotation(1, 2),
          delta_rotation(0, 2) - delta_rotation(2, 0),
          delta_rotation(1, 0) - delta_rotation(0, 1)) /
          (2.0 * epsilon);
      numerical.tail<3>() =
          (perturbed * lever - rotation * lever) / epsilon;
    }
    EXPECT_TRUE(numerical.isApprox(jacobian.col(column), 1.0e-6));
  }
}

TEST(V1Pose, StrictAdapterRejectsInvalidSchemaAndMatchesDenseTransform) {
  const V1Config config = SyntheticKnownConfig("pose-adapter-test");
  ExtrinsicCalibration extrinsic = config.resource.extrinsic;
  extrinsic.revision = "test-extrinsic-v1";
  extrinsic.digest = "test-extrinsic-digest";
  extrinsic.t_IL = Eigen::Vector3d(0.2, -0.1, 0.05);
  fast_lio::LioOdomCov message = ValidFastLioMessage();
  const PoseAdapterResult valid = AdaptFastLioPose(
      message, extrinsic, config.numerical);
  ASSERT_TRUE(valid.valid) << valid.detail;
  const geometry_msgs::Quaternion& q = message.odom.pose.pose.orientation;
  const Eigen::Matrix3d rotation =
      Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix();
  EXPECT_TRUE(valid.pose.origin_L_R.isApprox(
      Eigen::Vector3d(1.0, -2.0, 0.5) + rotation * extrinsic.t_IL,
      1.0e-14));
  Eigen::Matrix<double, 6, 6> source;
  for (int row = 0; row < 6; ++row)
    for (int column = 0; column < 6; ++column)
      source(row, column) = message.state_covariance[
          static_cast<size_t>(23 * row + column)];
  const Eigen::Matrix<double, 6, 6> jacobian =
      FastLioToLidarJacobian(rotation, extrinsic.t_IL);
  EXPECT_TRUE(valid.pose.covariance.isApprox(
      jacobian * source * jacobian.transpose(), 1.0e-14));

  fast_lio::LioOdomCov invalid = message;
  invalid.state_dim = 22U;
  EXPECT_EQ(AdaptFastLioPose(invalid, extrinsic, config.numerical).reason,
            ReasonCode::INPUT_SCHEMA_INVALID);
  invalid = message;
  invalid.state_order = "rot,pos";
  EXPECT_FALSE(AdaptFastLioPose(invalid, extrinsic, config.numerical).valid);
  invalid = message;
  invalid.state_covariance.pop_back();
  EXPECT_FALSE(AdaptFastLioPose(invalid, extrinsic, config.numerical).valid);
  invalid = message;
  invalid.state_covariance[0] =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(AdaptFastLioPose(invalid, extrinsic, config.numerical).reason,
            ReasonCode::INPUT_POSE_COVARIANCE_INVALID);
  invalid = message;
  invalid.state_covariance.assign(23U * 23U, 0.0);
  invalid.state_covariance[0] = -1.0;
  EXPECT_EQ(AdaptFastLioPose(invalid, extrinsic, config.numerical).reason,
            ReasonCode::INPUT_POSE_COVARIANCE_INVALID);
  extrinsic.online_estimation_enabled = true;
  EXPECT_EQ(AdaptFastLioPose(message, extrinsic, config.numerical).reason,
            ReasonCode::INPUT_EXTRINSIC_STATE_UNAVAILABLE);
}

TEST(V1Evidence, CanonicalPointIdsAndRolesArePermutationInvariant) {
  pcl::PointCloud<pcl::PointXYZI> first;
  for (const auto& values : std::vector<std::array<float, 4>>{{
           {{0.3F, -0.1F, 1.0F, 8.0F}},
           {{0.1F, 0.2F, 0.5F, 2.0F}},
           {{0.2F, 0.0F, 0.7F, 4.0F}}}}) {
    pcl::PointXYZI point;
    point.x = values[0]; point.y = values[1]; point.z = values[2];
    point.intensity = values[3]; first.push_back(point);
  }
  pcl::PointCloud<pcl::PointXYZI> second;
  second.push_back(first[2]); second.push_back(first[0]); second.push_back(first[1]);
  CanonicalPointVector first_output, second_output;
  StableId first_frame, second_frame;
  std::string error;
  ASSERT_TRUE(CanonicalizePointCloud(first, "point-permutation", 4U,
      ros::Time(20U, 30U), 7U, &first_output, &first_frame, &error));
  ASSERT_TRUE(CanonicalizePointCloud(second, "point-permutation", 4U,
      ros::Time(20U, 30U), 7U, &second_output, &second_frame, &error));
  ASSERT_EQ(first_frame, second_frame);
  ASSERT_EQ(first_output.size(), second_output.size());
  std::set<StableId> evidence_ids;
  for (size_t index = 0; index < first_output.size(); ++index) {
    EXPECT_TRUE(first_output[index].position_R.isApprox(
        second_output[index].position_R, 0.0));
    EXPECT_EQ(first_output[index].point_id, second_output[index].point_id);
    EXPECT_EQ(first_output[index].evidence_id,
              second_output[index].evidence_id);
    EXPECT_EQ(first_output[index].role, second_output[index].role);
    EXPECT_EQ(first_output[index].correlation_group_id, first_frame);
    EXPECT_EQ(second_output[index].correlation_group_id, second_frame);
    evidence_ids.insert(first_output[index].evidence_id);
  }
  EXPECT_EQ(evidence_ids.size(), first_output.size());
}

TEST(V1Evidence, DisabledIntensityCannotChangeOnlineIdentityOrInputContract) {
  pcl::PointCloud<pcl::PointXYZI> low_retro;
  pcl::PointCloud<pcl::PointXYZI> high_retro;
  const std::vector<Eigen::Vector3f> geometry = {
      Eigen::Vector3f(0.1F, 0.2F, 0.5F),
      // Exact duplicate geometry exercises the old intensity ordering
      // tie-break without changing the geometric detection input.
      Eigen::Vector3f(0.1F, 0.2F, 0.5F),
      Eigen::Vector3f(0.2F, -0.1F, 0.7F),
      Eigen::Vector3f(0.3F, 0.0F, 1.0F)};
  const std::array<float, 4> first_intensity{{1.0F, 2.0F, 3.0F, 4.0F}};
  const std::array<float, 4> second_intensity{{250.0F, -20.0F, 80.0F, 0.5F}};
  for (size_t index = 0; index < geometry.size(); ++index) {
    pcl::PointXYZI first;
    first.x = geometry[index].x();
    first.y = geometry[index].y();
    first.z = geometry[index].z();
    first.intensity = first_intensity[index];
    low_retro.push_back(first);
    pcl::PointXYZI second = first;
    second.intensity = second_intensity[index];
    high_retro.push_back(second);
  }

  CanonicalPointVector first_points, second_points;
  StableId first_frame_id, second_frame_id;
  std::string error;
  const ros::Time stamp(30U, 40U);
  ASSERT_TRUE(CanonicalizePointCloud(
      low_retro, "intensity-disabled", 5U, stamp, 9U,
      &first_points, &first_frame_id, &error));
  ASSERT_TRUE(CanonicalizePointCloud(
      high_retro, "intensity-disabled", 5U, stamp, 9U,
      &second_points, &second_frame_id, &error));

  FrameBundle first_bundle;
  first_bundle.session_id = "intensity-disabled";
  first_bundle.frame_id = first_frame_id;
  first_bundle.accepted_frame_index = 9U;
  first_bundle.stamp = stamp;
  first_bundle.world_frame = "camera_init";
  first_bundle.pose.origin_L_R = Eigen::Vector3d(1.0, 2.0, 3.0);
  first_bundle.pose.R_RL =
      Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  first_bundle.points = first_points;
  FrameBundle second_bundle = first_bundle;
  second_bundle.frame_id = second_frame_id;
  second_bundle.points = second_points;

  ASSERT_EQ(first_bundle.frame_id, second_bundle.frame_id);
  ASSERT_TRUE(first_bundle.pose.origin_L_R.isApprox(
      second_bundle.pose.origin_L_R, 0.0));
  ASSERT_TRUE(first_bundle.pose.R_RL.isApprox(
      second_bundle.pose.R_RL, 0.0));
  ASSERT_EQ(first_bundle.points.size(), second_bundle.points.size());
  for (size_t index = 0; index < first_bundle.points.size(); ++index) {
    const CanonicalPoint& first = first_bundle.points[index];
    const CanonicalPoint& second = second_bundle.points[index];
    EXPECT_TRUE(first.position_R.isApprox(second.position_R, 0.0));
    EXPECT_EQ(first.canonical_rank, second.canonical_rank);
    EXPECT_EQ(first.point_id, second.point_id);
    EXPECT_EQ(first.evidence_id, second.evidence_id);
    EXPECT_EQ(first.correlation_group_id, second.correlation_group_id);
    EXPECT_EQ(first.role, second.role);
    EXPECT_EQ(first.intensity, 0.0F);
    EXPECT_EQ(second.intensity, 0.0F);
  }
}

TEST(V1Association, MultinomialTopThreeTailIsPermutationInvariant) {
  V1Config config = SyntheticKnownConfig("association-test");
  config.resource.association_max_candidates = 3;
  AssociationModel model(config);
  std::vector<AssociationFeature> forward;
  for (uint64_t index = 0; index < 6U; ++index)
    forward.push_back(AssociationCandidate(index, 0.1 * index));
  std::vector<AssociationFeature> reverse = forward;
  std::reverse(reverse.begin(), reverse.end());
  const AssociationResult first = model.Evaluate(forward, true);
  const AssociationResult second = model.Evaluate(reverse, true);
  ASSERT_TRUE(first.valid);
  ASSERT_TRUE(second.valid);
  ASSERT_EQ(first.categories.size(), second.categories.size());
  double total = 0.0;
  for (size_t index = 0; index < first.categories.size(); ++index) {
    EXPECT_EQ(first.categories[index].candidate_id,
              second.categories[index].candidate_id);
    EXPECT_EQ(first.categories[index].semantics,
              second.categories[index].semantics);
    EXPECT_NEAR(first.categories[index].probability,
                second.categories[index].probability, 1.0e-15);
    total += first.categories[index].probability;
  }
  EXPECT_NEAR(total, 1.0, 1.0e-15);
  EXPECT_GT(first.other_mass, 0.0);
  const AssociationResult invisible = model.Evaluate({}, false);
  ASSERT_TRUE(invisible.valid);
  ASSERT_EQ(invisible.categories.size(), 2U);
  EXPECT_DOUBLE_EQ(invisible.categories[0].probability, 0.0);
  EXPECT_DOUBLE_EQ(invisible.other_mass, 1.0);
}

TEST(V1Association,
     DisjointRoleMatchingIsMaximumCardinalityUniqueAndPermutationInvariant) {
  DisjointPatchDescriptor a0;
  a0.id = TestId("disjoint-association", 0U);
  a0.gdu_index = 3U;
  a0.center_R = Eigen::Vector3d(0.0, 0.0, 0.0);
  DisjointPatchDescriptor a1;
  a1.id = TestId("disjoint-association", 1U);
  a1.gdu_index = 3U;
  a1.center_R = Eigen::Vector3d(0.09, 0.0, 0.0);
  DisjointPatchDescriptor m0;
  m0.id = TestId("disjoint-measurement", 0U);
  m0.gdu_index = 3U;
  m0.center_R = Eigen::Vector3d(0.01, 0.0, 0.0);
  DisjointPatchDescriptor m1;
  m1.id = TestId("disjoint-measurement", 1U);
  m1.gdu_index = 3U;
  m1.center_R = Eigen::Vector3d(-0.09, 0.0, 0.0);

  const auto first = MatchDisjointPatchPairs(
      DisjointPatchDescriptors{a0, a1},
      DisjointPatchDescriptors{m0, m1}, 0.11, 1.0e-12);
  const auto permuted = MatchDisjointPatchPairs(
      DisjointPatchDescriptors{a1, a0},
      DisjointPatchDescriptors{m1, m0}, 0.11, 1.0e-12);
  ASSERT_EQ(first.size(), 2U)
      << "augmenting-path matching must retain both feasible pairs";
  EXPECT_EQ(first, permuted);
  std::set<StableId> used_measurements;
  for (const auto& pair : first)
    EXPECT_TRUE(used_measurements.insert(pair.second).second);

  DisjointPatchDescriptor other_gdu = m0;
  other_gdu.id = TestId("disjoint-measurement", 2U);
  other_gdu.gdu_index = 4U;
  EXPECT_TRUE(MatchDisjointPatchPairs(
      DisjointPatchDescriptors{a0},
      DisjointPatchDescriptors{other_gdu}, 1.0, 1.0e-12).empty());
}

TEST(V1Association, MultiGduMatchingEqualsTheUnionOfIndependentGduMatches) {
  DisjointPatchDescriptors association;
  DisjointPatchDescriptors measurement;
  for (uint32_t gdu = 0U; gdu < 4U; ++gdu) {
    for (uint64_t index = 0U; index < 5U; ++index) {
      DisjointPatchDescriptor candidate;
      candidate.id = TestId("multi-gdu-association",
                            10U * static_cast<uint64_t>(gdu) + index);
      candidate.gdu_index = gdu;
      candidate.center_R = Eigen::Vector3d(
          3.0 * static_cast<double>(gdu) + 0.08 * index,
          0.01 * static_cast<double>(index % 2U), 0.0);
      association.push_back(candidate);
    }
    for (uint64_t index = 0U; index < 6U; ++index) {
      DisjointPatchDescriptor candidate;
      candidate.id = TestId("multi-gdu-measurement",
                            10U * static_cast<uint64_t>(gdu) + index);
      candidate.gdu_index = gdu;
      candidate.center_R = Eigen::Vector3d(
          3.0 * static_cast<double>(gdu) + 0.08 * index + 0.015,
          0.01 * static_cast<double>((index + 1U) % 2U), 0.0);
      measurement.push_back(candidate);
    }
  }
  const auto combined = MatchDisjointPatchPairs(
      association, measurement, 0.11, 1.0e-12);
  std::map<StableId, StableId> independent_union;
  for (uint32_t gdu = 0U; gdu < 4U; ++gdu) {
    DisjointPatchDescriptors local_association;
    DisjointPatchDescriptors local_measurement;
    for (const DisjointPatchDescriptor& patch : association)
      if (patch.gdu_index == gdu) local_association.push_back(patch);
    for (const DisjointPatchDescriptor& patch : measurement)
      if (patch.gdu_index == gdu) local_measurement.push_back(patch);
    const auto local = MatchDisjointPatchPairs(
        local_association, local_measurement, 0.11, 1.0e-12);
    independent_union.insert(local.begin(), local.end());
  }
  EXPECT_EQ(combined, independent_union);
}

TEST(V1Association, SparseSpatialMatchingEqualsExhaustiveMatching) {
  DisjointPatchDescriptors association;
  DisjointPatchDescriptors measurement;
  for (uint32_t gdu = 0U; gdu < 3U; ++gdu) {
    for (uint64_t index = 0U; index < 40U; ++index) {
      DisjointPatchDescriptor a;
      a.id = TestId("sparse-oracle-association",
                    100U * static_cast<uint64_t>(gdu) + index);
      a.gdu_index = gdu;
      a.center_R = Eigen::Vector3d(
          -3.7 + 0.31 * static_cast<double>(index),
          0.17 * static_cast<double>(gdu),
          0.03 * static_cast<double>(index % 3U));
      association.push_back(a);
      DisjointPatchDescriptor m;
      m.id = TestId("sparse-oracle-measurement",
                    100U * static_cast<uint64_t>(gdu) + index);
      m.gdu_index = gdu;
      m.center_R = a.center_R + Eigen::Vector3d(
          index % 2U == 0U ? 0.04 : -0.04, 0.01, -0.005);
      measurement.push_back(m);
    }
  }
  const auto exhaustive = [](DisjointPatchDescriptors association_input,
                             DisjointPatchDescriptors measurement_input,
                             double maximum_distance_m,
                             double tie_tolerance) {
    const auto by_id = [](const DisjointPatchDescriptor& lhs,
                          const DisjointPatchDescriptor& rhs) {
      return lhs.id < rhs.id;
    };
    std::sort(association_input.begin(), association_input.end(), by_id);
    std::sort(measurement_input.begin(), measurement_input.end(), by_id);
    struct Candidate { size_t measurement; double distance_m; };
    std::vector<std::vector<Candidate>> candidates(association_input.size());
    for (size_t a = 0U; a < association_input.size(); ++a) {
      for (size_t m = 0U; m < measurement_input.size(); ++m) {
        if (association_input[a].gdu_index !=
            measurement_input[m].gdu_index)
          continue;
        const double distance = (association_input[a].center_R -
                                 measurement_input[m].center_R).norm();
        if (distance <= maximum_distance_m + tie_tolerance)
          candidates[a].push_back(Candidate{m, distance});
      }
      std::sort(candidates[a].begin(), candidates[a].end(),
                [&](const Candidate& lhs, const Candidate& rhs) {
                  if (std::abs(lhs.distance_m - rhs.distance_m) >
                      tie_tolerance)
                    return lhs.distance_m < rhs.distance_m;
                  return measurement_input[lhs.measurement].id <
                         measurement_input[rhs.measurement].id;
                });
    }
    std::vector<size_t> order(association_input.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
      if (candidates[lhs].size() != candidates[rhs].size())
        return candidates[lhs].size() < candidates[rhs].size();
      return association_input[lhs].id < association_input[rhs].id;
    });
    const size_t unmatched = std::numeric_limits<size_t>::max();
    std::vector<size_t> owner(measurement_input.size(), unmatched);
    std::function<bool(size_t, std::vector<bool>*)> augment =
        [&](size_t a, std::vector<bool>* visited) {
          for (const Candidate& candidate : candidates[a]) {
            if ((*visited)[candidate.measurement]) continue;
            (*visited)[candidate.measurement] = true;
            const size_t previous = owner[candidate.measurement];
            if (previous == unmatched || augment(previous, visited)) {
              owner[candidate.measurement] = a;
              return true;
            }
          }
          return false;
        };
    for (size_t a : order) {
      std::vector<bool> visited(measurement_input.size(), false);
      static_cast<void>(augment(a, &visited));
    }
    std::map<StableId, StableId> result;
    for (size_t m = 0U; m < owner.size(); ++m)
      if (owner[m] != unmatched)
        result.emplace(association_input[owner[m]].id,
                       measurement_input[m].id);
    return result;
  };

  DisjointPatchMatchingStats sparse_stats;
  const auto sparse = MatchDisjointPatchPairs(
      association, measurement, 0.11, 1.0e-12, &sparse_stats);
  EXPECT_EQ(sparse, exhaustive(association, measurement, 0.11, 1.0e-12));
  EXPECT_LT(sparse_stats.distance_evaluations,
            3U * 40U * 40U);
}

TEST(V1Pipeline,
     UnmonitorableCarrierMatchingPruneIsPosteriorAndExplanationExact) {
  ros::Time::init();
  V1Config config = SyntheticKnownConfig(
      "unmonitorable-formal-matching-prune-test");
  config.resource.upstream_stationary_pose_freeze_verified = true;
  config.resource.pose_control_exclusion_radius_m =
      config.resource.graph_neighbor_radius_m;
  config.resource.max_tiles = 4096U;
  config.resource.max_points_per_frame = 4096U;
  config.resource.max_raw_buffer_points = 8192U;
  config.resource.deadline_ms = 100000.0;
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);

  ReferenceSnapshot reference = MetricLineReference(false);
  reference.id = TestId("unmonitorable-prune-reference", 0U);
  reference.begin_stamp = ros::Time(9U, 0U);
  reference.end_stamp = ros::Time(10U, 0U);
  for (Surfel& surfel : reference.surfels) {
    surfel.covariance_R =
        1.0e-6 * Eigen::Matrix3d::Identity();
    surfel.point_count = 8U;
    surfel.exposure_count = 6U;
  }

  constexpr uint32_t kUnmonitorableCarrierCount = 256U;
  GduDefinition unmonitorable;
  unmonitorable.id = TestId("unmonitorable-prune-gdu", 0U);
  unmonitorable.coordinate_rank = 0U;
  unmonitorable.coordinate_geometry = GeometryHypothesis::OTHER;
  unmonitorable.monitorable = false;
  for (uint32_t index = 0U; index < kUnmonitorableCarrierCount; ++index) {
    Surfel surfel;
    surfel.id = TestId("unmonitorable-prune-surfel", index);
    surfel.center_R = Eigen::Vector3d(
        10.0 + 0.30 * static_cast<double>(index), 0.0, 0.0);
    surfel.covariance_R =
        1.0e-6 * Eigen::Matrix3d::Identity();
    surfel.support_radius_m = 0.05;
    surfel.point_count = 8U;
    surfel.exposure_count = 6U;
    surfel.gdu_index = 1U;
    unmonitorable.surfel_indices.push_back(
        static_cast<uint32_t>(reference.surfels.size()));
    reference.surfels.push_back(surfel);
  }
  reference.gdus.push_back(unmonitorable);

  V1Pipeline pipeline(config);
  AxisAlignedRoi roi;
  roi.minimum_R = Eigen::Vector3d(-1.0, -1.0, -1.0);
  roi.maximum_R = Eigen::Vector3d(100.0, 1.0, 1.0);
  std::vector<ReasonCode> install_reasons;
  ASSERT_TRUE(V1PipelineTestAccess::InstallFrozenReference(
      &pipeline, reference, roi, &install_reasons));

  UacGaussianSumFilter predicted(config);
  std::vector<ReasonCode> initialization_reasons;
  ASSERT_TRUE(predicted.Initialize(
      reference, ros::Time(10U, 0U), &initialization_reasons));
  const ros::Time measurement_stamp(10U, 100000000U);
  ASSERT_TRUE(predicted.Predict(measurement_stamp, &initialization_reasons));

  FrameBundle frame;
  frame.session_id = config.resource.session_id;
  frame.frame_id = TestId("unmonitorable-prune-frame", 0U);
  frame.accepted_frame_index = 1U;
  frame.stamp = measurement_stamp;
  frame.world_frame = config.resource.world_frame;
  frame.pose.origin_L_R = Eigen::Vector3d::Zero();
  frame.pose.covariance =
      1.0e-8 * Eigen::Matrix<double, 6, 6>::Identity();
  frame.extrinsic_revision = config.resource.extrinsic.revision;
  frame.extrinsic_digest = config.resource.extrinsic.digest;
  frame.validity = Validity::VALID;
  uint64_t evidence_index = 0U;
  auto append_point = [&](const Eigen::Vector3d& position,
                          PointRole role) {
    CanonicalPoint point;
    point.position_R = position;
    point.original_index = evidence_index;
    point.canonical_rank = evidence_index;
    point.role = role;
    point.point_id = TestId("unmonitorable-prune-point", evidence_index);
    point.evidence_id = TestId(
        "unmonitorable-prune-evidence", evidence_index);
    point.correlation_group_id = frame.frame_id;
    frame.points.push_back(point);
    ++evidence_index;
  };
  for (uint32_t surfel = 0U; surfel < 6U; ++surfel) {
    const Eigen::Vector3d center = reference.surfels[surfel].center_R;
    append_point(center, PointRole::ASSOCIATION);
    append_point(center, PointRole::MEASUREMENT);
  }
  for (uint32_t offset : unmonitorable.surfel_indices) {
    const Eigen::Vector3d center = reference.surfels[offset].center_R;
    append_point(center, PointRole::ASSOCIATION);
    append_point(center, PointRole::MEASUREMENT);
  }
  // These unowned endpoints exercise the protective OTHER path.  They must be
  // byte-for-byte unaffected by pruning a valid but unmonitorable owner from
  // the separate formal matching boundary.
  append_point(Eigen::Vector3d(95.0, 0.0, 0.0),
               PointRole::ASSOCIATION);
  append_point(Eigen::Vector3d(95.0, 0.0, 0.0),
               PointRole::MEASUREMENT);

  std::vector<std::vector<AssociationMeasurementSet>> legacy_measurements;
  std::vector<std::vector<AssociationMeasurementSet>> pruned_measurements;
  std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>
      legacy_nuisances;
  std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>
      pruned_nuisances;
  std::vector<ObservationExplanation> legacy_explanations;
  std::vector<ObservationExplanation> pruned_explanations;
  std::vector<ReasonCode> legacy_reasons;
  std::vector<ReasonCode> pruned_reasons;
  MeasurementBuildStats legacy_stats;
  MeasurementBuildStats pruned_stats;
  ASSERT_TRUE(V1PipelineTestAccess::BuildMeasurements(
      pipeline, frame, predicted, true, &legacy_measurements,
      &legacy_nuisances, &legacy_explanations, &legacy_reasons,
      &legacy_stats));
  ASSERT_TRUE(V1PipelineTestAccess::BuildMeasurements(
      pipeline, frame, predicted, false, &pruned_measurements,
      &pruned_nuisances, &pruned_explanations, &pruned_reasons,
      &pruned_stats));

  EXPECT_EQ(legacy_stats.association_descriptor_count,
            6U + kUnmonitorableCarrierCount);
  EXPECT_EQ(legacy_stats.measurement_descriptor_count,
            6U + kUnmonitorableCarrierCount);
  EXPECT_EQ(pruned_stats.association_descriptor_count, 6U);
  EXPECT_EQ(pruned_stats.measurement_descriptor_count, 6U);
  EXPECT_LT(legacy_stats.matching_distance_evaluations,
            36U + static_cast<uint64_t>(kUnmonitorableCarrierCount) *
                kUnmonitorableCarrierCount);
  EXPECT_LT(pruned_stats.matching_distance_evaluations, 36U);
  EXPECT_GT(legacy_stats.matching_distance_evaluations,
            pruned_stats.matching_distance_evaluations);
  EXPECT_EQ(legacy_reasons, pruned_reasons);
  ASSERT_EQ(legacy_measurements.size(), pruned_measurements.size());
  size_t formal_candidate_count = 0U;
  for (size_t gdu = 0U; gdu < legacy_measurements.size(); ++gdu) {
    ASSERT_EQ(legacy_measurements[gdu].size(),
              pruned_measurements[gdu].size());
    for (size_t source = 0U; source < legacy_measurements[gdu].size();
         ++source) {
      const AssociationMeasurementSet& lhs =
          legacy_measurements[gdu][source];
      const AssociationMeasurementSet& rhs =
          pruned_measurements[gdu][source];
      EXPECT_EQ(lhs.source_evidence_id, rhs.source_evidence_id);
      ASSERT_EQ(lhs.selection.categories.size(),
                rhs.selection.categories.size());
      ASSERT_EQ(lhs.candidates.size(), rhs.candidates.size());
      formal_candidate_count += lhs.candidates.size();
      for (size_t category = 0U;
           category < lhs.selection.categories.size(); ++category) {
        EXPECT_EQ(lhs.selection.categories[category].candidate_id,
                  rhs.selection.categories[category].candidate_id);
        EXPECT_EQ(lhs.selection.categories[category].semantics,
                  rhs.selection.categories[category].semantics);
        EXPECT_DOUBLE_EQ(lhs.selection.categories[category].log_score,
                         rhs.selection.categories[category].log_score);
        EXPECT_DOUBLE_EQ(lhs.selection.categories[category].probability,
                         rhs.selection.categories[category].probability);
      }
      for (size_t candidate = 0U; candidate < lhs.candidates.size();
           ++candidate) {
        EXPECT_EQ(lhs.candidates[candidate].evidence_id,
                  rhs.candidates[candidate].evidence_id);
        EXPECT_EQ(lhs.candidates[candidate].candidate_id,
                  rhs.candidates[candidate].candidate_id);
        EXPECT_EQ(lhs.candidates[candidate].raw_residual_R,
                  rhs.candidates[candidate].raw_residual_R);
        EXPECT_DOUBLE_EQ(lhs.candidates[candidate].association_probability,
                         rhs.candidates[candidate].association_probability);
        for (size_t hypothesis = 0U; hypothesis < 4U; ++hypothesis) {
          const HypothesisMeasurement& left_model =
              lhs.candidates[candidate].hypothesis[hypothesis];
          const HypothesisMeasurement& right_model =
              rhs.candidates[candidate].hypothesis[hypothesis];
          EXPECT_EQ(left_model.observable_rank,
                    right_model.observable_rank);
          EXPECT_EQ(left_model.residual_m, right_model.residual_m);
          EXPECT_EQ(left_model.design, right_model.design);
          EXPECT_EQ(left_model.independent_covariance_m2,
                    right_model.independent_covariance_m2);
          EXPECT_EQ(left_model.reference_bias_design,
                    right_model.reference_bias_design);
          EXPECT_EQ(left_model.extrinsic_design,
                    right_model.extrinsic_design);
          EXPECT_EQ(left_model.pose_design, right_model.pose_design);
        }
      }
    }
  }
  EXPECT_GT(formal_candidate_count, 0U);

  ASSERT_EQ(legacy_explanations.size(), pruned_explanations.size());
  for (size_t index = 0U; index < legacy_explanations.size(); ++index) {
    const ObservationExplanation& lhs = legacy_explanations[index];
    const ObservationExplanation& rhs = pruned_explanations[index];
    EXPECT_EQ(lhs.cause_id, rhs.cause_id);
    EXPECT_EQ(lhs.source_evidence_id, rhs.source_evidence_id);
    EXPECT_EQ(lhs.semantics, rhs.semantics);
    EXPECT_DOUBLE_EQ(lhs.probability.lower, rhs.probability.lower);
    EXPECT_DOUBLE_EQ(lhs.probability.mean, rhs.probability.mean);
    EXPECT_DOUBLE_EQ(lhs.probability.upper, rhs.probability.upper);
    EXPECT_EQ(lhs.probability.calibrated, rhs.probability.calibrated);
    EXPECT_EQ(lhs.probability.mean_identified,
              rhs.probability.mean_identified);
    EXPECT_EQ(lhs.committed_to_posterior, rhs.committed_to_posterior);
    EXPECT_EQ(lhs.protective_geometry_available,
              rhs.protective_geometry_available);
    EXPECT_EQ(lhs.cross_gdu_assignment_envelope,
              rhs.cross_gdu_assignment_envelope);
    EXPECT_EQ(lhs.protective_center_R, rhs.protective_center_R);
    EXPECT_EQ(lhs.protective_covariance_R, rhs.protective_covariance_R);
    EXPECT_DOUBLE_EQ(lhs.protective_radius_m, rhs.protective_radius_m);
  }
  EXPECT_TRUE(std::any_of(
      pruned_explanations.begin(), pruned_explanations.end(),
      [](const ObservationExplanation& explanation) {
        return explanation.protective_geometry_available &&
            explanation.semantics == AssociationSemantics::OTHER;
      }));

  UacGaussianSumFilter legacy_filter = predicted;
  UacGaussianSumFilter pruned_filter = predicted;
  std::vector<ReasonCode> legacy_update_reasons;
  std::vector<ReasonCode> pruned_update_reasons;
  ASSERT_TRUE(legacy_filter.Update(
      legacy_measurements, legacy_nuisances, measurement_stamp,
      &legacy_update_reasons));
  ASSERT_TRUE(pruned_filter.Update(
      pruned_measurements, pruned_nuisances, measurement_stamp,
      &pruned_update_reasons));
  EXPECT_EQ(legacy_update_reasons, pruned_update_reasons);
  EXPECT_EQ(UacGaussianSumFilterTestAccess::MutableStateBytes(legacy_filter),
            UacGaussianSumFilterTestAccess::MutableStateBytes(pruned_filter));
}

TEST(V1Association, InvalidCandidatesFailClosed) {
  V1Config config = SyntheticKnownConfig("association-contract-test");
  AssociationModel model(config);
  AssociationFeature first = AssociationCandidate(0U, 0.1);
  AssociationFeature second = AssociationCandidate(1U, 0.2);
  ASSERT_TRUE(model.Evaluate({first, second}, true).valid);

  AssociationFeature nonfinite = second;
  nonfinite.overlap = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(model.Evaluate({first, nonfinite}, true).valid);
  EXPECT_FALSE(model.Evaluate({first, first}, true).valid);
  AssociationFeature unidentified = second;
  unidentified.candidate_id = StableId();
  EXPECT_FALSE(model.Evaluate({first, unidentified}, true).valid);
}

TEST(V1Geometry, MetricGraphPouIsContinuousAndReproducesConstants) {
  V1Config config = SyntheticKnownConfig("metric-graph-test");
  const ReferenceSnapshot reference = MetricLineReference(false);
  PouField field(&reference, &config);
  const Eigen::Vector3d constant(0.002, -0.003, 0.004);
  Eigen::VectorXd coefficients(18);
  for (int control = 0; control < 6; ++control)
    coefficients.segment<3>(3 * control) = constant;
  Eigen::MatrixXd previous;
  for (double x : {0.0, 0.05, 0.1, 0.15, 0.2}) {
    Validity validity = Validity::UNKNOWN;
    std::vector<ReasonCode> reasons;
    const Eigen::MatrixXd basis = field.VectorBasis(
        0U, Eigen::Vector3d(x, 0.0, 0.0), &validity, &reasons);
    ASSERT_EQ(validity, Validity::VALID);
    ASSERT_EQ(basis.rows(), 3);
    ASSERT_EQ(basis.cols(), 18);
    Eigen::Matrix3d weight_sum = Eigen::Matrix3d::Zero();
    for (int control = 0; control < 6; ++control)
      weight_sum += basis.middleCols<3>(3 * control);
    EXPECT_TRUE(weight_sum.isApprox(Eigen::Matrix3d::Identity(), 1.0e-14));
    EXPECT_TRUE((basis * coefficients).isApprox(constant, 1.0e-14));
    previous = basis;
  }
  Validity left_validity = Validity::UNKNOWN;
  Validity right_validity = Validity::UNKNOWN;
  std::vector<ReasonCode> reasons;
  const double epsilon = 1.0e-8;
  const Eigen::MatrixXd left = field.VectorBasis(
      0U, Eigen::Vector3d(0.1 - epsilon, 0.0, 0.0),
      &left_validity, &reasons);
  const Eigen::MatrixXd right = field.VectorBasis(
      0U, Eigen::Vector3d(0.1 + epsilon, 0.0, 0.0),
      &right_validity, &reasons);
  ASSERT_EQ(left_validity, Validity::VALID);
  ASSERT_EQ(right_validity, Validity::VALID);
  EXPECT_LT((left - right).norm(), 1.0e-5);

  double attachment_distance = -1.0;
  Validity offset_validity = Validity::UNKNOWN;
  const Eigen::MatrixXd offset = field.VectorBasis(
      0U, Eigen::Vector3d(0.05, 0.02, 0.0), &offset_validity, &reasons,
      &attachment_distance);
  EXPECT_EQ(offset_validity, Validity::VALID);
  EXPECT_GT(offset.size(), 0);
  EXPECT_NEAR(attachment_distance, 0.02, 1.0e-14);
  Validity outside_validity = Validity::VALID;
  EXPECT_EQ(field.VectorBasis(0U, Eigen::Vector3d(0.05, 0.11, 0.0),
                             &outside_validity, &reasons).size(), 0);
  EXPECT_EQ(outside_validity, Validity::UNKNOWN);
}

TEST(V1Geometry, PoseControlExclusionUsesFrozenSpatialNeighbourhood) {
  ReferenceSnapshot reference;
  reference.surfels.resize(4);
  reference.surfels[0].center_R = Eigen::Vector3d(0.0, 0.0, 0.0);
  reference.surfels[1].center_R = Eigen::Vector3d(0.5, 0.0, 0.0);
  // Different GDU, but still part of the same local physical surface.
  reference.surfels[2].center_R = Eigen::Vector3d(1.4, 0.0, 0.0);
  // Spatially separated stable reference carrier.
  reference.surfels[3].center_R = Eigen::Vector3d(3.1, 0.0, 0.0);
  reference.gdus.resize(3);
  reference.gdus[0].surfel_indices = {0U, 1U};
  reference.gdus[1].surfel_indices = {2U};
  reference.gdus[2].surfel_indices = {3U};

  const std::vector<bool> excluded =
      PoseControlSpatialExclusionMask(reference, 0U, 2.0);
  ASSERT_EQ(excluded.size(), 4U);
  EXPECT_TRUE(excluded[0]);
  EXPECT_TRUE(excluded[1]);
  EXPECT_TRUE(excluded[2]);
  EXPECT_FALSE(excluded[3]);

  const std::vector<bool> invalid =
      PoseControlSpatialExclusionMask(reference, 9U, 2.0);
  EXPECT_EQ(std::count(invalid.begin(), invalid.end(), true), 0);
}

TEST(V1Geometry,
     FrozenSurfelSpatialIndexMatchesExhaustiveOwnerAndExclusionDecisions) {
  ReferenceSnapshot reference;
  reference.surfels.resize(7U);
  reference.gdus.resize(3U);
  const std::array<Eigen::Vector3d, 7> centers{{
      Eigen::Vector3d(-0.26, 0.00, 0.00),
      Eigen::Vector3d(-0.01, 0.00, 0.00),
      Eigen::Vector3d(0.09, 0.00, 0.00),
      Eigen::Vector3d(0.34, 0.00, 0.00),
      Eigen::Vector3d(0.00, 0.24, 0.00),
      Eigen::Vector3d(0.00, 0.00, 0.26),
      Eigen::Vector3d(2.00, 0.00, 0.00)}};
  const std::array<uint32_t, 7> gdus{{0U, 0U, 1U, 1U, 2U, 2U, 2U}};
  for (uint32_t index = 0U; index < reference.surfels.size(); ++index) {
    reference.surfels[index].center_R = centers[index];
    reference.surfels[index].gdu_index = gdus[index];
    // Deliberately make StableId order differ from storage order.
    reference.surfels[index].id =
        TestId("spatial-index-surfel", 20U - index);
    reference.gdus[gdus[index]].surfel_indices.push_back(index);
  }
  ReferenceSurfelSpatialIndex owner_index;
  ASSERT_TRUE(owner_index.Build(reference, Eigen::Vector3d(-5.0, -5.0, -5.0),
                                0.25));
  ReferenceSurfelSpatialIndex cached_owner_index;
  ASSERT_TRUE(cached_owner_index.BuildOwnerCandidateCache(
      reference, Eigen::Vector3d(-5.0, -5.0, -5.0), 0.10, 0.25,
      1.0e-12));
  const auto exhaustive_owner = [&](const Eigen::Vector3d& position) {
    PatchOwnership result;
    double nearest = std::numeric_limits<double>::infinity();
    std::map<uint32_t, double> nearest_by_gdu;
    for (uint32_t index = 0U; index < reference.surfels.size(); ++index) {
      const double distance =
          (position - reference.surfels[index].center_R).norm();
      if (distance <= 0.25) {
        const uint32_t gdu = reference.surfels[index].gdu_index;
        const auto found = nearest_by_gdu.find(gdu);
        if (found == nearest_by_gdu.end() || distance < found->second)
          nearest_by_gdu[gdu] = distance;
      }
      if (distance < nearest - 1.0e-12 ||
          (std::abs(distance - nearest) <= 1.0e-12 &&
           (result.owner_surfel == std::numeric_limits<uint32_t>::max() ||
            reference.surfels[index].id <
                reference.surfels[result.owner_surfel].id))) {
        nearest = distance;
        result.owner_surfel = index;
      }
    }
    if (nearest > 0.25)
      result.owner_surfel = std::numeric_limits<uint32_t>::max();
    std::vector<std::pair<double, uint32_t>> ranked;
    for (const auto& candidate : nearest_by_gdu)
      ranked.emplace_back(candidate.second, candidate.first);
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs,
                                               const auto& rhs) {
      if (lhs.first != rhs.first) return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });
    if (ranked.size() > 1U && ranked[1].first - ranked[0].first <= 0.015) {
      result.owner_surfel = std::numeric_limits<uint32_t>::max();
      result.cross_gdu_ambiguous = true;
    }
    return result;
  };
  for (const Eigen::Vector3d& query : {
           Eigen::Vector3d(-0.25, 0.00, 0.00),
           Eigen::Vector3d(0.04, 0.00, 0.00),
           Eigen::Vector3d(0.00, 0.12, 0.00),
           Eigen::Vector3d(0.00, 0.00, 0.13),
           Eigen::Vector3d(1.60, 0.00, 0.00)}) {
    const PatchOwnership indexed = owner_index.QueryOwner(
        reference, query, 0.25, 1.0e-12, 0.015);
    const PatchOwnership cached = cached_owner_index.QueryOwner(
        reference, query, 0.25, 1.0e-12, 0.015);
    const PatchOwnership exhaustive = exhaustive_owner(query);
    EXPECT_EQ(indexed.owner_surfel, exhaustive.owner_surfel);
    EXPECT_EQ(indexed.cross_gdu_ambiguous,
              exhaustive.cross_gdu_ambiguous);
    EXPECT_EQ(cached.owner_surfel, exhaustive.owner_surfel);
    EXPECT_EQ(cached.cross_gdu_ambiguous,
              exhaustive.cross_gdu_ambiguous);
  }

  ReferenceSurfelSpatialIndex exclusion_index;
  ASSERT_TRUE(exclusion_index.Build(
      reference, Eigen::Vector3d(-5.0, -5.0, -5.0), 0.6));
  for (uint32_t gdu = 0U; gdu < reference.gdus.size(); ++gdu) {
    EXPECT_EQ(exclusion_index.PoseControlExclusionMask(reference, gdu, 0.6),
              PoseControlSpatialExclusionMask(reference, gdu, 0.6));
  }

  // Preserve the exhaustive storage-order semantics even for a deliberately
  // non-transitive near-tie chain crossing grid cells.  The final lower-Id
  // representative lies just outside the radius, so the historical exact
  // rule rejects the owner; cell traversal order must not turn it valid.
  ReferenceSnapshot tie_chain;
  tie_chain.surfels.resize(3U);
  tie_chain.gdus.resize(1U);
  tie_chain.surfels[0].center_R = Eigen::Vector3d(0.90, 0.0, 0.0);
  tie_chain.surfels[1].center_R = Eigen::Vector3d(-0.99, 0.0, 0.0);
  tie_chain.surfels[2].center_R = Eigen::Vector3d(-1.08, 0.0, 0.0);
  for (uint32_t index = 0U; index < 3U; ++index) {
    tie_chain.surfels[index].id.bytes.fill(0U);
    tie_chain.surfels[index].id.bytes[0] =
        static_cast<uint8_t>(3U - index);
    tie_chain.surfels[index].gdu_index = 0U;
    tie_chain.gdus[0].surfel_indices.push_back(index);
  }
  ReferenceSurfelSpatialIndex tie_index;
  ASSERT_TRUE(tie_index.Build(tie_chain, Eigen::Vector3d(-5.0, -5.0, -5.0),
                              0.5));
  const PatchOwnership tie_owner = tie_index.QueryOwner(
      tie_chain, Eigen::Vector3d::Zero(), 1.0, 0.1, 0.0);
  EXPECT_EQ(tie_owner.owner_surfel,
            std::numeric_limits<uint32_t>::max());
  EXPECT_FALSE(tie_owner.cross_gdu_ambiguous);
}

TEST(V1Geometry, RegisteredInterfaceHasTwoOneSidedJumpLimits) {
  V1Config config = SyntheticKnownConfig("metric-interface-test");
  const ReferenceSnapshot reference = MetricLineReference(true);
  PouField field(&reference, &config);
  Validity left_validity = Validity::UNKNOWN;
  Validity right_validity = Validity::UNKNOWN;
  std::vector<ReasonCode> reasons;
  const double epsilon = 1.0e-8;
  const Eigen::MatrixXd left = field.VectorBasis(
      0U, Eigen::Vector3d(0.25 - epsilon, 0.0, 0.0),
      &left_validity, &reasons);
  const Eigen::MatrixXd right = field.VectorBasis(
      0U, Eigen::Vector3d(0.25 + epsilon, 0.0, 0.0),
      &right_validity, &reasons);
  ASSERT_EQ(left_validity, Validity::VALID);
  ASSERT_EQ(right_validity, Validity::VALID);
  ASSERT_EQ(left.cols(), 21);
  ASSERT_EQ(right.cols(), 21);
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_NEAR(left(axis, 18 + axis), -0.5, 1.0e-6);
    EXPECT_NEAR(right(axis, 18 + axis), 0.5, 1.0e-6);
    EXPECT_NEAR(right(axis, 18 + axis) - left(axis, 18 + axis),
                1.0, 1.0e-6);
  }
}

TEST(V1Geometry,
     ReferenceGduBudgetUsesGenericSpatialCoverageInsteadOfLargestOnly) {
  ReferenceSnapshot reference;
  std::vector<GduDefinition, Eigen::aligned_allocator<GduDefinition>>
      candidates;
  const std::array<double, 6> centers{{0.2, 1.0, 2.0, 3.0, 4.0, 10.0}};
  for (size_t candidate = 0U; candidate < centers.size(); ++candidate) {
    GduDefinition gdu;
    gdu.id = TestId("coverage-gdu", candidate);
    // Different support counts exercise the frozen hybrid policy: closest,
    // bounded dense-evidence reserve, then generic geometric coverage.
    const size_t support_count = centers.size() - candidate;
    for (size_t support = 0U; support < support_count; ++support) {
      Surfel surfel;
      surfel.id = TestId("coverage-surfel", reference.surfels.size());
      surfel.center_R = Eigen::Vector3d(
          centers[candidate], 0.01 * static_cast<double>(support), 0.0);
      surfel.observable_projector_R =
          Eigen::Vector3d::UnitY() * Eigen::Vector3d::UnitY().transpose();
      gdu.surfel_indices.push_back(
          static_cast<uint32_t>(reference.surfels.size()));
      reference.surfels.push_back(surfel);
    }
    candidates.push_back(gdu);
  }
  const std::vector<size_t> selected = SelectReferenceGduCoverage(
      reference, candidates, Eigen::Vector3d::Zero(), 3U, 0.6, 1.0e-12);
  ASSERT_EQ(selected.size(), 3U);
  EXPECT_EQ(selected[0], 0U);  // nearest frozen carrier
  EXPECT_EQ(selected[1], 1U);  // half-budget dense-evidence reserve
  EXPECT_EQ(selected[2], 5U);  // remaining capacity keeps farthest coverage
  std::set<StableId> selected_ids;
  for (size_t index : selected) selected_ids.insert(candidates[index].id);

  std::reverse(candidates.begin(), candidates.end());
  const std::vector<size_t> reversed_selection = SelectReferenceGduCoverage(
      reference, candidates, Eigen::Vector3d::Zero(), 3U, 0.6, 1.0e-12);
  std::set<StableId> reversed_ids;
  for (size_t index : reversed_selection)
    reversed_ids.insert(candidates[index].id);
  EXPECT_EQ(reversed_ids, selected_ids);
}

TEST(V1Geometry,
     ReferenceGduBudgetReservesDenseEvidenceBeforeFarthestCoverage) {
  ReferenceSnapshot reference;
  std::vector<GduDefinition, Eigen::aligned_allocator<GduDefinition>>
      candidates;
  const std::array<double, 3> centers{{5.0, 0.1, 10.0}};
  const std::array<size_t, 3> supports{{10U, 1U, 1U}};
  for (size_t candidate = 0U; candidate < centers.size(); ++candidate) {
    GduDefinition gdu;
    gdu.id = TestId("dense-coverage-gdu", candidate);
    for (size_t support = 0U; support < supports[candidate]; ++support) {
      Surfel surfel;
      surfel.id = TestId("dense-coverage-surfel", reference.surfels.size());
      surfel.center_R = Eigen::Vector3d(
          centers[candidate], 0.01 * static_cast<double>(support), 0.0);
      surfel.observable_projector_R =
          Eigen::Vector3d::UnitY() * Eigen::Vector3d::UnitY().transpose();
      gdu.surfel_indices.push_back(
          static_cast<uint32_t>(reference.surfels.size()));
      reference.surfels.push_back(surfel);
    }
    candidates.push_back(gdu);
  }

  const std::vector<size_t> selected = SelectReferenceGduCoverage(
      reference, candidates, Eigen::Vector3d::Zero(), 2U, 0.6, 1.0e-12);
  ASSERT_EQ(selected.size(), 2U);
  EXPECT_EQ(selected[0], 1U);  // closest carrier
  EXPECT_EQ(selected[1], 0U);  // dense evidence reserve, not empty far edge

  const std::vector<size_t> bounded = SelectReferenceGduCoverage(
      reference, candidates, Eigen::Vector3d::Zero(), 1U, 0.6, 1.0e-12);
  ASSERT_EQ(bounded.size(), 1U);
  EXPECT_EQ(bounded[0], 1U);  // reserve must never exceed the total budget
}

TEST(V1Geometry,
     ReferenceGduCoveragePreservesStableIdsAndOtherCarrierPartition) {
  V1Config config = SyntheticKnownConfig("reference-coverage-partition-test");
  config.resource.reference_min_frames = 6;
  config.resource.reference_min_surfel_exposures = 6;
  config.resource.reference_min_duration_s = 0.50;
  config.resource.max_gdu = 3U;  // two monitored carriers plus one OTHER
  config.resource.max_tiles = 4096U;
  config.resource.graph_neighbor_radius_m = 0.25;
  config.resource.graph_max_neighbors = 36;
  config.resource.interface_scales_m.clear();
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  config.calibration.geometry_softmax_bias << 10.0, 0.0, 0.0, -10.0;

  std::deque<FrameBundle> frames;
  for (uint64_t frame_index = 0U; frame_index < 6U; ++frame_index) {
    FrameBundle frame;
    frame.session_id = config.resource.session_id;
    frame.frame_id = TestId("coverage-partition-frame", frame_index);
    frame.accepted_frame_index = frame_index;
    frame.stamp.fromSec(1.0 + 0.1 * static_cast<double>(frame_index));
    frame.world_frame = config.resource.world_frame;
    frame.pose.origin_L_R = Eigen::Vector3d::Zero();
    frame.pose.covariance = 1.0e-8 *
        Eigen::Matrix<double, 6, 6>::Identity();
    frame.validity = Validity::VALID;
    uint64_t local_index = 0U;
    for (uint64_t component = 0U; component < 5U; ++component) {
      const double x = 0.30 + 1.0 * static_cast<double>(component);
      for (uint64_t row = 0U; row < 2U; ++row) {
        for (uint64_t column = 0U; column < 3U; ++column) {
          CanonicalPoint point;
          point.position_R = Eigen::Vector3d(
              x, 0.10 * static_cast<double>(column),
              0.10 * static_cast<double>(row));
          point.original_index = local_index;
          point.canonical_rank = local_index;
          point.role = PointRole::MEASUREMENT;
          point.point_id = TestId(
              "coverage-partition-point",
              1000U * frame_index + local_index);
          point.evidence_id = TestId(
              "coverage-partition-evidence",
              1000U * frame_index + local_index);
          point.correlation_group_id = frame.frame_id;
          frame.points.push_back(point);
          ++local_index;
        }
      }
    }
    frames.push_back(frame);
  }

  const ReferenceSnapshot first = ReferenceBuilder(config).Build(frames, 21U);
  const ReferenceSnapshot repeated =
      ReferenceBuilder(config).Build(frames, 21U);
  ASSERT_TRUE(first.frozen);
  ASSERT_TRUE(repeated.frozen);
  EXPECT_EQ(first.id, repeated.id);
  ASSERT_EQ(first.gdus.size(), 3U);
  ASSERT_EQ(repeated.gdus.size(), first.gdus.size());
  size_t monitorable_count = 0U;
  size_t other_count = 0U;
  std::vector<size_t> surfel_membership(first.surfels.size(), 0U);
  for (size_t gdu_index = 0U; gdu_index < first.gdus.size(); ++gdu_index) {
    const GduDefinition& gdu = first.gdus[gdu_index];
    EXPECT_EQ(gdu.id, repeated.gdus[gdu_index].id);
    EXPECT_EQ(gdu.surfel_indices,
              repeated.gdus[gdu_index].surfel_indices);
    if (gdu.monitorable) {
      ++monitorable_count;
      EXPECT_EQ(gdu.surfel_indices.size(), 6U);
    } else {
      ++other_count;
      EXPECT_EQ(gdu.coordinate_geometry, GeometryHypothesis::OTHER);
      EXPECT_EQ(gdu.surfel_indices.size(), 18U);
    }
    for (uint32_t surfel : gdu.surfel_indices) {
      ASSERT_LT(surfel, surfel_membership.size());
      ++surfel_membership[surfel];
    }
  }
  EXPECT_EQ(monitorable_count, 2U);
  EXPECT_EQ(other_count, 1U);
  EXPECT_TRUE(std::all_of(
      surfel_membership.begin(), surfel_membership.end(),
      [](size_t count) { return count == 1U; }));
}

TEST(V1Geometry,
     ReferenceTangentialExposureSlidingRemainsSurfaceAndDeterministic) {
  V1Config config = SyntheticKnownConfig("reference-tangential-repeatability-test");
  config.resource.interface_scales_m.clear();
  config.resource.max_tiles = 4096U;
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  config.calibration.geometry_softmax_weights.setZero();
  config.calibration.geometry_softmax_bias << 8.0, -8.0, -8.0, -8.0;

  const ReferenceSnapshot first = ReferenceBuilder(config).Build(
      PlanarExposureFrames(config, 0.0, 0.006, false), 31U);
  const ReferenceSnapshot reversed = ReferenceBuilder(config).Build(
      PlanarExposureFrames(config, 0.0, 0.006, true), 31U);
  ASSERT_TRUE(first.frozen);
  ASSERT_FALSE(first.surfels.empty());
  EXPECT_EQ(first.id, reversed.id);
  EXPECT_EQ(first.surfels.size(), reversed.surfels.size());
  EXPECT_TRUE(std::any_of(
      first.surfels.begin(), first.surfels.end(), [](const Surfel& surfel) {
        return surfel.geometry_probability[0] > 0.5 &&
               surfel.hypothesis_projector_R[0].trace() > 0.5;
      }));
  EXPECT_TRUE(std::any_of(
      first.gdus.begin(), first.gdus.end(),
      [](const GduDefinition& gdu) { return gdu.monitorable; }));
}

TEST(V1Geometry,
     ReferenceNormalTwoModeExposureBecomesOtherWithoutDroppingCarrier) {
  V1Config config = SyntheticKnownConfig("reference-normal-two-mode-test");
  config.resource.interface_scales_m.clear();
  config.resource.max_tiles = 4096U;
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  // A deliberately soft surface probability reproduces the failure mode of
  // P_mix S P_mix^T: the old non-idempotent gate attenuated the normal
  // variation by the square of this probability.  Per-hypothesis gating must
  // reject it independent of that mixture weight.
  config.calibration.geometry_softmax_weights.setZero();
  config.calibration.geometry_softmax_bias << 0.0, -12.0, -12.0, 1.0;

  const ReferenceSnapshot reference = ReferenceBuilder(config).Build(
      PlanarExposureFrames(config, 0.008, 0.0, false), 32U);
  ASSERT_TRUE(reference.frozen);
  ASSERT_FALSE(reference.surfels.empty());
  for (const Surfel& surfel : reference.surfels) {
    EXPECT_DOUBLE_EQ(surfel.geometry_probability.head<3>().sum(), 0.0);
    EXPECT_DOUBLE_EQ(surfel.geometry_probability[3], 1.0);
    EXPECT_DOUBLE_EQ(surfel.observable_projector_R.norm(), 0.0);
  }
  EXPECT_TRUE(std::all_of(
      reference.gdus.begin(), reference.gdus.end(),
      [](const GduDefinition& gdu) {
        return !gdu.monitorable && gdu.coordinate_rank == 0 &&
               gdu.control_surfel_indices.empty();
      }));
}

TEST(V1Geometry, ReferenceBuildAndOptionalInterfaceLibraryAreBitDeterministic) {
  V1Config config = SyntheticKnownConfig("reference-determinism-test");
  config.resource.reference_min_frames = 6;
  config.resource.reference_min_duration_s = 0.50;
  config.resource.graph_neighbor_radius_m = 0.75;
  config.resource.interface_scales_m = {0.20};
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);

  std::deque<FrameBundle> frames;
  for (uint64_t frame_index = 0U; frame_index < 6U; ++frame_index) {
    FrameBundle frame = PipelineFrame(
        config, frame_index, 1.0 + 0.1 * frame_index, false);
    for (CanonicalPoint& point : frame.points) {
      const double original_x = point.position_R.x();
      point.position_R = Eigen::Vector3d(0.0, original_x, 0.0);
    }
    frames.push_back(frame);
  }
  const ReferenceSnapshot first = ReferenceBuilder(config).Build(frames, 7U);
  const ReferenceSnapshot second = ReferenceBuilder(config).Build(frames, 7U);
  ASSERT_TRUE(first.frozen);
  ASSERT_TRUE(second.frozen);
  EXPECT_EQ(first.id, second.id);
  ASSERT_EQ(first.surfels.size(), second.surfels.size());
  ASSERT_EQ(first.gdus.size(), second.gdus.size());
  ASSERT_EQ(first.interfaces.size(), second.interfaces.size());
  for (size_t index = 0; index < first.surfels.size(); ++index) {
    EXPECT_EQ(first.surfels[index].id, second.surfels[index].id);
    EXPECT_TRUE((first.surfels[index].center_R.array() ==
                 second.surfels[index].center_R.array()).all());
    EXPECT_TRUE((first.surfels[index].covariance_R.array() ==
                 second.surfels[index].covariance_R.array()).all());
    EXPECT_DOUBLE_EQ(first.surfels[index].support_radius_m,
                     second.surfels[index].support_radius_m);
    EXPECT_GE(first.surfels[index].support_radius_m, 0.0);
  }
  for (size_t index = 0; index < first.gdus.size(); ++index) {
    EXPECT_EQ(first.gdus[index].id, second.gdus[index].id);
    EXPECT_TRUE(first.gdus[index].reference_bias_covariance_R.allFinite());
    EXPECT_TRUE((first.gdus[index].reference_bias_covariance_R.array() ==
                 second.gdus[index].reference_bias_covariance_R.array()).all());
    EXPECT_LE(first.gdus[index].interface_indices.size(), 1U);
  }
  for (size_t index = 0; index < first.interfaces.size(); ++index) {
    const InterfaceDefinition& lhs = first.interfaces[index];
    const InterfaceDefinition& rhs = second.interfaces[index];
    EXPECT_EQ(lhs.id, rhs.id);
    EXPECT_TRUE((lhs.center_R.array() == rhs.center_R.array()).all());
    EXPECT_TRUE((lhs.normal_R.array() == rhs.normal_R.array()).all());
    EXPECT_TRUE(lhs.normal_R.allFinite());
    EXPECT_NEAR(lhs.normal_R.norm(), 1.0, 1.0e-14);
    const Eigen::Vector3d left_to_right =
        first.surfels[lhs.right_surfel].center_R -
        first.surfels[lhs.left_surfel].center_R;
    ASSERT_GT(left_to_right.norm(), 0.0);
    EXPECT_GT(lhs.normal_R.dot(left_to_right.normalized()), 1.0 - 1.0e-14);
    if (!lhs.identifiable)
      EXPECT_NE(std::find(first.reasons.begin(), first.reasons.end(),
                          ReasonCode::INTERFACE_UNIDENTIFIABLE),
                first.reasons.end());
  }
}

TEST(V1Geometry, ReferencePreservesInputAuthorityAndUnderResampledGeometryIsOther) {
  V1Config config = SyntheticKnownConfig("reference-authority-test");
  config.resource.reference_min_frames = 6;
  config.resource.reference_min_duration_s = 0.50;
  // Repeated exposure centers without stable local eigendirections must remain
  // OTHER; extra frames cannot turn a rank-deficient geometry into evidence.
  config.resource.graph_neighbor_radius_m = 0.11;
  config.resource.graph_max_neighbors = 12;
  config.resource.interface_scales_m = {0.20};
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);
  std::deque<FrameBundle> frames;
  for (uint64_t frame_index = 0U; frame_index < 6U; ++frame_index) {
    FrameBundle frame = PipelineFrame(
        config, frame_index, 1.0 + 0.1 * frame_index, false);
    for (CanonicalPoint& point : frame.points)
      point.position_R = Eigen::Vector3d(1.0, 0.2, 0.02);
    if (frame_index == 1U) {
      frame.validity = Validity::INCONCLUSIVE;
      frame.reasons.push_back(ReasonCode::INPUT_DESKEW_UNRESOLVED);
    }
    frames.push_back(frame);
  }
  const ReferenceSnapshot reference = ReferenceBuilder(config).Build(frames, 8U);
  ASSERT_TRUE(reference.frozen);
  EXPECT_EQ(reference.validity, Validity::INCONCLUSIVE);
  EXPECT_NE(std::find(reference.reasons.begin(), reference.reasons.end(),
                      ReasonCode::INPUT_DESKEW_UNRESOLVED),
            reference.reasons.end());
  ASSERT_FALSE(reference.surfels.empty());
  for (const Surfel& surfel : reference.surfels) {
    EXPECT_DOUBLE_EQ(surfel.geometry_probability[0], 0.0);
    EXPECT_DOUBLE_EQ(surfel.geometry_probability[1], 0.0);
    EXPECT_DOUBLE_EQ(surfel.geometry_probability[2], 0.0);
    EXPECT_DOUBLE_EQ(surfel.geometry_probability[3], 1.0);
  }
}

TEST(V1Geometry, ReferenceCarrierBudgetFailsClosedWithoutSupportTruncation) {
  V1Config config = SyntheticKnownConfig("reference-carrier-limit-test");
  config.resource.reference_min_frames = 6;
  config.resource.reference_min_duration_s = 0.50;
  config.resource.max_tiles = 1U;
  std::deque<FrameBundle> frames;
  for (uint64_t frame_index = 0U; frame_index < 6U; ++frame_index)
    frames.push_back(PipelineFrame(
        config, frame_index, 1.0 + 0.1 * frame_index, false));
  const ReferenceSnapshot reference = ReferenceBuilder(config).Build(frames, 1U);
  EXPECT_FALSE(reference.frozen);
  EXPECT_TRUE(reference.surfels.empty());
  EXPECT_NE(std::find(reference.reasons.begin(), reference.reasons.end(),
                      ReasonCode::REFERENCE_RESOURCE_LIMIT),
            reference.reasons.end());
}

TEST(V1Geometry, ReferenceControlCoverageBudgetFailsClosedWithoutSparseBasis) {
  V1Config config = SyntheticKnownConfig("reference-control-limit-test");
  config.resource.reference_min_frames = 6;
  config.resource.reference_min_duration_s = 0.50;
  config.resource.graph_neighbor_radius_m = 0.75;
  config.resource.control_spacing_m = 0.10;
  config.resource.pou_support_radius_m = 0.25;
  config.resource.max_controls_per_gdu = 1U;
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  config.calibration.geometry_softmax_bias << 10.0, 0.0, 0.0, -10.0;
  std::deque<FrameBundle> frames;
  for (uint64_t frame_index = 0U; frame_index < 6U; ++frame_index)
    frames.push_back(PipelineFrame(
        config, frame_index, 1.0 + 0.1 * frame_index, false));
  const ReferenceSnapshot reference = ReferenceBuilder(config).Build(frames, 1U);
  EXPECT_FALSE(reference.frozen);
  EXPECT_TRUE(reference.gdus.empty());
  EXPECT_NE(std::find(reference.reasons.begin(), reference.reasons.end(),
                      ReasonCode::REFERENCE_RESOURCE_LIMIT),
            reference.reasons.end());
}

TEST(V1Dynamics, UacPredictionMatchesCtmcMarginalAndPreservesPsd) {
  V1Config config = SyntheticKnownConfig("uac-transition-test");
  const ReferenceSnapshot reference = MetricLineReference(false);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(5U, 0U), &reasons));
  ASSERT_TRUE(filter.Predict(ros::Time(5U, 100000000U), &reasons));
  const auto posterior = filter.Posterior();
  ASSERT_EQ(posterior.size(), 1U);
  const Eigen::Matrix3d transition = CtmcTransition(
      UacGenerator(config.calibration.ctmc_rates), 0.1);
  for (int state = 0; state < 3; ++state)
    EXPECT_NEAR(posterior[0].state_probability[static_cast<size_t>(state)],
                transition(0, state), 1.0e-12);
  double total = posterior[0].other_mass;
  for (const GaussianComponent& component : posterior[0].components) {
    total += component.weight;
    EXPECT_TRUE(ValidateSymmetricPsd(
        component.covariance, config.numerical.psd_eigenvalue_tolerance));
  }
  EXPECT_NEAR(total, 1.0, 1.0e-12);
  const FieldQueryResult query = filter.Query(
      0U, Eigen::Vector3d(0.05, 0.02, 0.0));
  EXPECT_NEAR(query.interpolation_error_upper_m,
              0.02 + config.calibration.prediction_model_error_m +
                  config.calibration.ctmc_halving_error_upper_m,
              1.0e-14);
}

TEST(V1Dynamics, FirstObservationReanchorMovesOnlyTheFreshFilterClock) {
  V1Config config = SyntheticKnownConfig("uac-first-observation-reanchor-test");
  const ReferenceSnapshot reference = MetricLineReference(false);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(5U, 0U), &reasons));
  const auto before = filter.Posterior();
  ASSERT_TRUE(filter.ReanchorFirstObservation(ros::Time(8U, 0U), &reasons));
  const auto after = filter.Posterior();
  ASSERT_EQ(before.size(), after.size());
  for (size_t gdu = 0U; gdu < before.size(); ++gdu) {
    EXPECT_EQ(before[gdu].gdu_id, after[gdu].gdu_id);
    EXPECT_EQ(before[gdu].state_probability, after[gdu].state_probability);
    ASSERT_EQ(before[gdu].components.size(), after[gdu].components.size());
    for (size_t component = 0U;
         component < before[gdu].components.size(); ++component) {
      EXPECT_EQ(before[gdu].components[component].component_id,
                after[gdu].components[component].component_id);
      EXPECT_DOUBLE_EQ(before[gdu].components[component].weight,
                       after[gdu].components[component].weight);
      EXPECT_TRUE(before[gdu].components[component].mean.isApprox(
          after[gdu].components[component].mean, 0.0));
      EXPECT_TRUE(before[gdu].components[component].covariance.isApprox(
          after[gdu].components[component].covariance, 0.0));
    }
  }
  ASSERT_TRUE(filter.Predict(ros::Time(8U, 100000000U), &reasons));
  reasons.clear();
  EXPECT_FALSE(filter.ReanchorFirstObservation(ros::Time(9U, 0U), &reasons));
  EXPECT_NE(std::find(reasons.begin(), reasons.end(),
                      ReasonCode::INPUT_TIME_INVALID), reasons.end());
}

TEST(V1Dynamics,
     TransactionCopySharesOnlyFrozenCacheAndPreservesExactRollback) {
  V1Config config = SyntheticKnownConfig(
      "uac-frozen-cache-transaction-copy-test");
  const ReferenceSnapshot reference =
      ReplicatedMetricLineReference(8U, true);
  UacGaussianSumFilter committed(config);
  UacGaussianSumFilter independent(config);
  std::vector<ReasonCode> committed_reasons;
  std::vector<ReasonCode> independent_reasons;
  const ros::Time initial_stamp(17U, 0U);
  const ros::Time prediction_stamp(17U, 100000000U);
  ASSERT_TRUE(committed.Initialize(
      reference, initial_stamp, &committed_reasons));
  ASSERT_TRUE(independent.Initialize(
      reference, initial_stamp, &independent_reasons));

  const std::vector<uint8_t> committed_before =
      UacGaussianSumFilterTestAccess::MutableStateBytes(committed);
  const std::vector<const void*> committed_cache =
      UacGaussianSumFilterTestAccess::FrozenCacheIdentities(committed);
  ASSERT_FALSE(committed_cache.empty());
  EXPECT_TRUE(std::all_of(
      committed_cache.begin(), committed_cache.end(),
      [](const void* cache) { return cache != nullptr; }));

  // This is the same copy operation used to open a pipeline transaction.
  // Only the const reference-derived cache is shared; all dynamic state is
  // copied by value and must roll back when the candidate is discarded.
  UacGaussianSumFilter candidate = committed;
  EXPECT_EQ(committed_cache,
            UacGaussianSumFilterTestAccess::FrozenCacheIdentities(candidate));
  committed_reasons.clear();
  independent_reasons.clear();
  ASSERT_TRUE(candidate.Predict(prediction_stamp, &committed_reasons));
  ASSERT_TRUE(independent.Predict(prediction_stamp, &independent_reasons));
  EXPECT_EQ(committed_reasons, independent_reasons);
  EXPECT_EQ(committed_before,
            UacGaussianSumFilterTestAccess::MutableStateBytes(committed));
  EXPECT_EQ(UacGaussianSumFilterTestAccess::MutableStateBytes(candidate),
            UacGaussianSumFilterTestAccess::MutableStateBytes(independent));
  EXPECT_EQ(committed_cache,
            UacGaussianSumFilterTestAccess::FrozenCacheIdentities(candidate));

  // Successful commit can take ownership of the candidate.  Moving it must
  // preserve every dynamic byte and the exact immutable-cache identity.
  const std::vector<uint8_t> candidate_bytes =
      UacGaussianSumFilterTestAccess::MutableStateBytes(candidate);
  UacGaussianSumFilter accepted(config);
  accepted = std::move(candidate);
  EXPECT_EQ(candidate_bytes,
            UacGaussianSumFilterTestAccess::MutableStateBytes(accepted));
  EXPECT_EQ(committed_cache,
            UacGaussianSumFilterTestAccess::FrozenCacheIdentities(accepted));
}

TEST(V1Dynamics,
     ExactIntervalCachesMatchTheUncachedTransitionFieldByField) {
  V1Config config = SyntheticKnownConfig("uac-exact-predict-cache-test");
  const ReferenceSnapshot reference =
      ReplicatedMetricLineReference(8U, true);
  UacGaussianSumFilter cached(config);
  UacGaussianSumFilter cached_repeat(config);
  UacGaussianSumFilter cached_serial(config);
  UacGaussianSumFilter uncached(config);
  std::vector<ReasonCode> cached_reasons;
  std::vector<ReasonCode> cached_repeat_reasons;
  std::vector<ReasonCode> cached_serial_reasons;
  std::vector<ReasonCode> uncached_reasons;
  ASSERT_TRUE(cached.Initialize(reference, ros::Time(5U, 0U),
                                &cached_reasons));
  ASSERT_TRUE(cached_repeat.Initialize(reference, ros::Time(5U, 0U),
                                       &cached_repeat_reasons));
  ASSERT_TRUE(uncached.Initialize(reference, ros::Time(5U, 0U),
                                  &uncached_reasons));
  ASSERT_TRUE(cached_serial.Initialize(reference, ros::Time(5U, 0U),
                                       &cached_serial_reasons));
  cached_reasons.clear();
  cached_repeat_reasons.clear();
  cached_serial_reasons.clear();
  uncached_reasons.clear();

  // Forty 0.05 s substeps are long enough to repeatedly revisit every
  // discrete transition/mode kernel while keeping this regression lightweight.
  const ros::Time destination_stamp(7U, 0U);
  const auto cached_begin = std::chrono::steady_clock::now();
  ASSERT_TRUE(cached.Predict(destination_stamp, &cached_reasons));
  const auto cached_end = std::chrono::steady_clock::now();
  ASSERT_TRUE(cached_repeat.Predict(destination_stamp,
                                    &cached_repeat_reasons));
  const auto cached_serial_begin = std::chrono::steady_clock::now();
  ASSERT_TRUE(UacGaussianSumFilterTestAccess::PredictCachedSerial(
      &cached_serial, destination_stamp, &cached_serial_reasons));
  const auto cached_serial_end = std::chrono::steady_clock::now();
  const auto uncached_begin = std::chrono::steady_clock::now();
  ASSERT_TRUE(UacGaussianSumFilterTestAccess::PredictUncached(
      &uncached, destination_stamp, &uncached_reasons));
  const auto uncached_end = std::chrono::steady_clock::now();

  EXPECT_EQ(cached_reasons, uncached_reasons);
  EXPECT_EQ(cached_reasons, cached_repeat_reasons);
  EXPECT_EQ(cached_reasons, cached_serial_reasons);
  EXPECT_EQ(cached.approximation_flags(), uncached.approximation_flags());
  EXPECT_EQ(cached.approximation_flags(),
            cached_repeat.approximation_flags());
  EXPECT_EQ(cached.approximation_flags(),
            cached_serial.approximation_flags());
  const auto cached_us = std::chrono::duration_cast<std::chrono::microseconds>(
      cached_end - cached_begin).count();
  const auto uncached_us = std::chrono::duration_cast<std::chrono::microseconds>(
      uncached_end - uncached_begin).count();
  const auto cached_serial_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          cached_serial_end - cached_serial_begin).count();
  std::cout << "[predict-exact-cache-benchmark] cached_us=" << cached_us
            << " cached_serial_us=" << cached_serial_us
            << " uncached_us=" << uncached_us << std::endl;
  // This fixture repeatedly uses the same finite kernel library.  Reusing
  // those exact matrices must avoid work rather than merely move it around.
  EXPECT_LT(cached_us, uncached_us);

  const auto cached_posterior = cached.Posterior();
  const auto cached_repeat_posterior = cached_repeat.Posterior();
  const auto cached_serial_posterior = cached_serial.Posterior();
  const auto uncached_posterior = uncached.Posterior();
  const auto expect_component_equal = [](
      const GaussianComponent& lhs, const GaussianComponent& rhs) {
    EXPECT_EQ(lhs.state, rhs.state);
    EXPECT_EQ(lhs.geometry, rhs.geometry);
    EXPECT_EQ(lhs.interface_modes, rhs.interface_modes);
    EXPECT_EQ(lhs.association_lineage, rhs.association_lineage);
    EXPECT_EQ(lhs.active_fixed_lag_lineage, rhs.active_fixed_lag_lineage);
    EXPECT_EQ(lhs.contamination_scale, rhs.contamination_scale);
    EXPECT_EQ(lhs.parent_id, rhs.parent_id);
    EXPECT_EQ(lhs.component_id, rhs.component_id);
    EXPECT_DOUBLE_EQ(lhs.weight, rhs.weight);
    EXPECT_TRUE(lhs.mean.isApprox(rhs.mean, 0.0));
    EXPECT_TRUE(lhs.covariance.isApprox(rhs.covariance, 0.0));
    EXPECT_TRUE(lhs.sqrt_information.isApprox(rhs.sqrt_information, 0.0));
    EXPECT_EQ(lhs.observable_rank, rhs.observable_rank);
    EXPECT_TRUE(lhs.observable_projector.isApprox(
        rhs.observable_projector, 0.0));
    EXPECT_TRUE(lhs.observable_nullspace.isApprox(
        rhs.observable_nullspace, 0.0));
    EXPECT_DOUBLE_EQ(lhs.minimum_detectable_effect_m,
                     rhs.minimum_detectable_effect_m);
    EXPECT_EQ(lhs.minimum_detectable_effect_defined,
              rhs.minimum_detectable_effect_defined);
  };
  const auto expect_posterior_equal = [&](const auto& cached_result,
                                          const auto& reference_result) {
    ASSERT_EQ(cached_result.size(), reference_result.size());
  for (size_t g = 0U; g < cached_result.size(); ++g) {
    const GduPosterior& lhs = cached_result[g];
    const GduPosterior& rhs = reference_result[g];
    EXPECT_EQ(lhs.gdu_id, rhs.gdu_id);
    EXPECT_EQ(lhs.nuisance_correlation_id, rhs.nuisance_correlation_id);
    EXPECT_EQ(lhs.state_probability, rhs.state_probability);
    EXPECT_DOUBLE_EQ(lhs.other_mass, rhs.other_mass);
    EXPECT_DOUBLE_EQ(lhs.task_scale_deformation_probability,
                     rhs.task_scale_deformation_probability);
    EXPECT_DOUBLE_EQ(lhs.task_scale_deformation_lower,
                     rhs.task_scale_deformation_lower);
    EXPECT_DOUBLE_EQ(lhs.task_scale_deformation_upper,
                     rhs.task_scale_deformation_upper);
    EXPECT_EQ(lhs.task_scale_deformation_mean_identified,
              rhs.task_scale_deformation_mean_identified);
    EXPECT_DOUBLE_EQ(lhs.task_displacement_probability,
                     rhs.task_displacement_probability);
    EXPECT_DOUBLE_EQ(lhs.current_velocity_exceedance_probability,
                     rhs.current_velocity_exceedance_probability);
    EXPECT_DOUBLE_EQ(lhs.continuous_valid_observation_duration_s,
                     rhs.continuous_valid_observation_duration_s);
    EXPECT_EQ(lhs.velocity_duration_requirement_met,
              rhs.velocity_duration_requirement_met);
    EXPECT_DOUBLE_EQ(lhs.task_interface_jump_probability,
                     rhs.task_interface_jump_probability);
    EXPECT_DOUBLE_EQ(lhs.minimum_detectable_effect_m,
                     rhs.minimum_detectable_effect_m);
    EXPECT_EQ(lhs.minimum_detectable_effect_defined,
              rhs.minimum_detectable_effect_defined);
    EXPECT_DOUBLE_EQ(lhs.velocity_minimum_detectable_effect_mps,
                     rhs.velocity_minimum_detectable_effect_mps);
    EXPECT_EQ(lhs.velocity_minimum_detectable_effect_defined,
              rhs.velocity_minimum_detectable_effect_defined);
    EXPECT_DOUBLE_EQ(lhs.interface_jump_minimum_detectable_effect_m,
                     rhs.interface_jump_minimum_detectable_effect_m);
    EXPECT_EQ(lhs.interface_jump_minimum_detectable_effect_defined,
              rhs.interface_jump_minimum_detectable_effect_defined);
    EXPECT_DOUBLE_EQ(lhs.task_power_worst_normalized_mde,
                     rhs.task_power_worst_normalized_mde);
    EXPECT_EQ(lhs.any_task_direction_detectable,
              rhs.any_task_direction_detectable);
    EXPECT_EQ(lhs.all_task_directions_detectable,
              rhs.all_task_directions_detectable);
    EXPECT_EQ(lhs.detection_allowed, rhs.detection_allowed);
    EXPECT_EQ(lhs.clearing_allowed, rhs.clearing_allowed);
    EXPECT_EQ(lhs.task_power_complete, rhs.task_power_complete);
    EXPECT_EQ(lhs.observable_rank, rhs.observable_rank);
    EXPECT_TRUE(lhs.observable_projector.isApprox(
        rhs.observable_projector, 0.0));
    EXPECT_TRUE(lhs.nullspace.isApprox(rhs.nullspace, 0.0));
    EXPECT_DOUBLE_EQ(lhs.reference_information_ratio,
                     rhs.reference_information_ratio);
    EXPECT_EQ(lhs.reference_confounding, rhs.reference_confounding);
    EXPECT_DOUBLE_EQ(lhs.effective_spatial_support,
                     rhs.effective_spatial_support);
    EXPECT_DOUBLE_EQ(lhs.effective_temporal_support,
                     rhs.effective_temporal_support);
    EXPECT_EQ(lhs.admission, rhs.admission);
    ASSERT_EQ(lhs.components.size(), rhs.components.size());
    for (size_t component = 0U; component < lhs.components.size(); ++component)
      expect_component_equal(lhs.components[component],
                             rhs.components[component]);
  }
  };
  expect_posterior_equal(cached_posterior, uncached_posterior);
  expect_posterior_equal(cached_posterior, cached_serial_posterior);
  expect_posterior_equal(cached_posterior, cached_repeat_posterior);

  const auto cached_interfaces = cached.InterfacePosteriors();
  const auto cached_reused_interfaces =
      cached.InterfacePosteriors(cached_posterior);
  const auto cached_repeat_interfaces = cached_repeat.InterfacePosteriors();
  const auto cached_serial_interfaces = cached_serial.InterfacePosteriors();
  const auto uncached_interfaces = uncached.InterfacePosteriors();
  const auto expect_interfaces_equal = [&](const auto& cached_result,
                                           const auto& reference_result) {
  ASSERT_EQ(cached_result.size(), reference_result.size());
  for (size_t interface = 0U; interface < cached_result.size();
       ++interface) {
    const InterfacePosterior& lhs = cached_result[interface];
    const InterfacePosterior& rhs = reference_result[interface];
    EXPECT_EQ(lhs.interface_id, rhs.interface_id);
    EXPECT_EQ(lhs.joint_probability, rhs.joint_probability);
    EXPECT_TRUE(lhs.jump_mean_R.isApprox(rhs.jump_mean_R, 0.0));
    EXPECT_TRUE(lhs.jump_covariance_R.isApprox(
        rhs.jump_covariance_R, 0.0));
    EXPECT_DOUBLE_EQ(lhs.other_mass, rhs.other_mass);
    EXPECT_EQ(lhs.connectivity_coherence_confounding,
              rhs.connectivity_coherence_confounding);
    EXPECT_EQ(lhs.validity, rhs.validity);
    EXPECT_TRUE(lhs.left_query_position_R.isApprox(
        rhs.left_query_position_R, 0.0));
    EXPECT_TRUE(lhs.right_query_position_R.isApprox(
        rhs.right_query_position_R, 0.0));
    EXPECT_EQ(lhs.left_field.validity, rhs.left_field.validity);
    EXPECT_EQ(lhs.left_field.reasons, rhs.left_field.reasons);
    EXPECT_TRUE(lhs.left_field.mean_displacement_R.isApprox(
        rhs.left_field.mean_displacement_R, 0.0));
    EXPECT_TRUE(lhs.left_field.mean_velocity_R.isApprox(
        rhs.left_field.mean_velocity_R, 0.0));
    EXPECT_EQ(lhs.right_field.validity, rhs.right_field.validity);
    EXPECT_EQ(lhs.right_field.reasons, rhs.right_field.reasons);
    EXPECT_TRUE(lhs.right_field.mean_displacement_R.isApprox(
        rhs.right_field.mean_displacement_R, 0.0));
    EXPECT_TRUE(lhs.right_field.mean_velocity_R.isApprox(
        rhs.right_field.mean_velocity_R, 0.0));
  }
  };
  expect_interfaces_equal(cached_interfaces, cached_reused_interfaces);
  expect_interfaces_equal(cached_interfaces, uncached_interfaces);
  expect_interfaces_equal(cached_interfaces, cached_serial_interfaces);
  expect_interfaces_equal(cached_interfaces, cached_repeat_interfaces);
}

TEST(V1Dynamics,
     ZeroHorizonMixturePreservesAssociationStateAndFieldMoments) {
  V1Config config = SyntheticKnownConfig(
      "uac-association-mixture-equivalence-test");
  const ReferenceSnapshot reference = MetricLineReference(false);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(5U, 0U), &reasons));
  // Prediction creates non-U mixture branches and nonzero process covariance,
  // so this checks more than the trivial freshly initialized U posterior.
  ASSERT_TRUE(filter.Predict(ros::Time(5U, 200000000U), &reasons));

  const auto full = filter.Posterior();
  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>> light;
  reasons.clear();
  ASSERT_TRUE(filter.PredictiveMixture(0.0, &light, &reasons));
  EXPECT_TRUE(reasons.empty());
  ASSERT_EQ(light.size(), full.size());
  for (size_t g = 0U; g < full.size(); ++g) {
    EXPECT_EQ(light[g].gdu_id, full[g].gdu_id);
    EXPECT_EQ(light[g].state_probability, full[g].state_probability);
    EXPECT_DOUBLE_EQ(light[g].other_mass, full[g].other_mass);
    ASSERT_EQ(light[g].components.size(), full[g].components.size());
    for (size_t component = 0U; component < full[g].components.size();
         ++component) {
      EXPECT_EQ(light[g].components[component].component_id,
                full[g].components[component].component_id);
      EXPECT_DOUBLE_EQ(light[g].components[component].weight,
                       full[g].components[component].weight);
      EXPECT_TRUE(light[g].components[component].mean.isApprox(
          full[g].components[component].mean, 0.0));
      EXPECT_TRUE(light[g].components[component].covariance.isApprox(
          full[g].components[component].covariance, 0.0));
    }
    for (uint32_t surfel_index : reference.gdus[g].surfel_indices) {
      const Eigen::Vector3d& position =
          reference.surfels[surfel_index].center_R;
      const FieldQueryResult full_query =
          filter.QueryWithPosterior(static_cast<uint32_t>(g), position,
                                    full[g]);
      const FieldQueryResult light_query =
          filter.QueryWithPosterior(static_cast<uint32_t>(g), position,
                                    light[g]);
      EXPECT_TRUE(light_query.mean_displacement_R.isApprox(
          full_query.mean_displacement_R, 0.0));
      EXPECT_TRUE(light_query.mean_velocity_R.isApprox(
          full_query.mean_velocity_R, 0.0));
      EXPECT_TRUE(light_query.displacement_covariance_R.isApprox(
          full_query.displacement_covariance_R, 0.0));
      EXPECT_TRUE(light_query.velocity_covariance_R.isApprox(
          full_query.velocity_covariance_R, 0.0));
      EXPECT_DOUBLE_EQ(light_query.active_probability.lower,
                       full_query.active_probability.lower);
      EXPECT_DOUBLE_EQ(light_query.active_probability.mean,
                       full_query.active_probability.mean);
      EXPECT_DOUBLE_EQ(light_query.active_probability.upper,
                       full_query.active_probability.upper);
      EXPECT_EQ(light_query.active_probability.mean_identified,
                full_query.active_probability.mean_identified);
      EXPECT_DOUBLE_EQ(light_query.other_mass, full_query.other_mass);
      EXPECT_EQ(light_query.validity, full_query.validity);
    }
  }
}

TEST(V1Dynamics,
     ParallelMeasurementUpdateMatchesSerialBitForBitAndIsRepeatable) {
  V1Config config = SyntheticKnownConfig("uac-parallel-update-exact-test");
  config.resource.max_update_gap_s = 0.5;
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);
  constexpr uint32_t kGduCount = 8U;
  constexpr uint32_t kSourcesPerGdu = 5U;
  constexpr int kCoefficientCount = 21;
  const ReferenceSnapshot reference =
      ReplicatedMetricLineReference(kGduCount, true);
  UacGaussianSumFilter parallel(config);
  UacGaussianSumFilter parallel_repeat(config);
  UacGaussianSumFilter serial(config);
  std::vector<ReasonCode> parallel_reasons;
  std::vector<ReasonCode> repeat_reasons;
  std::vector<ReasonCode> serial_reasons;
  const ros::Time initial_stamp(31U, 0U);
  const ros::Time update_stamp(31U, 200000000U);
  ASSERT_TRUE(parallel.Initialize(reference, initial_stamp, &parallel_reasons));
  ASSERT_TRUE(parallel_repeat.Initialize(
      reference, initial_stamp, &repeat_reasons));
  ASSERT_TRUE(serial.Initialize(reference, initial_stamp, &serial_reasons));
  parallel_reasons.clear();
  repeat_reasons.clear();
  serial_reasons.clear();
  ASSERT_TRUE(parallel.Predict(update_stamp, &parallel_reasons));
  ASSERT_TRUE(parallel_repeat.Predict(update_stamp, &repeat_reasons));
  ASSERT_TRUE(serial.Predict(update_stamp, &serial_reasons));
  ASSERT_EQ(parallel_reasons, serial_reasons);
  ASSERT_EQ(parallel_reasons, repeat_reasons);

  std::vector<std::vector<AssociationMeasurementSet>> measurements(
      kGduCount);
  std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>
      nuisances(kGduCount);
  for (uint32_t g = 0U; g < kGduCount; ++g) {
    FrameNuisance& nuisance = nuisances[g];
    nuisance.pose_covariance =
        (1.0e-8 + 1.0e-10 * static_cast<double>(g)) *
        Eigen::Matrix<double, 6, 6>::Identity();
    nuisance.nuisance_correlation_id = TestId("parallel-update-nuisance", g);
    nuisance.control_registration_valid = true;
    nuisance.deformation_measurement_valid = true;
    for (uint32_t source = 0U; source < kSourcesPerGdu; ++source) {
      AssociationMeasurementSet set;
      set.source_evidence_id = TestId(
          "parallel-update-source", kSourcesPerGdu * g + source);
      AssociationCategory candidate;
      candidate.candidate_id = TestId(
          "parallel-update-candidate", kSourcesPerGdu * g + source);
      candidate.semantics = AssociationSemantics::OTHER;
      candidate.geometric_candidate = true;
      candidate.probability = 0.9;
      AssociationCategory other;
      other.semantics = AssociationSemantics::OTHER;
      other.geometric_candidate = false;
      other.probability = 0.1;
      set.selection.categories = {candidate, other};
      set.selection.entropy = -0.9 * std::log(0.9) - 0.1 * std::log(0.1);
      set.selection.other_mass = 0.1;
      set.selection.valid = true;
      FieldMeasurement measurement;
      measurement.evidence_id = TestId(
          "parallel-update-measurement", kSourcesPerGdu * g + source);
      measurement.candidate_id = candidate.candidate_id;
      measurement.gdu_index = g;
      measurement.position_R = Eigen::Vector3d(
          2.0 * static_cast<double>(g) + 0.1 * static_cast<double>(source),
          0.0, 0.0);
      for (size_t geometry = 0U; geometry < measurement.hypothesis.size();
           ++geometry) {
        HypothesisMeasurement& hypothesis = measurement.hypothesis[geometry];
        hypothesis.residual_m = Eigen::Vector3d(
            1.0e-4 * static_cast<double>(source + 1U),
            -2.0e-5 * static_cast<double>(g + 1U),
            1.0e-5 * static_cast<double>(geometry));
        hypothesis.design =
            Eigen::MatrixXd::Zero(3, kCoefficientCount);
        hypothesis.design.block<3, 3>(
            0, 3 * static_cast<int>(source)) =
            Eigen::Matrix3d::Identity();
        hypothesis.reference_bias_design =
            0.05 * Eigen::Matrix3d::Identity();
        hypothesis.extrinsic_design = Eigen::MatrixXd::Zero(3, 6);
        hypothesis.pose_design = Eigen::MatrixXd::Zero(3, 6);
        hypothesis.independent_covariance_m2 =
            (1.0e-6 + 1.0e-8 * static_cast<double>(geometry)) *
            Eigen::Matrix3d::Identity();
        hypothesis.observable_rank = 3;
      }
      set.candidates.push_back(std::move(measurement));
      measurements[g].push_back(std::move(set));
    }
  }

  parallel_reasons.clear();
  repeat_reasons.clear();
  serial_reasons.clear();
  const auto serial_begin = std::chrono::steady_clock::now();
  ASSERT_TRUE(UacGaussianSumFilterTestAccess::UpdateSerial(
      &serial, measurements, nuisances, update_stamp, &serial_reasons));
  const auto serial_end = std::chrono::steady_clock::now();
  const auto parallel_begin = std::chrono::steady_clock::now();
  ASSERT_TRUE(parallel.Update(
      measurements, nuisances, update_stamp, &parallel_reasons));
  const auto parallel_end = std::chrono::steady_clock::now();
  ASSERT_TRUE(parallel_repeat.Update(
      measurements, nuisances, update_stamp, &repeat_reasons));
  const auto serial_us = std::chrono::duration_cast<std::chrono::microseconds>(
      serial_end - serial_begin).count();
  const auto parallel_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          parallel_end - parallel_begin).count();
  std::cout << "[update-exact-parallel-benchmark] parallel_us="
            << parallel_us << " serial_us=" << serial_us << std::endl;

  EXPECT_EQ(parallel_reasons, serial_reasons);
  EXPECT_EQ(parallel_reasons, repeat_reasons);
  EXPECT_EQ(parallel.approximation_flags(), serial.approximation_flags());
  EXPECT_EQ(parallel.approximation_flags(),
            parallel_repeat.approximation_flags());
  const std::vector<uint8_t> parallel_bytes =
      UacGaussianSumFilterTestAccess::MutableStateBytes(parallel);
  EXPECT_EQ(parallel_bytes,
            UacGaussianSumFilterTestAccess::MutableStateBytes(serial));
  EXPECT_EQ(parallel_bytes,
            UacGaussianSumFilterTestAccess::MutableStateBytes(
                parallel_repeat));
}

TEST(V1Dynamics,
     ParallelMeasurementFailureKeepsCanonicalReasonOrderAndAtomicState) {
  V1Config config = SyntheticKnownConfig("uac-parallel-update-failure-test");
  config.resource.source_missing_consecutive_limit = 1U;
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);
  constexpr uint32_t kGduCount = 4U;
  const ReferenceSnapshot reference =
      ReplicatedMetricLineReference(kGduCount, false);
  UacGaussianSumFilter parallel(config);
  UacGaussianSumFilter serial(config);
  std::vector<ReasonCode> parallel_reasons;
  std::vector<ReasonCode> serial_reasons;
  const ros::Time initial_stamp(41U, 0U);
  const ros::Time update_stamp(41U, 100000000U);
  ASSERT_TRUE(parallel.Initialize(reference, initial_stamp, &parallel_reasons));
  ASSERT_TRUE(serial.Initialize(reference, initial_stamp, &serial_reasons));
  parallel_reasons.clear();
  serial_reasons.clear();
  ASSERT_TRUE(parallel.Predict(update_stamp, &parallel_reasons));
  ASSERT_TRUE(serial.Predict(update_stamp, &serial_reasons));
  ASSERT_EQ(parallel_reasons, serial_reasons);
  const std::vector<uint8_t> predicted_bytes =
      UacGaussianSumFilterTestAccess::MutableStateBytes(parallel);

  std::vector<std::vector<AssociationMeasurementSet>> measurements(
      kGduCount);
  AssociationMeasurementSet missing_set;
  missing_set.source_evidence_id = TestId("parallel-failure-missing", 0U);
  AssociationCategory missing;
  missing.semantics = AssociationSemantics::SOURCE_MISSING;
  missing.probability = 1.0;
  missing_set.selection.categories.push_back(missing);
  missing_set.selection.valid = true;
  measurements[0].push_back(missing_set);
  AssociationMeasurementSet later_ambiguous;
  later_ambiguous.source_evidence_id = TestId(
      "parallel-failure-later-ambiguous", 0U);
  later_ambiguous.selection.valid = false;
  measurements[2].push_back(later_ambiguous);

  std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>
      nuisances(kGduCount);
  for (uint32_t g = 0U; g < kGduCount; ++g) {
    nuisances[g].pose_covariance =
        1.0e-8 * Eigen::Matrix<double, 6, 6>::Identity();
    nuisances[g].nuisance_correlation_id = TestId(
        "parallel-failure-nuisance", g);
    nuisances[g].control_registration_valid = true;
    nuisances[g].deformation_measurement_valid = true;
  }
  nuisances[1].pose_covariance =
      -Eigen::Matrix<double, 6, 6>::Identity();

  parallel_reasons.clear();
  serial_reasons.clear();
  EXPECT_FALSE(parallel.Update(
      measurements, nuisances, update_stamp, &parallel_reasons));
  EXPECT_FALSE(UacGaussianSumFilterTestAccess::UpdateSerial(
      &serial, measurements, nuisances, update_stamp, &serial_reasons));
  EXPECT_EQ(parallel_reasons, serial_reasons);
  const std::vector<ReasonCode> expected_reasons{
      ReasonCode::SOURCE_MISSING_LIMIT,
      ReasonCode::INPUT_POSE_COVARIANCE_INVALID};
  EXPECT_EQ(parallel_reasons, expected_reasons);
  EXPECT_EQ(parallel.approximation_flags(), serial.approximation_flags());
  EXPECT_EQ(predicted_bytes,
            UacGaussianSumFilterTestAccess::MutableStateBytes(parallel));
  EXPECT_EQ(predicted_bytes,
            UacGaussianSumFilterTestAccess::MutableStateBytes(serial));
}

TEST(V1Dynamics, FrozenTaskGeometryKeepsRepeatedPosteriorBitStable) {
  V1Config config = SyntheticKnownConfig("uac-frozen-task-cache-test");
  // Include an interface so the regression also covers the invariant that
  // jump columns are excluded from the cached smooth task norm.
  const ReferenceSnapshot reference = MetricLineReference(true);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(5U, 0U), &reasons));
  ASSERT_TRUE(filter.Predict(ros::Time(5U, 100000000U), &reasons));

  const auto first = filter.Posterior();
  ASSERT_EQ(first.size(), 1U);
  // Exercise a dynamic point query between reads.  It may build a separate
  // point-local POU basis, but must neither modify nor invalidate the frozen
  // task quadrature used by Posterior().
  const FieldQueryResult query = filter.Query(
      0U, Eigen::Vector3d(0.15, 0.01, 0.0));
  EXPECT_EQ(query.validity, Validity::VALID);
  const FieldQueryResult generic_surfel_query = filter.QueryWithPosterior(
      0U, reference.surfels[2U].center_R, first.front());
  const FieldQueryResult cached_surfel_query =
      filter.QueryReferenceSurfelWithPosterior(
          0U, 2U, first.front());
  PouField field(&reference, &config);
  Validity generic_basis_validity = Validity::UNKNOWN;
  std::vector<ReasonCode> generic_basis_reasons;
  double generic_attachment_distance =
      std::numeric_limits<double>::infinity();
  const Eigen::MatrixXd generic_basis = field.VectorBasis(
      0U, reference.surfels[2U].center_R, &generic_basis_validity,
      &generic_basis_reasons, &generic_attachment_distance);
  Eigen::MatrixXd cached_basis;
  Validity cached_basis_validity = Validity::UNKNOWN;
  std::vector<ReasonCode> cached_basis_reasons;
  double cached_attachment_distance =
      std::numeric_limits<double>::infinity();
  ASSERT_TRUE(filter.ReferenceSurfelBasis(
      0U, 2U, &cached_basis, &cached_basis_validity,
      &cached_basis_reasons, &cached_attachment_distance));
  EXPECT_EQ(cached_basis_validity, generic_basis_validity);
  EXPECT_EQ(cached_basis_reasons, generic_basis_reasons);
  EXPECT_DOUBLE_EQ(cached_attachment_distance, generic_attachment_distance);
  EXPECT_TRUE(cached_basis.isApprox(generic_basis, 0.0));
  EXPECT_EQ(cached_surfel_query.validity, generic_surfel_query.validity);
  EXPECT_EQ(cached_surfel_query.reasons, generic_surfel_query.reasons);
  EXPECT_DOUBLE_EQ(cached_surfel_query.interpolation_error_upper_m,
                   generic_surfel_query.interpolation_error_upper_m);
  EXPECT_TRUE(cached_surfel_query.mean_displacement_R.isApprox(
      generic_surfel_query.mean_displacement_R, 0.0));
  EXPECT_TRUE(cached_surfel_query.mean_velocity_R.isApprox(
      generic_surfel_query.mean_velocity_R, 0.0));
  EXPECT_TRUE(cached_surfel_query.displacement_covariance_R.isApprox(
      generic_surfel_query.displacement_covariance_R, 0.0));
  EXPECT_TRUE(cached_surfel_query.velocity_covariance_R.isApprox(
      generic_surfel_query.velocity_covariance_R, 0.0));
  EXPECT_TRUE(cached_surfel_query.observable_projector_R.isApprox(
      generic_surfel_query.observable_projector_R, 0.0));
  EXPECT_DOUBLE_EQ(cached_surfel_query.active_probability.mean,
                   generic_surfel_query.active_probability.mean);
  EXPECT_DOUBLE_EQ(cached_surfel_query.other_mass,
                   generic_surfel_query.other_mass);
  ASSERT_EQ(cached_surfel_query.components.size(),
            generic_surfel_query.components.size());
  for (size_t component = 0U;
       component < cached_surfel_query.components.size(); ++component) {
    const FieldComponentQuery& cached_component =
        cached_surfel_query.components[component];
    const FieldComponentQuery& generic_component =
        generic_surfel_query.components[component];
    EXPECT_EQ(cached_component.component_id, generic_component.component_id);
    EXPECT_TRUE(cached_component.mean_displacement_R.isApprox(
        generic_component.mean_displacement_R, 0.0));
    EXPECT_TRUE(cached_component.mean_velocity_R.isApprox(
        generic_component.mean_velocity_R, 0.0));
    EXPECT_TRUE(cached_component.displacement_covariance_R.isApprox(
        generic_component.displacement_covariance_R, 0.0));
    EXPECT_TRUE(cached_component.velocity_covariance_R.isApprox(
        generic_component.velocity_covariance_R, 0.0));
  }
  const FieldQueryResult invalid_cached_surfel =
      filter.QueryReferenceSurfelWithPosterior(
          0U, static_cast<uint32_t>(reference.surfels.size()), first.front());
  EXPECT_EQ(invalid_cached_surfel.validity, Validity::INVALID);
  EXPECT_NE(std::find(invalid_cached_surfel.reasons.begin(),
                      invalid_cached_surfel.reasons.end(),
                      ReasonCode::INVALID_NUMERICAL),
            invalid_cached_surfel.reasons.end());
  const auto repeated = filter.Posterior();
  ASSERT_EQ(repeated.size(), first.size());

  const GduPosterior& lhs = first.front();
  const GduPosterior& rhs = repeated.front();
  EXPECT_EQ(lhs.gdu_id, rhs.gdu_id);
  EXPECT_EQ(lhs.state_probability, rhs.state_probability);
  EXPECT_DOUBLE_EQ(lhs.task_scale_deformation_probability,
                   rhs.task_scale_deformation_probability);
  EXPECT_DOUBLE_EQ(lhs.task_displacement_probability,
                   rhs.task_displacement_probability);
  EXPECT_DOUBLE_EQ(lhs.current_velocity_exceedance_probability,
                   rhs.current_velocity_exceedance_probability);
  EXPECT_DOUBLE_EQ(lhs.task_interface_jump_probability,
                   rhs.task_interface_jump_probability);
  EXPECT_DOUBLE_EQ(lhs.minimum_detectable_effect_m,
                   rhs.minimum_detectable_effect_m);
  EXPECT_DOUBLE_EQ(lhs.velocity_minimum_detectable_effect_mps,
                   rhs.velocity_minimum_detectable_effect_mps);
  EXPECT_DOUBLE_EQ(lhs.interface_jump_minimum_detectable_effect_m,
                   rhs.interface_jump_minimum_detectable_effect_m);
  EXPECT_DOUBLE_EQ(lhs.task_power_worst_normalized_mde,
                   rhs.task_power_worst_normalized_mde);
  EXPECT_EQ(lhs.task_power_complete, rhs.task_power_complete);
  EXPECT_EQ(lhs.any_task_direction_detectable,
            rhs.any_task_direction_detectable);
  EXPECT_EQ(lhs.all_task_directions_detectable,
            rhs.all_task_directions_detectable);
  EXPECT_TRUE(lhs.observable_projector.isApprox(
      rhs.observable_projector, 0.0));
  EXPECT_TRUE(lhs.nullspace.isApprox(rhs.nullspace, 0.0));
  ASSERT_EQ(lhs.components.size(), rhs.components.size());
  for (size_t component = 0U; component < lhs.components.size(); ++component) {
    EXPECT_EQ(lhs.components[component].component_id,
              rhs.components[component].component_id);
    EXPECT_DOUBLE_EQ(lhs.components[component].weight,
                     rhs.components[component].weight);
    EXPECT_TRUE(lhs.components[component].mean.isApprox(
        rhs.components[component].mean, 0.0));
    EXPECT_TRUE(lhs.components[component].covariance.isApprox(
        rhs.components[component].covariance, 0.0));
  }
}

TEST(V1Dynamics, PlannerHorizonPredictionUsesFullCtmcWithoutMutatingSnapshot) {
  V1Config config = SyntheticKnownConfig("uac-planner-horizon-test");
  const ReferenceSnapshot reference = MetricLineReference(false);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(5U, 0U), &reasons));

  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>> predicted;
  ASSERT_TRUE(filter.PredictiveMixture(2.0, &predicted, &reasons));
  ASSERT_EQ(predicted.size(), 1U);
  const Eigen::Matrix3d transition = CtmcTransition(
      UacGenerator(config.calibration.ctmc_rates), 2.0);
  for (int state = 0; state < 3; ++state)
    EXPECT_NEAR(predicted.front().state_probability[static_cast<size_t>(state)],
                transition(0, state), 1.0e-11);
  EXPECT_GT(predicted.front().state_probability[1] +
                predicted.front().state_probability[2],
            0.0);

  // 0.75 s is deliberately not a published geometry node.  Querying it must
  // propagate the same CTMC directly to the requested ETA rather than
  // interpolate the 0.5 s and 1.0 s posteriors.
  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>>
      arbitrary_eta;
  ASSERT_TRUE(filter.PredictiveMixture(0.75, &arbitrary_eta, &reasons));
  ASSERT_EQ(arbitrary_eta.size(), 1U);
  const Eigen::Matrix3d arbitrary_transition = CtmcTransition(
      UacGenerator(config.calibration.ctmc_rates), 0.75);
  for (int state = 0; state < 3; ++state)
    EXPECT_NEAR(
        arbitrary_eta.front().state_probability[static_cast<size_t>(state)],
        arbitrary_transition(0, state), 1.0e-11);

  const FieldQueryResult future = filter.QueryAtHorizon(
      0U, Eigen::Vector3d(0.05, 0.02, 0.0), 2.0);
  EXPECT_NEAR(future.active_probability.mean,
              transition(0, 1) + transition(0, 2), 1.0e-11);
  const auto current = filter.Posterior();
  ASSERT_EQ(current.size(), 1U);
  EXPECT_NEAR(current.front().state_probability[0], 1.0, 1.0e-14);
  EXPECT_NEAR(current.front().state_probability[1], 0.0, 1.0e-14);
  EXPECT_NEAR(current.front().state_probability[2], 0.0, 1.0e-14);
}

TEST(V1Dynamics, InterfaceBirthKernelsRespectCoherentRelativeAndUnresolvedRoles) {
  V1Config config = SyntheticKnownConfig("interface-kernel-test");
  config.resource.mixture_max_components = 256;
  const ReferenceSnapshot reference = MetricLineReference(true);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(5U, 0U), &reasons));
  ASSERT_TRUE(filter.Predict(ros::Time(5U, 50000000U), &reasons));
  const auto posterior = filter.Posterior();
  ASSERT_EQ(posterior.size(), 1U);
  bool found_changed_coherent = false;
  bool found_changed_relative = false;
  bool found_changed_unresolved = false;
  bool found_active_relative = false;
  for (const GaussianComponent& component : posterior.front().components) {
    if (component.interface_modes.size() != 1U ||
        component.covariance.rows() < 21)
      continue;
    const double smooth_variance = component.covariance(0, 0);
    const double jump_variance = component.covariance(18, 18);
    if (component.state == StructuralState::C &&
        component.interface_modes.front() ==
            InterfaceJointState::LINK_COHERENT) {
      found_changed_coherent = true;
      EXPECT_LT(jump_variance, smooth_variance);
      EXPECT_NEAR(jump_variance,
          1.0 / (1.0 /
                     config.calibration.changed_birth_displacement_variance +
                 1.0 /
                     config.calibration.interface_coherent_process_variance),
          1.0e-14);
    }
    if (component.state == StructuralState::C &&
        component.interface_modes.front() ==
            InterfaceJointState::LINK_RELATIVE_MOTION) {
      found_changed_relative = true;
      EXPECT_GT(jump_variance,
                config.calibration.interface_coherent_process_variance);
      EXPECT_LT(jump_variance, smooth_variance);
    }
    if (component.state == StructuralState::C &&
        component.interface_modes.front() ==
            InterfaceJointState::LINK_UNRESOLVED) {
      found_changed_unresolved = true;
      EXPECT_GT(jump_variance, smooth_variance);
      EXPECT_NEAR(jump_variance,
          config.calibration.changed_birth_displacement_variance +
              config.calibration.interface_unresolved_process_variance,
          1.0e-14);
    }
    if (component.state == StructuralState::A &&
        component.interface_modes.front() ==
            InterfaceJointState::LINK_RELATIVE_MOTION) {
      found_active_relative = true;
      EXPECT_NEAR(jump_variance,
          config.calibration.interface_relative_process_variance,
          1.0e-14);
    }
  }
  EXPECT_TRUE(found_changed_coherent);
  EXPECT_TRUE(found_changed_relative);
  EXPECT_TRUE(found_changed_unresolved);
  EXPECT_TRUE(found_active_relative);
}

TEST(V1Dynamics, RejectsReferencesOutsideFrozenInterfaceContract) {
  const V1Config config = SyntheticKnownConfig("interface-contract-test");
  ReferenceSnapshot multiple = MetricLineReference(true);
  InterfaceDefinition second = multiple.interfaces.front();
  second.id = TestId("metric-line-interface", 1U);
  multiple.interfaces.push_back(second);
  multiple.gdus.front().interface_indices.push_back(1U);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  EXPECT_FALSE(filter.Initialize(multiple, ros::Time(5U, 0U), &reasons));
  EXPECT_NE(std::find(reasons.begin(), reasons.end(),
                      ReasonCode::INVALID_NUMERICAL), reasons.end());

  ReferenceSnapshot cross_gdu = MetricLineReference(true);
  cross_gdu.interfaces.front().cross_gdu = true;
  reasons.clear();
  EXPECT_FALSE(filter.Initialize(cross_gdu, ros::Time(5U, 0U), &reasons));
  EXPECT_NE(std::find(reasons.begin(), reasons.end(),
                      ReasonCode::INVALID_NUMERICAL), reasons.end());
}

TEST(V1Dynamics, PosteriorPreservesEmptyCoordinatesForUnmonitorableGdu) {
  const V1Config config = SyntheticKnownConfig("unmonitorable-posterior-test");
  ReferenceSnapshot reference;
  reference.frozen = true;
  reference.id = TestId("unmonitorable-reference", 0U);
  reference.world_frame = config.resource.world_frame;
  GduDefinition other;
  other.id = TestId("unmonitorable-gdu", 0U);
  other.monitorable = false;
  other.coordinate_rank = 0U;
  reference.gdus.push_back(other);

  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(5U, 0U), &reasons));
  const auto posterior = filter.Posterior();
  ASSERT_EQ(posterior.size(), 1U);
  EXPECT_EQ(posterior.front().observable_projector.rows(), 0);
  EXPECT_EQ(posterior.front().observable_projector.cols(), 0);
  EXPECT_EQ(posterior.front().nullspace.rows(), 0);
  EXPECT_EQ(posterior.front().nullspace.cols(), 0);

  RiskQueryEngine engine(config);
  std::vector<HazardGeometry, Eigen::aligned_allocator<HazardGeometry>> hazards;
  EXPECT_TRUE(engine.BuildHazards(CommonHeader(), reference, filter,
                                  config.resource.max_hazard_primitives,
                                  &hazards, &reasons));
  EXPECT_TRUE(hazards.empty())
      << "an OTHER carrier belongs to occupancy, not deformation prediction";
}

TEST(V1Pipeline,
     DetectionTilesKeepOtherCoverageWithoutPhysicalDeformationSamples) {
  const V1Config config = SyntheticKnownConfig(
      "unmonitorable-detection-tile-contract-test");
  ReferenceSnapshot reference = MetricLineReference(false);
  const uint32_t other_surfel_index =
      static_cast<uint32_t>(reference.surfels.size());
  Surfel other_surfel;
  other_surfel.id = TestId("unmonitorable-tile-surfel", 0U);
  other_surfel.center_R = Eigen::Vector3d(1.0, 0.5, 0.25);
  other_surfel.support_radius_m = 0.05;
  other_surfel.gdu_index = 1U;
  other_surfel.geometry_probability[3] = 1.0;
  reference.surfels.push_back(other_surfel);
  GduDefinition other_gdu;
  other_gdu.id = TestId("unmonitorable-tile-gdu", 0U);
  other_gdu.surfel_indices.push_back(other_surfel_index);
  other_gdu.coordinate_geometry = GeometryHypothesis::OTHER;
  other_gdu.coordinate_rank = 0U;
  other_gdu.monitorable = false;
  reference.gdus.push_back(other_gdu);

  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(6U, 0U), &reasons));
  V1Snapshot snapshot;
  snapshot.common.snapshot_id = TestId("unmonitorable-tile-snapshot", 0U);
  snapshot.common.validity = Validity::VALID;
  snapshot.reference = reference;
  snapshot.deformation = filter.Posterior();
  const auto tiles = BuildDetectionTileMessages(snapshot, filter, config);
  ASSERT_EQ(tiles.size(), 3U);

  const auto monitorable_id = BuildStableIdMessage(reference.gdus[0].id);
  const auto other_id = BuildStableIdMessage(other_gdu.id);
  const auto same_id = [](const auto& lhs, const auto& rhs) {
    return lhs.high == rhs.high && lhs.low == rhs.low;
  };
  ASSERT_EQ(tiles[2].reference_support.size(), reference.surfels.size());
  EXPECT_NE(std::find_if(
                tiles[2].reference_support.begin(),
                tiles[2].reference_support.end(),
                [&](const auto& support) {
                  return same_id(support.gdu_id, other_id);
                }),
            tiles[2].reference_support.end())
      << "OTHER frozen geometry remains an explicit coverage product";
  ASSERT_EQ(tiles[2].physical_deformation.size(),
            reference.gdus[0].surfel_indices.size());
  for (const auto& sample : tiles[2].physical_deformation) {
    EXPECT_TRUE(same_id(sample.gdu_id, monitorable_id));
    EXPECT_FALSE(same_id(sample.gdu_id, other_id));
  }

  const auto find_other_evidence = [&](const auto& tile) {
    return std::find_if(
        tile.detection_evidence.begin(), tile.detection_evidence.end(),
        [&](const auto& evidence) {
          return same_id(evidence.gdu_id, other_id);
        });
  };
  const auto lod1_other = find_other_evidence(tiles[1]);
  const auto lod2_other = find_other_evidence(tiles[2]);
  ASSERT_NE(lod1_other, tiles[1].detection_evidence.end());
  ASSERT_NE(lod2_other, tiles[2].detection_evidence.end());
  EXPECT_EQ(lod2_other->validity,
            static_cast<uint8_t>(Validity::INCONCLUSIVE));
  EXPECT_EQ(lod2_other->observable_rank, 0U);
  EXPECT_FALSE(lod2_other->detection_allowed);
  EXPECT_FALSE(lod2_other->clearing_allowed);
}

TEST(V1Pipeline,
     ParallelDetectionTileSerializationMatchesSerialBytesAndOrder) {
  const V1Config config = SyntheticKnownConfig(
      "parallel-detection-tile-serialization-test");
  constexpr uint32_t kGduCount = 8U;
  const ReferenceSnapshot reference =
      ReplicatedMetricLineReference(kGduCount, true);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(12U, 0U), &reasons));

  V1Snapshot snapshot;
  snapshot.common.snapshot_id = TestId(
      "parallel-detection-tile-snapshot", 0U);
  snapshot.common.measurement_stamp = ros::Time(12U, 100000000U);
  snapshot.common.validity = Validity::VALID;
  snapshot.reference = reference;
  snapshot.deformation = filter.Posterior();
  snapshot.interfaces = filter.InterfacePosteriors(snapshot.deformation);

  const auto serial = BuildDetectionTileMessagesSerialForTesting(
      snapshot, filter, config);
  const auto parallel = BuildDetectionTileMessages(snapshot, filter, config);
  const auto parallel_repeat = BuildDetectionTileMessages(
      snapshot, filter, config);
  ASSERT_EQ(serial.size(), 3U);
  ASSERT_EQ(parallel.size(), serial.size());
  ASSERT_EQ(parallel_repeat.size(), serial.size());

  const auto serialized_bytes = [](const auto& message) {
    const uint32_t length =
        ros::serialization::serializationLength(message);
    std::vector<uint8_t> bytes(length);
    ros::serialization::OStream stream(bytes.data(), length);
    ros::serialization::serialize(stream, message);
    return bytes;
  };
  for (size_t lod = 0U; lod < serial.size(); ++lod) {
    const std::vector<uint8_t> serial_bytes = serialized_bytes(serial[lod]);
    EXPECT_EQ(serial_bytes, serialized_bytes(parallel[lod]));
    EXPECT_EQ(serial_bytes, serialized_bytes(parallel_repeat[lod]));
  }

  ASSERT_EQ(parallel[2].reference_support.size(),
            reference.surfels.size());
  size_t output_index = 0U;
  for (uint32_t g = 0U; g < kGduCount; ++g) {
    for (uint32_t surfel_index : reference.gdus[g].surfel_indices) {
      ASSERT_LT(output_index, parallel[2].reference_support.size());
      const auto expected = BuildStableIdMessage(
          reference.surfels[surfel_index].id);
      const auto& actual = parallel[2].reference_support[output_index];
      EXPECT_EQ(actual.surfel_id.high, expected.high);
      EXPECT_EQ(actual.surfel_id.low, expected.low);
      ++output_index;
    }
  }
  EXPECT_EQ(output_index, parallel[2].reference_support.size());
  ASSERT_EQ(parallel[1].detection_evidence.size(), kGduCount);
  for (uint32_t g = 0U; g < kGduCount; ++g) {
    const auto expected = BuildStableIdMessage(reference.gdus[g].id);
    EXPECT_EQ(parallel[1].detection_evidence[g].gdu_id.high, expected.high);
    EXPECT_EQ(parallel[1].detection_evidence[g].gdu_id.low, expected.low);
  }
}

TEST(V1Detection, EffectiveSupportUsesIndependentSourcesAndOuFramePenalty) {
  V1Config config = SyntheticKnownConfig("effective-support-test");
  config.resource.fixed_lag_frames = 3;
  config.resource.max_update_gap_s = 0.2;
  config.calibration.pose_ou_tau_s = 0.5;
  config.calibration.detection_min_effective_spatial_support = 2.0;
  config.calibration.detection_min_effective_temporal_support = 1.2;
  config.calibration.task_velocity_required_duration_s = 0.4;
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);
  const ReferenceSnapshot reference = MetricLineReference(false);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(5U, 0U), &reasons));

  const int coefficient_count = 18;
  std::vector<AssociationMeasurementSet> gdu_measurements;
  for (uint64_t source = 0U; source < 6U; ++source) {
    AssociationMeasurementSet set;
    set.source_evidence_id = TestId("support-source", source);
    AssociationCategory category;
    category.candidate_id = TestId("support-candidate", source);
    category.semantics = AssociationSemantics::OTHER;
    category.geometric_candidate = true;
    category.probability = 1.0;
    set.selection.categories.push_back(category);
    set.selection.entropy = 0.0;
    set.selection.other_mass = 0.0;
    set.selection.valid = true;
    FieldMeasurement measurement;
    measurement.evidence_id = TestId("support-measurement", source);
    measurement.candidate_id = category.candidate_id;
    measurement.gdu_index = 0U;
    for (HypothesisMeasurement& hypothesis : measurement.hypothesis) {
      hypothesis.residual_m = Eigen::Vector3d::Zero();
      hypothesis.design = Eigen::MatrixXd::Zero(3, coefficient_count);
      hypothesis.design.block<3, 3>(0, 3 * static_cast<int>(source)) =
          Eigen::Matrix3d::Identity();
      hypothesis.reference_bias_design = Eigen::Matrix3d::Zero();
      hypothesis.extrinsic_design = Eigen::MatrixXd::Zero(3, 6);
      hypothesis.pose_design = Eigen::MatrixXd::Zero(3, 6);
      hypothesis.independent_covariance_m2 =
          1.0e-6 * Eigen::Matrix3d::Identity();
      hypothesis.observable_rank = 3;
    }
    set.candidates.push_back(measurement);
    gdu_measurements.push_back(set);
  }
  const std::vector<std::vector<AssociationMeasurementSet>> measurements{{
      gdu_measurements}};
  FrameNuisance nuisance;
  nuisance.pose_covariance =
      1.0e-8 * Eigen::Matrix<double, 6, 6>::Identity();
  nuisance.nuisance_correlation_id = TestId("support-nuisance", 0U);
  nuisance.control_registration_valid = true;
  nuisance.deformation_measurement_valid = true;
  const std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>
      nuisances{nuisance};

  std::array<double, 3> temporal_support{{0.0, 0.0, 0.0}};
  GduPosterior final_posterior;
  for (uint32_t update = 0U; update < 3U; ++update) {
    const ros::Time stamp(5U, 200000000U * (update + 1U));
    ASSERT_TRUE(filter.Predict(stamp, &reasons));
    ASSERT_TRUE(filter.Update(measurements, nuisances, stamp, &reasons));
    const auto posterior = filter.Posterior();
    ASSERT_EQ(posterior.size(), 1U);
    EXPECT_NEAR(posterior[0].effective_spatial_support, 6.0, 1.0e-12);
    temporal_support[update] = posterior[0].effective_temporal_support;
    final_posterior = posterior[0];
  }
  EXPECT_NEAR(temporal_support[0], 1.0, 1.0e-12);
  EXPECT_GT(temporal_support[1], temporal_support[0]);
  EXPECT_GT(temporal_support[2], temporal_support[1]);
  EXPECT_GE(temporal_support[2],
            config.calibration.detection_min_effective_temporal_support);
  EXPECT_TRUE(final_posterior.velocity_minimum_detectable_effect_defined);
  EXPECT_TRUE(final_posterior.task_power_complete);
  EXPECT_TRUE(final_posterior.any_task_direction_detectable);
  EXPECT_FALSE(final_posterior.all_task_directions_detectable);
  EXPECT_TRUE(final_posterior.detection_allowed);
  EXPECT_FALSE(final_posterior.clearing_allowed);
  EXPECT_FALSE(final_posterior.interface_jump_minimum_detectable_effect_defined);
  const auto expect_unit_probability = [](double probability) {
    EXPECT_TRUE(std::isfinite(probability));
    EXPECT_GE(probability, 0.0);
    EXPECT_LE(probability, 1.0);
  };
  for (double probability : final_posterior.state_probability)
    expect_unit_probability(probability);
  expect_unit_probability(final_posterior.other_mass);
  expect_unit_probability(final_posterior.task_scale_deformation_probability);
  expect_unit_probability(final_posterior.task_scale_deformation_lower);
  expect_unit_probability(final_posterior.task_scale_deformation_upper);
  expect_unit_probability(final_posterior.task_displacement_probability);
  expect_unit_probability(
      final_posterior.current_velocity_exceedance_probability);
  expect_unit_probability(final_posterior.task_interface_jump_probability);
  EXPECT_GE(final_posterior.task_scale_deformation_probability,
            final_posterior.task_displacement_probability - 1.0e-15);
  EXPECT_GT(final_posterior.current_velocity_exceedance_probability, 0.0);
  EXPECT_GE(final_posterior.task_scale_deformation_probability,
            final_posterior.task_interface_jump_probability - 1.0e-15);
  EXPECT_LE(final_posterior.task_scale_deformation_probability,
            final_posterior.task_displacement_probability +
                final_posterior.current_velocity_exceedance_probability +
                final_posterior.task_interface_jump_probability + 1.0e-15);
  EXPECT_FALSE(final_posterior.velocity_duration_requirement_met);
  EXPECT_DOUBLE_EQ(
      final_posterior.continuous_valid_observation_duration_s, 0.0);
  const auto repeated = filter.Posterior();
  ASSERT_EQ(repeated.size(), 1U);
  EXPECT_DOUBLE_EQ(repeated[0].task_scale_deformation_probability,
                   final_posterior.task_scale_deformation_probability);
  EXPECT_DOUBLE_EQ(final_posterior.other_mass, 0.0);
  EXPECT_LE(final_posterior.components.size(),
            static_cast<size_t>(config.resource.mixture_max_components));
}

TEST(V1Detection, DominantSourceMissingSkipsUpdateAndCountsConsecutively) {
  V1Config config = SyntheticKnownConfig("source-missing-test");
  config.resource.source_missing_consecutive_limit = 2U;
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);
  const ReferenceSnapshot reference = MetricLineReference(false);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(9U, 0U), &reasons));

  AssociationMeasurementSet missing_set;
  missing_set.source_evidence_id = TestId("missing-source", 0U);
  AssociationCategory missing;
  missing.semantics = AssociationSemantics::SOURCE_MISSING;
  missing.geometric_candidate = false;
  missing.probability = 1.0;
  missing_set.selection.categories.push_back(missing);
  missing_set.selection.entropy = 0.0;
  missing_set.selection.other_mass = 0.0;
  missing_set.selection.valid = true;
  const std::vector<std::vector<AssociationMeasurementSet>> measurements{{
      missing_set}};

  FrameNuisance nuisance;
  nuisance.pose_covariance =
      1.0e-8 * Eigen::Matrix<double, 6, 6>::Identity();
  nuisance.nuisance_correlation_id = TestId("missing-nuisance", 0U);
  nuisance.control_registration_valid = true;
  nuisance.deformation_measurement_valid = true;
  const std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>
      nuisances{nuisance};

  for (uint32_t update = 0U; update < 2U; ++update) {
    const ros::Time stamp(9U, 100000000U * (update + 1U));
    reasons.clear();
    ASSERT_TRUE(filter.Predict(stamp, &reasons));
    const auto predicted = filter.Posterior();
    ASSERT_EQ(predicted.size(), 1U);
    ASSERT_TRUE(filter.Update(measurements, nuisances, stamp, &reasons));
    const auto skipped = filter.Posterior();
    ASSERT_EQ(skipped.size(), 1U);
    for (size_t state = 0U; state < 3U; ++state)
      EXPECT_NEAR(skipped[0].state_probability[state],
                  predicted[0].state_probability[state], 1.0e-15);
    EXPECT_EQ(skipped[0].admission,
              DetectionAdmission::INCONCLUSIVE_ASSOCIATION);
    EXPECT_FALSE(skipped[0].detection_allowed);
    EXPECT_FALSE(skipped[0].clearing_allowed);
    EXPECT_DOUBLE_EQ(skipped[0].effective_spatial_support, 0.0);
    const bool limit_reported = std::find(
        reasons.begin(), reasons.end(), ReasonCode::SOURCE_MISSING_LIMIT) !=
        reasons.end();
    EXPECT_EQ(limit_reported,
              update + 1U >= config.resource.source_missing_consecutive_limit);
  }
}

TEST(V1Detection, SourceLocalOtherAndUnpairedCandidateAreIdentityBranches) {
  V1Config config = SyntheticKnownConfig("source-local-other-test");
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(
      MetricLineReference(false), ros::Time(11U, 0U), &reasons));

  FrameNuisance nuisance;
  nuisance.pose_covariance =
      1.0e-8 * Eigen::Matrix<double, 6, 6>::Identity();
  nuisance.nuisance_correlation_id = TestId("other-nuisance", 0U);
  nuisance.control_registration_valid = true;
  nuisance.deformation_measurement_valid = true;
  const std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>
      nuisances{nuisance};

  for (int branch = 0; branch < 2; ++branch) {
    const ros::Time stamp(11U, 100000000U * static_cast<uint32_t>(branch + 1));
    ASSERT_TRUE(filter.Predict(stamp, &reasons));
    const GduPosterior predicted = filter.Posterior().front();

    AssociationMeasurementSet set;
    set.source_evidence_id = TestId("other-source", branch);
    AssociationCategory category;
    category.semantics = AssociationSemantics::OTHER;
    category.probability = 1.0;
    category.geometric_candidate = branch == 1;
    if (category.geometric_candidate)
      category.candidate_id = TestId("unpaired-candidate", 0U);
    set.selection.categories.push_back(category);
    set.selection.other_mass = branch == 0 ? 1.0 : 0.0;
    set.selection.valid = true;
    const std::vector<std::vector<AssociationMeasurementSet>> measurements{{set}};
    ASSERT_TRUE(filter.Update(measurements, nuisances, stamp, &reasons));
    const GduPosterior posterior = filter.Posterior().front();
    for (size_t state = 0U; state < 3U; ++state)
      EXPECT_NEAR(posterior.state_probability[state],
                  predicted.state_probability[state], 1.0e-14);
    EXPECT_DOUBLE_EQ(posterior.other_mass, 0.0)
        << "per-source association uncertainty must not accumulate into the "
           "persistent global model-inadequacy mass";
    EXPECT_LE(posterior.components.size(),
              static_cast<size_t>(config.resource.mixture_max_components));
  }
}

TEST(V1Detection, RegisteredInterfaceHasSeparateJumpProbabilityAndPower) {
  V1Config config = SyntheticKnownConfig("interface-task-test");
  config.resource.fixed_lag_frames = 3;
  config.resource.max_update_gap_s = 0.2;
  config.calibration.pose_ou_tau_s = 0.5;
  config.calibration.ctmc_rates[0] = 1.0;
  config.calibration.ctmc_rates[1] = 1.0;
  config.calibration.detection_min_effective_spatial_support = 2.0;
  config.calibration.detection_min_effective_temporal_support = 1.2;
  config.calibration.task_interface_jump_scale_m = 0.001;
  config.calibration.task_velocity_required_duration_s = 0.4;
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);
  const ReferenceSnapshot reference = MetricLineReference(true);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(7U, 0U), &reasons));

  constexpr int coefficient_count = 21;
  std::vector<AssociationMeasurementSet> gdu_measurements;
  for (uint64_t source = 0U; source < 7U; ++source) {
    AssociationMeasurementSet set;
    set.source_evidence_id = TestId("interface-source", source);
    AssociationCategory category;
    category.candidate_id = TestId("interface-candidate", source);
    category.semantics = AssociationSemantics::OTHER;
    category.geometric_candidate = true;
    category.probability = 1.0;
    set.selection.categories.push_back(category);
    set.selection.entropy = 0.0;
    set.selection.other_mass = 0.0;
    set.selection.valid = true;
    FieldMeasurement measurement;
    measurement.evidence_id = TestId("interface-measurement", source);
    measurement.candidate_id = category.candidate_id;
    measurement.gdu_index = 0U;
    for (HypothesisMeasurement& hypothesis : measurement.hypothesis) {
      hypothesis.residual_m = Eigen::Vector3d::Zero();
      hypothesis.design = Eigen::MatrixXd::Zero(3, coefficient_count);
      hypothesis.design.block<3, 3>(0, 3 * static_cast<int>(source)) =
          Eigen::Matrix3d::Identity();
      hypothesis.reference_bias_design = Eigen::Matrix3d::Zero();
      hypothesis.extrinsic_design = Eigen::MatrixXd::Zero(3, 6);
      hypothesis.pose_design = Eigen::MatrixXd::Zero(3, 6);
      hypothesis.independent_covariance_m2 =
          1.0e-6 * Eigen::Matrix3d::Identity();
      hypothesis.observable_rank = 3;
    }
    set.candidates.push_back(measurement);
    gdu_measurements.push_back(set);
  }
  const std::vector<std::vector<AssociationMeasurementSet>> measurements{{
      gdu_measurements}};
  FrameNuisance nuisance;
  nuisance.pose_covariance =
      1.0e-8 * Eigen::Matrix<double, 6, 6>::Identity();
  nuisance.nuisance_correlation_id = TestId("interface-nuisance", 0U);
  nuisance.control_registration_valid = true;
  nuisance.deformation_measurement_valid = true;
  const std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>
      nuisances{nuisance};

  GduPosterior posterior;
  for (uint32_t update = 0U; update < 3U; ++update) {
    const ros::Time stamp(7U, 200000000U * (update + 1U));
    ASSERT_TRUE(filter.Predict(stamp, &reasons));
    ASSERT_TRUE(filter.Update(measurements, nuisances, stamp, &reasons));
    const auto current = filter.Posterior();
    ASSERT_EQ(current.size(), 1U);
    posterior = current.front();
  }
  EXPECT_TRUE(posterior.minimum_detectable_effect_defined);
  EXPECT_TRUE(posterior.velocity_minimum_detectable_effect_defined);
  EXPECT_TRUE(posterior.interface_jump_minimum_detectable_effect_defined);
  EXPECT_TRUE(posterior.task_power_complete);
  EXPECT_TRUE(posterior.any_task_direction_detectable);
  EXPECT_FALSE(posterior.all_task_directions_detectable);
  EXPECT_TRUE(posterior.detection_allowed);
  EXPECT_FALSE(posterior.clearing_allowed);
  EXPECT_GT(posterior.task_interface_jump_probability, 0.0);
  EXPECT_GE(posterior.task_scale_deformation_probability,
            posterior.task_displacement_probability - 1.0e-15);
  EXPECT_GT(posterior.current_velocity_exceedance_probability, 0.0);
  EXPECT_GE(posterior.task_scale_deformation_probability,
            posterior.task_interface_jump_probability - 1.0e-15);
  EXPECT_LE(posterior.task_scale_deformation_probability,
            posterior.task_displacement_probability +
                posterior.current_velocity_exceedance_probability +
                posterior.task_interface_jump_probability + 1.0e-15);
  const auto repeated = filter.Posterior();
  ASSERT_EQ(repeated.size(), 1U);
  EXPECT_DOUBLE_EQ(repeated.front().task_displacement_probability,
                   posterior.task_displacement_probability);
  EXPECT_DOUBLE_EQ(repeated.front().current_velocity_exceedance_probability,
                   posterior.current_velocity_exceedance_probability);
  EXPECT_DOUBLE_EQ(repeated.front().task_interface_jump_probability,
                   posterior.task_interface_jump_probability);
  EXPECT_DOUBLE_EQ(repeated.front().task_scale_deformation_probability,
                   posterior.task_scale_deformation_probability);
  EXPECT_DOUBLE_EQ(repeated.front().interface_jump_minimum_detectable_effect_m,
                   posterior.interface_jump_minimum_detectable_effect_m);
}

TEST(V1Pipeline, BoundedHistoryBuildsOnceAndPreservesSelectedReferenceGeometry) {
  ros::Time::init();
  V1Config config = SyntheticKnownConfig("bounded-history-test");
  config.resource.bounded_reference_search = true;
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  config.calibration.geometry_softmax_bias << 10.0, 0.0, 0.0, -10.0;
  config.config_digest = CanonicalConfigDigest(config);
  V1Pipeline pipeline(config);
  std::deque<FrameBundle> latest;
  for (uint64_t i = 0; i < 40U; ++i) {
    FrameBundle frame = PipelineFrame(config, i, 1.0 + 0.1 * i, true);
    V1Snapshot ignored;
    std::vector<ReasonCode> reasons;
    ASSERT_TRUE(pipeline.AcceptFrame(frame, &ignored, &reasons));
    if (i >= 10U) latest.push_back(frame);
  }
  PhaseRequest request;
  request.session_id = config.resource.session_id;
  request.target = MonitoringPhase::MONITOR_REFERENCE;
  request.effective_stamp = ros::Time(4.95);
  request.selection_support_cutoff = request.effective_stamp;
  request.roi.minimum_R = Eigen::Vector3d(-1, -1, -1);
  request.roi.maximum_R = Eigen::Vector3d(3, 1, 1);
  ASSERT_TRUE(pipeline.SetPhase(request).accepted);
  ASSERT_TRUE(pipeline.reference().frozen) << pipeline.reference_diagnostics();
  EXPECT_EQ(V1PipelineTestAccess::ReferenceBuildAttempts(pipeline), 1U);
  EXPECT_EQ(pipeline.reference().begin_stamp, ros::Time(2.0));
  EXPECT_EQ(pipeline.reference().end_stamp, ros::Time(4.9));
  const auto direct = ReferenceBuilder(config).Build(latest, pipeline.reference_epoch());
  ASSERT_TRUE(direct.frozen);
  EXPECT_EQ(pipeline.reference().id, direct.id);
  EXPECT_EQ(pipeline.reference().surfels.size(), direct.surfels.size());
  EXPECT_GT(std::count_if(direct.gdus.begin(), direct.gdus.end(),
                         [](const GduDefinition& gdu) { return gdu.monitorable; }), 0)
      << "Repeated samples must not consume the entire spatial geometry budget";
  EXPECT_NE(pipeline.reference_diagnostics().find("monitorable_surfels="), std::string::npos);
}

TEST(V1Pipeline, BoundedHistoryCanMeetMinimumDurationAtHighScanRate) {
  ros::Time::init();
  V1Config config = SyntheticKnownConfig("bounded-high-rate-test");
  config.resource.bounded_reference_search = true;
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  config.calibration.geometry_softmax_bias << 10.0, 0.0, 0.0, -10.0;
  V1Pipeline pipeline(config);
  for (uint64_t i = 0; i < 60U; ++i) {
    V1Snapshot ignored;
    std::vector<ReasonCode> reasons;
    ASSERT_TRUE(pipeline.AcceptFrame(PipelineFrame(config, i, 1.0 + 0.01 * i, true),
                                    &ignored, &reasons));
  }
  PhaseRequest request;
  request.session_id = config.resource.session_id;
  request.target = MonitoringPhase::MONITOR_REFERENCE;
  request.effective_stamp = ros::Time(1.595);
  request.selection_support_cutoff = request.effective_stamp;
  request.roi.minimum_R = Eigen::Vector3d(-1, -1, -1);
  request.roi.maximum_R = Eigen::Vector3d(3, 1, 1);
  ASSERT_TRUE(pipeline.SetPhase(request).accepted);
  ASSERT_TRUE(pipeline.reference().frozen) << pipeline.reference_diagnostics();
  EXPECT_EQ(pipeline.reference().end_stamp, ros::Time(1.59));
  EXPECT_GT(std::count_if(pipeline.reference().gdus.begin(), pipeline.reference().gdus.end(),
                         [](const GduDefinition& gdu) { return gdu.monitorable; }), 0);
  EXPECT_GE((pipeline.reference().end_stamp - pipeline.reference().begin_stamp).toSec(), 0.5);
  EXPECT_EQ(V1PipelineTestAccess::ReferenceBuildAttempts(pipeline), 1U);
}

TEST(V1Pipeline, BoundedFallbackStartsAtMinimumDataAndThrottlesFailedBuilds) {
  ros::Time::init();
  for (bool empty_roi : {false, true}) {
    V1Config config = SyntheticKnownConfig(empty_roi ? "bounded-empty-test" : "bounded-dwell-test");
    config.resource.bounded_reference_search = true;
    config.calibration.reference_link_logit_bias = 100.0;
    config.calibration.reference_link_distance_weight = 0.0;
    config.calibration.reference_link_projector_weight = 0.0;
    config.calibration.geometry_softmax_bias << 10.0, 0.0, 0.0, -10.0;
    V1Pipeline pipeline(config);
    PhaseRequest request;
    request.session_id = config.resource.session_id;
    request.target = MonitoringPhase::MONITOR_REFERENCE;
    request.effective_stamp = ros::Time(2.0);
    request.selection_support_cutoff = request.effective_stamp;
    request.roi.minimum_R = empty_roi ? Eigen::Vector3d(100, 100, 100) : Eigen::Vector3d(-1, -1, -1);
    request.roi.maximum_R = empty_roi ? Eigen::Vector3d(101, 101, 101) : Eigen::Vector3d(3, 1, 1);
    ASSERT_TRUE(pipeline.SetPhase(request).accepted);
    EXPECT_EQ(V1PipelineTestAccess::ReferenceBuildAttempts(pipeline), 0U);
    for (uint64_t i = 0; i < 12U; ++i) {
      V1Snapshot ignored;
      std::vector<ReasonCode> reasons;
      ASSERT_TRUE(pipeline.AcceptFrame(PipelineFrame(config, i, 2.0 + 0.1 * i, true),
                                       &ignored, &reasons));
      EXPECT_EQ(V1PipelineTestAccess::ReferenceBuildAttempts(pipeline),
                i < 5U ? 0U : (empty_roi && i == 11U ? 2U : 1U));
    }
    EXPECT_EQ(pipeline.reference().frozen, !empty_roi);
    if (!empty_roi) {
      EXPECT_EQ(pipeline.reference().end_stamp, ros::Time(2.5));
      EXPECT_NE(std::find(pipeline.reference().reasons.begin(), pipeline.reference().reasons.end(),
                          ReasonCode::PRE_REFERENCE_CHANGE_UNOBSERVABLE), pipeline.reference().reasons.end());
    }
  }
}

TEST(V1Pipeline, ReferenceUpdatePublishesOneAtomicSnapshot) {
  ros::Time::init();
  V1Config config = SyntheticKnownConfig("pipeline-transaction-test");
  config.resource.reference_min_frames = 6;
  config.resource.reference_min_duration_s = 0.50;
  config.resource.max_update_gap_s = 0.25;
  config.resource.deadline_ms = 100000.0;
  // This compact synthetic scene is only a few metres wide. Use the minimum
  // valid graph-neighbour exclusion here; the frozen research profile's 3 m
  // neighbourhood is covered by the dedicated mask and integration tests.
  config.resource.pose_control_exclusion_radius_m =
      config.resource.graph_neighbor_radius_m;
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  config.calibration.geometry_softmax_bias << 10.0, 0.0, 0.0, -10.0;
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);

  std::deque<FrameBundle> direct_reference_frames;
  for (uint64_t frame_index = 0U; frame_index < 6U; ++frame_index)
    direct_reference_frames.push_back(PipelineFrame(
        config, frame_index, 1.0 + 0.1 * frame_index, false));
  const ReferenceSnapshot direct_reference =
      ReferenceBuilder(config).Build(direct_reference_frames, 1U);
  ASSERT_TRUE(direct_reference.frozen)
      << "reason="
      << (direct_reference.reasons.empty()
              ? -1 : static_cast<int>(direct_reference.reasons.front()));
  for (const Surfel& surfel : direct_reference.surfels) {
    EXPECT_TRUE(ValidateSymmetricPsd(
        surfel.covariance_R,
        config.numerical.psd_eigenvalue_tolerance));
    EXPECT_TRUE(surfel.geometry_probability.allFinite());
    EXPECT_GE(surfel.geometry_probability.minCoeff(), 0.0);
    EXPECT_NEAR(surfel.geometry_probability.sum(), 1.0,
                config.numerical.normalization_tolerance);
  }
  for (const GduDefinition& gdu : direct_reference.gdus) {
    EXPECT_TRUE(gdu.reference_bias_covariance_R.allFinite());
    EXPECT_TRUE(ValidateSymmetricPsd(
        gdu.reference_bias_covariance_R,
        config.numerical.psd_eigenvalue_tolerance));
  }

  V1Pipeline pipeline(config);
  for (uint64_t frame_index = 0U; frame_index < 6U; ++frame_index) {
    V1Snapshot ignored;
    std::vector<ReasonCode> reasons;
    ASSERT_TRUE(pipeline.AcceptFrame(
        PipelineFrame(config, frame_index, 1.0 + 0.1 * frame_index, true),
        &ignored, &reasons));
    EXPECT_TRUE(ignored.common.snapshot_id.empty());
  }

  PhaseRequest reference_request;
  reference_request.session_id = config.resource.session_id;
  reference_request.target = MonitoringPhase::MONITOR_REFERENCE;
  reference_request.effective_stamp.fromSec(1.55);
  reference_request.selection_support_cutoff =
      reference_request.effective_stamp;
  reference_request.roi.minimum_R = Eigen::Vector3d(-1.0, -1.0, -1.0);
  reference_request.roi.maximum_R = Eigen::Vector3d(3.0, 1.0, 1.0);
  reference_request.trigger_source = TriggerSource::HUMAN_OR_EXTERNAL_TASK;
  const PhaseResponse reference_response = pipeline.SetPhase(reference_request);
  ASSERT_TRUE(reference_response.accepted);
  ASSERT_TRUE(pipeline.reference().frozen)
      << "reason="
      << (reference_response.reasons.empty()
              ? -1 : static_cast<int>(reference_response.reasons.front()));
  ASSERT_FALSE(reference_response.reference_snapshot_id.empty());

  // Model the real ROS hand-off: the frozen-status message and phase service
  // share a callback queue, so several phase-1 sensor frames can be accepted
  // before MONITOR_UPDATE is serviced.  They must not make the first update
  // permanently fail max_update_gap_s.
  for (uint64_t frame_index = 6U; frame_index < 10U; ++frame_index) {
    V1Snapshot ignored;
    std::vector<ReasonCode> handoff_reasons;
    ASSERT_TRUE(pipeline.AcceptFrame(
        PipelineFrame(config, frame_index,
                      1.0 + 0.1 * static_cast<double>(frame_index), true),
        &ignored, &handoff_reasons));
    EXPECT_TRUE(ignored.common.snapshot_id.empty());
  }

  PhaseRequest update_request = reference_request;
  update_request.target = MonitoringPhase::MONITOR_UPDATE;
  const PhaseResponse update_response = pipeline.SetPhase(update_request);
  ASSERT_TRUE(update_response.accepted);
  ASSERT_EQ(update_response.reference_snapshot_id,
            reference_response.reference_snapshot_id);

  V1Snapshot snapshot;
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(pipeline.AcceptFrame(
      PipelineFrame(config, 10U, 2.50, true), &snapshot, &reasons));
  ASSERT_FALSE(snapshot.common.snapshot_id.empty())
      << "reason="
      << (reasons.empty() ? -1 : static_cast<int>(reasons.back()));
  EXPECT_EQ(snapshot.common.snapshot_id,
            pipeline.last_snapshot().common.snapshot_id);
  EXPECT_EQ(snapshot.reference.id, update_response.reference_snapshot_id);
  EXPECT_EQ(snapshot.deformation.size(), snapshot.reference.gdus.size());
  EXPECT_TRUE(snapshot.occupancy.empty());
  EXPECT_TRUE(snapshot.hazards.empty());
  EXPECT_EQ(snapshot.common.reference_epoch, pipeline.reference_epoch());
  EXPECT_EQ(snapshot.common.accepted_frame_index, 10U);
  EXPECT_EQ(snapshot.common.schema_version, 3U);
  EXPECT_EQ(snapshot.common.implementation_freeze_version,
            "implementation-freeze-v1.1-research");
  EXPECT_EQ(snapshot.common.sensor_scope,
            "LIDAR_ONLY_NO_SEMANTICS_NO_OBJECT_ID");
  EXPECT_EQ(snapshot.common.hazard_coverage_mask, 0U);
  EXPECT_EQ(snapshot.common.validity, Validity::INCONCLUSIVE);
  EXPECT_TRUE(snapshot.common.probabilities_calibrated);
  EXPECT_EQ(std::find(snapshot.common.reasons.begin(),
                      snapshot.common.reasons.end(),
                      ReasonCode::POSE_CONTROL_INSUFFICIENT),
            snapshot.common.reasons.end());
  for (const GduPosterior& posterior : snapshot.deformation)
    EXPECT_NE(posterior.admission,
              DetectionAdmission::INCONCLUSIVE_SLAM);

  // Exercise the documented spatially target-excluded SE(3) correction with a
  // non-zero common FAST-LIO registration error.  Every evidence role is
  // perturbed, but only POSE_CONTROL patches outside the target GDU may drive
  // the correction; the frame must still commit without a SLAM admission
  // failure.
  FrameBundle perturbed = PipelineFrame(config, 11U, 2.60, true);
  const Eigen::Matrix3d registration_error_R =
      Eigen::AngleAxisd(0.002, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Vector3d registration_error_t(0.004, -0.003, 0.001);
  for (CanonicalPoint& point : perturbed.points)
    point.position_R = registration_error_R * point.position_R +
                       registration_error_t;
  V1Snapshot perturbed_snapshot;
  reasons.clear();
  ASSERT_TRUE(pipeline.AcceptFrame(
      perturbed, &perturbed_snapshot, &reasons));
  ASSERT_FALSE(perturbed_snapshot.common.snapshot_id.empty());
  EXPECT_EQ(std::find(perturbed_snapshot.common.reasons.begin(),
                      perturbed_snapshot.common.reasons.end(),
                      ReasonCode::POSE_CONTROL_INSUFFICIENT),
            perturbed_snapshot.common.reasons.end());
  EXPECT_EQ(std::find(perturbed_snapshot.common.reasons.begin(),
                      perturbed_snapshot.common.reasons.end(),
                      ReasonCode::INVALID_NUMERICAL),
            perturbed_snapshot.common.reasons.end());
  for (const GduPosterior& posterior : perturbed_snapshot.deformation)
    EXPECT_NE(posterior.admission,
              DetectionAdmission::INCONCLUSIVE_SLAM);

  FrameBundle outside = PipelineFrame(config, 12U, 2.70, true);
  auto outside_point = std::find_if(
      outside.points.begin(), outside.points.end(),
      [](const CanonicalPoint& point) {
        return point.role == PointRole::ASSOCIATION;
      });
  ASSERT_NE(outside_point, outside.points.end());
  outside_point->position_R = Eigen::Vector3d(2.9, 0.0, 0.0);
  V1Snapshot outside_snapshot;
  reasons.clear();
  ASSERT_TRUE(pipeline.AcceptFrame(outside, &outside_snapshot, &reasons));
  const auto protective = std::find_if(
      outside_snapshot.explanations.begin(),
      outside_snapshot.explanations.end(),
      [](const ObservationExplanation& explanation) {
        return explanation.protective_geometry_available &&
            std::abs(explanation.protective_center_R[0] - 2.9) < 0.1;
      });
  ASSERT_NE(protective, outside_snapshot.explanations.end());
  EXPECT_EQ(protective->semantics, AssociationSemantics::OTHER);
  EXPECT_FALSE(protective->cross_gdu_assignment_envelope);
  EXPECT_EQ(outside_snapshot.common.approximation_flags &
                APPROX_CROSS_GDU_ASSOCIATION_TO_ENVELOPE,
            0U);
  // An unassigned association patch remains detector diagnostics only.  The
  // pure detector must not promote it to a navigation hazard or live
  // occupancy event.
  EXPECT_TRUE(outside_snapshot.hazards.empty());
  EXPECT_TRUE(outside_snapshot.occupancy.empty());

  const size_t registry_size_before_messages = StableIdRegistrySize();
  const auto deformation_message = BuildDeformationMessage(outside_snapshot);
  const auto interface_message = BuildInterfaceMessage(outside_snapshot);
  const auto tile_messages = BuildDetectionTileMessages(
      outside_snapshot, pipeline.filter(), config);
  const auto status_message = BuildDetectionStatusMessage(
      outside_snapshot, pipeline.phase(), pipeline.deadline_failure_count(),
      pipeline.resource_failure_count());
  const auto expected_snapshot = BuildStableIdMessage(
      outside_snapshot.common.snapshot_id);
  EXPECT_EQ(deformation_message.common.snapshot_id.high,
            expected_snapshot.high);
  EXPECT_EQ(interface_message.common.snapshot_id.high,
            expected_snapshot.high);
  EXPECT_EQ(status_message.common.snapshot_id.high, expected_snapshot.high);
  EXPECT_EQ(deformation_message.common.snapshot_id.low, expected_snapshot.low);
  EXPECT_EQ(interface_message.common.snapshot_id.low, expected_snapshot.low);
  EXPECT_EQ(status_message.common.snapshot_id.low, expected_snapshot.low);
  ASSERT_EQ(tile_messages.size(), 3U);
  for (const auto& tile : tile_messages) {
    EXPECT_EQ(tile.common.snapshot_id.high, expected_snapshot.high);
    EXPECT_EQ(tile.common.snapshot_id.low, expected_snapshot.low);
    EXPECT_EQ(tile.source_snapshot_id.high, expected_snapshot.high);
    EXPECT_EQ(tile.source_snapshot_id.low, expected_snapshot.low);
  }
  EXPECT_EQ(StableIdRegistrySize(), registry_size_before_messages)
      << "serialization must not create post-commit identities";

  bool found_rectangular_information_factor = false;
  ASSERT_EQ(deformation_message.gdus.size(),
            outside_snapshot.deformation.size());
  for (size_t g = 0U; g < deformation_message.gdus.size(); ++g) {
    ASSERT_EQ(deformation_message.gdus[g].components.size(),
              outside_snapshot.deformation[g].components.size());
    for (size_t c = 0U;
         c < deformation_message.gdus[g].components.size(); ++c) {
      const auto& serialized =
          deformation_message.gdus[g].components[c].sqrt_information;
      const Eigen::MatrixXd& expected =
          outside_snapshot.deformation[g].components[c].sqrt_information;
      EXPECT_EQ(serialized.rows, static_cast<uint32_t>(expected.rows()));
      EXPECT_EQ(serialized.cols, static_cast<uint32_t>(expected.cols()));
      ASSERT_EQ(serialized.row_ptr.size(),
                static_cast<size_t>(serialized.rows) + 1U);
      ASSERT_EQ(serialized.col_idx.size(), serialized.data.size());
      ASSERT_EQ(serialized.row_ptr.back(), serialized.data.size());
      Eigen::MatrixXd reconstructed = Eigen::MatrixXd::Zero(
          static_cast<int>(serialized.rows),
          static_cast<int>(serialized.cols));
      for (uint32_t row = 0U; row < serialized.rows; ++row) {
        ASSERT_LE(serialized.row_ptr[row], serialized.row_ptr[row + 1U]);
        for (uint64_t item = serialized.row_ptr[row];
             item < serialized.row_ptr[row + 1U]; ++item) {
          ASSERT_LT(serialized.col_idx[item], serialized.cols);
          reconstructed(static_cast<int>(row),
                        static_cast<int>(serialized.col_idx[item])) =
              serialized.data[item];
        }
      }
      EXPECT_EQ(reconstructed, expected);
      found_rectangular_information_factor =
          found_rectangular_information_factor ||
          serialized.rows < serialized.cols;
    }
  }
  EXPECT_TRUE(found_rectangular_information_factor);
  // A later approximation must never overwrite a more severe pre-existing
  // validity state.
  FrameBundle stale = PipelineFrame(config, 13U, 2.80, true);
  stale.validity = Validity::STALE;
  stale.reasons.push_back(ReasonCode::OUTSIDE_CALIBRATED_RANGE);
  V1Snapshot stale_snapshot;
  reasons.clear();
  ASSERT_TRUE(pipeline.AcceptFrame(stale, &stale_snapshot, &reasons));
  ASSERT_FALSE(stale_snapshot.common.snapshot_id.empty());
  EXPECT_EQ(stale_snapshot.common.validity, Validity::STALE);
  EXPECT_NE(std::find(stale_snapshot.common.reasons.begin(),
                      stale_snapshot.common.reasons.end(),
                      ReasonCode::OUTSIDE_CALIBRATED_RANGE),
            stale_snapshot.common.reasons.end());
}

TEST(V1Pipeline, OccupancyLineageStartsFromZeroAndDeduplicatesPrimitives) {
  V1Config config = SyntheticKnownConfig("lineage-probability-test");
  V1Snapshot snapshot;
  snapshot.common.snapshot_id = TestId("lineage-snapshot", 1U);
  const StableId cause = TestId("lineage-cause", 1U);
  const StableId first_component = TestId("lineage-component", 1U);
  const StableId second_component = TestId("lineage-component", 2U);
  const auto append = [&](const StableId& component, double probability) {
    HazardGeometry hazard;
    hazard.cause_id = cause;
    hazard.component_id = component;
    hazard.existence = ProbabilityInterval{
        probability, probability, probability, true, true};
    hazard.other_mass = 0.1;
    snapshot.hazards.push_back(hazard);
  };
  append(first_component, 0.2);
  append(first_component, 0.2);  // second primitive, same mixture component
  append(second_component, 0.3);

  const auto message = BuildOccupancyMessage(snapshot, config);
  ASSERT_EQ(message.event_lineage.size(), 1U);
  EXPECT_NEAR(message.event_lineage.front().probability_lower, 0.5, 1.0e-15);
  EXPECT_NEAR(message.event_lineage.front().probability_mean, 0.6, 1.0e-15);
  EXPECT_NEAR(message.event_lineage.front().probability_upper, 0.6, 1.0e-15);
  EXPECT_FALSE(
      message.event_lineage.front().probability_mean_identified);
  EXPECT_NEAR(message.event_lineage.front().other_mass, 0.1, 1.0e-15);
}

TEST(V1Pipeline, SpatialAndInterfaceProductsInheritCommonSnapshotValidity) {
  V1Config config = SyntheticKnownConfig("product-validity-test");
  const ReferenceSnapshot reference = MetricLineReference(true);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(20U, 0U), &reasons));
  V1Snapshot snapshot;
  snapshot.common.snapshot_id = TestId("product-validity-snapshot", 1U);
  snapshot.common.validity = Validity::INCONCLUSIVE;
  snapshot.common.reasons.push_back(ReasonCode::INPUT_DESKEW_UNRESOLVED);
  snapshot.reference = reference;
  snapshot.deformation = filter.Posterior();
  snapshot.interfaces = filter.InterfacePosteriors();
  const auto tiles = BuildDetectionTileMessages(snapshot, filter, config);
  ASSERT_EQ(tiles.size(), 3U);
  ASSERT_FALSE(tiles[2].reference_support.empty());
  ASSERT_FALSE(tiles[2].physical_deformation.empty());
  EXPECT_EQ(tiles[2].reference_support.front().validity,
            static_cast<uint8_t>(Validity::INCONCLUSIVE));
  EXPECT_EQ(tiles[2].physical_deformation.front().validity,
            static_cast<uint8_t>(Validity::INCONCLUSIVE));
  ASSERT_FALSE(tiles[2].detection_evidence.empty());
  EXPECT_EQ(tiles[2].detection_evidence.front().validity,
            static_cast<uint8_t>(Validity::INCONCLUSIVE));
  const auto interface_message = BuildInterfaceMessage(snapshot);
  ASSERT_FALSE(interface_message.interfaces.empty());
  EXPECT_EQ(interface_message.interfaces.front().validity,
            static_cast<uint8_t>(Validity::INCONCLUSIVE));
  EXPECT_EQ(interface_message.interfaces.front().left_query_validity,
            static_cast<uint8_t>(Validity::INCONCLUSIVE));
  EXPECT_EQ(interface_message.interfaces.front().right_query_validity,
            static_cast<uint8_t>(Validity::INCONCLUSIVE));
}

TEST(V1Pipeline, OversizedInputFrameIsRejectedBeforeIdentityCommit) {
  V1Config config = SyntheticKnownConfig("input-point-limit-test");
  config.resource.max_points_per_frame = 1U;
  config.resource.max_raw_buffer_points = 2U;
  V1Pipeline pipeline(config);
  FrameBundle oversized = PipelineFrame(config, 0U, 1.0, false);
  ASSERT_GT(oversized.points.size(), 1U);
  std::vector<ReasonCode> reasons;
  V1Snapshot snapshot;
  EXPECT_FALSE(pipeline.AcceptFrame(oversized, &snapshot, &reasons));
  EXPECT_NE(std::find(reasons.begin(), reasons.end(),
                      ReasonCode::RESOURCE_LIMIT),
            reasons.end());
  EXPECT_EQ(pipeline.resource_failure_count(), 1U);
}

TEST(V1Occupancy, PersistentEventUsesExposureAndSingleTotalNewCarrier) {
  V1Config config = SyntheticKnownConfig("occupancy-test");
  config.resource.reference_min_frames = 6;
  config.resource.max_events = 1024;
  OccupancyEventGraph graph(config);
  std::deque<FrameBundle> reference_frames;
  for (uint64_t frame = 0; frame < 100U; ++frame)
    reference_frames.push_back(ReferenceRayFrame(frame));
  ReferenceSnapshot structural_reference;
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(graph.BuildReference(reference_frames, structural_reference,
                                   7U, &reasons));
  const VoxelKey target{4, 0, 0};

  const FrameBundle hit = CurrentRayFrame(1U, 0.45, 10);
  ASSERT_TRUE(graph.Update(hit, {}, &reasons));
  const OccupancyCell* after_hit = graph.Find(target);
  ASSERT_NE(after_hit, nullptr);
  ASSERT_TRUE(after_hit->new_occupancy_applicable);
  ASSERT_EQ(after_hit->cause_ids.size(), 1U);
  EXPECT_NEAR(after_hit->total_occupancy.mean,
              after_hit->new_occupancy.mean, 1.0e-15);
  const double hit_probability = after_hit->total_occupancy.mean;
  const StableId hit_cause = after_hit->cause_ids.front();
  EXPECT_GT(hit_probability, config.calibration.new_occupancy_prior);

  const FrameBundle miss = CurrentRayFrame(2U, 1.05, 10);
  ASSERT_TRUE(graph.Update(miss, {}, &reasons));
  const OccupancyCell* after_miss = graph.Find(target);
  ASSERT_NE(after_miss, nullptr);
  ASSERT_EQ(after_miss->cause_ids.size(), 1U);
  EXPECT_LT(after_miss->total_occupancy.mean, hit_probability);
  EXPECT_EQ(after_miss->cause_ids.front(), hit_cause);
  EXPECT_EQ(after_miss->last_observed, miss.stamp);
  const ProbabilityInterval future = graph.PredictCauseProbability(
      target, hit_cause, miss.stamp + ros::Duration(2.0));
  const double birth = config.calibration.novel_occupancy_birth_rate_hz;
  const double death = config.calibration.novel_occupancy_death_rate_hz;
  const double stationary = birth / (birth + death);
  const double retention = std::exp(-(birth + death) * 2.0);
  EXPECT_NEAR(future.mean,
              stationary +
                  (after_miss->total_occupancy.mean - stationary) * retention,
              1.0e-14);
  EXPECT_NEAR(future.lower,
              stationary +
                  (after_miss->total_occupancy.lower - stationary) * retention,
              1.0e-14);
  EXPECT_NEAR(future.upper,
              stationary +
                  (after_miss->total_occupancy.upper - stationary) * retention,
              1.0e-14);
}

TEST(V1Occupancy, DdaCornerCrossingDoesNotFabricateFreeCells) {
  V1Config config = SyntheticKnownConfig("dda-corner-test");
  config.resource.grid_origin_m.setZero();
  config.resource.occupancy_voxel_m = 1.0;
  OccupancyEventGraph graph(config);
  const std::vector<VoxelKey> cells = graph.TraceRay(
      Eigen::Vector3d(0.5, 0.5, 0.5),
      Eigen::Vector3d(2.5, 2.5, 2.5), true);
  ASSERT_EQ(cells.size(), 3U);
  EXPECT_EQ(cells[0], (VoxelKey{0, 0, 0}));
  EXPECT_EQ(cells[1], (VoxelKey{1, 1, 1}));
  EXPECT_EQ(cells[2], (VoxelKey{2, 2, 2}));
}

TEST(V1Occupancy, DdaRejectsOutOfRangeAndOverBudgetRays) {
  V1Config config = SyntheticKnownConfig("dda-resource-test");
  config.resource.grid_origin_m.setZero();
  config.resource.occupancy_voxel_m = 1.0;
  config.resource.max_tiles = 4U;
  OccupancyEventGraph graph(config);
  bool exceeded = false;
  EXPECT_TRUE(graph.TraceRay(Eigen::Vector3d(0.5, 0.5, 0.5),
                             Eigen::Vector3d(10.5, 0.5, 0.5), true,
                             &exceeded).empty());
  EXPECT_TRUE(exceeded);
  exceeded = false;
  EXPECT_TRUE(graph.TraceRay(Eigen::Vector3d(0.5, 0.5, 0.5),
                             Eigen::Vector3d(1.0e300, 0.5, 0.5), true,
                             &exceeded).empty());
  EXPECT_TRUE(exceeded);
}

TEST(V1Occupancy, EvidenceToVoxelLinksHaveAnIndependentHardBudget) {
  V1Config config = SyntheticKnownConfig("occupancy-evidence-link-test");
  config.resource.max_evidence_links = 2U;
  OccupancyEventGraph graph(config);
  std::deque<FrameBundle> reference_frames;
  for (uint64_t frame = 0; frame < 6U; ++frame)
    reference_frames.push_back(ReferenceRayFrame(frame));
  ReferenceSnapshot structural_reference;
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(graph.BuildReference(reference_frames, structural_reference,
                                   9U, &reasons));
  reasons.clear();
  EXPECT_FALSE(graph.Update(CurrentRayFrame(1U, 0.45, 1), {}, &reasons));
  EXPECT_NE(std::find(reasons.begin(), reasons.end(),
                      ReasonCode::RESOURCE_LIMIT),
            reasons.end());
}

TEST(V1Occupancy, ReferenceUsesOnlyDisjointOccupancySupport) {
  V1Config config = SyntheticKnownConfig("occupancy-reference-role-test");
  config.resource.max_tiles = 64U;
  OccupancyEventGraph graph(config);
  std::deque<FrameBundle> reference_frames;
  for (uint64_t frame = 0; frame < 6U; ++frame) {
    FrameBundle sample = ReferenceRayFrame(frame);
    sample.points.push_back(
        TestPoint(100.05, PointRole::MEASUREMENT, 1000U + frame));
    reference_frames.push_back(sample);
  }
  ReferenceSnapshot structural_reference;
  std::vector<ReasonCode> reasons;
  EXPECT_TRUE(graph.BuildReference(reference_frames, structural_reference,
                                   9U, &reasons));
  EXPECT_EQ(std::find(reasons.begin(), reasons.end(),
                      ReasonCode::REFERENCE_RESOURCE_LIMIT),
            reasons.end());
}

TEST(V1Occupancy, UnresolvedInputProtectsEndpointWithoutFreeRayEvidence) {
  V1Config config = SyntheticKnownConfig("occupancy-input-quality-test");
  config.resource.reference_min_frames = 6;
  config.resource.max_events = 1024;
  OccupancyEventGraph graph(config);
  std::deque<FrameBundle> reference_frames;
  for (uint64_t frame = 0; frame < 100U; ++frame)
    reference_frames.push_back(ReferenceRayFrame(frame));
  ReferenceSnapshot structural_reference;
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(graph.BuildReference(reference_frames, structural_reference,
                                   9U, &reasons));

  FrameBundle unresolved = CurrentRayFrame(1U, 0.45, 10);
  unresolved.validity = Validity::INCONCLUSIVE;
  unresolved.reasons.push_back(ReasonCode::INPUT_DESKEW_UNRESOLVED);
  ASSERT_TRUE(graph.Update(unresolved, {}, &reasons));
  EXPECT_EQ(std::find(reasons.begin(), reasons.end(),
                      ReasonCode::OUTSIDE_CALIBRATED_RANGE), reasons.end());
  const OccupancyCell* endpoint = graph.Find(VoxelKey{4, 0, 0});
  ASSERT_NE(endpoint, nullptr);
  ASSERT_FALSE(endpoint->cause_ids.empty());
  EXPECT_DOUBLE_EQ(endpoint->total_occupancy.upper, 1.0);
  EXPECT_FALSE(endpoint->total_occupancy.calibrated);
  EXPECT_EQ(endpoint->cause_semantics.front(),
            HazardSemantics::OTHER_ENVELOPE);
  const OccupancyCell* traversed = graph.Find(VoxelKey{2, 0, 0});
  ASSERT_NE(traversed, nullptr);
  EXPECT_TRUE(traversed->cause_ids.empty());
  EXPECT_TRUE(traversed->last_observed.isZero());
}

TEST(V1Occupancy,
     SurfelCarrierOwnershipPreventsReferencePersistenceDoubleCause) {
  V1Config config = SyntheticKnownConfig("occupancy-test");
  config.resource.reference_min_frames = 6;
  OccupancyEventGraph graph(config);
  std::deque<FrameBundle> reference_frames;
  for (uint64_t frame = 0; frame < 100U; ++frame)
    reference_frames.push_back(ReferenceRayFrame(frame));

  ReferenceSnapshot structural_reference;
  Surfel surfel;
  surfel.id = TestId("support-owner-surfel", 1U);
  // The center belongs to voxel x=9, while the frozen carrier also intersects
  // the reference endpoint voxel x=10.  Center-only ownership would create an
  // independent REFERENCE_PERSISTENCE cause in x=10.
  surfel.center_R = Eigen::Vector3d(0.94, 0.05, 0.05);
  surfel.support_radius_m = 0.12;
  surfel.gdu_index = 0U;
  structural_reference.surfels.push_back(surfel);
  GduDefinition gdu;
  gdu.id = TestId("support-owner-gdu", 1U);
  gdu.surfel_indices.push_back(0U);
  structural_reference.gdus.push_back(gdu);

  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(graph.BuildReference(reference_frames, structural_reference,
                                   10U, &reasons));
  CanonicalBytes cause_fields;
  cause_fields.AddString(config.resource.session_id);
  cause_fields.AddId(gdu.id);
  HazardGeometry claim;
  claim.cause_id = MakeStableId("hazard-cause-v1", cause_fields);
  claim.component_id = TestId("support-owner-component", 1U);
  claim.semantics = HazardSemantics::STRUCTURAL_DEFORMATION;
  claim.structural_state = StructuralState::U;
  claim.center_R = surfel.center_R;
  claim.radius_m = surfel.support_radius_m;
  claim.existence = ProbabilityInterval{0.8, 0.8, 0.8, true, true};

  ASSERT_TRUE(graph.Update(CurrentRayFrame(4U, 1.05, 10), {claim},
                           &reasons));
  const VoxelKey endpoint{10, 0, 0};
  const OccupancyCell* before_projection = graph.Find(endpoint);
  ASSERT_NE(before_projection, nullptr);
  EXPECT_TRUE(before_projection->cause_ids.empty());
  ASSERT_TRUE(graph.ProjectStructuralHazards({claim}, &reasons));
  const OccupancyCell* after_projection = graph.Find(endpoint);
  ASSERT_NE(after_projection, nullptr);
  ASSERT_EQ(after_projection->cause_ids.size(), 1U);
  EXPECT_EQ(after_projection->cause_ids.front(), claim.cause_id);
  EXPECT_EQ(after_projection->cause_semantics.front(),
            HazardSemantics::STRUCTURAL_DEFORMATION);
  EXPECT_NEAR(after_projection->total_occupancy.mean, 0.8, 1.0e-15);
}

TEST(V1Occupancy, StructuralClaimPreventsIndependentNovelDoubleCause) {
  V1Config config = SyntheticKnownConfig("occupancy-test");
  config.resource.max_events = 1024;
  OccupancyEventGraph graph(config);
  std::deque<FrameBundle> reference_frames;
  for (uint64_t frame = 0; frame < 100U; ++frame)
    reference_frames.push_back(ReferenceRayFrame(frame));
  ReferenceSnapshot structural_reference;
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(graph.BuildReference(reference_frames, structural_reference,
                                   8U, &reasons));
  HazardGeometry claim;
  claim.cause_id = TestId("structural-cause", 1U);
  claim.component_id = TestId("structural-component", 1U);
  claim.semantics = HazardSemantics::STRUCTURAL_DEFORMATION;
  claim.structural_state = StructuralState::A;
  claim.center_R = Eigen::Vector3d(0.45, 0.05, 0.05);
  claim.radius_m = 0.08;
  claim.existence = ProbabilityInterval{0.8, 0.8, 0.8, true, true};
  ASSERT_TRUE(graph.Update(CurrentRayFrame(3U, 0.45, 10), {claim},
                           &reasons));
  const VoxelKey target{4, 0, 0};
  const OccupancyCell* before_projection = graph.Find(target);
  ASSERT_NE(before_projection, nullptr);
  EXPECT_TRUE(before_projection->cause_ids.empty());
  EXPECT_DOUBLE_EQ(before_projection->new_occupancy.mean, 0.0);
  ASSERT_TRUE(graph.ProjectStructuralHazards({claim}, &reasons));
  const OccupancyCell* after_projection = graph.Find(target);
  ASSERT_NE(after_projection, nullptr);
  ASSERT_EQ(after_projection->cause_ids.size(), 1U);
  EXPECT_EQ(after_projection->cause_ids.front(), claim.cause_id);
  EXPECT_NEAR(after_projection->total_occupancy.mean, 0.8, 1.0e-15);
  EXPECT_NEAR(after_projection->new_occupancy.mean, 0.8, 1.0e-15);
  EXPECT_TRUE(after_projection->total_occupancy.mean_identified);
  EXPECT_TRUE(after_projection->new_occupancy.mean_identified);
}

TEST(V1Risk, CanonicalMotionSamplingBoundsRelativeSweepAndResources) {
  V1Config config = SyntheticKnownConfig("motion-test");
  config.policy.motion_sample_interval_s = 1.0;
  config.calibration.hazard_sweep_speed_upper_mps = 0.5;
  config.resource.max_motion_query_samples = 64U;
  RiskQueryEngine engine(config);
  std::vector<TimedRobotState, Eigen::aligned_allocator<TimedRobotState>> motion(2);
  motion[0].stamp = ros::Time(1U, 0U);
  motion[1].stamp = ros::Time(2U, 0U);
  motion[0].T_RB = Eigen::Isometry3d::Identity();
  motion[1].T_RB = Eigen::Isometry3d::Identity();
  motion[1].T_RB.translation().x() = 1.0;
  const auto samples = engine.CanonicalMotionSamples(
      config.robot_geometry, motion, 0.1);
  // Relative speed is 1.0 + 0.5 m/s, hence dt <= 1/15 s.
  ASSERT_EQ(samples.size(), 17U);
  for (size_t index = 1; index < samples.size(); ++index)
    EXPECT_LE((samples[index].stamp - samples[index - 1U].stamp).toSec(),
              1.0 / 15.0 + 1.0e-12);

  config.resource.max_motion_query_samples = 8U;
  RiskQueryEngine limited(config);
  EXPECT_TRUE(limited.CanonicalMotionSamples(
      config.robot_geometry, motion, 0.1).empty());
}

TEST(V1Risk, MotionInterpolationUsesUnknownCorrelationCovarianceEnvelope) {
  V1Config config = SyntheticKnownConfig("motion-covariance-envelope-test");
  config.policy.motion_sample_interval_s = 1.0;
  config.calibration.hazard_sweep_speed_upper_mps = 0.0;
  config.resource.max_motion_query_samples = 16U;
  RiskQueryEngine engine(config);
  std::vector<TimedRobotState, Eigen::aligned_allocator<TimedRobotState>> motion(2);
  motion[0].stamp = ros::Time(1U, 0U);
  motion[1].stamp = ros::Time(2U, 0U);
  motion[0].T_RB = Eigen::Isometry3d::Identity();
  motion[1].T_RB = Eigen::Isometry3d::Identity();
  motion[1].T_RB.translation().x() = 0.1;
  motion[0].covariance.setZero();
  motion[1].covariance.setZero();
  motion[0].covariance(0, 0) = 1.0;
  motion[1].covariance(1, 1) = 9.0;
  const auto samples = engine.CanonicalMotionSamples(
      config.robot_geometry, motion, 1.0);
  ASSERT_EQ(samples.size(), 3U);
  EXPECT_EQ(samples.front().covariance, motion.front().covariance);
  EXPECT_EQ(samples.back().covariance, motion.back().covariance);
  const double expected_variance = 4.0;
  EXPECT_NEAR((samples[1].covariance - expected_variance *
               Eigen::Matrix<double, 6, 6>::Identity()).norm(),
              0.0, 1.0e-12);
}

TEST(V1Risk, BodyVolumeQueryFailsClosedWhenSamplingResourceIsExceeded) {
  V1Config config = SyntheticKnownConfig("body-query-resource-test");
  config.resource.max_body_query_samples = 64U;
  RiskQueryEngine engine(config);
  CommonHeader header;
  header.snapshot_id = TestId("body-resource-snapshot", 1U);
  header.measurement_stamp = ros::Time(10U, 0U);
  header.publish_stamp = header.measurement_stamp;
  header.valid_until = ros::Time(11U, 0U);
  header.prediction_valid_until = ros::Time(12U, 0U);
  header.validity = Validity::VALID;
  header.probabilities_calibrated = true;
  TimedRobotState state;
  state.stamp = header.measurement_stamp;
  const ReferenceSnapshot reference = MetricLineReference(false);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> initialize_reasons;
  ASSERT_TRUE(filter.Initialize(reference, header.measurement_stamp,
                                &initialize_reasons));
  OccupancyEventGraph occupancy(config);
  const RiskAssessment result = engine.AssessState(
      header, reference, filter, occupancy,
      config.robot_geometry, state);
  EXPECT_EQ(result.validity, Validity::UNKNOWN);
  EXPECT_EQ(result.hard, HardState::UNKNOWN);
  EXPECT_NE(std::find(result.reasons.begin(), result.reasons.end(),
                      ReasonCode::RESOURCE_LIMIT),
            result.reasons.end());
  EXPECT_DOUBLE_EQ(result.soft.total, config.policy.unknown_weight);
}

TEST(V1Risk, DeterministicQueryWorkBudgetFailsClosedBeforeSampling) {
  V1Config config = SyntheticKnownConfig("risk-operation-budget-test");
  config.resource.max_risk_query_operations = 1U;
  RiskQueryEngine engine(config);
  CommonHeader header;
  header.snapshot_id = TestId("query-budget-snapshot", 1U);
  header.measurement_stamp = ros::Time(10U, 0U);
  header.publish_stamp = header.measurement_stamp;
  header.valid_until = ros::Time(11U, 0U);
  header.prediction_valid_until = ros::Time(12U, 0U);
  header.validity = Validity::VALID;
  header.probabilities_calibrated = true;
  TimedRobotState state;
  state.stamp = header.measurement_stamp;
  const ReferenceSnapshot reference = MetricLineReference(false);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> initialize_reasons;
  ASSERT_TRUE(filter.Initialize(reference, header.measurement_stamp,
                                &initialize_reasons));
  OccupancyEventGraph occupancy(config);
  const RiskAssessment result = engine.AssessState(
      header, reference, filter, occupancy,
      config.robot_geometry, state);
  EXPECT_EQ(result.validity, Validity::UNKNOWN);
  EXPECT_EQ(result.hard, HardState::UNKNOWN);
  EXPECT_NE(std::find(result.reasons.begin(), result.reasons.end(),
                      ReasonCode::RESOURCE_LIMIT),
            result.reasons.end());
  EXPECT_DOUBLE_EQ(result.soft.total, config.policy.unknown_weight);
}

TEST(V1Risk, HazardGeometryBudgetRejectsWholeCandidateWithoutTruncation) {
  V1Config config = SyntheticKnownConfig("hazard-geometry-budget-test");
  const ReferenceSnapshot reference = MetricLineReference(false);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(10U, 0U), &reasons));
  RiskQueryEngine engine(config);
  std::vector<HazardGeometry, Eigen::aligned_allocator<HazardGeometry>> hazards;
  ASSERT_TRUE(engine.BuildHazards(
      CommonHeader(), reference, filter,
      config.resource.max_hazard_primitives, &hazards,
      &reasons));
  ASSERT_FALSE(hazards.empty());
  for (const HazardGeometry& hazard : hazards)
    EXPECT_DOUBLE_EQ(hazard.radius_m, 0.05);
  hazards.clear();
  reasons.clear();
  EXPECT_FALSE(engine.BuildHazards(
      CommonHeader(), reference, filter, 1U, &hazards,
      &reasons));
  EXPECT_TRUE(hazards.empty());
  EXPECT_NE(std::find(reasons.begin(), reasons.end(),
                      ReasonCode::RESOURCE_LIMIT),
            reasons.end());
}

TEST(V1Risk, MotionAggregatePreservesWorstConfigurationValidity) {
  V1Config config = SyntheticKnownConfig("motion-validity-test");
  config.policy.unknown_is_blocked = false;
  RiskQueryEngine engine(config);
  CommonHeader header;
  header.snapshot_id = TestId("motion-validity-snapshot", 1U);
  header.measurement_stamp = ros::Time(10U, 0U);
  header.publish_stamp = header.measurement_stamp;
  header.valid_until = ros::Time(11U, 0U);
  header.prediction_valid_until = ros::Time(12U, 0U);
  header.validity = Validity::VALID;
  header.probabilities_calibrated = true;
  ReferenceSnapshot reference;
  OccupancyEventGraph occupancy(config);
  UacGaussianSumFilter filter(config);
  TimedRobotState outside_prediction;
  outside_prediction.stamp = ros::Time(13U, 0U);
  std::vector<TimedRobotState, Eigen::aligned_allocator<TimedRobotState>> motion{
      outside_prediction};
  RiskAssessment result = engine.AssessMotion(
      header, reference, filter, occupancy,
      config.robot_geometry, motion, 0.05);
  EXPECT_EQ(result.validity, Validity::UNKNOWN);
  EXPECT_EQ(result.hard, HardState::UNKNOWN);
  EXPECT_NE(std::find(result.reasons.begin(), result.reasons.end(),
                      ReasonCode::UNKNOWN_PREDICTION),
            result.reasons.end());

  RobotGeometry wrong_revision = config.robot_geometry;
  wrong_revision.revision = "wrong-revision";
  motion.front().stamp = header.measurement_stamp;
  result = engine.AssessMotion(
      header, reference, filter, occupancy,
      wrong_revision, motion, 0.05);
  EXPECT_EQ(result.validity, Validity::INVALID);
  EXPECT_EQ(result.hard, HardState::UNKNOWN);
  EXPECT_NE(std::find(result.reasons.begin(), result.reasons.end(),
                      ReasonCode::QUERY_CONTRACT_INVALID),
            result.reasons.end());
}

TEST(V1Risk, CauseDedupAndCanonicalPathAreSubdivisionInvariant) {
  V1Config config = SyntheticKnownConfig("risk-dedup-test");
  config.policy.unknown_is_blocked = false;
  config.policy.motion_sample_interval_s = 0.1;
  config.policy.motion_sample_spacing_m = 0.05;
  config.resource.max_motion_query_samples = 1024U;
  const ReferenceSnapshot reference = MetricLineReference(false);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(201U, 0U), &reasons));

  OccupancyEventGraph occupancy(config);
  std::deque<FrameBundle> reference_frames;
  for (uint64_t frame = 0; frame < 100U; ++frame)
    reference_frames.push_back(ReferenceRayFrame(frame));
  ASSERT_TRUE(occupancy.BuildReference(reference_frames, reference, 3U,
                                       &reasons));
  ASSERT_TRUE(occupancy.Update(CurrentRayFrame(1U, 1.05, 10), {}, &reasons));

  CommonHeader header;
  header.session_id = config.resource.session_id;
  header.snapshot_id = TestId("risk-snapshot", 1U);
  header.measurement_stamp = ros::Time(201U, 0U);
  header.publish_stamp = ros::Time(201U, 0U);
  header.valid_from = header.measurement_stamp;
  header.valid_until = header.measurement_stamp + ros::Duration(1.0);
  header.prediction_valid_until =
      header.measurement_stamp + ros::Duration(2.0);
  header.validity = Validity::VALID;
  header.probabilities_calibrated = true;

  RiskQueryEngine engine(config);
  TimedRobotState at_hazard;
  at_hazard.stamp = header.measurement_stamp;
  at_hazard.T_RB = Eigen::Isometry3d::Identity();
  at_hazard.T_RB.translation().x() = 1.05;
  const RiskAssessment state = engine.AssessState(
      header, reference, filter, occupancy, config.robot_geometry,
      at_hazard);
  const RiskAssessment repeated_state = engine.AssessState(
      header, reference, filter, occupancy, config.robot_geometry,
      at_hazard);
  EXPECT_EQ(state.assessment_scope, "HAZARD_COLLISION_OVERLAY_ONLY");
  EXPECT_FALSE(state.geometric_support_evaluated);
  EXPECT_EQ(state.geometric_support_validity, Validity::UNKNOWN);
  EXPECT_EQ(state.validity, Validity::VALID);
  EXPECT_EQ(state.hard, HardState::BLOCK);
  ASSERT_EQ(state.cause_ids.size(), state.cause_probabilities.size());
  EXPECT_EQ(repeated_state.cause_ids, state.cause_ids);
  ASSERT_EQ(repeated_state.cause_probabilities.size(),
            state.cause_probabilities.size());
  for (size_t cause = 0U; cause < state.cause_probabilities.size(); ++cause) {
    EXPECT_DOUBLE_EQ(repeated_state.cause_probabilities[cause].lower,
                     state.cause_probabilities[cause].lower);
    EXPECT_DOUBLE_EQ(repeated_state.cause_probabilities[cause].mean,
                     state.cause_probabilities[cause].mean);
    EXPECT_DOUBLE_EQ(repeated_state.cause_probabilities[cause].upper,
                     state.cause_probabilities[cause].upper);
  }
  std::set<StableId> unique_causes(state.cause_ids.begin(), state.cause_ids.end());
  EXPECT_EQ(unique_causes.size(), state.cause_ids.size());

  TimedRobotState begin = at_hazard;
  begin.T_RB.translation().x() = 0.7;
  TimedRobotState end = begin;
  end.stamp = header.measurement_stamp + ros::Duration(1.0);
  end.T_RB.translation().x() = 1.4;
  std::vector<TimedRobotState, Eigen::aligned_allocator<TimedRobotState>> coarse{
      begin, end};
  TimedRobotState middle = begin;
  middle.stamp = header.measurement_stamp + ros::Duration(0.5);
  middle.T_RB.translation().x() = 1.05;
  std::vector<TimedRobotState, Eigen::aligned_allocator<TimedRobotState>> refined{
      begin, middle, end};
  std::vector<TimedRobotState, Eigen::aligned_allocator<TimedRobotState>>
      canonical_states;
  std::vector<RiskAssessment> state_assessments;
  std::vector<double> contributions;
  const RiskAssessment coarse_result = engine.AssessMotion(
      header, reference, filter, occupancy, config.robot_geometry, coarse,
      config.policy.motion_sample_spacing_m, &canonical_states,
      &state_assessments, &contributions);
  const RiskAssessment refined_result = engine.AssessMotion(
      header, reference, filter, occupancy, config.robot_geometry, refined,
      config.policy.motion_sample_spacing_m);
  EXPECT_EQ(coarse_result.hard, HardState::BLOCK);
  EXPECT_EQ(refined_result.hard, HardState::BLOCK);
  EXPECT_NEAR(coarse_result.path_risk.lower,
              refined_result.path_risk.lower, 1.0e-15);
  EXPECT_NEAR(coarse_result.path_risk.mean,
              refined_result.path_risk.mean, 1.0e-15);
  EXPECT_NEAR(coarse_result.path_risk.upper,
              refined_result.path_risk.upper, 1.0e-15);
  EXPECT_EQ(coarse_result.path_risk.mean_identified,
            refined_result.path_risk.mean_identified);
  EXPECT_EQ(coarse_result.cause_ids, refined_result.cause_ids);
  ASSERT_EQ(canonical_states.size(), state_assessments.size());
  ASSERT_EQ(canonical_states.size(), contributions.size());
  EXPECT_NEAR(std::accumulate(contributions.begin(), contributions.end(), 0.0),
              coarse_result.path_risk.upper, 1.0e-15);
  EXPECT_TRUE(coarse_result.first_violation_configuration_defined);
  EXPECT_FALSE(coarse_result.first_violation_time.isZero());
  EXPECT_DOUBLE_EQ(state_assessments.back().soft.stale_cost, 0.0);

  const ros::Time evaluation_stamp =
      header.measurement_stamp + ros::Duration(0.5);
  const RiskAssessment stale_now = engine.AssessState(
      header, reference, filter, occupancy, config.robot_geometry, at_hazard,
      evaluation_stamp);
  TimedRobotState future_eta = at_hazard;
  future_eta.stamp = header.measurement_stamp + ros::Duration(1.0);
  const RiskAssessment stale_future = engine.AssessState(
      header, reference, filter, occupancy, config.robot_geometry, future_eta,
      evaluation_stamp);
  const double expected_stale = config.policy.stale_weight *
      0.5 / config.resource.map_ttl_s;
  EXPECT_NEAR(stale_now.soft.stale_cost, expected_stale, 1.0e-15);
  EXPECT_NEAR(stale_future.soft.stale_cost, expected_stale, 1.0e-15);

  const RiskAssessment expired = engine.AssessState(
      header, reference, filter, occupancy, config.robot_geometry, at_hazard,
      header.valid_until + ros::Duration(0.001));
  EXPECT_EQ(expired.validity, Validity::STALE);
  EXPECT_EQ(expired.hard, HardState::UNKNOWN);
  EXPECT_NE(std::find(expired.reasons.begin(), expired.reasons.end(),
                      ReasonCode::SNAPSHOT_STALE),
            expired.reasons.end());
}

TEST(V1Pipeline, AlgorithmTriggeredReferenceExcludesSelectionSupportFrames) {
  V1Config config = SyntheticKnownConfig("selection-reference-test");
  config.resource.reference_min_frames = 6;
  config.resource.reference_min_duration_s = 0.50;
  // The selection-support cutoff excludes frames at/after 1.6 s, while the
  // reference transaction starts at 1.8 s.  Keep that intentional 0.3 s
  // separation inside the registered-propagation limit; this test exercises
  // evidence separation, not the stale-history fallback tested below.
  config.resource.max_update_gap_s = 0.31;
  config.resource.max_tiles = 4096U;
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  config.calibration.geometry_softmax_bias << 10.0, 0.0, 0.0, -10.0;
  V1Pipeline pipeline(config);
  for (uint64_t index = 0U; index < 8U; ++index) {
    FrameBundle frame = PipelineFrame(
        config, index, 1.0 + 0.1 * static_cast<double>(index), true);
    V1Snapshot snapshot;
    std::vector<ReasonCode> reasons;
    ASSERT_TRUE(pipeline.AcceptFrame(frame, &snapshot, &reasons));
  }
  PhaseRequest request;
  request.session_id = config.resource.session_id;
  request.target = MonitoringPhase::MONITOR_REFERENCE;
  request.roi.minimum_R = Eigen::Vector3d(-1.0, -1.0, -1.0);
  request.roi.maximum_R = Eigen::Vector3d(3.0, 1.0, 1.0);
  request.effective_stamp.fromSec(1.8);
  request.selection_support_cutoff.fromSec(1.6);
  request.trigger_source = TriggerSource::ALGORITHM_RESIDUAL;
  const PhaseResponse response = pipeline.SetPhase(request);
  ASSERT_TRUE(response.accepted);
  ASSERT_TRUE(pipeline.reference().frozen);
  EXPECT_LT(pipeline.reference().end_stamp,
            request.selection_support_cutoff);
}

TEST(V1Pipeline,
     ExternalTaskReferenceUsesBoundedLongerWindowForSparseCarrierCoverage) {
  ros::Time::init();
  V1Config config = SyntheticKnownConfig("external-reference-coverage-test");
  config.resource.reference_min_frames = 6;
  config.resource.reference_min_surfel_exposures = 6;
  config.resource.reference_min_duration_s = 0.50;
  config.resource.max_update_gap_s = 5.0;
  config.resource.max_tiles = 8192U;
  config.resource.deadline_ms = 100000.0;
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  config.calibration.geometry_softmax_bias << 10.0, 0.0, 0.0, -10.0;
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);

  V1Pipeline pipeline(config);
  for (uint64_t index = 0U; index < 12U; ++index) {
    FrameBundle frame = PipelineFrame(
        config, index, 1.0 + 0.1 * static_cast<double>(index), true);
    if (index % 2U != 0U) {
      frame.points.erase(std::remove_if(
          frame.points.begin(), frame.points.end(),
          [](const CanonicalPoint& point) {
            return point.position_R.x() > 1.5;
          }), frame.points.end());
    }
    V1Snapshot ignored;
    std::vector<ReasonCode> reasons;
    ASSERT_TRUE(pipeline.AcceptFrame(frame, &ignored, &reasons));
  }

  PhaseRequest request;
  request.session_id = config.resource.session_id;
  request.target = MonitoringPhase::MONITOR_REFERENCE;
  request.effective_stamp.fromSec(2.2);
  request.selection_support_cutoff = request.effective_stamp;
  request.roi.minimum_R = Eigen::Vector3d(-1.0, -1.0, -1.0);
  request.roi.maximum_R = Eigen::Vector3d(3.0, 1.0, 1.0);
  request.trigger_source = TriggerSource::HUMAN_OR_EXTERNAL_TASK;
  const PhaseResponse response = pipeline.SetPhase(request);
  ASSERT_TRUE(response.accepted);
  ASSERT_TRUE(pipeline.reference().frozen);
  const size_t monitorable = static_cast<size_t>(std::count_if(
      pipeline.reference().gdus.begin(), pipeline.reference().gdus.end(),
      [](const GduDefinition& gdu) { return gdu.monitorable; }));
  EXPECT_GE(monitorable, 2U);
  EXPECT_EQ(pipeline.reference().begin_stamp, ros::Time(1.0));
  EXPECT_EQ(pipeline.reference().end_stamp, ros::Time(2.1));
}

TEST(V1Pipeline,
     ExternalCurrentDwellReferenceExcludesOtherwiseValidHistoricalContext) {
  ros::Time::init();
  V1Config config = SyntheticKnownConfig("external-current-dwell-test");
  config.resource.reference_min_frames = 6;
  config.resource.reference_min_surfel_exposures = 6;
  config.resource.reference_min_duration_s = 0.50;
  config.resource.max_update_gap_s = 5.0;
  config.resource.max_tiles = 8192U;
  config.resource.deadline_ms = 100000.0;
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  config.calibration.geometry_softmax_bias << 10.0, 0.0, 0.0, -10.0;
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);

  V1Pipeline pipeline(config);
  for (uint64_t index = 0U; index < 6U; ++index) {
    V1Snapshot ignored;
    std::vector<ReasonCode> reasons;
    ASSERT_TRUE(pipeline.AcceptFrame(
        PipelineFrame(config, index, 1.0 + 0.1 * index, true),
        &ignored, &reasons));
  }
  PhaseRequest request;
  request.session_id = config.resource.session_id;
  request.target = MonitoringPhase::MONITOR_REFERENCE;
  request.effective_stamp.fromSec(2.0);
  request.selection_support_cutoff = request.effective_stamp;
  request.roi.minimum_R = Eigen::Vector3d(-1.0, -1.0, -1.0);
  request.roi.maximum_R = Eigen::Vector3d(3.0, 1.0, 1.0);
  request.trigger_source = TriggerSource::EXTERNAL_TASK_CURRENT_DWELL;
  const PhaseResponse response = pipeline.SetPhase(request);
  ASSERT_TRUE(response.accepted);
  EXPECT_FALSE(pipeline.reference().frozen);

  for (uint64_t index = 6U; index < 12U; ++index) {
    V1Snapshot ignored;
    std::vector<ReasonCode> reasons;
    ASSERT_TRUE(pipeline.AcceptFrame(
        PipelineFrame(config, index, 2.0 + 0.1 * (index - 6U), true),
        &ignored, &reasons));
  }
  EXPECT_FALSE(pipeline.reference().frozen);
  for (uint64_t index = 12U; index < 35U; ++index) {
    V1Snapshot ignored;
    std::vector<ReasonCode> reasons;
    ASSERT_TRUE(pipeline.AcceptFrame(
        PipelineFrame(config, index, 2.0 + 0.1 * (index - 6U), true),
        &ignored, &reasons));
  }
  EXPECT_FALSE(pipeline.reference().frozen);
  {
    V1Snapshot ignored;
    std::vector<ReasonCode> reasons;
    ASSERT_TRUE(pipeline.AcceptFrame(
        PipelineFrame(config, 35U, 4.9, true), &ignored, &reasons));
  }
  ASSERT_TRUE(pipeline.reference().frozen);
  EXPECT_GE(pipeline.reference().begin_stamp, ros::Time(2.0));
  EXPECT_EQ(pipeline.reference().end_stamp, ros::Time(4.9));
  EXPECT_EQ(std::find(pipeline.reference().reasons.begin(),
                      pipeline.reference().reasons.end(),
                      ReasonCode::PRE_REFERENCE_CHANGE_UNOBSERVABLE),
            pipeline.reference().reasons.end());
}

TEST(V1Pipeline, StaleHistoricalContextFallsBackToFirstContinuousDwellWindow) {
  ros::Time::init();
  V1Config config = SyntheticKnownConfig("reference-continuity-test");
  config.resource.reference_min_frames = 6;
  config.resource.reference_min_duration_s = 0.50;
  config.resource.max_update_gap_s = 0.25;
  config.resource.deadline_ms = 100000.0;
  config.calibration.reference_link_logit_bias = 100.0;
  config.calibration.reference_link_distance_weight = 0.0;
  config.calibration.reference_link_projector_weight = 0.0;
  config.calibration.geometry_softmax_bias << 10.0, 0.0, 0.0, -10.0;
  config.config_digest = CanonicalConfigDigest(config);
  ASSERT_TRUE(ValidateConfig(config).valid);

  V1Pipeline pipeline(config);
  for (uint64_t frame_index = 0U; frame_index < 6U; ++frame_index) {
    V1Snapshot ignored;
    std::vector<ReasonCode> reasons;
    ASSERT_TRUE(pipeline.AcceptFrame(
        PipelineFrame(config, frame_index, 1.0 + 0.1 * frame_index, true),
        &ignored, &reasons));
  }

  PhaseRequest request;
  request.session_id = config.resource.session_id;
  request.target = MonitoringPhase::MONITOR_REFERENCE;
  request.effective_stamp.fromSec(2.0);
  request.selection_support_cutoff = request.effective_stamp;
  request.roi.minimum_R = Eigen::Vector3d(-1.0, -1.0, -1.0);
  request.roi.maximum_R = Eigen::Vector3d(3.0, 1.0, 1.0);
  request.trigger_source = TriggerSource::HUMAN_OR_EXTERNAL_TASK;
  const PhaseResponse reference_response = pipeline.SetPhase(request);
  ASSERT_TRUE(reference_response.accepted);
  EXPECT_TRUE(reference_response.reference_snapshot_id.empty());

  for (uint64_t frame_index = 6U; frame_index < 12U; ++frame_index) {
    V1Snapshot ignored;
    std::vector<ReasonCode> reasons;
    ASSERT_TRUE(pipeline.AcceptFrame(
        PipelineFrame(config, frame_index,
                      2.0 + 0.1 * static_cast<double>(frame_index - 6U),
                      true),
        &ignored, &reasons));
  }
  ASSERT_TRUE(pipeline.reference().frozen);
  EXPECT_NE(std::find(pipeline.reference().reasons.begin(),
                      pipeline.reference().reasons.end(),
                      ReasonCode::PRE_REFERENCE_CHANGE_UNOBSERVABLE),
            pipeline.reference().reasons.end());

  request.target = MonitoringPhase::MONITOR_UPDATE;
  request.effective_stamp.fromSec(2.5);
  const PhaseResponse update_response = pipeline.SetPhase(request);
  EXPECT_TRUE(update_response.accepted);
  EXPECT_EQ(update_response.phase, MonitoringPhase::MONITOR_UPDATE);
}

TEST(V1Dynamics,
     ParallelPosteriorMatchesSerialFieldsBytesRepeatAndCanonicalOrder) {
  const V1Config config = SyntheticKnownConfig(
      "parallel-posterior-field-byte-order-test");
  constexpr uint32_t kGduCount = 8U;
  const ReferenceSnapshot reference =
      ReplicatedMetricLineReference(kGduCount, true);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(20U, 0U), &reasons));
  // Prediction creates the complete U/A/C switching mixture, so the
  // comparison covers task QMC, MDE, observable subspaces and every component
  // field rather than only the initial U state.
  ASSERT_TRUE(filter.Predict(ros::Time(20U, 100000000U), &reasons));

  const std::vector<uint8_t> state_before =
      UacGaussianSumFilterTestAccess::MutableStateBytes(filter);
  const auto serial =
      UacGaussianSumFilterTestAccess::PosteriorSerial(filter);
  const auto parallel = filter.Posterior();
  const auto parallel_repeat = filter.Posterior();
  // Zero successful child creation is the exact execution state after the
  // first std::thread constructor fails: the calling thread drains all GDUs.
  const auto creation_failure_zero =
      UacGaussianSumFilterTestAccess::PosteriorAfterWorkerCreationFailure(
          filter, 0U);
  // Also cover a partial worker set followed by creation failure.
  const auto creation_failure_partial =
      UacGaussianSumFilterTestAccess::PosteriorAfterWorkerCreationFailure(
          filter, 2U);
  EXPECT_EQ(state_before,
            UacGaussianSumFilterTestAccess::MutableStateBytes(filter));

  const auto expect_matrix_exact = [](const auto& lhs, const auto& rhs) {
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    EXPECT_TRUE(lhs.isApprox(rhs, 0.0));
  };
  const auto expect_component_exact = [&](const GaussianComponent& lhs,
                                          const GaussianComponent& rhs) {
    EXPECT_EQ(lhs.state, rhs.state);
    EXPECT_EQ(lhs.geometry, rhs.geometry);
    EXPECT_EQ(lhs.interface_modes, rhs.interface_modes);
    EXPECT_EQ(lhs.association_lineage, rhs.association_lineage);
    EXPECT_EQ(lhs.active_fixed_lag_lineage, rhs.active_fixed_lag_lineage);
    EXPECT_EQ(lhs.contamination_scale, rhs.contamination_scale);
    EXPECT_EQ(lhs.parent_id, rhs.parent_id);
    EXPECT_EQ(lhs.component_id, rhs.component_id);
    EXPECT_DOUBLE_EQ(lhs.weight, rhs.weight);
    expect_matrix_exact(lhs.mean, rhs.mean);
    expect_matrix_exact(lhs.covariance, rhs.covariance);
    expect_matrix_exact(lhs.sqrt_information, rhs.sqrt_information);
    EXPECT_EQ(lhs.observable_rank, rhs.observable_rank);
    expect_matrix_exact(lhs.observable_projector,
                        rhs.observable_projector);
    expect_matrix_exact(lhs.observable_nullspace,
                        rhs.observable_nullspace);
    EXPECT_DOUBLE_EQ(lhs.minimum_detectable_effect_m,
                     rhs.minimum_detectable_effect_m);
    EXPECT_EQ(lhs.minimum_detectable_effect_defined,
              rhs.minimum_detectable_effect_defined);
  };
  const auto expect_posterior_exact = [&](const auto& actual) {
    ASSERT_EQ(actual.size(), serial.size());
    ASSERT_EQ(actual.size(), reference.gdus.size());
    for (size_t g = 0U; g < actual.size(); ++g) {
      const GduPosterior& lhs = actual[g];
      const GduPosterior& rhs = serial[g];
      // Index ownership, rather than worker completion order, is the public
      // canonical ordering contract.
      EXPECT_EQ(lhs.gdu_id, reference.gdus[g].id);
      EXPECT_EQ(lhs.gdu_id, rhs.gdu_id);
      EXPECT_EQ(lhs.nuisance_correlation_id, rhs.nuisance_correlation_id);
      EXPECT_EQ(lhs.state_probability, rhs.state_probability);
      EXPECT_DOUBLE_EQ(lhs.other_mass, rhs.other_mass);
      EXPECT_DOUBLE_EQ(lhs.task_scale_deformation_probability,
                       rhs.task_scale_deformation_probability);
      EXPECT_DOUBLE_EQ(lhs.task_scale_deformation_lower,
                       rhs.task_scale_deformation_lower);
      EXPECT_DOUBLE_EQ(lhs.task_scale_deformation_upper,
                       rhs.task_scale_deformation_upper);
      EXPECT_EQ(lhs.task_scale_deformation_mean_identified,
                rhs.task_scale_deformation_mean_identified);
      EXPECT_DOUBLE_EQ(lhs.task_displacement_probability,
                       rhs.task_displacement_probability);
      EXPECT_DOUBLE_EQ(lhs.current_velocity_exceedance_probability,
                       rhs.current_velocity_exceedance_probability);
      EXPECT_DOUBLE_EQ(lhs.continuous_valid_observation_duration_s,
                       rhs.continuous_valid_observation_duration_s);
      EXPECT_EQ(lhs.velocity_duration_requirement_met,
                rhs.velocity_duration_requirement_met);
      EXPECT_DOUBLE_EQ(lhs.task_interface_jump_probability,
                       rhs.task_interface_jump_probability);
      EXPECT_DOUBLE_EQ(lhs.minimum_detectable_effect_m,
                       rhs.minimum_detectable_effect_m);
      EXPECT_EQ(lhs.minimum_detectable_effect_defined,
                rhs.minimum_detectable_effect_defined);
      EXPECT_DOUBLE_EQ(lhs.velocity_minimum_detectable_effect_mps,
                       rhs.velocity_minimum_detectable_effect_mps);
      EXPECT_EQ(lhs.velocity_minimum_detectable_effect_defined,
                rhs.velocity_minimum_detectable_effect_defined);
      EXPECT_DOUBLE_EQ(lhs.interface_jump_minimum_detectable_effect_m,
                       rhs.interface_jump_minimum_detectable_effect_m);
      EXPECT_EQ(lhs.interface_jump_minimum_detectable_effect_defined,
                rhs.interface_jump_minimum_detectable_effect_defined);
      EXPECT_DOUBLE_EQ(lhs.task_power_worst_normalized_mde,
                       rhs.task_power_worst_normalized_mde);
      EXPECT_EQ(lhs.any_task_direction_detectable,
                rhs.any_task_direction_detectable);
      EXPECT_EQ(lhs.all_task_directions_detectable,
                rhs.all_task_directions_detectable);
      EXPECT_EQ(lhs.detection_allowed, rhs.detection_allowed);
      EXPECT_EQ(lhs.clearing_allowed, rhs.clearing_allowed);
      EXPECT_EQ(lhs.task_power_complete, rhs.task_power_complete);
      EXPECT_EQ(lhs.observable_rank, rhs.observable_rank);
      expect_matrix_exact(lhs.observable_projector,
                          rhs.observable_projector);
      expect_matrix_exact(lhs.nullspace, rhs.nullspace);
      EXPECT_DOUBLE_EQ(lhs.reference_information_ratio,
                       rhs.reference_information_ratio);
      EXPECT_EQ(lhs.reference_confounding, rhs.reference_confounding);
      EXPECT_DOUBLE_EQ(lhs.effective_spatial_support,
                       rhs.effective_spatial_support);
      EXPECT_DOUBLE_EQ(lhs.effective_temporal_support,
                       rhs.effective_temporal_support);
      EXPECT_EQ(lhs.admission, rhs.admission);
      ASSERT_EQ(lhs.components.size(), rhs.components.size());
      for (size_t component = 0U; component < lhs.components.size();
           ++component)
        expect_component_exact(lhs.components[component],
                               rhs.components[component]);
    }
  };
  expect_posterior_exact(parallel);
  expect_posterior_exact(parallel_repeat);
  expect_posterior_exact(creation_failure_zero);
  expect_posterior_exact(creation_failure_partial);

  const auto serialized_bytes = [](const auto& message) {
    const uint32_t length =
        ros::serialization::serializationLength(message);
    std::vector<uint8_t> bytes(length);
    ros::serialization::OStream stream(bytes.data(), length);
    ros::serialization::serialize(stream, message);
    return bytes;
  };
  V1Snapshot serial_snapshot;
  serial_snapshot.common.snapshot_id = TestId(
      "parallel-posterior-wire-snapshot", 0U);
  serial_snapshot.common.measurement_stamp = ros::Time(20U, 100000000U);
  serial_snapshot.reference = reference;
  serial_snapshot.deformation = serial;
  V1Snapshot parallel_snapshot = serial_snapshot;
  parallel_snapshot.deformation = parallel;
  V1Snapshot repeat_snapshot = serial_snapshot;
  repeat_snapshot.deformation = parallel_repeat;
  const std::vector<uint8_t> serial_wire = serialized_bytes(
      BuildDeformationMessage(serial_snapshot));
  EXPECT_EQ(serial_wire, serialized_bytes(
      BuildDeformationMessage(parallel_snapshot)));
  EXPECT_EQ(serial_wire, serialized_bytes(
      BuildDeformationMessage(repeat_snapshot)));
}

TEST(V1Dynamics,
     FrozenInterfaceEndpointCacheMatchesGenericFieldsBytesAndFallback) {
  const V1Config config = SyntheticKnownConfig(
      "interface-frozen-endpoint-cache-test");
  constexpr uint32_t kGduCount = 8U;
  const ReferenceSnapshot reference =
      ReplicatedMetricLineReference(kGduCount, true);
  UacGaussianSumFilter filter(config);
  std::vector<ReasonCode> reasons;
  ASSERT_TRUE(filter.Initialize(reference, ros::Time(30U, 0U), &reasons));
  ASSERT_TRUE(filter.Predict(ros::Time(30U, 100000000U), &reasons));
  const auto posterior = filter.Posterior();
  ASSERT_EQ(posterior.size(), reference.gdus.size());
  const std::vector<uint8_t> state_before =
      UacGaussianSumFilterTestAccess::MutableStateBytes(filter);
  const auto generic =
      UacGaussianSumFilterTestAccess::InterfacePosteriorsGeneric(
          filter, posterior);
  const auto cached = filter.InterfacePosteriors(posterior);
  const auto cached_repeat = filter.InterfacePosteriors(posterior);
  const auto missing_cache =
      UacGaussianSumFilterTestAccess::InterfacePosteriorsWithMissingCache(
          &filter, posterior, 0U);
  EXPECT_EQ(state_before,
            UacGaussianSumFilterTestAccess::MutableStateBytes(filter));

  const auto expect_matrix_exact = [](const auto& lhs, const auto& rhs) {
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    EXPECT_TRUE(lhs.isApprox(rhs, 0.0));
  };
  const auto expect_interval_exact = [](const ProbabilityInterval& lhs,
                                        const ProbabilityInterval& rhs) {
    EXPECT_DOUBLE_EQ(lhs.lower, rhs.lower);
    EXPECT_DOUBLE_EQ(lhs.mean, rhs.mean);
    EXPECT_DOUBLE_EQ(lhs.upper, rhs.upper);
    EXPECT_EQ(lhs.calibrated, rhs.calibrated);
    EXPECT_EQ(lhs.mean_identified, rhs.mean_identified);
  };
  const auto expect_field_component_exact = [&](
      const FieldComponentQuery& lhs, const FieldComponentQuery& rhs) {
    EXPECT_EQ(lhs.component_id, rhs.component_id);
    EXPECT_DOUBLE_EQ(lhs.weight, rhs.weight);
    EXPECT_EQ(lhs.state, rhs.state);
    EXPECT_EQ(lhs.geometry, rhs.geometry);
    expect_matrix_exact(lhs.mean_displacement_R, rhs.mean_displacement_R);
    expect_matrix_exact(lhs.mean_velocity_R, rhs.mean_velocity_R);
    expect_matrix_exact(lhs.displacement_covariance_R,
                        rhs.displacement_covariance_R);
    expect_matrix_exact(lhs.velocity_covariance_R,
                        rhs.velocity_covariance_R);
    expect_matrix_exact(lhs.observable_projector_R,
                        rhs.observable_projector_R);
    EXPECT_DOUBLE_EQ(lhs.minimum_detectable_effect_m,
                     rhs.minimum_detectable_effect_m);
    EXPECT_EQ(lhs.minimum_detectable_effect_defined,
              rhs.minimum_detectable_effect_defined);
  };
  const auto expect_field_exact = [&](const FieldQueryResult& lhs,
                                      const FieldQueryResult& rhs) {
    expect_matrix_exact(lhs.mean_displacement_R, rhs.mean_displacement_R);
    expect_matrix_exact(lhs.mean_velocity_R, rhs.mean_velocity_R);
    expect_matrix_exact(lhs.displacement_covariance_R,
                        rhs.displacement_covariance_R);
    expect_matrix_exact(lhs.velocity_covariance_R,
                        rhs.velocity_covariance_R);
    expect_matrix_exact(lhs.observable_projector_R,
                        rhs.observable_projector_R);
    expect_interval_exact(lhs.active_probability, rhs.active_probability);
    EXPECT_DOUBLE_EQ(lhs.other_mass, rhs.other_mass);
    EXPECT_DOUBLE_EQ(lhs.interpolation_error_upper_m,
                     rhs.interpolation_error_upper_m);
    EXPECT_EQ(lhs.validity, rhs.validity);
    EXPECT_EQ(lhs.reasons, rhs.reasons);
    ASSERT_EQ(lhs.components.size(), rhs.components.size());
    for (size_t component = 0U; component < lhs.components.size();
         ++component)
      expect_field_component_exact(lhs.components[component],
                                   rhs.components[component]);
  };
  const auto expect_interfaces_exact = [&](const auto& actual) {
    ASSERT_EQ(actual.size(), generic.size());
    ASSERT_EQ(actual.size(), reference.interfaces.size());
    for (size_t interface = 0U; interface < actual.size(); ++interface) {
      const InterfacePosterior& lhs = actual[interface];
      const InterfacePosterior& rhs = generic[interface];
      EXPECT_EQ(lhs.interface_id, reference.interfaces[interface].id);
      EXPECT_EQ(lhs.interface_id, rhs.interface_id);
      EXPECT_EQ(lhs.joint_probability, rhs.joint_probability);
      expect_matrix_exact(lhs.jump_mean_R, rhs.jump_mean_R);
      expect_matrix_exact(lhs.jump_covariance_R, rhs.jump_covariance_R);
      EXPECT_DOUBLE_EQ(lhs.other_mass, rhs.other_mass);
      EXPECT_EQ(lhs.connectivity_coherence_confounding,
                rhs.connectivity_coherence_confounding);
      EXPECT_EQ(lhs.validity, rhs.validity);
      expect_field_exact(lhs.left_field, rhs.left_field);
      expect_field_exact(lhs.right_field, rhs.right_field);
      expect_matrix_exact(lhs.left_query_position_R,
                          rhs.left_query_position_R);
      expect_matrix_exact(lhs.right_query_position_R,
                          rhs.right_query_position_R);
    }
  };
  expect_interfaces_exact(cached);
  expect_interfaces_exact(cached_repeat);
  // When one immutable cache is unavailable, the production path must fall
  // back to the former generic query for every affected endpoint.
  expect_interfaces_exact(missing_cache.first);
  expect_interfaces_exact(missing_cache.second);

  const auto serialized_bytes = [](const auto& message) {
    const uint32_t length =
        ros::serialization::serializationLength(message);
    std::vector<uint8_t> bytes(length);
    ros::serialization::OStream stream(bytes.data(), length);
    ros::serialization::serialize(stream, message);
    return bytes;
  };
  V1Snapshot generic_snapshot;
  generic_snapshot.common.snapshot_id = TestId(
      "interface-cache-wire-snapshot", 0U);
  generic_snapshot.common.measurement_stamp = ros::Time(30U, 100000000U);
  generic_snapshot.common.validity = Validity::VALID;
  generic_snapshot.reference = reference;
  generic_snapshot.deformation = posterior;
  generic_snapshot.interfaces = generic;
  V1Snapshot cached_snapshot = generic_snapshot;
  cached_snapshot.interfaces = cached;
  V1Snapshot repeat_snapshot = generic_snapshot;
  repeat_snapshot.interfaces = cached_repeat;
  const std::vector<uint8_t> generic_wire = serialized_bytes(
      BuildInterfaceMessage(generic_snapshot));
  EXPECT_EQ(generic_wire, serialized_bytes(
      BuildInterfaceMessage(cached_snapshot)));
  EXPECT_EQ(generic_wire, serialized_bytes(
      BuildInterfaceMessage(repeat_snapshot)));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
