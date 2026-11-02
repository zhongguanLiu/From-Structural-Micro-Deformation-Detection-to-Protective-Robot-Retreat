#include "deform_monitor_v3/risk_map/channel_adapter.hpp"

#include <cmath>
#include <unordered_map>

namespace deform_monitor_v3 { namespace risk_map {

ChannelFrame ChannelAdapter::BuildFrame(const AnchorReferenceVector& anchors,
                                        const CurrentObservationVector& observations,
                                        const PoseCov6D& pose_cov,
                                        const Eigen::Vector3d& sensor_origin_R,
                                        const ros::Time& stamp,
                                        uint32_t reference_epoch,
                                        const FrozenModeVector& modes) const {
  ChannelFrame out;
  out.stamp = stamp;
  out.pose_cov = pose_cov;
  out.sensor_origin_R = sensor_origin_R;
  std::unordered_map<int, Eigen::Vector3d> mode_by_id;
  for (const auto& mode : modes) mode_by_id[mode.anchor_id] = mode.direction_R;
  std::unordered_map<int,const CurrentObservation*> observation_by_id;
  for(const auto& observation:observations)
    if(observation.anchor_id>=0)observation_by_id[observation.anchor_id]=&observation;
  for (const auto& anchor:anchors) {
    const auto observation_it=observation_by_id.find(anchor.id);
    if(observation_it==observation_by_id.end())continue;
    const CurrentObservation& observation=*observation_it->second;
    const Eigen::Vector3d rho = anchor.center_R - sensor_origin_R;
    const auto mit = mode_by_id.find(anchor.id);
    const Eigen::Vector3d mode = mit == mode_by_id.end()
                                     ? Eigen::Vector3d::UnitZ() : mit->second;
    for (size_t axis = 0; axis < observation.scalars.size(); ++axis) {
      const ScalarMeasurement& scalar = observation.scalars[axis];
      if (!std::isfinite(scalar.z) || !std::isfinite(scalar.r) || scalar.r <= 0.0 ||
          !scalar.h_R.allFinite() || scalar.h_R.norm() <= 1.0e-12) continue;
      ChannelObservation row;
      row.key.anchor_id = anchor.id;
      // ScalarMeasurement::type is the stable observable-axis identity
      // (normal=0, edge=1, band=3); vector position is not an identity.
      row.key.axis_index = scalar.type;
      row.key.reference_epoch = reference_epoch;
      row.stamp = stamp;
      row.h_R = scalar.h_R.normalized();
      row.g.head<3>() = -row.h_R.cross(rho);
      row.g.tail<3>() = row.h_R;
      row.z = scalar.z;
      row.independent_variance = std::isfinite(scalar.independent_r)
                                     ? scalar.independent_r : scalar.r;
      row.reference_covariance_R = scalar.reference_covariance_R;
      row.reference_covariance_group = scalar.reference_covariance_group;
      row.psi = row.h_R.dot(mode);
      row.matched = observation.comparable;
      out.channels.push_back(row);
    }
  }
  std::sort(out.channels.begin(), out.channels.end(),
            [](const ChannelObservation& a, const ChannelObservation& b) {
              return a.key < b.key;
            });
  return out;
}

FrozenModeVector ChannelAdapter::FreezeModes(const AnchorReferenceVector& anchors,
                                             const AnchorStateVector& states,
                                             const ros::Time& source_stamp,
                                             const Eigen::Vector3d& gravity_R) {
  FrozenModeVector modes;
  modes.reserve(anchors.size());
  const Eigen::Vector3d fallback = gravity_R.norm() > 1e-9
                                       ? gravity_R.normalized() : Eigen::Vector3d::UnitZ();
  std::unordered_map<int,const AnchorTrackState*> state_by_id;
  for (const auto& state:states) state_by_id[state.id]=&state;
  for (size_t i = 0; i < anchors.size(); ++i) {
    FrozenMode mode;
    mode.anchor_id = anchors[i].id;
    mode.source_stamp = source_stamp;
    mode.direction_R = fallback;
    const auto state_it=state_by_id.find(anchors[i].id);
    if (state_it!=state_by_id.end()) {
      const AnchorTrackState& state=*state_it->second;
      if (!state.last_update.isZero()) mode.source_stamp = state.last_update;
      const Eigen::Vector3d displacement = state.x_mix.head<3>();
      if (displacement.allFinite() && displacement.norm() > 1e-5) {
        mode.direction_R = displacement.normalized();
        mode.from_previous_state = true;
        const Eigen::Matrix3d covariance=.5*(state.P_mix.topLeftCorner<3,3>()+
                                             state.P_mix.topLeftCorner<3,3>().transpose());
        const double directional_variance=mode.direction_R.dot(covariance*mode.direction_R);
        if(std::isfinite(directional_variance)&&directional_variance>0.0){
          const double snr2=displacement.squaredNorm()/directional_variance;
          mode.direction_confidence=std::max(0.0,std::min(1.0,-std::expm1(-.5*snr2)));
        }
      }
    }
    modes.push_back(mode);
  }
  return modes;
}

} }
