#include "deform_monitor_v3/risk_map/current_occupancy.hpp"

#include <cmath>

namespace deform_monitor_v3 { namespace risk_map {

void CurrentOccupancyFrame::Build(const ObservationFrameDeque& frames) {
  voxels_.clear();
  for (const auto& frame : frames) {
    if (!frame.cloud) continue;
    for (const auto& p : frame.cloud->points) {
      const Eigen::Vector3d endpoint(p.x,p.y,p.z);
      if (!endpoint.allFinite()) continue;
      for (const auto& key : geometry_.TraceRay(frame.lidar_origin_R, endpoint, false)) {
        auto& cell=voxels_[key]; ++cell.traversed_count; cell.stamp=frame.stamp;
      }
      auto& cell=voxels_[geometry_.Key(endpoint)]; ++cell.hit_count; cell.stamp=frame.stamp;
    }
  }
  for (auto& item : voxels_) {
    auto& cell=item.second;
    // These are block-level observation probabilities, not fractions whose
    // denominator is the number of frames.  The exponential occupancy form
    // is stable for irregular scan density and remains explicitly calibratable.
    cell.beam_coverage=1.0-std::exp(-static_cast<double>(cell.traversed_count));
    cell.visibility_probability=1.0-std::exp(-static_cast<double>(
        cell.hit_count+cell.traversed_count));
    // lambda_1 is an exposure-conditioned model parameter.  It must not use
    // k=hit_count, otherwise the tested observation defines its own likelihood.
    const double beam_exposure=std::max(1.0,static_cast<double>(cell.traversed_count));
    cell.expected_hit_count=std::max(params_.probability_floor,
        params_.occupied_rate_scale*beam_exposure);
  }
}

const CurrentVoxelObservation* CurrentOccupancyFrame::Find(const VoxelKey& key) const {
  const auto it=voxels_.find(key); return it==voxels_.end()?nullptr:&it->second;
}

bool CurrentOccupancyFrame::IsOccluded(const Eigen::Vector3d& origin,
                                       const Eigen::Vector3d& target,
                                       const VoxelKey* excluded) const {
  for (const auto& key : geometry_.TraceRay(origin,target,false)) {
    if (excluded && key==*excluded) continue;
    const auto* cell=Find(key); if (cell && cell->hit_count>0) return true;
  }
  return false;
}

} }
