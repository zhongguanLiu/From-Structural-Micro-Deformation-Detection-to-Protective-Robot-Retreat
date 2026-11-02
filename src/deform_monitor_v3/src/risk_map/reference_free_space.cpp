#include "deform_monitor_v3/risk_map/reference_free_space.hpp"

#include <cmath>
#include <limits>

namespace deform_monitor_v3 { namespace risk_map {

void ReferenceFreeSpaceMap::Reset(uint32_t epoch) { voxels_.clear(); epoch_ = epoch; frozen_ = false; }

VoxelKey ReferenceFreeSpaceMap::Key(const Eigen::Vector3d& p) const {
  const double s = params_.voxel_size;
  return {static_cast<int32_t>(std::floor(p.x()/s)), static_cast<int32_t>(std::floor(p.y()/s)),
          static_cast<int32_t>(std::floor(p.z()/s))};
}
Eigen::Vector3d ReferenceFreeSpaceMap::Center(const VoxelKey& k) const {
  return params_.voxel_size * Eigen::Vector3d(k.x + 0.5, k.y + 0.5, k.z + 0.5);
}

std::vector<VoxelKey> ReferenceFreeSpaceMap::TraceRay(const Eigen::Vector3d& origin,
                                                       const Eigen::Vector3d& endpoint,
                                                       bool include_endpoint) const {
  std::vector<VoxelKey> keys;
  VoxelKey current = Key(origin), target = Key(endpoint);
  keys.push_back(current);
  const Eigen::Vector3d direction = endpoint - origin;
  int step[3]; double t_max[3], t_delta[3];
  int* c[3] = {&current.x, &current.y, &current.z};
  const int target_values[3] = {target.x, target.y, target.z};
  for (int axis=0; axis<3; ++axis) {
    const double d = direction[axis];
    step[axis] = d > 0 ? 1 : (d < 0 ? -1 : 0);
    if (step[axis] == 0) { t_max[axis] = t_delta[axis] = std::numeric_limits<double>::infinity(); continue; }
    const double boundary = params_.voxel_size * (*c[axis] + (step[axis] > 0 ? 1 : 0));
    t_max[axis] = (boundary - origin[axis]) / d;
    t_delta[axis] = params_.voxel_size / std::abs(d);
  }
  const size_t safety_limit = 1 + static_cast<size_t>((direction.cwiseAbs().array()/params_.voxel_size).sum()) + 6;
  while (current != target && keys.size() < safety_limit) {
    const double crossing = std::min(t_max[0], std::min(t_max[1], t_max[2]));
    for (int axis = 0; axis < 3; ++axis) {
      if (std::abs(t_max[axis] - crossing) <= 1e-12) {
        *c[axis] += step[axis];
        t_max[axis] += t_delta[axis];
      }
    }
    keys.push_back(current);
  }
  if (!include_endpoint && !keys.empty() && keys.back() == target) keys.pop_back();
  return keys;
}

void ReferenceFreeSpaceMap::IntegrateInitializationFrame(const ObservationFrame& frame) {
  if (frozen_ || !frame.cloud) return;
  for (const auto& point : frame.cloud->points) {
    const Eigen::Vector3d endpoint(point.x, point.y, point.z);
    if (!endpoint.allFinite()) continue;
    for (const auto& key : TraceRay(frame.lidar_origin_R, endpoint, false)) {
      auto& v = voxels_[key]; ++v.traversed_ray_count; v.reference_epoch = epoch_;
    }
    auto& v = voxels_[Key(endpoint)]; ++v.endpoint_count; v.reference_epoch = epoch_;
  }
}

void ReferenceFreeSpaceMap::Freeze() {
  for (auto& item : voxels_) {
    auto& v = item.second;
    const double hits = params_.reference_beta_a + v.endpoint_count;
    const double total = params_.reference_beta_a + params_.reference_beta_b +
                         v.endpoint_count + v.traversed_ray_count;
    v.occupancy_probability = hits / total;
    const uint32_t coverage = v.endpoint_count + v.traversed_ray_count;
    v.state = coverage < static_cast<uint32_t>(params_.reference_min_coverage)
                  ? KnowledgeState::UNKNOWN
                  : (v.occupancy_probability <= params_.reference_free_threshold
                         ? KnowledgeState::FREE
                         : (v.occupancy_probability >= params_.reference_occupied_threshold
                                ? KnowledgeState::OCCUPIED : KnowledgeState::UNKNOWN));
  }
  frozen_ = true;
}

const ReferenceVoxel* ReferenceFreeSpaceMap::Find(const VoxelKey& key) const {
  const auto it = voxels_.find(key); return it == voxels_.end() ? nullptr : &it->second;
}

} }
