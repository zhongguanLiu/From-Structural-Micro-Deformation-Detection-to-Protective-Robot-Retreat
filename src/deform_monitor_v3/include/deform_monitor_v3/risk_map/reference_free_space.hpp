#ifndef DEFORM_MONITOR_V3_RISK_MAP_REFERENCE_FREE_SPACE_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_REFERENCE_FREE_SPACE_HPP

#include "deform_monitor_v3/risk_map/risk_types.hpp"

namespace deform_monitor_v3 { namespace risk_map {

class ReferenceFreeSpaceMap {
 public:
  explicit ReferenceFreeSpaceMap(const RiskMapParams& params = RiskMapParams()) : params_(params) {}
  void SetParams(const RiskMapParams& params) { params_ = params; }
  void Reset(uint32_t epoch);
  void IntegrateInitializationFrame(const ObservationFrame& frame);
  void Freeze();
  const ReferenceVoxel* Find(const VoxelKey& key) const;
  const AlignedUnorderedMap<VoxelKey, ReferenceVoxel, VoxelKeyHash>& voxels() const { return voxels_; }
  VoxelKey Key(const Eigen::Vector3d& point) const;
  Eigen::Vector3d Center(const VoxelKey& key) const;
  std::vector<VoxelKey> TraceRay(const Eigen::Vector3d& origin,
                                 const Eigen::Vector3d& endpoint,
                                 bool include_endpoint) const;
  bool frozen() const { return frozen_; }

 private:
  RiskMapParams params_;
  uint32_t epoch_ = 0;
  bool frozen_ = false;
  AlignedUnorderedMap<VoxelKey, ReferenceVoxel, VoxelKeyHash> voxels_;
};

} }
#endif
