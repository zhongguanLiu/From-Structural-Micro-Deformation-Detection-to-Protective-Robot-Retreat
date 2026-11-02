#ifndef DEFORM_MONITOR_V3_RISK_MAP_CURRENT_OCCUPANCY_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_CURRENT_OCCUPANCY_HPP

#include "deform_monitor_v3/risk_map/reference_free_space.hpp"

namespace deform_monitor_v3 { namespace risk_map {

class CurrentOccupancyFrame {
 public:
  explicit CurrentOccupancyFrame(const RiskMapParams& params = RiskMapParams()) : params_(params), geometry_(params) {}
  void SetParams(const RiskMapParams& params) { params_=params; geometry_.SetParams(params); }
  void Build(const ObservationFrameDeque& frames);
  const CurrentVoxelObservation* Find(const VoxelKey& key) const;
  const AlignedUnorderedMap<VoxelKey, CurrentVoxelObservation, VoxelKeyHash>& voxels() const { return voxels_; }
  bool IsOccluded(const Eigen::Vector3d& origin, const Eigen::Vector3d& target,
                  const VoxelKey* excluded_target = nullptr) const;

 private:
  RiskMapParams params_;
  ReferenceFreeSpaceMap geometry_;
  AlignedUnorderedMap<VoxelKey, CurrentVoxelObservation, VoxelKeyHash> voxels_;
};

} }
#endif
