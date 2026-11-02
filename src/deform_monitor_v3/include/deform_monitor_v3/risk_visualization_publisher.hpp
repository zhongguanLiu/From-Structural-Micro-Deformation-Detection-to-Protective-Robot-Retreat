#ifndef DEFORM_MONITOR_V3_RISK_VISUALIZATION_PUBLISHER_HPP
#define DEFORM_MONITOR_V3_RISK_VISUALIZATION_PUBLISHER_HPP

#include <visualization_msgs/MarkerArray.h>

#include "deform_monitor_v3/RiskEvidenceArray.h"
#include "deform_monitor_v3/RiskRegion.h"
#include "deform_monitor_v3/RiskRegions.h"
#include "deform_monitor_v3/RiskVoxelField.h"
#include "deform_monitor_v3/PersistentRiskRegion.h"
#include "deform_monitor_v3/PersistentRiskRegions.h"
#include "deform_monitor_v3/data_types.hpp"

namespace deform_monitor_v3 {

class RiskVisualizationPublisher {
public:
  void SetParams(const RiskVisualizationParams& params);

  deform_monitor_v3::RiskEvidenceArray BuildRiskEvidenceMsg(
      const RiskEvidenceVector& evidences,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  deform_monitor_v3::RiskVoxelField BuildRiskVoxelFieldMsg(
      const RiskVoxelVector& voxels,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  deform_monitor_v3::RiskRegions BuildRiskRegionsMsg(
      const RiskRegionVector& regions,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  deform_monitor_v3::PersistentRiskRegions BuildPersistentRiskRegionsMsg(
      const PersistentRiskTrackVector& tracks,
      const ros::Time& stamp,
      const std::string& frame_id,
      uint32_t reference_epoch) const;

  visualization_msgs::MarkerArray BuildRiskMarkers(
      const RiskVoxelVector& voxels,
      const RiskRegionVector& regions,
      const ros::Time& stamp,
      const std::string& frame_id) const;

  visualization_msgs::MarkerArray BuildPersistentRiskMarkers(
      const PersistentRiskTrackVector& tracks,
      const ros::Time& stamp,
      const std::string& frame_id) const;

private:
  RiskVisualizationParams params_;
};

}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_RISK_VISUALIZATION_PUBLISHER_HPP
