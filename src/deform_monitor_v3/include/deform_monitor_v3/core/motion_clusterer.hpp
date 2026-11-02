#ifndef DEFORM_MONITOR_V3_CORE_MOTION_CLUSTERER_HPP
#define DEFORM_MONITOR_V3_CORE_MOTION_CLUSTERER_HPP

#include "deform_monitor_v3/data_types.hpp"

namespace deform_monitor_v3 {

class MotionClusterer {
public:
  void SetParams(const ClusterParams& params);
  MotionClusterVector Cluster(
      const AnchorReferenceVector& anchors,
      const AnchorStateVector& states) const;

private:
  ClusterParams params_;
};

}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_CORE_MOTION_CLUSTERER_HPP
