#ifndef BASELINE_CORE_REFERENCE_MANAGER_HPP
#define BASELINE_CORE_REFERENCE_MANAGER_HPP

#include "baseline/common/config.hpp"
#include "baseline/common/types.hpp"

namespace baseline {

class ReferenceManager {
public:
  void SetConfig(const ReferenceConfig& config);
  bool AddCloud(const PointCloudPtr& cloud);
  bool IsFrozen() const;
  void Reset();
  bool ForceFreeze();
  PointCloudPtr GetReferenceCloud() const;

private:
  void Freeze();

  ReferenceConfig config_;
  int accumulated_frames_ = 0;
  bool frozen_ = false;
  PointCloudPtr accumulated_cloud_{new PointCloud()};
  PointCloudPtr frozen_cloud_{new PointCloud()};
};

}  // namespace baseline

#endif  // BASELINE_CORE_REFERENCE_MANAGER_HPP
