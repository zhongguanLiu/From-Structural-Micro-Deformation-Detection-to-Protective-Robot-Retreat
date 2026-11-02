#ifndef BASELINE_CORE_TEMPORAL_WINDOW_FUSION_HPP
#define BASELINE_CORE_TEMPORAL_WINDOW_FUSION_HPP

#include "baseline/common/config.hpp"
#include "baseline/common/types.hpp"

#include <deque>
#include <utility>

namespace baseline {

class TemporalWindowFusion {
public:
  void SetConfig(const TemporalFusionConfig& config);
  void PushCloud(const PointCloudPtr& cloud, double stamp_sec);
  size_t FrameCount() const;
  void Reset();
  PointCloudPtr BuildFusedCloud() const;

private:
  TemporalFusionConfig config_;
  std::deque<std::pair<double, PointCloudPtr>> frames_;
};

}  // namespace baseline

#endif  // BASELINE_CORE_TEMPORAL_WINDOW_FUSION_HPP
