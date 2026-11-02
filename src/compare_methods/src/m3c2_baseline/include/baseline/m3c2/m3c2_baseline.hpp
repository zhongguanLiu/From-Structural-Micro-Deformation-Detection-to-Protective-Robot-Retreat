#ifndef BASELINE_M3C2_M3C2_BASELINE_HPP
#define BASELINE_M3C2_M3C2_BASELINE_HPP

#include "baseline/common/config.hpp"
#include "baseline/common/types.hpp"
#include "baseline/core/cluster_extractor.hpp"

namespace baseline {

class M3C2Baseline {
public:
  void SetConfig(const M3C2Config& config);

  M3C2Result Run(const PointCloudPtr& reference_cloud,
                 const PointCloudPtr& query_cloud) const;

private:
  M3C2Config config_;
};

}  // namespace baseline

#endif  // BASELINE_M3C2_M3C2_BASELINE_HPP
