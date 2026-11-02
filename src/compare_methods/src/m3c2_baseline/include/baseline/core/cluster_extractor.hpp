#ifndef BASELINE_CORE_CLUSTER_EXTRACTOR_HPP
#define BASELINE_CORE_CLUSTER_EXTRACTOR_HPP

#include "baseline/common/config.hpp"
#include "baseline/common/types.hpp"

#include <vector>

namespace baseline {

class ClusterExtractor {
public:
  void SetConfig(const ClusterConfig& config);

  std::vector<ClusterRecord> Extract(const PointCloudPtr& cloud,
                                     const std::string& method) const;

private:
  ClusterConfig config_;
};

}  // namespace baseline

#endif  // BASELINE_CORE_CLUSTER_EXTRACTOR_HPP
