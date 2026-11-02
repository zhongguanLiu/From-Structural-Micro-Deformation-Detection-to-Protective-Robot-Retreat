#ifndef DEFORM_MONITOR_V3_RISK_MAP_POSE_COVARIANCE_MODEL_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_POSE_COVARIANCE_MODEL_HPP

#include "deform_monitor_v3/risk_map/risk_types.hpp"

namespace deform_monitor_v3 { namespace risk_map {

struct PoseCovarianceResult {
  Eigen::MatrixXd covariance;
  bool valid = false;
  double minimum_eigenvalue = 0.0;
};

class PoseCovarianceModel {
 public:
  explicit PoseCovarianceModel(const RiskMapParams& params = RiskMapParams()) : params_(params) {}
  void SetParams(const RiskMapParams& params) { params_ = params; }
  PoseCovarianceResult Build(const ChannelFrameDeque& frames) const;

 private:
  RiskMapParams params_;
};

} }
#endif
