#ifndef DEFORM_MONITOR_V3_RISK_MAP_DEFORMATION_EVIDENCE_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_DEFORMATION_EVIDENCE_HPP

#include "deform_monitor_v3/risk_map/pose_covariance_model.hpp"

namespace deform_monitor_v3 { namespace risk_map {

class DeformationEvidence {
 public:
  explicit DeformationEvidence(const RiskMapParams& params = RiskMapParams()) : params_(params), pose_model_(params) {}
  void SetParams(const RiskMapParams& params) { params_ = params; pose_model_.SetParams(params); }
  EvidenceResult Evaluate(const EvidenceBlock& block, int anchor_id,
                          const Eigen::MatrixXd& shared_covariance = Eigen::MatrixXd()) const;

 private:
  RiskMapParams params_;
  PoseCovarianceModel pose_model_;
};

} }
#endif
