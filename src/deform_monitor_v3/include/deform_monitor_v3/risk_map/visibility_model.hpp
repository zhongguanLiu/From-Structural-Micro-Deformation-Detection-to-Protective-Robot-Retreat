#ifndef DEFORM_MONITOR_V3_RISK_MAP_VISIBILITY_MODEL_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_VISIBILITY_MODEL_HPP
#include "deform_monitor_v3/risk_map/current_occupancy.hpp"
namespace deform_monitor_v3 { namespace risk_map {
class VisibilityModel {
 public:
  explicit VisibilityModel(const RiskMapParams& p=RiskMapParams()):params_(p){}
  void SetParams(const RiskMapParams& p){params_=p;}
  VisibilityEstimate Estimate(const Eigen::Vector3d& origin, const Eigen::Vector3d& target,
                              const Eigen::Vector3d& normal,
                              const CurrentOccupancyFrame& occupancy) const;
 private: RiskMapParams params_;
};
} }
#endif
