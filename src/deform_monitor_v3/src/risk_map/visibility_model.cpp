#include "deform_monitor_v3/risk_map/visibility_model.hpp"
#include "deform_monitor_v3/risk_map/risk_math.hpp"
#include <cmath>
namespace deform_monitor_v3 { namespace risk_map {
VisibilityEstimate VisibilityModel::Estimate(const Eigen::Vector3d& origin,const Eigen::Vector3d& target,
 const Eigen::Vector3d& normal,const CurrentOccupancyFrame& occupancy) const {
  VisibilityEstimate v; Eigen::Vector3d ray=target-origin; v.range=ray.norm();
  if(v.range<1e-9) return v; ray/=v.range;
  const double elevation=std::asin(std::max(-1.0,std::min(1.0,ray.z())))*180.0/M_PI;
  const double width=std::max(.01,params_.fov_soft_width_deg);
  v.fov_feature=Sigmoid((elevation-params_.fov_vertical_min_deg)/width)*
                Sigmoid((params_.fov_vertical_max_deg-elevation)/width)*
                Sigmoid((v.range-params_.min_range)/.1)*Sigmoid((params_.max_range-v.range)/.5);
  const VoxelKey target_key={static_cast<int32_t>(std::floor(target.x()/params_.voxel_size)),
                             static_cast<int32_t>(std::floor(target.y()/params_.voxel_size)),
                             static_cast<int32_t>(std::floor(target.z()/params_.voxel_size))};
  v.occlusion_feature=occupancy.IsOccluded(origin,target,&target_key)?0.0:1.0;
  const Eigen::Vector3d unit_normal =
      normal.allFinite() && normal.norm() > 1.0e-9 ? normal.normalized()
                                                     : Eigen::Vector3d::UnitZ();
  v.incidence_cos=std::abs(unit_normal.dot(-ray));
  const CurrentVoxelObservation* target_observation = occupancy.Find(target_key);
  // Coverage represents traversing beams; a direct return is independent
  // evidence that the target cell was observable in this block.
  v.density_feature = target_observation
      ? std::max(std::max(0.0, std::min(1.0, target_observation->beam_coverage)),
                 1.0 - std::exp(-static_cast<double>(target_observation->hit_count)))
      : 0.0;
  const double raw=v.fov_feature*v.occlusion_feature*std::max(.05,v.density_feature);
  v.probability=ClampProbability(Sigmoid(params_.visibility_logit_bias+
      params_.visibility_logit_scale*Logit(ClampProbability(raw,params_.probability_floor))),
      params_.probability_floor);
  v.calibrated=params_.calibration_version!="uncalibrated"; v.model_version=params_.calibration_version;
  return v;
}
} }
