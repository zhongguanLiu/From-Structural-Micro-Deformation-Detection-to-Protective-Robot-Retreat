#ifndef DEFORM_MONITOR_V3_CORE_SCALAR_MEASUREMENT_BUILDER_HPP
#define DEFORM_MONITOR_V3_CORE_SCALAR_MEASUREMENT_BUILDER_HPP

#include "deform_monitor_v3/data_types.hpp"

namespace deform_monitor_v3 {

class ScalarMeasurementBuilder {
public:
  void SetParams(const ObservationParams& observation_params,
                 const NoiseParams& noise_params);

  AlignedVector<ScalarMeasurement> BuildMeasurements(
      const AnchorReference& anchor,
      const LocalSupportData& support,
      const PoseCov6D& pose_cov,
      const Eigen::Vector3d& lidar_origin_R,
      bool apply_innovation_gate = true) const;

private:
  ObservationParams observation_params_;
  NoiseParams noise_params_;
};

}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_CORE_SCALAR_MEASUREMENT_BUILDER_HPP
