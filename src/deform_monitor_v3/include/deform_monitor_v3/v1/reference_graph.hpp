#ifndef DEFORM_MONITOR_V3_V1_REFERENCE_GRAPH_HPP
#define DEFORM_MONITOR_V3_V1_REFERENCE_GRAPH_HPP

#include "deform_monitor_v3/v1/types.hpp"

#include <map>

namespace deform_monitor_v3 {
namespace v1 {

// Returns -1/+1 for the frozen left/right side and 0 only when a point cannot
// be assigned to either side within the supplied GDU support.
int InterfaceSideForPoint(const ReferenceSnapshot& reference,
                          const InterfaceDefinition& interface,
                          uint32_t gdu_index,
                          const Eigen::Vector3d& position_R,
                          double tie_tolerance);

// Reference-only mask used by a target GDU's pose-control solve. A surfel is
// excluded when it lies within radius_m of any surfel belonging to the target
// GDU, irrespective of its GDU label.
std::vector<bool> PoseControlSpatialExclusionMask(
    const ReferenceSnapshot& reference, uint32_t target_gdu_index,
    double radius_m);

// Select a bounded subset of otherwise monitorable frozen components without
// using object identity, a target region, intensity or a current residual.
// The first carrier is nearest the frozen reference sensor origin; subsequent
// carriers maximize their minimum joint position/projector distance from the
// selected set. Surface area (surfel count) is only a deterministic tie
// breaker, so broad walls cannot consume the complete GDU budget merely by
// containing more sampled cells. Returned values index `candidates`.
std::vector<size_t> SelectReferenceGduCoverage(
    const ReferenceSnapshot& reference,
    const std::vector<GduDefinition,
                      Eigen::aligned_allocator<GduDefinition>>& candidates,
    const Eigen::Vector3d& frozen_sensor_origin_R, size_t maximum_count,
    double orientation_scale_m, double tie_tolerance);

class ReferenceBuilder {
 public:
  explicit ReferenceBuilder(const V1Config& config) : config_(config) {}

  ReferenceSnapshot Build(const std::deque<FrameBundle>& frames,
                          uint32_t reference_epoch) const;

 private:
  V1Config config_;
};

class PouField {
 public:
  PouField(const ReferenceSnapshot* reference, const V1Config* config)
      : reference_(reference), config_(config) {}

  int CoefficientCount(uint32_t gdu_index) const;
  Eigen::MatrixXd VectorBasis(uint32_t gdu_index,
                              const Eigen::Vector3d& position_R,
                              Validity* validity,
                              std::vector<ReasonCode>* reasons,
                              double* attachment_distance_m = nullptr) const;
  Eigen::VectorXd ScalarBasis(uint32_t gdu_index,
                              const Eigen::Vector3d& position_R,
                              const Eigen::Vector3d& observation_axis_R,
                              Validity* validity,
                              std::vector<ReasonCode>* reasons,
                              double* attachment_distance_m = nullptr) const;

 private:
  std::vector<double> GraphDistances(uint32_t gdu_index,
                                     uint32_t source_surfel) const;
  const ReferenceSnapshot* reference_ = nullptr;
  const V1Config* config_ = nullptr;
  mutable std::map<std::pair<uint32_t, uint32_t>, std::vector<double>>
      graph_distance_cache_;
};

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_REFERENCE_GRAPH_HPP
