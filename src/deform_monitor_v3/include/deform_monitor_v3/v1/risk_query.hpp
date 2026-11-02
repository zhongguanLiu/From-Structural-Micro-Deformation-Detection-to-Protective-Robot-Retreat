#ifndef DEFORM_MONITOR_V3_V1_RISK_QUERY_HPP
#define DEFORM_MONITOR_V3_V1_RISK_QUERY_HPP

#include "deform_monitor_v3/v1/occupancy.hpp"
#include "deform_monitor_v3/v1/uac_filter.hpp"

namespace deform_monitor_v3 {
namespace v1 {

class RiskQueryEngine {
 public:
  explicit RiskQueryEngine(const V1Config& config) : config_(config) {}

  bool BuildHazards(
      const CommonHeader& header, const ReferenceSnapshot& reference,
      const UacGaussianSumFilter& filter,
      size_t maximum_hazards,
      std::vector<HazardGeometry,
          Eigen::aligned_allocator<HazardGeometry>>* hazards,
      std::vector<ReasonCode>* reasons) const;

  RiskAssessment AssessState(const CommonHeader& header,
                             const ReferenceSnapshot& reference,
                             const UacGaussianSumFilter& filter,
                             const OccupancyEventGraph& occupancy,
                             const RobotGeometry& robot,
                             const TimedRobotState& state,
                             const ros::Time& evaluation_stamp = ros::Time()) const;

  RiskAssessment AssessPath(const CommonHeader& header,
                            const ReferenceSnapshot& reference,
                            const UacGaussianSumFilter& filter,
                            const OccupancyEventGraph& occupancy,
                            const RobotGeometry& robot,
                            const std::vector<TimedRobotState,
                                Eigen::aligned_allocator<TimedRobotState>>& path,
                            std::vector<TimedRobotState,
                                Eigen::aligned_allocator<TimedRobotState>>*
                                canonical_states = nullptr,
                            std::vector<RiskAssessment>* state_assessments = nullptr,
                            std::vector<double>* path_upper_contributions = nullptr,
                            const ros::Time& evaluation_stamp = ros::Time()) const;

  RiskAssessment AssessMotion(const CommonHeader& header,
                              const ReferenceSnapshot& reference,
                              const UacGaussianSumFilter& filter,
                              const OccupancyEventGraph& occupancy,
                              const RobotGeometry& robot,
                              const std::vector<TimedRobotState,
                                  Eigen::aligned_allocator<TimedRobotState>>& motion,
                              double body_sweep_tolerance_m,
                              std::vector<TimedRobotState,
                                  Eigen::aligned_allocator<TimedRobotState>>*
                                  canonical_states = nullptr,
                              std::vector<RiskAssessment>* state_assessments = nullptr,
                              std::vector<double>* path_upper_contributions = nullptr,
                              const ros::Time& evaluation_stamp = ros::Time()) const;

  std::vector<TimedRobotState, Eigen::aligned_allocator<TimedRobotState>>
  CanonicalMotionSamples(
      const RobotGeometry& robot,
      const std::vector<TimedRobotState,
          Eigen::aligned_allocator<TimedRobotState>>& motion,
      double body_sweep_tolerance_m,
      bool* resource_exceeded = nullptr) const;

  bool WithinOperationBudget(
      const ReferenceSnapshot& reference,
      const std::vector<GduPosterior,
          Eigen::aligned_allocator<GduPosterior>>& posterior,
      const OccupancyEventGraph& occupancy,
      const RobotGeometry& robot,
      size_t temporal_sample_count = 1U) const;

 private:
  struct CauseRisk {
    ProbabilityInterval collision;
    ProbabilityInterval encounter;
  };
  CauseRisk StructuralCauseRisk(
      const CommonHeader& header, const ReferenceSnapshot& reference,
      const GduDefinition& gdu, uint32_t gdu_index,
      const GduPosterior& posterior, const RobotGeometry& robot,
      const TimedRobotState& state) const;
  ProbabilityInterval OccupancyCauseRisk(
      const CommonHeader& header, const StableId& cause_id,
      const std::vector<std::pair<const OccupancyCell*, ProbabilityInterval>>& cells,
      HazardSemantics semantics,
      const RobotGeometry& robot, const TimedRobotState& state) const;
  StableId GduCauseId(const StableId& gdu_id) const;
  size_t EstimateStateQueryOperations(
      const ReferenceSnapshot& reference,
      const std::vector<GduPosterior,
          Eigen::aligned_allocator<GduPosterior>>& posterior,
      const OccupancyEventGraph& occupancy,
      const RobotGeometry& robot) const;
  V1Config config_;
};

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_RISK_QUERY_HPP
