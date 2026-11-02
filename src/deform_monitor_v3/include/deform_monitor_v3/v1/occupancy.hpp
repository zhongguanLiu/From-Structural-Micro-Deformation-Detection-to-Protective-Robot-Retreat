#ifndef DEFORM_MONITOR_V3_V1_OCCUPANCY_HPP
#define DEFORM_MONITOR_V3_V1_OCCUPANCY_HPP

#include "deform_monitor_v3/v1/types.hpp"

#include <map>

namespace deform_monitor_v3 {
namespace v1 {

class OccupancyEventGraph {
 public:
  explicit OccupancyEventGraph(const V1Config& config) : config_(config) {}

  bool BuildReference(const std::deque<FrameBundle>& frames,
                      const ReferenceSnapshot& structural_reference,
                      uint32_t reference_epoch,
                      std::vector<ReasonCode>* reasons);
  bool Update(
      const FrameBundle& frame,
      const std::vector<HazardGeometry,
          Eigen::aligned_allocator<HazardGeometry>>& structural_claims,
      std::vector<ReasonCode>* reasons);
  // Projects already inferred structural causes into reference FREE space.
  // This consumes no LiDAR evidence and therefore cannot double-count a ray.
  bool ProjectStructuralHazards(
      const std::vector<HazardGeometry, Eigen::aligned_allocator<HazardGeometry>>& hazards,
      std::vector<ReasonCode>* reasons);

  const std::vector<OccupancyCell>& cells() const { return output_; }
  const OccupancyCell* Find(const VoxelKey& key) const;
  ProbabilityInterval PredictCauseProbability(
      const VoxelKey& key, const StableId& cause_id,
      const ros::Time& arrival_stamp) const;
  std::vector<VoxelKey> TraceRay(const Eigen::Vector3d& origin_R,
                                 const Eigen::Vector3d& endpoint_R,
                                 bool include_endpoint,
                                 bool* resource_exceeded = nullptr) const;
  bool ReferenceLineOfSightVisible(const Eigen::Vector3d& origin_R,
                                   const Eigen::Vector3d& endpoint_R) const;

 private:
  struct ReferenceCount {
    // Fractional sufficient statistics, with at most one total unit per
    // accepted frame/exposure group for a voxel.
    double occupied_groups = 0.0;
    double free_groups = 0.0;
    uint32_t exposures = 0;
    ProbabilityInterval occupancy;
    KnowledgeState knowledge = KnowledgeState::UNKNOWN;
    // True when this reference voxel intersects at least one frozen surfel
    // carrier.  Such occupancy is already represented by the structural GDU
    // mixture and must never birth a second REFERENCE_PERSISTENCE cause.
    bool structural_geometry_represented = false;
    // Populated only for a unique structural owner.  Ambiguous overlaps keep
    // this ID empty but still set structural_geometry_represented=true; every
    // intersecting GDU is then projected with its own canonical cause.
    StableId owner_cause_id;
  };
  struct CurrentCount {
    uint32_t traversed = 0;
    uint32_t endpoints = 0;
    double visibility = 0.0;
    std::set<StableId> evidence_ids;
  };
  struct EventState {
    ProbabilityInterval posterior;
    ros::Time last_stamp;
    ros::Time last_evidence_stamp;
    HazardSemantics semantics = HazardSemantics::OTHER_ENVELOPE;
    std::set<StableId> source_ids;
  };

  ProbabilityInterval BetaPosterior(double successes, double failures) const;
  ProbabilityInterval PredictEvent(const EventState& event,
                                   const ros::Time& stamp) const;
  ProbabilityInterval UpdateEventPosterior(
      const ProbabilityInterval& prior, const CurrentCount& current) const;
  std::pair<double, double> EventRates(HazardSemantics semantics) const;
  ProbabilityInterval AggregateCauses(
      const std::map<StableId, ProbabilityInterval>& causes) const;
  void RefreshOutputProbabilities();
  StableId CauseId(const VoxelKey& key) const;
  StableId GduCauseId(const StableId& gdu_id) const;
  StableId ReferenceCauseId(const VoxelKey& key) const;

  V1Config config_;
  uint32_t reference_epoch_ = 0;
  std::map<VoxelKey, ReferenceCount> reference_;
  std::map<VoxelKey, std::map<StableId, ProbabilityInterval>> total_causes_;
  std::map<VoxelKey, std::map<StableId, ProbabilityInterval>> new_causes_;
  std::map<VoxelKey, std::map<StableId, HazardSemantics>> cause_semantics_;
  std::map<VoxelKey, std::map<StableId, EventState>> event_states_;
  std::vector<OccupancyCell> output_;
};

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_OCCUPANCY_HPP
