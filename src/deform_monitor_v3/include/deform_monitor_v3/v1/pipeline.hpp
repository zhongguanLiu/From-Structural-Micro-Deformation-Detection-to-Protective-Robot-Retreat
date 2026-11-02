#ifndef DEFORM_MONITOR_V3_V1_PIPELINE_HPP
#define DEFORM_MONITOR_V3_V1_PIPELINE_HPP

#include "deform_monitor_v3/v1/association.hpp"
#include "deform_monitor_v3/v1/occupancy.hpp"
#include "deform_monitor_v3/v1/reference_graph.hpp"
#include "deform_monitor_v3/v1/uac_filter.hpp"

#include <deque>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>

namespace deform_monitor_v3 {
namespace v1 {

// The ASSOCIATION and MEASUREMENT roles are disjoint samples of the same
// current surface.  Requiring their independently voxelized patches to own
// the exact same frozen surfel makes a formal update disappear whenever the
// two samples fall in adjacent seed cells.  Match them once, within one GDU
// and a bounded local radius, so every patch remains unique evidence.
struct DisjointPatchDescriptor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StableId id;
  uint32_t gdu_index = std::numeric_limits<uint32_t>::max();
  Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
};

using DisjointPatchDescriptors = std::vector<
    DisjointPatchDescriptor,
    Eigen::aligned_allocator<DisjointPatchDescriptor>>;

struct DisjointPatchMatchingStats {
  uint64_t distance_evaluations = 0U;
};

// Returns association-id -> measurement-id.  The result is a deterministic
// maximum-cardinality bipartite matching; candidate order is distance then
// StableId, and no patch occurs in more than one pair.
std::map<StableId, StableId> MatchDisjointPatchPairs(
    DisjointPatchDescriptors association,
    DisjointPatchDescriptors measurement,
    double maximum_distance_m,
    double tie_tolerance,
    DisjointPatchMatchingStats* stats = nullptr);

// Test-visible accounting for the formal measurement boundary.  These counts
// include only descriptors that can enter a monitorable GDU posterior; patches
// assigned to an unmonitorable reference carrier remain available to the
// diagnostic/protective explanation path but cannot participate in matching.
struct MeasurementBuildStats {
  uint64_t association_descriptor_count = 0U;
  uint64_t measurement_descriptor_count = 0U;
  uint64_t matching_distance_evaluations = 0U;
};

// Frozen-reference spatial lookup used by the online measurement builder.
// Every returned owner/exclusion decision is still accepted only after the
// same exact Euclidean-distance and StableId tie tests used by the exhaustive
// implementation; the grid only removes surfels that cannot possibly pass
// those tests.
struct PatchOwnership {
  uint32_t owner_surfel = std::numeric_limits<uint32_t>::max();
  bool cross_gdu_ambiguous = false;
};

class ReferenceSurfelSpatialIndex {
 public:
  bool Build(const ReferenceSnapshot& reference,
             const Eigen::Vector3d& grid_origin_R,
             double cell_size_m);

  // Precompute, at the frozen-reference boundary, the complete set of surfels
  // that can be within owner_radius_m of any point in each 10 cm patch cell.
  // Runtime ownership then needs one lookup and the unchanged exact distance
  // tests, rather than 27 tree lookups plus a per-query sort.
  bool BuildOwnerCandidateCache(
      const ReferenceSnapshot& reference,
      const Eigen::Vector3d& grid_origin_R,
      double patch_cell_size_m,
      double owner_radius_m,
      double tie_tolerance);

  PatchOwnership QueryOwner(
      const ReferenceSnapshot& reference,
      const Eigen::Vector3d& position_R,
      double neighborhood_radius_m,
      double tie_tolerance,
      double ownership_resolution_m) const;

  std::vector<bool> PoseControlExclusionMask(
      const ReferenceSnapshot& reference,
      uint32_t target_gdu_index,
      double radius_m) const;

 private:
  struct VoxelKeyHash {
    size_t operator()(const VoxelKey& key) const {
      size_t seed = std::hash<int32_t>()(key.x);
      seed ^= std::hash<int32_t>()(key.y) + 0x9e3779b9U +
              (seed << 6U) + (seed >> 2U);
      seed ^= std::hash<int32_t>()(key.z) + 0x9e3779b9U +
              (seed << 6U) + (seed >> 2U);
      return seed;
    }
  };

  bool NearbySurfels(const Eigen::Vector3d& position_R,
                     double radius_m,
                     std::vector<uint32_t>* surfel_indices) const;

  Eigen::Vector3d grid_origin_R_ = Eigen::Vector3d::Zero();
  double cell_size_m_ = 0.0;
  std::map<VoxelKey, std::vector<uint32_t>> surfels_by_cell_;
  std::unordered_map<VoxelKey, std::vector<uint32_t>, VoxelKeyHash>
      owner_candidates_by_patch_cell_;
  double owner_patch_cell_size_m_ = 0.0;
  double owner_search_radius_m_ = 0.0;
  bool owner_candidate_cache_valid_ = false;
  bool valid_ = false;
};

class V1Pipeline {
 public:
  explicit V1Pipeline(const V1Config& config);

  PhaseResponse SetPhase(const PhaseRequest& request);
  bool AcceptFrame(const FrameBundle& frame, V1Snapshot* published_snapshot,
                   std::vector<ReasonCode>* reasons);

  MonitoringPhase phase() const { return phase_; }
  // The active identity domain is a property of the monitoring transaction,
  // not of the last successfully frozen ReferenceSnapshot.  During the
  // MONITOR_REFERENCE fallback dwell the latter is intentionally still empty,
  // while newly accepted frame/point/role IDs must already use the new epoch.
  uint32_t reference_epoch() const { return reference_epoch_; }
  const ReferenceSnapshot& reference() const { return reference_; }
  const std::string& reference_diagnostics() const { return reference_diagnostics_; }
  const V1Snapshot& last_snapshot() const { return last_snapshot_; }
  const UacGaussianSumFilter& filter() const { return filter_; }
  // Frozen reference ray visibility is used only to decide whether a missing
  // deformation measurement was geometrically observable.  It is not a live
  // occupancy map and is never published as one by the detector.
  const OccupancyEventGraph& reference_visibility() const {
    return reference_visibility_;
  }
  // Deprecated source-compatibility alias.  The standard node does not
  // publish or update live occupancy; this object contains frozen visibility
  // evidence only.
  const OccupancyEventGraph& occupancy() const { return reference_visibility_; }
  void RecordDroppedInput() { ++dropped_input_count_; }
  void RecordResourceFailure() { ++resource_failure_count_; }
  uint64_t deadline_failure_count() const { return deadline_failure_count_; }
  uint64_t resource_failure_count() const { return resource_failure_count_; }

 private:
  friend struct V1PipelineTestAccess;

  struct CurrentPatch {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StableId id;
    VoxelKey key;
    PointRole role = PointRole::MEASUREMENT;
    Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
    Eigen::Matrix3d centroid_covariance_R = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d projector_R = Eigen::Matrix3d::Identity();
    uint32_t count = 0;
    uint32_t owner_surfel = std::numeric_limits<uint32_t>::max();
    bool cross_gdu_ambiguous = false;
  };

  std::deque<FrameBundle> CropFrames(const std::deque<FrameBundle>& frames) const;
  bool SelectHistoricalReference(std::vector<ReasonCode>* reasons);
  bool TryFallbackReference(std::vector<ReasonCode>* reasons);
  bool TryBoundedReference(bool historical, std::vector<ReasonCode>* reasons);
  bool ValidateReferenceSelfConsistency(const std::deque<FrameBundle>& frames,
                                        const ReferenceSnapshot& reference,
                                        std::vector<ReasonCode>* reasons) const;
  std::vector<CurrentPatch, Eigen::aligned_allocator<CurrentPatch>> BuildPatches(
      const FrameBundle& frame, PointRole role) const;
  void AssignPatchOwners(std::vector<CurrentPatch, Eigen::aligned_allocator<CurrentPatch>>* patches) const;
  bool PrepareMeasurementReferenceCache(std::vector<ReasonCode>* reasons);
  void ClearMeasurementReferenceCache();
  bool BuildMeasurements(
      const FrameBundle& frame,
      const UacGaussianSumFilter& predicted_filter,
      std::vector<std::vector<AssociationMeasurementSet>>* measurements_by_gdu,
      std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>* nuisance_by_gdu,
      std::vector<ObservationExplanation>* explanations,
      std::vector<ReasonCode>* reasons,
      bool include_unmonitorable_matching_for_test = false,
      MeasurementBuildStats* build_stats = nullptr) const;
  CommonHeader BuildCommonHeader(const FrameBundle& frame,
                                 uint32_t approximation_flags,
                                 const std::vector<ReasonCode>& reasons) const;
  void TrimRawBuffer(std::vector<ReasonCode>* reasons);

  V1Config config_;
  MonitoringPhase phase_ = MonitoringPhase::MAPPING_CONTEXT;
  uint32_t reference_epoch_ = 0;
  uint64_t last_accepted_frame_index_ = std::numeric_limits<uint64_t>::max();
  bool has_accepted_frame_ = false;
  ros::Time last_accepted_stamp_;
  std::set<StableId> accepted_frame_ids_;
  std::deque<StableId> accepted_frame_id_order_;
  std::deque<FrameBundle> raw_buffer_;
  std::deque<FrameBundle> reference_frames_;
  AxisAlignedRoi roi_;
  ros::Time reference_cutoff_;
  ros::Time selection_support_cutoff_;
  TriggerSource trigger_source_ = TriggerSource::HUMAN_OR_EXTERNAL_TASK;
  bool reference_from_fallback_ = false;
  bool reference_prepared_ = false;
  ros::Time bounded_last_attempt_stamp_;
  uint64_t reference_build_attempts_ = 0U;
  std::string reference_diagnostics_;
  bool first_monitor_update_pending_ = false;
  ReferenceSnapshot reference_;
  ReferenceSurfelSpatialIndex reference_owner_index_;
  ReferenceSurfelSpatialIndex pose_control_index_;
  std::vector<std::vector<bool>> pose_control_excluded_cache_;
  std::vector<std::vector<uint32_t>> gdu_surfels_by_id_cache_;
  bool measurement_reference_cache_valid_ = false;
  // The reference is immutable throughout MONITOR_UPDATE. Its O(N-surfel)
  // finite/PSD audit is completed once while preparing the frozen-reference
  // caches; per-frame validation then audits only dynamic products plus the
  // reference identity/shape contract.
  bool frozen_reference_numerically_validated_ = false;
  UacGaussianSumFilter filter_;
  OccupancyEventGraph reference_visibility_;
  AssociationModel association_;
  V1Snapshot last_snapshot_;
  uint64_t dropped_input_count_ = 0;
  uint64_t deadline_failure_count_ = 0;
  uint64_t resource_failure_count_ = 0;
};

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_PIPELINE_HPP
