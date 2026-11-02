#ifndef DEFORM_MONITOR_V3_V1_UAC_FILTER_HPP
#define DEFORM_MONITOR_V3_V1_UAC_FILTER_HPP

#include "deform_monitor_v3/v1/reference_graph.hpp"
#include "deform_monitor_v3/v1/types.hpp"

#include <memory>

namespace deform_monitor_v3 {
namespace v1 {

class UacGaussianSumFilter {
 public:
  explicit UacGaussianSumFilter(const V1Config& config) : config_(config) {}

  bool Initialize(const ReferenceSnapshot& reference, const ros::Time& stamp,
                  std::vector<ReasonCode>* reasons);
  // Move the clock of a freshly initialized, still-unobserved reference
  // posterior to the first measurement stamp without inventing dynamics over
  // the ROS phase hand-off interval.  This is intentionally legal only before
  // any measurement/history has entered the filter.
  bool ReanchorFirstObservation(const ros::Time& stamp,
                                std::vector<ReasonCode>* reasons);
  bool Predict(const ros::Time& stamp, std::vector<ReasonCode>* reasons);
  bool Update(const std::vector<std::vector<AssociationMeasurementSet>>& measurements_by_gdu,
              const std::vector<FrameNuisance, Eigen::aligned_allocator<FrameNuisance>>& nuisance_by_gdu,
              const ros::Time& stamp, std::vector<ReasonCode>* reasons);

  // Return the complete DEC-030 predictive mixture at one elapsed horizon.
  // This is intentionally a copy-only operation: planner queries must never
  // advance or otherwise mutate the monitoring posterior.
  bool PredictiveMixture(
      double horizon_s,
      std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>>* mixture,
      std::vector<ReasonCode>* reasons) const;

  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>> Posterior() const;
  std::vector<InterfacePosterior, Eigen::aligned_allocator<InterfacePosterior>>
  InterfacePosteriors() const;
  // Build interface products against an already materialized full posterior
  // from this exact filter state.  Snapshot construction uses this overload so
  // interface endpoint queries do not repeat fixed-lag/MDE/QMC work that was
  // completed for the deformation product immediately beforehand.
  std::vector<InterfacePosterior, Eigen::aligned_allocator<InterfacePosterior>>
  InterfacePosteriors(
      const std::vector<GduPosterior,
          Eigen::aligned_allocator<GduPosterior>>& frozen_posterior) const;
  FieldQueryResult Query(uint32_t gdu_index, const Eigen::Vector3d& position_R) const;
  // Evaluate the current field against an already frozen posterior from this
  // filter state.  Bulk serializers use this to avoid recomputing global
  // MDE/task-power diagnostics once per spatial sample.
  FieldQueryResult QueryWithPosterior(
      uint32_t gdu_index, const Eigen::Vector3d& position_R,
      const GduPosterior& posterior) const;
  // Exact query at a frozen reference surfel.  This uses the full (including
  // interface columns) basis cached by Initialize and therefore returns the
  // same field as QueryWithPosterior at that surfel center without rebuilding
  // the reference-only metric-graph POU basis during serialization.
  FieldQueryResult QueryReferenceSurfelWithPosterior(
      uint32_t gdu_index, uint32_t surfel_index,
      const GduPosterior& posterior) const;
  // Return the immutable POU basis already computed by Initialize() for a
  // frozen reference surfel. Measurement construction uses this instead of
  // repeating metric-graph attachment and graph-distance scans every frame.
  bool ReferenceSurfelBasis(
      uint32_t gdu_index, uint32_t surfel_index, Eigen::MatrixXd* basis,
      Validity* validity, std::vector<ReasonCode>* reasons,
      double* attachment_distance_m = nullptr) const;
  FieldQueryResult QueryAtHorizon(uint32_t gdu_index,
                                  const Eigen::Vector3d& position_R,
                                  double horizon_s) const;
  double other_mass() const;
  uint32_t approximation_flags() const { return approximation_flags_; }

 private:
  // Test access is deliberately limited to selecting the retained uncached
  // reference path.  Production always enters the cached path through
  // Predict(); both paths execute the same transition algebra.
  friend struct UacGaussianSumFilterTestAccess;

  struct ObservableFrameBlock {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ros::Time stamp;
    Eigen::MatrixXd design;
    Eigen::MatrixXd independent_covariance;
    Eigen::MatrixXd reference_bias_design;
    Eigen::MatrixXd extrinsic_design;
    Eigen::MatrixXd pose_design;
    Eigen::Matrix<double, 6, 6> pose_covariance =
        Eigen::Matrix<double, 6, 6>::Zero();
  };

  struct GduState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    struct FrozenReferenceSurfelBasis {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      Eigen::MatrixXd basis;
      Validity validity = Validity::UNKNOWN;
      std::vector<ReasonCode> reasons;
      double attachment_distance_m =
          std::numeric_limits<double>::infinity();
    };

    // All members of this cache are functions only of the immutable frozen
    // ReferenceSnapshot and V1Config accepted by Initialize().  A monitoring
    // transaction must copy the mutable Gaussian/history state for atomic
    // rollback, but deep-copying these matrices on every frame is redundant.
    // Sharing a const cache therefore changes neither arithmetic nor mutation
    // order while keeping candidate-filter rollback isolation exact.
    struct FrozenReferenceCache {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      std::vector<Eigen::MatrixXd> task_bases;
      Eigen::MatrixXd task_metric;
      size_t task_quadrature_count = 0U;
      std::vector<FrozenReferenceSurfelBasis,
          Eigen::aligned_allocator<FrozenReferenceSurfelBasis>>
          surfel_bases;
      std::map<uint32_t, size_t> surfel_basis_offsets;
    };

    int coefficient_count = 0;
    // Frozen-reference quadrature geometry.  These matrices depend only on
    // the immutable ReferenceSnapshot and V1Config accepted by Initialize;
    // no Gaussian-component state enters their construction.  Keeping both
    // the individual smooth bases and their equal-surfel Gram metric avoids
    // rebuilding the same POU field once per posterior/component.
    std::shared_ptr<const FrozenReferenceCache> frozen_reference_cache;
    Eigen::Matrix3d reference_bias_prior_covariance = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, 6, 6> last_pose_prior_covariance =
        Eigen::Matrix<double, 6, 6>::Zero();
    ros::Time last_pose_stamp;
    bool has_pose_prior = false;
    StableId nuisance_correlation_id;
    std::array<Eigen::MatrixXd, 4> current_observable_information;
    std::array<double, 4> reference_information_ratio{{1.0, 1.0, 1.0, 1.0}};
    std::array<bool, 4> reference_confounding{{false, false, false, false}};
    std::array<double, 4> effective_spatial_support{{0.0, 0.0, 0.0, 0.0}};
    std::array<double, 4> effective_temporal_support{{0.0, 0.0, 0.0, 0.0}};
    std::array<std::deque<std::pair<ros::Time, double>>, 4>
        information_weight_history;
    std::array<std::deque<ObservableFrameBlock,
        Eigen::aligned_allocator<ObservableFrameBlock>>, 4>
        observable_frame_history;
    double other_mass = 0.0;
    int consecutive_source_missing = 0;
    bool current_measurement_valid = false;
    double continuous_valid_velocity_duration_s = 0.0;
    ros::Time last_valid_velocity_stamp;
    std::vector<GaussianComponent, Eigen::aligned_allocator<GaussianComponent>> components;
  };

  struct LinearKernel {
    Eigen::MatrixXd F;
    Eigen::MatrixXd Q;
    bool valid = true;
  };

  struct SharedInterfaceState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::array<double, 4> probability{{1.0, 0.0, 0.0, 0.0}};
    std::array<Eigen::Vector3d, 4> mean{{
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}};
    std::array<Eigen::Matrix3d, 4> covariance{{
        Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(),
        Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()}};
    double other_mass = 0.0;
  };

  struct TaskEventProbability {
    double displacement = 0.0;
    double current_velocity = 0.0;
    double interface_jump = 0.0;
    double event_union_without_velocity = 0.0;
    double event_union = 0.0;
  };

  struct CoherenceCompetitionGroup {
    StableId id;
    std::vector<size_t> local_interface_indices;
    std::vector<double> normalized_weights;
    double p_link = 1.0;
  };

  LinearKernel BuildKernel(StructuralState source, StructuralState destination,
                           const std::vector<InterfaceJointState>& destination_modes,
                           uint32_t gdu_index, double delta_t) const;
  std::vector<std::pair<std::vector<InterfaceJointState>, double>>
  CoherenceBranches(const GaussianComponent& parent, StructuralState destination,
                    uint32_t gdu_index,
                    double* pruned_probability) const;
  std::vector<std::pair<std::vector<InterfaceJointState>, double>>
  CoherenceBranchesFromGroups(
      const GaussianComponent& parent, StructuralState destination,
      uint32_t gdu_index,
      const std::vector<CoherenceCompetitionGroup>& groups,
      double* pruned_probability) const;
  std::vector<CoherenceCompetitionGroup> CoherenceCompetitionGroups(
      uint32_t gdu_index) const;
  GaussianComponent ApplyKernel(const GaussianComponent& parent,
                                StructuralState destination,
                                const std::vector<InterfaceJointState>& modes,
                                const LinearKernel& kernel,
                                bool assign_transient_component_id,
                                bool* valid) const;
  bool AdvancePoseNuisance(
      GduState* state, const Eigen::Matrix<double, 6, 6>& current_pose_covariance,
      const ros::Time& stamp, std::vector<ReasonCode>* reasons) const;
  bool BuildCurrentObservableInformation(
      GduState* state, const std::vector<AssociationMeasurementSet>& measurements,
      const FrameNuisance& nuisance, const ros::Time& stamp,
      std::vector<ReasonCode>* reasons) const;
  TaskEventProbability ComponentTaskScaleProbability(
      uint32_t gdu_index, const GaussianComponent& component) const;
  bool SynchronizeSharedInterfaces(std::vector<ReasonCode>* reasons);
  std::vector<GaussianComponent, Eigen::aligned_allocator<GaussianComponent>>
  RobustMeasurementUpdate(const GaussianComponent& component,
                          const FieldMeasurement& measurement,
                          double category_probability,
                          double* log_normalizer,
                          bool* numerical_failure) const;
  void MergePruneNormalize(GduState* state,
                           uint32_t* deferred_approximation_flags = nullptr);
  bool RefreshDerived(GduState* state,
                      std::vector<ReasonCode>* reasons) const;
  StableId ComponentId(const GaussianComponent& component) const;
  std::string DiscreteKey(const GaussianComponent& component) const;
  bool PredictInterval(const ros::Time& stamp, bool enforce_update_gap,
                       std::vector<ReasonCode>* reasons);
  bool PredictIntervalImpl(const ros::Time& stamp, bool enforce_update_gap,
                           bool use_exact_interval_caches,
                           bool parallelize_independent_gdus,
                           std::vector<ReasonCode>* reasons);
  bool UpdateImpl(
      const std::vector<std::vector<AssociationMeasurementSet>>&
          measurements_by_gdu,
      const std::vector<FrameNuisance,
          Eigen::aligned_allocator<FrameNuisance>>& nuisance_by_gdu,
      const ros::Time& stamp, bool parallelize_independent_gdus,
      std::vector<ReasonCode>* reasons);
  // Posterior products are GDU-local.  Production evaluates independent GDUs
  // in a bounded worker set, while the retained serial path is used only by
  // field-exact regression tests.  Both paths preserve canonical GDU order.
  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>>
  PosteriorImpl(
      bool parallelize_independent_gdus,
      size_t successful_worker_creation_limit =
          std::numeric_limits<size_t>::max()) const;
  // Interface endpoints are frozen reference surfels.  Production reuses
  // their immutable full POU bases; the generic-query path is retained as an
  // exact regression oracle and as the fallback for an unavailable cache.
  std::vector<InterfacePosterior, Eigen::aligned_allocator<InterfacePosterior>>
  InterfacePosteriorsImpl(
      const std::vector<GduPosterior,
          Eigen::aligned_allocator<GduPosterior>>& frozen_posterior,
      bool use_frozen_surfel_cache) const;
  std::vector<GduPosterior, Eigen::aligned_allocator<GduPosterior>>
  MixtureOnlyPosterior() const;
  FieldQueryResult QueryCurrent(
      uint32_t gdu_index, const Eigen::Vector3d& position_R,
      const GduPosterior* frozen_posterior) const;
  FieldQueryResult QueryCurrentWithBasis(
      uint32_t gdu_index, const Eigen::MatrixXd& basis,
      Validity basis_validity, const std::vector<ReasonCode>& basis_reasons,
      double attachment_distance_m,
      const GduPosterior* frozen_posterior) const;

  V1Config config_;
  ReferenceSnapshot reference_;
  std::vector<GduState, Eigen::aligned_allocator<GduState>> states_;
  std::vector<SharedInterfaceState,
      Eigen::aligned_allocator<SharedInterfaceState>> shared_interfaces_;
  ros::Time last_stamp_;
  bool initialized_ = false;
  bool first_observation_reanchor_available_ = false;
  uint32_t approximation_flags_ = APPROX_NONE;
};

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_UAC_FILTER_HPP
