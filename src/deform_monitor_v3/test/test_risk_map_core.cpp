#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include "deform_monitor_v3/core/scalar_measurement_builder.hpp"
#include "deform_monitor_v3/risk_map/channel_adapter.hpp"
#include "deform_monitor_v3/risk_map/belief_map.hpp"
#include "deform_monitor_v3/risk_map/current_occupancy.hpp"
#include "deform_monitor_v3/risk_map/deformation_evidence.hpp"
#include "deform_monitor_v3/risk_map/exclusive_association.hpp"
#include "deform_monitor_v3/risk_map/match_evidence.hpp"
#include "deform_monitor_v3/risk_map/pose_covariance_model.hpp"
#include "deform_monitor_v3/risk_map/reference_free_space.hpp"
#include "deform_monitor_v3/risk_map/risk_math.hpp"

using namespace deform_monitor_v3;
using namespace deform_monitor_v3::risk_map;

TEST(RiskMapMath, StableProbabilityOperations) {
  EXPECT_NEAR(Sigmoid(Logit(0.123)), 0.123, 1e-12);
  EXPECT_NEAR(LogSumExp({-1000.0, -1000.0}), -1000.0 + std::log(2.0), 1e-12);
}

TEST(RiskMapChannels, UsesSemanticAxisTypeAndUncensoredRiskResidual) {
  ObservationParams observation_params; observation_params.min_support_scalar=1;
  observation_params.tau_nis_scalar=0.1;
  NoiseParams noise_params; ScalarMeasurementBuilder builder;
  builder.SetParams(observation_params,noise_params);
  AnchorReference anchor; anchor.id=7; anchor.type=AnchorType::PLANE;
  anchor.normal_R=Eigen::Vector3d::UnitZ(); anchor.basis_R=Eigen::Matrix3d::Identity();
  LocalSupportData support; support.valid=true; support.support_count=10;
  support.centroid_R=Eigen::Vector3d(0,0,1); support.centroid_cov.setZero();
  support.view_dir_R=Eigen::Vector3d::UnitX(); support.range=1.0;
  PoseCov6D pose;
  EXPECT_TRUE(builder.BuildMeasurements(anchor,support,pose,Eigen::Vector3d::Zero(),true).empty());
  const auto raw=builder.BuildMeasurements(anchor,support,pose,Eigen::Vector3d::Zero(),false);
  ASSERT_EQ(raw.size(),1u);

  CurrentObservation observation; observation.anchor_id=anchor.id;observation.comparable=true; observation.scalars=raw;
  ScalarMeasurement band_axis=raw.front(); band_axis.type=3; observation.scalars.push_back(band_axis);
  AnchorReferenceVector anchors{anchor}; CurrentObservationVector observations{observation};
  FrozenMode mode; mode.anchor_id=anchor.id; FrozenModeVector modes{mode};
  const ChannelFrame frame=ChannelAdapter().BuildFrame(
      anchors,observations,pose,Eigen::Vector3d::Zero(),ros::Time(1.0),2,modes);
  ASSERT_EQ(frame.channels.size(),2u);
  EXPECT_EQ(frame.channels[0].key.axis_index,0u);
  EXPECT_EQ(frame.channels[1].key.axis_index,3u);
}

TEST(RiskMapEmergence, OccupiedRateDoesNotDependOnObservedHitCount) {
  RiskMapParams params; params.voxel_size=1.0; params.occupied_rate_scale=2.0;
  auto build=[&](int copies){
    ObservationFrame frame; frame.lidar_origin_R=Eigen::Vector3d::Zero();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    for(int i=0;i<copies;++i){pcl::PointXYZI p;p.x=1.2F;p.y=.2F;p.z=.2F;cloud->push_back(p);}frame.cloud=cloud;
    ObservationFrameDeque frames{frame};CurrentOccupancyFrame occupancy(params);occupancy.Build(frames);
    return *occupancy.Find(VoxelKey{1,0,0});};
  const auto one=build(1),many=build(5);
  EXPECT_EQ(one.hit_count,1u);EXPECT_EQ(many.hit_count,5u);
  EXPECT_DOUBLE_EQ(one.expected_hit_count,many.expected_hit_count);
}

TEST(RiskMapMatch, MatchedAndMissingBranchesHaveConsistentSigns) {
  RiskMapParams params;MatchEvidence model(params);
  const auto matched=model.Evaluate(true,.8,5.0,.9,AnchorType::PLANE,1.0,1,.1,0.0);
  const auto missing=model.Evaluate(false,.8,5.0,.9,AnchorType::PLANE,1.0,1,.1,0.0);
  EXPECT_LT(matched.q1,matched.q0);EXPECT_LT(matched.log_bayes_factor,0.0);
  EXPECT_GT(missing.log_bayes_factor,0.0);
  const auto weak_edge=model.Evaluate(false,.8,5.0,.9,AnchorType::EDGE,0.0,2,.1,0.0);
  EXPECT_LT(weak_edge.q0,missing.q0);
}

TEST(RiskMapGeometry, DdaHasUniqueOrderedCellsAcrossNegativeCoordinates) {
  RiskMapParams params; params.voxel_size = 1.0;
  ReferenceFreeSpaceMap map(params);
  const auto cells = map.TraceRay(Eigen::Vector3d(-1.2, -0.2, 0.2),
                                  Eigen::Vector3d(2.2, -0.2, 0.2), true);
  ASSERT_EQ(cells.size(), 5u);
  for (size_t i = 1; i < cells.size(); ++i) EXPECT_NE(cells[i], cells[i - 1]);
  EXPECT_EQ(cells.front().x, -2); EXPECT_EQ(cells.back().x, 2);
}

TEST(RiskMapBelief, EventIdIsConsumedOncePerUpdate) {
  RiskMapParams params; BeliefMap beliefs(params); beliefs.Reset(7);
  EvidenceEvent event; event.event_id = 42; event.log_bayes_factor = 1.0;
  EvidenceEventVector events{event, event};
  const BeliefCell& cell = beliefs.Update(VoxelKey{0,0,0}, EventSemantics::DEFORMATION,
                                          events, ros::Time(10.0));
  EXPECT_EQ(cell.evidence_count, 1u);
}

TEST(RiskMapBelief, UnidentifiedSymmetricModeDoesNotCreateDirection) {
  RiskMapParams params;BeliefMap beliefs(params);beliefs.Reset(1);
  EvidenceEvent event;event.event_id=1;event.log_bayes_factor=2.0;
  event.direction_R=Eigen::Vector3d::UnitZ();event.direction_confidence=0.0;
  const auto& unidentified=beliefs.Update(VoxelKey{0,0,0},EventSemantics::DEFORMATION,
                                          EvidenceEventVector{event},ros::Time(1.0));
  EXPECT_DOUBLE_EQ(unidentified.direction_weight_sum,0.0);
  event.event_id=2;event.direction_confidence=1.0;
  const auto& identified=beliefs.Update(VoxelKey{0,0,0},EventSemantics::DEFORMATION,
                                        EvidenceEventVector{event},ros::Time(2.0));
  EXPECT_GT(identified.direction_weight_sum,0.0);
}

TEST(RiskMapCovariance, MarginalBlocksAndRankOneEvidenceMatchDenseGaussian) {
  RiskMapParams params;
  params.pose_temporal_model = PoseTemporalModel::INDEPENDENT_POSE;
  params.sigma_mu = 0.2;
  EvidenceBlock block; block.complete = true; block.reference_epoch = 3;
  for (int t = 0; t < 2; ++t) {
    ChannelFrame frame; frame.stamp = ros::Time(10.0 + t);
    frame.pose_cov.stamp = frame.stamp;
    frame.pose_cov.Sigma_xi = (0.01 + 0.01*t) * Eigen::Matrix<double,6,6>::Identity();
    ChannelObservation row; row.key.anchor_id = 9; row.key.reference_epoch = 3;
    row.g << 0.1, -0.2, 0.05, 0.3, 0.1, -0.1;
    row.psi = 0.6 - 0.2*t; row.z = 0.15 + 0.05*t;
    row.independent_variance = 0.04 + 0.01*t;
    frame.channels.push_back(row); block.frames.push_back(frame);
  }
  PoseCovarianceResult pose = PoseCovarianceModel(params).Build(block.frames);
  ASSERT_TRUE(pose.valid);
  const Eigen::Matrix<double,6,6> marginal0=pose.covariance.block<6,6>(0,0);
  const Eigen::Matrix<double,6,6> marginal1=pose.covariance.block<6,6>(6,6);
  const Eigen::Matrix<double,6,6> cross=pose.covariance.block<6,6>(0,6);
  EXPECT_TRUE(marginal0.isApprox(block.frames[0].pose_cov.Sigma_xi,1e-12));
  EXPECT_TRUE(marginal1.isApprox(block.frames[1].pose_cov.Sigma_xi,1e-12));
  EXPECT_NEAR(cross.norm(),0.0,1e-14);

  const EvidenceResult result = DeformationEvidence(params).Evaluate(block, 9);
  ASSERT_TRUE(result.valid);
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2,12);
  A.block<1,6>(0,0) = block.frames[0].channels[0].g.transpose();
  A.block<1,6>(1,6) = block.frames[1].channels[0].g.transpose();
  Eigen::Matrix2d sigma0 = A*pose.covariance*A.transpose();
  sigma0(0,0) += 0.04; sigma0(1,1) += 0.05;
  Eigen::Vector2d psi(0.6,0.4), z(0.15,0.20);
  const Eigen::Matrix2d sigma1 = sigma0 + params.sigma_mu*params.sigma_mu*psi*psi.transpose();
  Eigen::LLT<Eigen::Matrix2d> l0(sigma0), l1(sigma1);
  const double logdet0 = 2.0*l0.matrixL().toDenseMatrix().diagonal().array().log().sum();
  const double logdet1 = 2.0*l1.matrixL().toDenseMatrix().diagonal().array().log().sum();
  const double dense = -0.5*((logdet1-logdet0) + z.dot(l1.solve(z))-z.dot(l0.solve(z)));
  EXPECT_NEAR(result.log_bayes_factor, dense, 1e-11);
}

TEST(RiskMapCovariance, ReferenceFitUncertaintyIsSharedAcrossTime) {
  RiskMapParams params;params.pose_temporal_model=PoseTemporalModel::INDEPENDENT_POSE;
  params.sigma_mu=.2;EvidenceBlock block;block.complete=true;
  for(int t=0;t<2;++t){ChannelFrame frame;frame.stamp=ros::Time(1.0+t);
    ChannelObservation row;row.key.anchor_id=4;row.independent_variance=.04;
    row.h_R=Eigen::Vector3d::UnitZ();row.reference_covariance_group=0;
    row.reference_covariance_R=Eigen::Matrix3d::Zero();row.reference_covariance_R(2,2)=.01;
    row.psi=1.0;row.z=.1+.02*t;frame.channels.push_back(row);block.frames.push_back(frame);}
  const EvidenceResult result=DeformationEvidence(params).Evaluate(block,4);ASSERT_TRUE(result.valid);
  Eigen::Matrix2d sigma0;sigma0<<.05,.01,.01,.05;
  Eigen::Vector2d psi=Eigen::Vector2d::Ones(),z(.1,.12);
  const Eigen::Matrix2d sigma1=sigma0+params.sigma_mu*params.sigma_mu*psi*psi.transpose();
  Eigen::LLT<Eigen::Matrix2d> l0(sigma0),l1(sigma1);
  const double dense=-.5*(2.0*l1.matrixL().toDenseMatrix().diagonal().array().log().sum()-
      2.0*l0.matrixL().toDenseMatrix().diagonal().array().log().sum()+
      z.dot(l1.solve(z))-z.dot(l0.solve(z)));
  EXPECT_NEAR(result.log_bayes_factor,dense,1e-11);
}

TEST(RiskMapAssociation, NormalizesAndIsPermutationInvariant) {
  ExclusiveAssociation association;
  std::vector<AssociationBranch> branches(3);
  branches[0].prior_probability=.2; branches[0].full_log_bayes_factor=-.3;
  branches[1].prior_probability=.5; branches[1].full_log_bayes_factor=1.2;
  branches[2].prior_probability=.3; branches[2].full_log_bayes_factor=.4;
  const AssociationResult first = association.Combine(branches);
  std::reverse(branches.begin(), branches.end());
  const AssociationResult second = association.Combine(branches);
  ASSERT_TRUE(first.valid); ASSERT_TRUE(second.valid);
  EXPECT_NEAR(first.joint_log_bayes_factor, second.joint_log_bayes_factor, 1e-14);
  EXPECT_NEAR(std::accumulate(first.responsibilities.begin(), first.responsibilities.end(), 0.0),
              1.0, 1e-14);
}

TEST(RiskMapBelief, ContinuousTimePredictionHasSemigroupProperty) {
  RiskMapParams params; params.deformation_hazard_on=.03; params.deformation_hazard_off=.2;
  BeliefMap direct(params), split(params); direct.Reset(1); split.Reset(1);
  EvidenceEvent event; event.event_id=1; event.log_bayes_factor=2.0;
  EvidenceEventVector evidence{event}, none;
  direct.Update(VoxelKey{1,2,3},EventSemantics::DEFORMATION,evidence,ros::Time(1.0));
  split.Update(VoxelKey{1,2,3},EventSemantics::DEFORMATION,evidence,ros::Time(1.0));
  const double direct_log_odds = direct.Update(VoxelKey{1,2,3},EventSemantics::DEFORMATION,
                                                none,ros::Time(3.0)).deformation_log_odds;
  split.Update(VoxelKey{1,2,3},EventSemantics::DEFORMATION,none,ros::Time(2.0));
  const double split_log_odds = split.Update(VoxelKey{1,2,3},EventSemantics::DEFORMATION,
                                              none,ros::Time(3.0)).deformation_log_odds;
  EXPECT_NEAR(direct_log_odds, split_log_odds, 1e-12);
}

TEST(RiskMapBelief, EventTypesKeepIndependentTransitionClocks) {
  RiskMapParams params; params.novel_hazard_on=.03; params.novel_hazard_off=.2;
  BeliefMap baseline(params), interleaved(params); baseline.Reset(1); interleaved.Reset(1);
  EvidenceEvent event; event.event_id=1; event.log_bayes_factor=2.0;
  EvidenceEventVector evidence{event}, none;
  baseline.Update(VoxelKey{1,2,3},EventSemantics::NOVEL_OCCUPANCY,evidence,ros::Time(1.0));
  interleaved.Update(VoxelKey{1,2,3},EventSemantics::NOVEL_OCCUPANCY,evidence,ros::Time(1.0));
  interleaved.Update(VoxelKey{1,2,3},EventSemantics::DEFORMATION,none,ros::Time(2.0));
  const double expected=baseline.Update(VoxelKey{1,2,3},EventSemantics::NOVEL_OCCUPANCY,
                                        none,ros::Time(3.0)).novel_log_odds;
  const double actual=interleaved.Update(VoxelKey{1,2,3},EventSemantics::NOVEL_OCCUPANCY,
                                         none,ros::Time(3.0)).novel_log_odds;
  EXPECT_NEAR(expected,actual,1e-12);
}

TEST(RiskMapBelief, ZeroHazardsHaveFiniteExplicitPrior) {
  RiskMapParams params;params.deformation_hazard_on=0.0;params.deformation_hazard_off=0.0;
  BeliefMap beliefs(params);beliefs.Reset(1);EvidenceEventVector none;
  const auto& cell=beliefs.Update(VoxelKey{0,0,0},EventSemantics::DEFORMATION,none,ros::Time(1.0));
  EXPECT_TRUE(std::isfinite(cell.deformation_log_odds));
  EXPECT_NEAR(Sigmoid(cell.deformation_log_odds),params.probability_floor,1e-12);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
