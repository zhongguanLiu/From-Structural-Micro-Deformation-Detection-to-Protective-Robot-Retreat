#include "deform_monitor_v3/risk_map/risk_map_pipeline.hpp"
#include "deform_monitor_v3/risk_map/risk_math.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
namespace deform_monitor_v3 { namespace risk_map {
void RiskMapPipeline::SetParams(const RiskMapParams&p){p_=p;window_.SetLimits(p.evidence_block_frames,p.max_block_duration);deformation_.SetParams(p);reference_.SetParams(p);current_.SetParams(p);visibility_.SetParams(p);match_=MatchEvidence(p);emergence_=EmergenceEvidence(p);beliefs_.SetParams(p);projector_=HazardProjector(p);}
void RiskMapPipeline::Reset(uint32_t e){epoch_=e;frame_index_=0;event_id_=1;last_input_stamp_=ros::Time();active_block_modes_.clear();window_.Reset(e);reference_.Reset(e);beliefs_.Reset(e);}
void RiskMapPipeline::BuildReference(const ObservationFrameDeque& f,uint32_t e){Reset(e);for(const auto&x:f)reference_.IntegrateInitializationFrame(x);reference_.Freeze();}
bool RiskMapPipeline::Process(const ObservationFrameDeque& frames,const std::vector<CurrentObservationVector>& observations,
 const AnchorReferenceVector&anchors,const AnchorStateVector&states,uint32_t epoch,RiskMapFrame*out){
 if(!out||frames.empty()||observations.size()!=frames.size()||!reference_.frozen())return false;if(epoch!=epoch_)return false;
 bool mode_layout_matches=active_block_modes_.size()==anchors.size();if(mode_layout_matches)for(size_t i=0;i<anchors.size();++i)if(active_block_modes_[i].anchor_id!=anchors[i].id){mode_layout_matches=false;break;}if(!mode_layout_matches&&window_.size()>0)window_.DiscardIncomplete();if(window_.size()==0&&!mode_layout_matches)active_block_modes_=ChannelAdapter::FreezeModes(anchors,states,frames.front().stamp,Eigen::Vector3d(0,0,-1));if(active_block_modes_.empty()&& !anchors.empty())active_block_modes_=ChannelAdapter::FreezeModes(anchors,states,frames.front().stamp,Eigen::Vector3d(0,0,-1));const FrozenModeVector modes=active_block_modes_;EvidenceBlock block;bool emitted=false;
 for(size_t t=0;t<frames.size();++t){if(!last_input_stamp_.isZero()&&frames[t].stamp<=last_input_stamp_)continue;ChannelFrame cf=adapter_.BuildFrame(anchors,observations[t],frames[t].pose_cov,frames[t].lidar_origin_R,frames[t].stamp,epoch,modes);last_input_stamp_=frames[t].stamp;if(window_.Push(std::move(cf),modes,&block)){emitted=true;break;}}
 if(!emitted)return false;active_block_modes_.clear();*out=RiskMapFrame();out->frame_index=frame_index_++;out->stamp=block.end_stamp;out->voxel_size=p_.voxel_size;out->reference_epoch=epoch;out->model_version=p_.model_version;out->calibration_version=p_.calibration_version;out->probabilities_calibrated=p_.calibration_version!="uncalibrated";out->approximate_joint_model=true;
 size_t end_index=frames.size()-1;for(size_t t=0;t<frames.size();++t)if(frames[t].stamp==block.end_stamp){end_index=t;break;}
 ObservationFrameDeque current_cycle;current_cycle.push_back(frames[end_index]);current_.Build(current_cycle);association_.BeginCycle();
 const CurrentObservationVector& end_observations=observations[end_index];const ObservationFrame& end_frame=frames[end_index];
 std::unordered_map<int,const CurrentObservation*> end_observation_by_anchor;for(const auto&observation:end_observations)if(observation.anchor_id>=0)end_observation_by_anchor[observation.anchor_id]=&observation;
 std::unordered_map<VoxelKey,uint64_t,VoxelKeyHash> occupancy_event_ids;for(const auto&item:current_.voxels())occupancy_event_ids[item.first]=event_id_++;
 std::unordered_map<int,size_t> anchor_index;for(size_t i=0;i<anchors.size();++i)anchor_index[anchors[i].id]=i;
 AlignedUnorderedMap<VoxelKey,EvidenceEventVector,VoxelKeyHash> structural_events;
 std::unordered_map<VoxelKey,double,VoxelKeyHash> structural_visibility;
 // One deterministic anchor owns each structural voxel.  This prevents
 // anchors backed by overlapping raw points from being summed as if their
 // Bayes factors were conditionally independent.
 std::unordered_map<VoxelKey,int,VoxelKeyHash> structural_owner;
 for(const auto& a:anchors){const VoxelKey key=reference_.Key(a.center_R);const auto it=structural_owner.find(key);if(it==structural_owner.end()||a.id<it->second)structural_owner[key]=a.id;}
 // Assign each possible migrated return to one missing anchor using geometry
 // only.  The ownership is fixed before any candidate LLR is evaluated, so a
 // raw occupancy event cannot enter two association mixtures or later reappear
 // as an independent novel event.
 std::unordered_map<VoxelKey,int,VoxelKeyHash> migration_owner;
 std::unordered_map<VoxelKey,double,VoxelKeyHash> migration_owner_distance;
 for(const auto& hit:current_.voxels()){
   if(hit.second.hit_count==0)continue;const ReferenceVoxel*rv=reference_.Find(hit.first);if(!rv||rv->state!=KnowledgeState::FREE)continue;
   for(const auto& a:anchors){const VoxelKey structural_key=reference_.Key(a.center_R);if(structural_owner[structural_key]!=a.id)continue;const auto observation=end_observation_by_anchor.find(a.id);const bool matched=observation!=end_observation_by_anchor.end()&&observation->second->comparable;if(matched)continue;const double distance=(reference_.Center(hit.first)-a.center_R).norm();if(distance>p_.migration_search_radius)continue;const auto owner=migration_owner.find(hit.first);if(owner==migration_owner.end()||distance<migration_owner_distance[hit.first]-1.0e-12||(std::abs(distance-migration_owner_distance[hit.first])<=1.0e-12&&a.id<owner->second)){migration_owner[hit.first]=a.id;migration_owner_distance[hit.first]=distance;}}
 }
 for(const auto&a:anchors){const VoxelKey key=reference_.Key(a.center_R);if(structural_owner[key]!=a.id)continue;EvidenceResult d=deformation_.Evaluate(block,a.id);auto ii=anchor_index.find(a.id);if(ii==anchor_index.end())continue;const size_t i=ii->second;bool matched=false;int rank=a.type==AnchorType::PLANE?1:2;const auto observation=end_observation_by_anchor.find(a.id);if(observation!=end_observation_by_anchor.end()){matched=observation->second->comparable;if(observation->second->dof_obs>0)rank=observation->second->dof_obs;}VisibilityEstimate vis=visibility_.Estimate(end_frame.lidar_origin_R,a.center_R,a.normal_R,current_);MatchEvidenceResult m=match_.Evaluate(matched,vis.probability,vis.range,vis.incidence_cos,a.type,a.ref_quality,rank,a.support_radius,d.valid?d.log_bayes_factor:0.0);EvidenceEvent e;e.event_id=event_id_++;e.cell=key;e.semantics=EventSemantics::DEFORMATION;e.log_bayes_factor=m.log_bayes_factor;e.direction_R=modes[i].direction_R;e.direction_confidence=modes[i].direction_confidence;e.stamp=block.end_stamp;if(matched&&!d.valid){structural_events[key];structural_visibility[key]=std::max(structural_visibility[key],vis.probability);continue;}
   if(!matched){std::vector<AssociationBranch> branches;AssociationBranch gone;gone.prior_probability=p_.gone_prior;gone.full_log_bayes_factor=m.log_bayes_factor;gone.semantics=EventSemantics::DEFORMATION;branches.push_back(gone);std::vector<double> raw;std::vector<VoxelKey> candidates;for(const auto&hit:current_.voxels()){if(hit.second.hit_count==0)continue;const auto owner=migration_owner.find(hit.first);if(owner==migration_owner.end()||owner->second!=a.id)continue;const double distance=migration_owner_distance.at(hit.first);candidates.push_back(hit.first);raw.push_back(std::exp(-distance/std::max(1e-6,p_.migration_distance_scale)));association_.Claim(occupancy_event_ids.at(hit.first));}double mass=0;for(double w:raw)mass+=w;for(size_t c=0;c<candidates.size();++c){const auto*hit=current_.Find(candidates[c]);AssociationBranch moved;moved.target=candidates[c];moved.event_id=occupancy_event_ids.at(candidates[c]);moved.prior_probability=(1.0-p_.gone_prior)*(mass>0?raw[c]/mass:0);moved.full_log_bayes_factor=m.log_bayes_factor+emergence_.Evaluate(hit->hit_count,hit->visibility_probability,std::max(p_.probability_floor,hit->expected_hit_count));moved.semantics=EventSemantics::DEFORMATION;branches.push_back(moved);}AssociationResult joint=association_.Combine(branches);if(joint.valid){e.log_bayes_factor=joint.joint_log_bayes_factor;if(joint.responsibilities.size()>1){size_t best=1;for(size_t c=2;c<joint.responsibilities.size();++c)if(joint.responsibilities[c]>joint.responsibilities[best])best=c;if(joint.responsibilities[best]>joint.responsibilities[0]){const VoxelKey&target=branches[best].target;e.direction_R=reference_.Center(target)-a.center_R;if(e.direction_R.norm()>1e-9)e.direction_R.normalize();e.posterior_responsibility=joint.responsibilities[best];e.direction_confidence=1.0;}}}}
   structural_events[key].push_back(e);structural_visibility[key]=std::max(structural_visibility[key],vis.probability);}
 for(const auto&item:structural_events){const VoxelKey&key=item.first;const BeliefCell&b=beliefs_.Update(key,EventSemantics::DEFORMATION,item.second,block.end_stamp);RiskMapVoxel v;v.key=key;v.center_R=reference_.Center(key);v.p_deformation=Sigmoid(b.deformation_log_odds);v.deformation_log_odds=b.deformation_log_odds;if(item.second.empty())v.last_log_bayes_factor=std::numeric_limits<float>::quiet_NaN();else for(const auto&e:item.second)v.last_log_bayes_factor+=e.log_bayes_factor;v.visibility_probability=structural_visibility[key];v.knowledge_state=KnowledgeState::OCCUPIED;v.semantics=EventSemantics::DEFORMATION;if(b.direction_weight_sum>0){Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(b.direction_second_moment);Eigen::Vector3d principal=es.eigenvectors().col(2);if(b.signed_direction_sum.norm()>1.0e-12&&principal.dot(b.signed_direction_sum)<0.0)principal=-principal;v.principal_direction=principal.cast<float>();const double coherence=es.eigenvalues()[2]>0?(es.eigenvalues()[2]-es.eigenvalues()[1])/es.eigenvalues()[2]:0;const double mean_quality=b.direction_support_weight_sum>0?b.direction_weight_sum/b.direction_support_weight_sum:0;v.direction_confidence=coherence*mean_quality;}out->voxels.push_back(v);}
 for(const auto&item:reference_.voxels()){const VoxelKey&key=item.first;const ReferenceVoxel&r=item.second;if(r.state!=KnowledgeState::FREE||structural_events.count(key))continue;const CurrentVoxelObservation*c=current_.Find(key);uint32_t hits=c?c->hit_count:0;double pi=c?c->visibility_probability:0;if(!c)continue;const uint64_t occupancy_id=occupancy_event_ids.at(key);if(!association_.Claim(occupancy_id))continue;double llr=emergence_.Evaluate(hits,pi,std::max(p_.probability_floor,c->expected_hit_count));EvidenceEvent e;e.event_id=occupancy_id;e.cell=key;e.semantics=EventSemantics::NOVEL_OCCUPANCY;e.log_bayes_factor=llr;e.stamp=block.end_stamp;EvidenceEventVector ev{e};const BeliefCell&b=beliefs_.Update(key,e.semantics,ev,block.end_stamp);RiskMapVoxel v;v.key=key;v.center_R=reference_.Center(key);v.p_novel_occupancy=Sigmoid(b.novel_log_odds);v.novel_log_odds=b.novel_log_odds;v.last_log_bayes_factor=llr;v.visibility_probability=pi;v.knowledge_state=KnowledgeState::FREE;v.semantics=EventSemantics::NOVEL_OCCUPANCY;out->voxels.push_back(v);}
 std::unordered_map<VoxelKey,bool,VoxelKeyHash> output_keys;for(const auto&voxel:out->voxels)output_keys[voxel.key]=true;
 for(const auto&item:reference_.voxels()){if(output_keys.count(item.first))continue;RiskMapVoxel v;v.key=item.first;v.center_R=reference_.Center(item.first);v.knowledge_state=item.second.state;v.semantics=item.second.state==KnowledgeState::UNKNOWN?EventSemantics::UNKNOWN:EventSemantics::NONE;out->voxels.push_back(v);}projector_.Project(&out->voxels);ExtractRegions(out);return true;}
void RiskMapPipeline::ExtractRegions(RiskMapFrame* f) const {
  std::unordered_map<VoxelKey,size_t,VoxelKeyHash> active;
  for(size_t i=0;i<f->voxels.size();++i)
    if(f->voxels[i].decision_cost>=p_.region_cost_threshold)active[f->voxels[i].key]=i;
  std::unordered_map<VoxelKey,bool,VoxelKeyHash> seen;
  int id=0;
  const int neighbors[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
  for(const auto& start:active){
    if(seen[start.first])continue;
    std::queue<VoxelKey> q;q.push(start.first);seen[start.first]=true;
    std::vector<size_t> ids;
    while(!q.empty()){
      const VoxelKey key=q.front();q.pop();ids.push_back(active[key]);
      for(const auto& offset:neighbors){const VoxelKey next{key.x+offset[0],key.y+offset[1],key.z+offset[2]};
        if(active.count(next)&&!seen[next]){seen[next]=true;q.push(next);}}
    }
    if(static_cast<int>(ids.size())<p_.region_min_voxels)continue;
    RiskMapRegion region;region.id=id;
    region.bbox_min_R=Eigen::Vector3d::Constant(1e30);
    region.bbox_max_R=Eigen::Vector3d::Constant(-1e30);
    double cost_sum=0.0,unknown_count=0.0,direction_weight=0.0;
    Eigen::Matrix3d direction_moment=Eigen::Matrix3d::Zero();
    Eigen::Vector3d signed_direction=Eigen::Vector3d::Zero();
    for(const size_t index:ids){
      auto& voxel=f->voxels[index];voxel.region_id=id;region.center_R+=voxel.center_R;
      region.bbox_min_R=region.bbox_min_R.cwiseMin(voxel.center_R);
      region.bbox_max_R=region.bbox_max_R.cwiseMax(voxel.center_R);
      region.peak_decision_cost=std::max<double>(region.peak_decision_cost,voxel.decision_cost);
      if(std::isfinite(voxel.p_deformation))region.peak_deformation_probability=
          std::max<double>(region.peak_deformation_probability,voxel.p_deformation);
      if(std::isfinite(voxel.p_novel_occupancy))region.peak_novel_probability=
          std::max<double>(region.peak_novel_probability,voxel.p_novel_occupancy);
      cost_sum+=voxel.decision_cost;unknown_count+=voxel.knowledge_state==KnowledgeState::UNKNOWN;
      if(voxel.principal_direction.allFinite()&&voxel.principal_direction.norm()>1.0e-6F){
        const Eigen::Vector3d direction=voxel.principal_direction.cast<double>().normalized();
        const double weight=std::max(0.0,static_cast<double>(voxel.direction_confidence))*
                            std::max(0.0,static_cast<double>(voxel.decision_cost));
        direction_moment+=weight*direction*direction.transpose();signed_direction+=weight*direction;
        direction_weight+=weight;}
    }
    region.voxel_count=ids.size();region.center_R/=ids.size();
    region.mean_decision_cost=cost_sum/ids.size();region.unknown_fraction=unknown_count/ids.size();
    region.semantics=region.peak_deformation_probability>=region.peak_novel_probability
                         ?EventSemantics::DEFORMATION:EventSemantics::NOVEL_OCCUPANCY;
    if(direction_weight>0.0){Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(direction_moment);
      Eigen::Vector3d principal=eig.eigenvectors().col(2);if(principal.dot(signed_direction)<0.0)principal=-principal;
      region.principal_direction_R=principal;const auto values=eig.eigenvalues();
      region.direction_confidence=values[2]>0.0?(values[2]-values[1])/values[2]:0.0;}
    f->regions.push_back(region);++id;
  }
}
} }
