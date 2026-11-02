#include "deform_monitor_v3/risk_map/belief_map.hpp"
#include "deform_monitor_v3/risk_map/risk_math.hpp"
#include <cmath>
namespace deform_monitor_v3 { namespace risk_map {
void BeliefMap::Reset(uint32_t e){cells_.clear();epoch_=e;}
double BeliefMap::Predict(double l,double dt,double on,double off)const{
 on=std::max(0.0,on);off=std::max(0.0,off);const double p=Sigmoid(l),sum=on+off;if(!(dt>0)||sum<=0)return l;
 const double q=-std::expm1(-sum*dt),p01=on/sum*q,p10=off/sum*q;
 return Logit(p01*(1-p)+(1-p10)*p,p_.probability_floor);}
const BeliefCell& BeliefMap::Update(const VoxelKey&key,EventSemantics sem,const EvidenceEventVector&e,const ros::Time&t){
 auto&c=cells_[key];if(c.reference_epoch!=epoch_){c=BeliefCell();c.reference_epoch=epoch_;const double deformation_sum=std::max(0.0,p_.deformation_hazard_on)+std::max(0.0,p_.deformation_hazard_off);const double novel_sum=std::max(0.0,p_.novel_hazard_on)+std::max(0.0,p_.novel_hazard_off);const double deformation_prior=deformation_sum>0.0?std::max(0.0,p_.deformation_hazard_on)/deformation_sum:p_.probability_floor;const double novel_prior=novel_sum>0.0?std::max(0.0,p_.novel_hazard_on)/novel_sum:p_.probability_floor;c.deformation_log_odds=Logit(deformation_prior,p_.probability_floor);c.novel_log_odds=Logit(novel_prior,p_.probability_floor);}
 const double deformation_dt=c.deformation_last_update.isZero()?0:(t-c.deformation_last_update).toSec();
 const double novel_dt=c.novel_last_update.isZero()?0:(t-c.novel_last_update).toSec();
 if(sem==EventSemantics::DEFORMATION||sem==EventSemantics::MIXED){c.deformation_log_odds=Predict(c.deformation_log_odds,deformation_dt,p_.deformation_hazard_on,p_.deformation_hazard_off);c.deformation_last_update=t;}
 if(sem==EventSemantics::NOVEL_OCCUPANCY||sem==EventSemantics::MIXED){c.novel_log_odds=Predict(c.novel_log_odds,novel_dt,p_.novel_hazard_on,p_.novel_hazard_off);c.novel_last_update=t;}
 std::unordered_set<uint64_t>seen;double sum=0,comp=0;for(const auto&x:e){if(!seen.insert(x.event_id).second)continue;double y=x.log_bayes_factor-comp,tmp=sum+y;comp=(tmp-sum)-y;sum=tmp;++c.evidence_count;if(x.log_bayes_factor>=0)c.positive_evidence_sum+=x.log_bayes_factor;else c.negative_evidence_sum+=x.log_bayes_factor;const double posterior_increment=2.0*(Sigmoid(x.log_bayes_factor)-0.5);const double support_weight=std::max(0.0,x.posterior_responsibility*posterior_increment);const double direction_quality=std::max(0.0,std::min(1.0,x.direction_confidence));const double w=support_weight*direction_quality;if(x.direction_R.allFinite()&&x.direction_R.norm()>1e-9&&w>0.0){Eigen::Vector3d d=x.direction_R.normalized();c.direction_second_moment+=w*d*d.transpose();c.signed_direction_sum+=w*d;c.direction_weight_sum+=w;c.direction_support_weight_sum+=support_weight;}}
 if(sem==EventSemantics::DEFORMATION)c.deformation_log_odds+=sum;else if(sem==EventSemantics::NOVEL_OCCUPANCY)c.novel_log_odds+=sum;else if(sem==EventSemantics::MIXED){c.deformation_log_odds+=sum;c.novel_log_odds+=sum;}return c;}
const BeliefCell* BeliefMap::Find(const VoxelKey&k)const{auto i=cells_.find(k);return i==cells_.end()?nullptr:&i->second;}
} }
