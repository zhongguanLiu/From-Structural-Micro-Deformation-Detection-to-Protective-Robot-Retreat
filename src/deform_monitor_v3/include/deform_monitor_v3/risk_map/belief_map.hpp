#ifndef DEFORM_MONITOR_V3_RISK_MAP_BELIEF_MAP_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_BELIEF_MAP_HPP
#include "deform_monitor_v3/risk_map/risk_types.hpp"
#include <unordered_set>
namespace deform_monitor_v3 { namespace risk_map {
class BeliefMap {
 public: explicit BeliefMap(const RiskMapParams&p=RiskMapParams()):p_(p){}
 void SetParams(const RiskMapParams&p){p_=p;} void Reset(uint32_t epoch);
 const BeliefCell& Update(const VoxelKey& key,EventSemantics semantics,const EvidenceEventVector& events,
                          const ros::Time& stamp);
 const BeliefCell* Find(const VoxelKey& key) const;
 const AlignedUnorderedMap<VoxelKey,BeliefCell,VoxelKeyHash>& cells()const{return cells_;}
 private: double Predict(double log_odds,double dt,double on,double off)const;
 RiskMapParams p_; uint32_t epoch_=0; AlignedUnorderedMap<VoxelKey,BeliefCell,VoxelKeyHash> cells_; };
} }
#endif
