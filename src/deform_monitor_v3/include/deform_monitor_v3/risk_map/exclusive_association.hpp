#ifndef DEFORM_MONITOR_V3_RISK_MAP_EXCLUSIVE_ASSOCIATION_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_EXCLUSIVE_ASSOCIATION_HPP
#include "deform_monitor_v3/risk_map/risk_types.hpp"
#include <unordered_set>
namespace deform_monitor_v3 { namespace risk_map {
struct AssociationBranch { uint64_t event_id=0; VoxelKey target; double prior_probability=0; double full_log_bayes_factor=0; EventSemantics semantics=EventSemantics::NONE; };
struct AssociationResult { double joint_log_bayes_factor=0; std::vector<double> responsibilities; bool valid=false; };
class ExclusiveAssociation { public:
 AssociationResult Combine(const std::vector<AssociationBranch>& branches) const;
 void BeginCycle(){consumed_.clear();} bool Claim(uint64_t id){return consumed_.insert(id).second;}
 private: std::unordered_set<uint64_t> consumed_; };
} }
#endif
