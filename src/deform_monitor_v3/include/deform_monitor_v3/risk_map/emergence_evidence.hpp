#ifndef DEFORM_MONITOR_V3_RISK_MAP_EMERGENCE_EVIDENCE_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_EMERGENCE_EVIDENCE_HPP
#include "deform_monitor_v3/risk_map/risk_types.hpp"
namespace deform_monitor_v3 { namespace risk_map {
class EmergenceEvidence { public: explicit EmergenceEvidence(const RiskMapParams&p=RiskMapParams()):p_(p){}
 double Evaluate(uint32_t hit_count,double probability_observed,double expected_occupied_count) const;
 private: RiskMapParams p_; };
} }
#endif
