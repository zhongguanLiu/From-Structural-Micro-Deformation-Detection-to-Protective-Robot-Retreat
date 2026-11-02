#ifndef DEFORM_MONITOR_V3_RISK_MAP_MATCH_EVIDENCE_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_MATCH_EVIDENCE_HPP
#include "deform_monitor_v3/risk_map/risk_types.hpp"
namespace deform_monitor_v3 { namespace risk_map {
struct MatchEvidenceResult { double log_bayes_factor=0; double q0=0; double q1=0; bool approximate_joint_model=true; };
class MatchEvidence {
 public: explicit MatchEvidence(const RiskMapParams&p=RiskMapParams()):p_(p){}
 MatchEvidenceResult Evaluate(bool matched,double pi_visible,double range,double incidence_cos,
                              AnchorType anchor_type,double reference_support_quality,
                              int rank,double support_radius,double displacement_llr) const;
 private: RiskMapParams p_;
};
} }
#endif
