#include "deform_monitor_v3/risk_map/match_evidence.hpp"
#include "deform_monitor_v3/risk_map/risk_math.hpp"
#include <cmath>
namespace deform_monitor_v3 { namespace risk_map {
MatchEvidenceResult MatchEvidence::Evaluate(bool matched,double pi,double range,double incidence,
 AnchorType anchor_type,double support_quality,int rank,double radius,double disp) const { MatchEvidenceResult o;
 const double q_min=std::max(p_.probability_floor,std::min(0.49,p_.q0_min));
 const double type_offset=anchor_type==AnchorType::EDGE?p_.q0_edge_offset:
                          (anchor_type==AnchorType::BAND?p_.q0_band_offset:0.0);
 o.q0=std::max(q_min,std::min(1.0-q_min,
      p_.q0_default-p_.q0_range_coefficient*range-
      p_.q0_incidence_coefficient*(1.0-incidence)-
      p_.q0_support_quality_coefficient*(1.0-std::max(0.0,std::min(1.0,support_quality)))+
      type_offset));
 const double ratio=p_.sigma_mu/std::max(radius,1e-4);
 o.q1=ClampProbability(o.q0*std::pow(1.0+ratio*ratio,-.5*std::max(1,rank)),p_.probability_floor);
 pi=ClampProbability(pi,p_.probability_floor);
 o.log_bayes_factor=matched ? std::log(o.q1/o.q0)+disp :
   std::log(ClampProbability(1.0-pi*o.q1,p_.probability_floor)/
            ClampProbability(1.0-pi*o.q0,p_.probability_floor)); return o; }
} }
