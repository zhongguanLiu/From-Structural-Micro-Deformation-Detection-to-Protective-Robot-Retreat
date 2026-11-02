#include "deform_monitor_v3/risk_map/emergence_evidence.hpp"
#include "deform_monitor_v3/risk_map/risk_math.hpp"
namespace deform_monitor_v3 { namespace risk_map {
double EmergenceEvidence::Evaluate(uint32_t k,double pi,double expected) const {
 const double l0=std::max(p_.probability_floor,p_.clutter_rate);
 const double l1=std::max(l0+p_.probability_floor,expected);
 return LogMixtureObservedCount(k,l1,pi,p_.probability_floor)-
        LogMixtureObservedCount(k,l0,pi,p_.probability_floor); }
} }
