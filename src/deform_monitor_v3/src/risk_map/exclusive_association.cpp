#include "deform_monitor_v3/risk_map/exclusive_association.hpp"
#include "deform_monitor_v3/risk_map/risk_math.hpp"
#include <cmath>
namespace deform_monitor_v3 { namespace risk_map {
AssociationResult ExclusiveAssociation::Combine(const std::vector<AssociationBranch>& b) const {
 AssociationResult o;if(b.empty())return o; double sum=0;for(const auto&x:b)if(x.prior_probability>0)sum+=x.prior_probability;
 if(!(sum>0))return o;std::vector<double>a;for(const auto&x:b)a.push_back(std::log(std::max(1e-300,x.prior_probability/sum))+x.full_log_bayes_factor);
 o.joint_log_bayes_factor=LogSumExp(a);for(double x:a)o.responsibilities.push_back(std::exp(x-o.joint_log_bayes_factor));o.valid=std::isfinite(o.joint_log_bayes_factor);return o; }
} }
