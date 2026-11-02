#ifndef DEFORM_MONITOR_V3_RISK_MAP_PUBLISHER_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_PUBLISHER_HPP
#include "deform_monitor_v3/risk_map/risk_types.hpp"
#include "deform_monitor_v3/ProbabilisticRiskMap.h"
#include "deform_monitor_v3/ProbabilisticRiskRegions.h"
namespace deform_monitor_v3 { namespace risk_map {
deform_monitor_v3::ProbabilisticRiskMap BuildRiskMapMessage(const RiskMapFrame& frame);
deform_monitor_v3::ProbabilisticRiskRegions BuildRiskRegionsMessage(const RiskMapFrame& frame);
} }
#endif
