#ifndef DEFORM_MONITOR_V3_RISK_MAP_HAZARD_PROJECTOR_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_HAZARD_PROJECTOR_HPP
#include "deform_monitor_v3/risk_map/risk_types.hpp"
namespace deform_monitor_v3 { namespace risk_map {
class HazardProjector{public:explicit HazardProjector(const RiskMapParams&p=RiskMapParams()):p_(p){}
 void Project(RiskMapVoxelVector* voxels)const; private:RiskMapParams p_;};
} }
#endif
