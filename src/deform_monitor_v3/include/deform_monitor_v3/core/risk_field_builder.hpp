#ifndef DEFORM_MONITOR_V3_RISK_FIELD_BUILDER_HPP
#define DEFORM_MONITOR_V3_RISK_FIELD_BUILDER_HPP

#include "deform_monitor_v3/data_types.hpp"

namespace deform_monitor_v3 {

class RiskFieldBuilder {
public:
  void SetParams(const RiskVisualizationParams& params);

  RiskVoxelVector Build(const AnchorReferenceVector& anchors,
                        const RiskEvidenceVector& evidences) const;

  RiskRegionVector ExtractRegions(const RiskVoxelVector& voxels) const;

private:
  RiskVisualizationParams params_;
};

}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_RISK_FIELD_BUILDER_HPP
