#ifndef DEFORM_MONITOR_V3_CORE_REFERENCE_MANAGER_HPP
#define DEFORM_MONITOR_V3_CORE_REFERENCE_MANAGER_HPP

#include "deform_monitor_v3/data_types.hpp"

namespace deform_monitor_v3 {

class ReferenceManager {
public:
  void SetParams(const ReferenceParams& params);
  void UpdateReferenceStatistics(
      AnchorReferenceVector* anchors,
      const CurrentObservationVector& observations,
      AnchorStateVector* states);

private:
  ReferenceParams params_;
};

}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_CORE_REFERENCE_MANAGER_HPP
