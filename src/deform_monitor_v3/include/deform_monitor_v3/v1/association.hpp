#ifndef DEFORM_MONITOR_V3_V1_ASSOCIATION_HPP
#define DEFORM_MONITOR_V3_V1_ASSOCIATION_HPP

#include "deform_monitor_v3/v1/types.hpp"

namespace deform_monitor_v3 {
namespace v1 {

class AssociationModel {
 public:
  explicit AssociationModel(const V1Config& config) : config_(config) {}

  AssociationResult Evaluate(std::vector<AssociationFeature> candidates,
                             bool expected_visible) const;

 private:
  V1Config config_;
};

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_ASSOCIATION_HPP
