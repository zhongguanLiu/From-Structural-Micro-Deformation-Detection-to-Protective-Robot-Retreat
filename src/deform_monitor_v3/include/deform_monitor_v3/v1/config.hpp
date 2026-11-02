#ifndef DEFORM_MONITOR_V3_V1_CONFIG_HPP
#define DEFORM_MONITOR_V3_V1_CONFIG_HPP

#include "deform_monitor_v3/v1/types.hpp"

namespace deform_monitor_v3 {
namespace v1 {

ConfigValidation ValidateConfig(const V1Config& config);
// Production V1.1 profile closure. ValidateConfig permits bounded synthetic
// variations for low-level tests; the ROS loader additionally requires these
// schema-defining resource and numerical constants to match the freeze.
ConfigValidation ValidateFrozenV11Profile(const V1Config& config);
std::string CanonicalConfigDigest(const V1Config& config);
V1Config SyntheticKnownConfig(const std::string& session_id);

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_CONFIG_HPP
