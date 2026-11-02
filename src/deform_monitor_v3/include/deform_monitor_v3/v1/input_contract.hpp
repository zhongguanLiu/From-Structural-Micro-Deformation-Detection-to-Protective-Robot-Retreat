#ifndef DEFORM_MONITOR_V3_V1_INPUT_CONTRACT_HPP
#define DEFORM_MONITOR_V3_V1_INPUT_CONTRACT_HPP

#include "deform_monitor_v3/v1/types.hpp"

#include <sensor_msgs/PointCloud2.h>

#include <cstddef>
#include <string>

namespace deform_monitor_v3 {
namespace v1 {

// Strict FAST-LIO PointXYZINormal wire contract projected to the V1
// FRAME_XYZI_REGISTERED capability.  normal_* and curvature are admitted only
// because the actual /cloud_registered producer serializes that PCL type; V1
// never treats them as semantic normals, health, return identity, or timing.
// The check runs before pcl::fromROSMsg so malformed/oversized input cannot
// allocate an unbounded PCL cloud or silently synthesize a missing XYZI field.
// Validate the exact in-repository FAST-LIO PointXYZINormal wire layout.  V1
// validates x/y/z/intensity as the FRAME_XYZI_REGISTERED wire capability.
// Canonical V1 detection consumes XYZ only: intensity is schema-checked and
// then normalized out because intensity_likelihood_usage is frozen DISABLED.
// The normal and curvature fields are transport padding/producer fields and
// are likewise not promoted to geometric or timing observations.
bool ValidateFastLioRegisteredCloudSchema(
    const sensor_msgs::PointCloud2& message, size_t maximum_points,
    ReasonCode* reason, std::string* error);

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_INPUT_CONTRACT_HPP
