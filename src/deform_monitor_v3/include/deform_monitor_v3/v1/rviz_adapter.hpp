#ifndef DEFORM_MONITOR_V3_V1_RVIZ_ADAPTER_HPP
#define DEFORM_MONITOR_V3_V1_RVIZ_ADAPTER_HPP

#include "deform_monitor_v3/V1DetectionTile.h"
#include "deform_monitor_v3/V1DetectionStatus.h"

#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>

#include <string>

namespace deform_monitor_v3 {
namespace v1 {

// These functions are a presentation adapter only.  They never feed data back
// into the estimator and never alter the probabilities or physical states.
// Deformation samples that pass the display-length and optional A+C probability
// lower-bound gates are packed losslessly into one LINE_LIST marker (shaft plus
// two head wings per input sample). Neither gate is a detection decision, and
// the visualization marker count is constant even for a full LOD-2 tile.
visualization_msgs::MarkerArray BuildV1DeformationMarkers(
    const deform_monitor_v3::V1DetectionTile& tile,
    double displacement_display_scale,
    double minimum_display_arrow_m,
    double maximum_display_arrow_m,
    double minimum_arrow_active_probability = 0.0);

visualization_msgs::MarkerArray BuildV1StatusMarkers(
    const deform_monitor_v3::V1DetectionStatus& status,
    const geometry_msgs::Point& position,
    double text_height_m,
    double displacement_display_scale,
    const std::string& detail_text = "");

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_RVIZ_ADAPTER_HPP
