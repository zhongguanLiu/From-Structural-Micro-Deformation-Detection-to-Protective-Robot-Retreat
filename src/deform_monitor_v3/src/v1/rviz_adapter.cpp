#include "deform_monitor_v3/v1/rviz_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace deform_monitor_v3 {
namespace v1 {
namespace {

std_msgs::ColorRGBA ProbabilityColor(double probability, double alpha) {
  const double p = std::max(0.0, std::min(1.0, probability));
  std_msgs::ColorRGBA color;
  color.r = static_cast<float>(p);
  color.g = static_cast<float>(1.0 - p);
  color.b = static_cast<float>(0.15 * (1.0 - p));
  color.a = static_cast<float>(alpha);
  return color;
}

visualization_msgs::Marker DeleteAll(const std_msgs::Header& header) {
  visualization_msgs::Marker marker;
  marker.header = header;
  marker.action = visualization_msgs::Marker::DELETEALL;
  return marker;
}

void SetUnitPose(visualization_msgs::Marker* marker) {
  if (marker) marker->pose.orientation.w = 1.0;
}

geometry_msgs::Point AddScaled(const geometry_msgs::Point& point,
                               double x,
                               double y,
                               double z,
                               double scale) {
  geometry_msgs::Point result = point;
  result.x += scale * x;
  result.y += scale * y;
  result.z += scale * z;
  return result;
}

// Append one deterministic line arrow as three segments: shaft, left wing and
// right wing.  Packing every deformation vector into one LINE_LIST avoids one
// RViz/Ogre object (and one ROS Marker message) per physical sample without
// decimating or otherwise changing the estimator output.
void AppendLineArrow(const geometry_msgs::Point& start,
                     double ux,
                     double uy,
                     double uz,
                     double display_length,
                     const std_msgs::ColorRGBA& color,
                     visualization_msgs::Marker* marker) {
  if (!marker) return;

  const geometry_msgs::Point tip =
      AddScaled(start, ux, uy, uz, display_length);

  // Cross the direction with the least-aligned coordinate axis.  This makes
  // the wing orientation deterministic and keeps the cross product away from
  // its singularity for every non-zero direction.
  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;
  const double ax = std::abs(ux);
  const double ay = std::abs(uy);
  const double az = std::abs(uz);
  if (ax <= ay && ax <= az) {
    // u x (1, 0, 0)
    py = uz;
    pz = -uy;
  } else if (ay <= az) {
    // u x (0, 1, 0)
    px = -uz;
    pz = ux;
  } else {
    // u x (0, 0, 1)
    px = uy;
    py = -ux;
  }
  const double perpendicular_norm = std::sqrt(px * px + py * py + pz * pz);
  px /= perpendicular_norm;
  py /= perpendicular_norm;
  pz /= perpendicular_norm;

  // Match the former compact arrow appearance: proportional heads for the
  // shortest visible vectors and a 4 cm cap for long display-scaled vectors.
  const double head_length = std::min(0.04, 0.35 * display_length);
  const double half_width = 0.55 * head_length;
  geometry_msgs::Point head_base = AddScaled(tip, ux, uy, uz, -head_length);
  const geometry_msgs::Point left =
      AddScaled(head_base, px, py, pz, half_width);
  const geometry_msgs::Point right =
      AddScaled(head_base, px, py, pz, -half_width);

  marker->points.push_back(start);
  marker->points.push_back(tip);
  marker->points.push_back(tip);
  marker->points.push_back(left);
  marker->points.push_back(tip);
  marker->points.push_back(right);
  marker->colors.insert(marker->colors.end(), 6U, color);
}

std::string PhaseName(uint8_t phase) {
  switch (phase) {
    case 0U: return "SLAM MAPPING - DETECTION INACTIVE";
    case 1U: return "BUILDING DEFORMATION REFERENCE";
    case 2U: return "MICRO-DEFORMATION MONITORING ACTIVE";
    case 3U: return "DEFORMATION MONITOR CLOSED";
    default: return "INVALID";
  }
}

}  // namespace

visualization_msgs::MarkerArray BuildV1DeformationMarkers(
    const deform_monitor_v3::V1DetectionTile& tile,
    double displacement_display_scale,
    double minimum_display_arrow_m,
    double maximum_display_arrow_m,
    double minimum_arrow_active_probability) {
  visualization_msgs::MarkerArray output;
  output.markers.reserve(4U);
  output.markers.push_back(DeleteAll(tile.common.header));

  visualization_msgs::Marker reference;
  reference.header = tile.common.header;
  reference.ns = "reference_support";
  reference.id = 0;
  reference.type = visualization_msgs::Marker::SPHERE_LIST;
  reference.action = visualization_msgs::Marker::ADD;
  SetUnitPose(&reference);
  reference.scale.x = 0.035;
  reference.scale.y = 0.035;
  reference.scale.z = 0.035;
  reference.color.r = 0.1F;
  reference.color.g = 0.75F;
  reference.color.b = 1.0F;
  reference.color.a = 0.28F;
  reference.points.reserve(tile.reference_support.size());
  for (const auto& support : tile.reference_support)
    reference.points.push_back(support.position);
  output.markers.push_back(reference);

  visualization_msgs::Marker detection;
  detection.header = tile.common.header;
  detection.ns = "detection_probability";
  detection.id = 0;
  detection.type = visualization_msgs::Marker::SPHERE_LIST;
  detection.action = visualization_msgs::Marker::ADD;
  SetUnitPose(&detection);
  detection.scale.x = 0.09;
  detection.scale.y = 0.09;
  detection.scale.z = 0.09;
  detection.points.reserve(tile.detection_evidence.size());
  detection.colors.reserve(tile.detection_evidence.size());
  for (const auto& evidence : tile.detection_evidence) {
    detection.points.push_back(evidence.position);
    const double probability = evidence.task_scale_probability_mean_identified
        ? evidence.task_scale_probability_mean
        : evidence.task_scale_probability_upper;
    detection.colors.push_back(ProbabilityColor(probability, 0.85));
  }
  output.markers.push_back(detection);

  visualization_msgs::Marker displacement;
  displacement.header = tile.common.header;
  displacement.ns = "scaled_displacement";
  displacement.id = 0;
  displacement.type = visualization_msgs::Marker::LINE_LIST;
  displacement.action = visualization_msgs::Marker::ADD;
  SetUnitPose(&displacement);
  displacement.scale.x = 0.012;
  displacement.points.reserve(6U * tile.physical_deformation.size());
  displacement.colors.reserve(6U * tile.physical_deformation.size());

  const double display_scale = std::max(0.0, displacement_display_scale);
  const double minimum_length = std::max(0.0, minimum_display_arrow_m);
  const double maximum_length = std::max(minimum_length, maximum_display_arrow_m);
  for (const auto& sample : tile.physical_deformation) {
    // Unknown probability mass must not qualify a vector through its upper
    // bound. This optional presentation gate never alters detector messages.
    if (minimum_arrow_active_probability > 0.0 &&
        (!std::isfinite(sample.active_probability_lower) ||
         sample.active_probability_lower < minimum_arrow_active_probability ||
         sample.active_probability_lower > 1.0)) continue;
    const double dx = sample.displacement_mean.x;
    const double dy = sample.displacement_mean.y;
    const double dz = sample.displacement_mean.z;
    const double physical_norm = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!std::isfinite(physical_norm) || physical_norm <= 0.0) continue;
    double display_length = display_scale * physical_norm;
    if (display_length < minimum_length) continue;
    display_length = std::min(display_length, maximum_length);
    const double factor = display_length / physical_norm;

    const double probability = sample.active_probability_mean_identified
        ? sample.active_probability_mean : sample.active_probability_upper;
    AppendLineArrow(sample.position, dx / physical_norm, dy / physical_norm,
                    dz / physical_norm, factor * physical_norm,
                    ProbabilityColor(probability, 0.95), &displacement);
  }
  output.markers.push_back(displacement);
  return output;
}

visualization_msgs::MarkerArray BuildV1StatusMarkers(
    const deform_monitor_v3::V1DetectionStatus& status,
    const geometry_msgs::Point& position,
    double text_height_m,
    double displacement_display_scale,
    const std::string& detail_text) {
  visualization_msgs::MarkerArray output;
  output.markers.push_back(DeleteAll(status.common.header));
  visualization_msgs::Marker marker;
  marker.header = status.common.header;
  marker.ns = "monitor_status";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::Marker::ADD;
  SetUnitPose(&marker);
  marker.pose.position = position;
  marker.scale.z = std::max(0.1, text_height_m);
  marker.color.a = 1.0F;
  if (status.phase == 2U) {
    marker.color.r = 0.2F;
    marker.color.g = 1.0F;
    marker.color.b = 0.25F;
  } else {
    marker.color.r = 1.0F;
    marker.color.g = status.phase == 1U ? 0.75F : 0.25F;
    marker.color.b = 0.1F;
  }
  std::ostringstream text;
  text << "V1 " << PhaseName(status.phase)
       << " | reference=" << (status.reference_frozen ? "FROZEN" : "WAIT")
       << " | GDU=" << status.gdu_count
       << " | published=" << (status.last_attempt_published ? "YES" : "NO")
       << " | displacement x" << std::fixed << std::setprecision(0)
       << displacement_display_scale;
  if (!detail_text.empty()) text << '\n' << detail_text;
  if (!status.last_attempt_reason_codes.empty()) {
    text << " | reason=";
    for (size_t index = 0; index < status.last_attempt_reason_codes.size(); ++index) {
      if (index > 0U) text << ',';
      text << status.last_attempt_reason_codes[index];
    }
  }
  marker.text = text.str();
  output.markers.push_back(marker);
  return output;
}

}  // namespace v1
}  // namespace deform_monitor_v3
