#include "deform_monitor_v3/v1/input_contract.hpp"

#include <sensor_msgs/PointField.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>

namespace deform_monitor_v3 {
namespace v1 {

bool ValidateFastLioRegisteredCloudSchema(
    const sensor_msgs::PointCloud2& message, size_t maximum_points,
    ReasonCode* reason, std::string* error) {
  const auto reject = [&](ReasonCode code, const std::string& detail) {
    if (reason) *reason = code;
    if (error) *error = detail;
    return false;
  };
  if (reason) *reason = ReasonCode::INPUT_SCHEMA_INVALID;
  if (maximum_points == 0U)
    return reject(ReasonCode::RESOURCE_LIMIT,
                  "max_points_per_frame must be positive");
  if (message.width == 0U || message.height == 0U)
    return reject(ReasonCode::INPUT_SCHEMA_INVALID,
                  "PointCloud2 width and height must be nonzero");
  const uint64_t point_count = static_cast<uint64_t>(message.width) *
                               static_cast<uint64_t>(message.height);
  if (point_count > static_cast<uint64_t>(maximum_points))
    return reject(ReasonCode::RESOURCE_LIMIT,
                  "PointCloud2 exceeds max_points_per_frame before conversion");
  if (message.is_bigendian)
    return reject(ReasonCode::INPUT_SCHEMA_INVALID,
                  "FRAME_XYZI_REGISTERED accepts little-endian PointCloud2 only");
  // The in-repository FAST-LIO publisher uses pcl::PointXYZINormal:
  // xyz@0/4/8, normal@16/20/24, intensity@32, curvature@36, sizeof=48.
  // Fixing the layout also bounds payload bytes by 48*maximum_points.
  if (message.point_step != 48U)
    return reject(ReasonCode::INPUT_SCHEMA_INVALID,
                  "FAST-LIO PointXYZINormal point_step must equal 48 bytes");
  const uint64_t packed_row = static_cast<uint64_t>(message.width) *
                              static_cast<uint64_t>(message.point_step);
  if (packed_row > std::numeric_limits<uint32_t>::max() ||
      message.row_step != packed_row)
    return reject(ReasonCode::INPUT_SCHEMA_INVALID,
                  "PointCloud2 rows must be tightly packed");
  const uint64_t expected_bytes = static_cast<uint64_t>(message.height) *
                                  static_cast<uint64_t>(message.row_step);
  if (expected_bytes != static_cast<uint64_t>(message.data.size()))
    return reject(ReasonCode::INPUT_SCHEMA_INVALID,
                  "PointCloud2 data size is inconsistent with row_step");

  std::map<std::string, sensor_msgs::PointField> fields;
  for (const sensor_msgs::PointField& field : message.fields) {
    if (fields.count(field.name) > 0U)
      return reject(ReasonCode::INPUT_SCHEMA_INVALID,
                    "PointCloud2 contains duplicate field names");
    fields[field.name] = field;
  }
  const std::map<std::string, uint32_t> required{
      {"x", 0U}, {"y", 4U}, {"z", 8U}, {"normal_x", 16U},
      {"normal_y", 20U}, {"normal_z", 24U}, {"intensity", 32U},
      {"curvature", 36U}};
  if (fields.size() != required.size())
    return reject(ReasonCode::INPUT_SCHEMA_INVALID,
                  "FAST-LIO input requires the exact PointXYZINormal field set");
  std::vector<std::pair<uint32_t, uint32_t>> occupied_ranges;
  for (const auto& expected : required) {
    const auto found = fields.find(expected.first);
    if (found == fields.end() ||
        found->second.datatype != sensor_msgs::PointField::FLOAT32 ||
        found->second.count != 1U ||
        found->second.offset != expected.second)
      return reject(ReasonCode::INPUT_SCHEMA_INVALID,
                    std::string("PointCloud2 field ") + expected.first +
                        " has the wrong FLOAT32 layout");
    const uint64_t end = static_cast<uint64_t>(found->second.offset) + 4U;
    if (end > message.point_step)
      return reject(ReasonCode::INPUT_SCHEMA_INVALID,
                    std::string("PointCloud2 field ") + expected.first +
                        " exceeds point_step");
    occupied_ranges.emplace_back(found->second.offset,
                                 static_cast<uint32_t>(end));
  }
  std::sort(occupied_ranges.begin(), occupied_ranges.end());
  for (size_t index = 1U; index < occupied_ranges.size(); ++index)
    if (occupied_ranges[index].first < occupied_ranges[index - 1U].second)
      return reject(ReasonCode::INPUT_SCHEMA_INVALID,
                    "PointCloud2 FAST-LIO fields overlap");
  if (reason) *reason = ReasonCode::NONE;
  if (error) error->clear();
  return true;
}

}  // namespace v1
}  // namespace deform_monitor_v3
